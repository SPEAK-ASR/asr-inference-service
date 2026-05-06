"""WebSocket gateway for `/ws/transcribe`.

Contract for mobile/other clients: ``docs/websocket-protocol.md`` and
``docs/flutter-websocket-guide.md``.
Summary: text JSON only; PCM16 mono at ``ASR_TARGET_SAMPLE_RATE`` (default 16 kHz)
inside ``audio_chunk.audio_b64`` (Base64).

Flow: ``start`` (first message), then repeated ``audio_chunk``, optional ``end_utterance``,
``ping``, ``stop``. Partial/final hypotheses use ``partial_transcript`` /
``final_transcript`` (not a single ``isFinal`` flag).

Investigated-detail background: section 6 of ``investigated_detail.md``.

The route may run a background partial flush (``sliding_window`` mode) or use
Silero VAD + silence-triggered segments (``vad`` mode, default). Final decodes
run when a VAD segment completes, or on ``end_utterance`` / ``stop``, or after
each partial interval in sliding-window mode.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.asr.streaming_engine import StreamingEngine
from app.asr.vad import flush_segment, get_vad_model, process_stream_chunk
from app.core.config import get_settings
from app.core.logging import get_logger
from app.sessions.manager import SessionManager, SessionState
from app.sessions.schemas import (
    Ack,
    AudioChunkMsg,
    ClientMessage,
    EndUtteranceMsg,
    Error,
    ErrorCode,
    FinalTranscript,
    PartialTranscript,
    PingMsg,
    SessionSummary,
    StartMsg,
    StopMsg,
    Warning,
)

log = get_logger(__name__)
router = APIRouter()

_client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def _engine(ws: WebSocket) -> StreamingEngine:
    return ws.app.state.engine  # type: ignore[no-any-return]


def _manager(ws: WebSocket) -> SessionManager:
    return ws.app.state.sessions  # type: ignore[no-any-return]


def _postprocessing_runtime(ws: WebSocket) -> Any:
    return ws.app.state.postprocessing  # type: ignore[no-any-return]


def _prepare_segment_audio(state: SessionState, segment_audio: np.ndarray) -> np.ndarray:
    from app.postprocessing.enhance import apply_noise_removal

    arr = np.ascontiguousarray(segment_audio, dtype=np.float32)
    return apply_noise_removal(arr, enabled=state.noise_removal_enabled)


async def _speaker_label(ws: WebSocket, state: SessionState, audio: np.ndarray) -> str | None:
    if not state.diarization_enabled:
        return None
    svc = getattr(ws.app.state, "diarization", None)
    if svc is None:
        return None
    return await svc.label_segment(audio, get_settings().target_sample_rate)


async def _send_json(ws: WebSocket, payload: Any) -> None:
    """Serialize a Pydantic model (or dict) and send it as JSON text."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload
    try:
        await ws.send_json(data)
    except Exception:  # noqa: BLE001 - socket may be already closed
        log.debug("send_json_failed", extra={"payload_type": data.get("type")})


async def _recv_validated(ws: WebSocket) -> ClientMessage | None:
    """Receive a single client message, validating against the schema."""
    raw = await ws.receive_text()
    try:
        return _client_message_adapter.validate_json(raw)
    except ValidationError as exc:
        log.warning("invalid_client_message", extra={"errors": exc.errors()[:3]})
        await _send_json(
            ws,
            Error(
                code=ErrorCode.PROTOCOL_ERROR,
                message="Invalid message; check schema.",
            ),
        )
        return None


@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket) -> None:
    settings = get_settings()
    manager = _manager(ws)
    engine = _engine(ws)

    await ws.accept()
    log.info("ws_connected", extra={"client": str(ws.client)})

    state: SessionState | None = None
    partial_task: asyncio.Task[None] | None = None
    close_reason = "client_close"

    try:
        first = await _recv_validated(ws)
        if first is None:
            return
        if not isinstance(first, StartMsg):
            await _send_json(
                ws,
                Error(
                    code=ErrorCode.PROTOCOL_ERROR,
                    message="First message must be 'start'.",
                ),
            )
            return

        if first.sample_rate != settings.target_sample_rate:
            await _send_json(
                ws,
                Error(
                    code=ErrorCode.INVALID_AUDIO_FORMAT,
                    message=(
                        f"Expected sample_rate={settings.target_sample_rate}, "
                        f"got {first.sample_rate}."
                    ),
                ),
            )
            return

        state = SessionState(
            session_id=first.session_id,
            sample_rate=first.sample_rate,
            language_hint=first.language_hint or settings.language_hint,
        )
        defaults = await _postprocessing_runtime(ws).snapshot()
        pp_opts = first.postprocessing
        state.diarization_enabled = (
            pp_opts.diarization
            if pp_opts is not None and pp_opts.diarization is not None
            else defaults["diarization_enabled"]
        )
        state.noise_removal_enabled = (
            pp_opts.noise_removal
            if pp_opts is not None and pp_opts.noise_removal is not None
            else defaults["noise_removal_enabled"]
        )

        async def close_cb(reason: str) -> None:
            nonlocal close_reason
            close_reason = reason
            try:
                await _send_json(
                    ws,
                    Error(
                        code=ErrorCode.SESSION_TIMEOUT
                        if reason == "idle_timeout"
                        else ErrorCode.INTERNAL_ERROR,
                        message=f"Session closed: {reason}",
                    ),
                )
                await ws.close(code=1001)
            except Exception:  # noqa: BLE001
                pass

        state.close_callback = close_cb
        await manager.register(state)
        await _send_json(
            ws,
            Ack(session_id=state.session_id, message="stream_started"),
        )
        if state.noise_removal_enabled:
            await _send_json(
                ws,
                Warning(
                    code="NOISE_REMOVAL_NOT_IMPLEMENTED",
                    message="noise_removal is enabled but not implemented yet; audio is unchanged.",
                ),
            )
        if state.diarization_enabled and not settings.resolved_hf_token():
            await _send_json(
                ws,
                Warning(
                    code="DIARIZATION_NO_HF_TOKEN",
                    message=(
                        "diarization is enabled but no Hugging Face token is configured "
                        "(ASR_HF_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN); "
                        "final_transcript speaker fields will be null."
                    ),
                ),
            )

        if settings.streaming_mode == "sliding_window":
            partial_task = asyncio.create_task(
                _partial_flush_loop(ws, engine, state, settings.partial_interval_ms)
            )

        while True:
            msg = await _recv_validated(ws)
            if msg is None:
                continue
            if isinstance(msg, PingMsg):
                await _send_json(ws, Ack(session_id=state.session_id, message="pong"))
                continue
            if isinstance(msg, AudioChunkMsg):
                await _handle_audio_chunk(ws, engine, state, msg)
                continue
            if isinstance(msg, EndUtteranceMsg):
                await _finalize_utterance(ws, engine, state)
                continue
            if isinstance(msg, StopMsg):
                await _finalize_utterance(ws, engine, state)
                close_reason = "client_stop"
                break

    except WebSocketDisconnect:
        close_reason = "client_disconnect"
        log.info(
            "ws_disconnected",
            extra={"session_id": state.session_id if state else None},
        )
    except Exception:  # noqa: BLE001
        close_reason = "internal_error"
        log.exception("ws_unhandled_error")
        if state is not None:
            await _send_json(
                ws,
                Error(code=ErrorCode.INTERNAL_ERROR, message="Internal error."),
            )
    finally:
        if partial_task and not partial_task.done():
            partial_task.cancel()
            try:
                await partial_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if state is not None:
            await _send_json(
                ws,
                SessionSummary(
                    session_id=state.session_id,
                    utterances=state.utterances_finalized,
                    duration_ms=state.duration_ms(),
                    reason=close_reason,
                ),
            )
            await manager.unregister(state.session_id)

        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _emit_final_for_segment(
    ws: WebSocket,
    engine: StreamingEngine,
    state: SessionState,
    segment_audio: np.ndarray,
) -> None:
    """Run full decode on a VAD segment and send ``final_transcript``."""
    processed = _prepare_segment_audio(state, segment_audio)
    try:
        text, decode_seconds = await engine.transcribe_audio_array(
            processed,
            language_hint=state.language_hint,
        )
    except Exception:  # noqa: BLE001
        log.exception("final_decode_failed", extra={"session_id": state.session_id})
        await _send_json(
            ws,
            Error(code=ErrorCode.INTERNAL_ERROR, message="Final decode failed."),
        )
        return

    spk: str | None = None
    if state.diarization_enabled:
        spk = await _speaker_label(ws, state, processed)
        if spk:
            state.last_speaker = spk

    final_text = state.decoder.finalize(text)
    end_ms = state.duration_ms()
    sample_rate = state.sample_rate
    start_ms = max(0, end_ms - int(len(processed) / float(sample_rate) * 1000))
    utterance_id = state.utterance_id

    await _send_json(
        ws,
        FinalTranscript(
            session_id=state.session_id,
            utterance_id=utterance_id,
            text=final_text,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=spk,
        ),
    )
    log.info(
        "final_emitted",
        extra={
            "session_id": state.session_id,
            "utterance_id": utterance_id,
            "decode_ms": int(decode_seconds * 1000),
            "text_len": len(final_text),
        },
    )

    state.utterances_finalized += 1
    state.new_utterance()


async def _handle_audio_chunk(
    ws: WebSocket,
    engine: StreamingEngine,
    state: SessionState,
    msg: AudioChunkMsg,
) -> None:
    settings = get_settings()
    try:
        pcm = base64.b64decode(msg.audio_b64, validate=False)
    except Exception:  # noqa: BLE001
        await _send_json(
            ws,
            Error(code=ErrorCode.INVALID_AUDIO_FORMAT, message="Bad base64 in audio_chunk."),
        )
        return

    if len(pcm) > settings.max_chunk_bytes:
        await _send_json(
            ws,
            Error(
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                message=f"audio_chunk exceeds {settings.max_chunk_bytes} bytes.",
            ),
        )
        return

    samples = StreamingEngine.pcm16_bytes_to_float32(pcm)
    if samples.size == 0:
        return

    if settings.streaming_mode == "vad":
        vad = get_vad_model()
        segment = process_stream_chunk(
            samples,
            state.vad_segmenter,
            vad,
            sample_rate=settings.target_sample_rate,
            silence_trigger_sec=settings.silence_trigger_seconds,
            max_buffer_sec=settings.max_buffer_seconds,
            min_speech_sec=settings.min_speech_seconds,
            vad_threshold=settings.vad_threshold,
        )
        if segment is not None and segment.size > 0:
            await _emit_final_for_segment(ws, engine, state, segment)
    else:
        state.engine_buffer.append(samples)
        state.engine_buffer.trim_to(settings.max_buffer_seconds)

    state.last_audio_seq = msg.seq
    state.chunks_received += 1
    state.bytes_received += len(pcm)
    state.touch()


async def _partial_flush_loop(
    ws: WebSocket,
    engine: StreamingEngine,
    state: SessionState,
    interval_ms: int,
) -> None:
    """Background task: emit a partial transcript every `interval_ms`."""
    interval = max(0.05, interval_ms / 1000.0)
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

        if state.engine_buffer.duration_seconds <= 0:
            continue

        try:
            text, decode_seconds = await engine.transcribe_window(
                state.engine_buffer,
                language_hint=state.language_hint,
            )
        except Exception:  # noqa: BLE001
            log.exception("partial_decode_failed", extra={"session_id": state.session_id})
            continue

        if not text:
            continue

        emit = state.decoder.observe_hypothesis(text)
        if emit is None:
            continue

        now_ms = int(time.time() * 1000)
        end_ms = state.duration_ms()
        start_ms = max(0, end_ms - int(state.engine_buffer.duration_seconds * 1000))
        state.last_partial_emit_at = now_ms

        await _send_json(
            ws,
            PartialTranscript(
                session_id=state.session_id,
                utterance_id=state.utterance_id,
                seq=state.decoder.seq,
                text=emit.text,
                start_ms=start_ms,
                end_ms=end_ms,
                is_stable=emit.is_stable,
                speaker=state.last_speaker if state.diarization_enabled else None,
            ),
        )
        log.debug(
            "partial_emitted",
            extra={
                "session_id": state.session_id,
                "utterance_id": state.utterance_id,
                "decode_ms": int(decode_seconds * 1000),
                "text_len": len(emit.text),
            },
        )


async def _finalize_utterance(
    ws: WebSocket,
    engine: StreamingEngine,
    state: SessionState,
) -> None:
    settings = get_settings()
    if settings.streaming_mode == "vad":
        segment = flush_segment(
            state.vad_segmenter,
            sample_rate=settings.target_sample_rate,
            min_speech_sec=settings.min_speech_seconds,
            force=True,
        )
        if segment is not None and segment.size > 0:
            await _emit_final_for_segment(ws, engine, state, segment)
        else:
            state.new_utterance()
        return

    if state.engine_buffer.samples.size == 0:
        state.new_utterance()
        return

    raw = np.ascontiguousarray(state.engine_buffer.samples, dtype=np.float32)
    processed = _prepare_segment_audio(state, raw)

    try:
        text, decode_seconds = await engine.transcribe_audio_array(
            processed,
            language_hint=state.language_hint,
        )
    except Exception:  # noqa: BLE001
        log.exception("final_decode_failed", extra={"session_id": state.session_id})
        await _send_json(
            ws,
            Error(code=ErrorCode.INTERNAL_ERROR, message="Final decode failed."),
        )
        return

    spk: str | None = None
    if state.diarization_enabled:
        spk = await _speaker_label(ws, state, processed)
        if spk:
            state.last_speaker = spk

    final_text = state.decoder.finalize(text)
    end_ms = state.duration_ms()
    sample_rate = state.sample_rate
    start_ms = max(0, end_ms - int(len(processed) / float(sample_rate) * 1000))
    utterance_id = state.utterance_id

    await _send_json(
        ws,
        FinalTranscript(
            session_id=state.session_id,
            utterance_id=utterance_id,
            text=final_text,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker=spk,
        ),
    )
    log.info(
        "final_emitted",
        extra={
            "session_id": state.session_id,
            "utterance_id": utterance_id,
            "decode_ms": int(decode_seconds * 1000),
            "text_len": len(final_text),
        },
    )

    state.utterances_finalized += 1
    state.new_utterance()
