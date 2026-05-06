"""Silero VAD segmentation matching the Hugging Face Space realtime pattern.

Buffers audio until speech is detected, then keeps accumulating through a short
trailing silence window so Whisper sees natural phrase endings. A segment is
released when post-speech silence reaches ``silence_trigger_sec`` or the buffer
hits ``max_buffer_sec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from app.core.logging import get_logger

log = get_logger(__name__)

_vad_model: Any | None = None


def get_vad_model() -> Any:
    """Lazy singleton; loaded on first use or from app lifespan."""
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        log.info("vad_loading")
        _vad_model = load_silero_vad()
        _vad_model.eval()
        log.info("vad_ready")
    return _vad_model


def preload_vad() -> Any:
    """Eager load for startup / readiness."""
    return get_vad_model()


def chunk_has_speech(
    audio_chunk: np.ndarray,
    vad_model: Any,
    *,
    sampling_rate: int,
    threshold: float,
) -> bool:
    """Return True if VAD detects any speech in this chunk."""
    if len(audio_chunk) == 0:
        return False
    from silero_vad import get_speech_timestamps

    tensor = torch.from_numpy(np.ascontiguousarray(audio_chunk))
    timestamps = get_speech_timestamps(
        tensor,
        vad_model,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
    )
    return len(timestamps) > 0


@dataclass
class VadSegmenterState:
    buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    has_speech: bool = False
    silent_samples: int = 0


def reset_segment(state: VadSegmenterState) -> None:
    state.buffer = np.zeros(0, dtype=np.float32)
    state.has_speech = False
    state.silent_samples = 0


def process_stream_chunk(
    waveform: np.ndarray,
    state: VadSegmenterState,
    vad_model: Any,
    *,
    sample_rate: int,
    silence_trigger_sec: float,
    max_buffer_sec: float,
    min_speech_sec: float,
    vad_threshold: float,
) -> np.ndarray | None:
    """Feed one chunk of float32 mono audio. Return audio to transcribe or None.

    Mirrors ``stream_transcribe`` in the reference Gradio app: silence before
    any speech is dropped; after speech begins, trailing silence is kept in the
    buffer until the silence threshold or max duration fires a segment.
    """
    wf = np.ascontiguousarray(waveform, dtype=np.float32)
    speech_in_chunk = chunk_has_speech(
        wf,
        vad_model,
        sampling_rate=sample_rate,
        threshold=vad_threshold,
    )

    if speech_in_chunk:
        state.buffer = (
            np.concatenate([state.buffer, wf]) if state.buffer.size else wf.copy()
        )
        state.has_speech = True
        state.silent_samples = 0
    else:
        if state.has_speech:
            state.silent_samples += len(wf)
            state.buffer = (
                np.concatenate([state.buffer, wf])
                if state.buffer.size
                else wf.copy()
            )
        else:
            return None

    silent_sec = state.silent_samples / float(sample_rate)
    buffer_sec = len(state.buffer) / float(sample_rate)

    silence_triggered = state.has_speech and silent_sec >= silence_trigger_sec
    buffer_maxed = buffer_sec >= max_buffer_sec

    if not (silence_triggered or buffer_maxed):
        return None

    buf = state.buffer
    speech_sec = buffer_sec - silent_sec

    segment: np.ndarray | None = None
    if speech_sec >= min_speech_sec:
        segment = np.ascontiguousarray(buf, dtype=np.float32)

    reset_segment(state)
    return segment


def flush_segment(
    state: VadSegmenterState,
    *,
    sample_rate: int,
    min_speech_sec: float,
    force: bool,
) -> np.ndarray | None:
    """Return remaining buffered audio (e.g. ``end_utterance`` / ``stop``)."""
    if state.buffer.size == 0 or not state.has_speech:
        reset_segment(state)
        return None

    buf_sec = len(state.buffer) / float(sample_rate)
    if force or buf_sec >= min_speech_sec:
        segment = np.ascontiguousarray(state.buffer, dtype=np.float32)
        reset_segment(state)
        return segment

    reset_segment(state)
    return None
