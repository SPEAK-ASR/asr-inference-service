"""HTTP endpoint for synchronous audio file transcription."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.api.schemas_http import TranscribeHttpResponse
from app.asr.audio_io import decode_uploaded_audio
from app.core.config import Settings
from app.core.logging import get_logger

router = APIRouter(prefix="/api", tags=["transcription"])
log = get_logger(__name__)


def _validate_upload(file: UploadFile, payload_size: int, settings: Settings) -> None:
    if payload_size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")
    if payload_size > settings.http_transcribe_max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio exceeds {settings.http_transcribe_max_upload_bytes} bytes.",
        )
    if file.content_type not in settings.http_transcribe_allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio content type: {file.content_type!r}.",
        )


@router.post("/transcribe", response_model=TranscribeHttpResponse)
async def transcribe_uploaded_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> TranscribeHttpResponse:
    settings: Settings = request.app.state.settings
    engine = request.app.state.engine

    payload = await audio_file.read()
    _validate_upload(audio_file, len(payload), settings)

    try:
        audio = decode_uploaded_audio(payload, target_sample_rate=settings.target_sample_rate)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "http_transcribe_audio_decode_failed",
            extra={"content_type": audio_file.content_type, "upload_filename": audio_file.filename, "error": str(exc)},
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or corrupt audio payload.") from exc

    if audio.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No decodable audio samples found.")

    language_hint = language or settings.language_hint
    t0 = time.perf_counter()
    try:
        text, _decode_seconds = await asyncio.wait_for(
            engine.transcribe_audio_array(audio, language_hint=language_hint),
            timeout=settings.http_transcribe_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription timed out. Try a shorter audio clip.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "http_transcribe_failed",
            extra={
                "upload_filename": audio_file.filename,
                "content_type": audio_file.content_type,
                "payload_size": len(payload),
            },
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transcription failed.") from exc

    duration_ms = int((time.perf_counter() - t0) * 1000)
    clean_text = (text or "").strip()
    log.info(
        "http_transcribe_success",
        extra={
            "upload_filename": audio_file.filename,
            "content_type": audio_file.content_type,
            "payload_size": len(payload),
            "duration_ms": duration_ms,
            "text_len": len(clean_text),
        },
    )
    return TranscribeHttpResponse(
        text=clean_text,
        language=language_hint,
        duration_ms=duration_ms,
        model_kind=request.app.state.asr.model_kind,
    )
