"""HTTP endpoint for synchronous audio file transcription."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.api.schemas_http import TranscribeHttpResponse
from app.asr.audio_io import decode_uploaded_audio
from app.core.config import Settings
from app.core.logging import get_logger

router = APIRouter(prefix="/api", tags=["transcription"])
log = get_logger(__name__)


def _content_type_allowed(content_type: str | None, allowed_patterns: list[str]) -> bool:
    """Match MIME types against exact values and wildcard patterns."""
    normalized = (content_type or "").strip().lower()
    if not normalized:
        return False

    for pattern in allowed_patterns:
        token = pattern.strip().lower()
        if not token:
            continue
        if token == "*/*":
            return True
        if token.endswith("/*"):
            prefix = token[:-1]
            if normalized.startswith(prefix):
                return True
        elif normalized == token:
            return True
    return False


def _build_upload_log_context(
    file: UploadFile,
    payload_size: int,
    language_hint: str | None,
) -> dict[str, str | int | float | None]:
    payload_size_mb = round(payload_size / (1024 * 1024), 3)
    return {
        "upload_filename": file.filename,
        "file_extension": Path(file.filename or "").suffix.lower() or None,
        "content_type": file.content_type,
        "payload_size_bytes": payload_size,
        "payload_size_mb": payload_size_mb,
        "language_hint": language_hint,
    }


def _validate_upload(
    file: UploadFile,
    payload_size: int,
    settings: Settings,
    log_context: dict[str, str | int | float | None],
) -> None:
    if payload_size == 0:
        log.warning(
            "http_transcribe_upload_rejected",
            extra={**log_context, "status": "error", "error_type": "empty_upload", "error_message": "Uploaded audio is empty."},
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")
    if payload_size > settings.http_transcribe_max_upload_bytes:
        limit_mb = round(settings.http_transcribe_max_upload_bytes / (1024 * 1024), 3)
        message = (
            f"Audio exceeds {settings.http_transcribe_max_upload_bytes} bytes "
            f"({limit_mb} MB)."
        )
        log.warning(
            "http_transcribe_upload_rejected",
            extra={**log_context, "status": "error", "error_type": "payload_too_large", "error_message": message},
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=message,
        )
    if not _content_type_allowed(file.content_type, settings.http_transcribe_allowed_mime_types):
        message = f"Unsupported audio content type: {file.content_type!r}."
        log.warning(
            "http_transcribe_upload_rejected",
            extra={**log_context, "status": "error", "error_type": "unsupported_content_type", "error_message": message},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
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
    language_hint = language or settings.language_hint
    log_context = _build_upload_log_context(audio_file, len(payload), language_hint)
    _validate_upload(audio_file, len(payload), settings, log_context)

    try:
        audio = decode_uploaded_audio(payload, target_sample_rate=settings.target_sample_rate)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "http_transcribe_audio_decode_failed",
            extra={
                **log_context,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or corrupt audio payload.") from exc

    if audio.size == 0:
        log.warning(
            "http_transcribe_audio_decode_failed",
            extra={
                **log_context,
                "status": "error",
                "error_type": "empty_decoded_audio",
                "error_message": "No decodable audio samples found.",
            },
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No decodable audio samples found.")

    t0 = time.perf_counter()
    try:
        text, _decode_seconds = await asyncio.wait_for(
            engine.transcribe_audio_array(audio, language_hint=language_hint),
            timeout=settings.http_transcribe_timeout_seconds,
        )
    except TimeoutError as exc:
        log.warning(
            "http_transcribe_failed",
            extra={
                **log_context,
                "status": "error",
                "error_type": "timeout",
                "error_message": "Transcription timed out.",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription timed out. Try a shorter audio clip.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "http_transcribe_failed",
            extra={
                **log_context,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Transcription failed.") from exc

    duration_ms = int((time.perf_counter() - t0) * 1000)
    clean_text = (text or "").strip()
    log.info(
        "http_transcribe_success",
        extra={
            **log_context,
            "status": "success",
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
