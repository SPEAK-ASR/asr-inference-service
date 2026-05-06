"""Admin REST endpoints for runtime configuration.

Current endpoints:

    GET  /admin/diarization  -> returns current diarization status
    POST /admin/diarization  -> toggle diarization on or off at runtime

The pyannote pipeline is loaded lazily on the first enable request and kept
in memory for fast re-enabling.  Disabling only flips the flag; the model
is not unloaded.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.sessions.schemas import DiarizationStatus, DiarizationToggle

log = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/diarization", response_model=DiarizationStatus)
async def get_diarization_status(request: Request) -> DiarizationStatus:
    """Return whether diarization is currently enabled and whether the model is loaded."""
    return DiarizationStatus(
        enabled=request.app.state.diarization_enabled,
        model_loaded=request.app.state.diarizer is not None,
    )


@router.post("/diarization", response_model=DiarizationStatus)
async def set_diarization(request: Request, body: DiarizationToggle) -> JSONResponse:
    """Enable or disable speaker diarization for all new sessions.

    - Enabling triggers a lazy load of the pyannote pipeline (first call only).
    - Disabling flips the flag immediately; the loaded model stays in memory.
    - Sessions already in progress keep the diarization state they started with.
    """
    app = request.app

    if body.enabled and app.state.diarizer is None:
        settings = get_settings()
        if not settings.diarization_hf_token:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": (
                        "ASR_DIARIZATION_HF_TOKEN is not configured. "
                        "Set it in your .env file or as an environment variable "
                        "and restart the server, then try again."
                    )
                },
            )

        try:
            from app.asr.diarizer import load_diarizer

            loop = asyncio.get_running_loop()
            diarizer = await loop.run_in_executor(None, load_diarizer, settings)
            app.state.diarizer = diarizer
            log.info("diarizer_lazy_loaded")
        except Exception as exc:  # noqa: BLE001
            log.exception("diarizer_load_failed")
            return JSONResponse(
                status_code=503,
                content={"detail": f"Failed to load diarization model: {exc}"},
            )

    app.state.diarization_enabled = body.enabled
    log.info("diarization_toggled", extra={"enabled": body.enabled})

    return JSONResponse(
        content=DiarizationStatus(
            enabled=app.state.diarization_enabled,
            model_loaded=app.state.diarizer is not None,
        ).model_dump()
    )
