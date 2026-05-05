"""FastAPI entrypoint for the realtime ASR backend.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                 -> root summary
    GET  /health/live      -> liveness (process up)
    GET  /health/ready     -> readiness (model loaded, device available)
    WS   /ws/transcribe    -> realtime transcription
    GET  /client           -> manual mic test page (tests/manual/client.html)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.ws_transcribe import router as ws_router
from app.asr.model_loader import load_asr
from app.asr.streaming_engine import StreamingEngine
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.sessions.manager import SessionManager

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ASR model and start the session manager on startup."""
    settings = get_settings()
    log.info("app_startup", extra={"model_id": settings.model_id})

    asr = load_asr(settings)
    engine = StreamingEngine(asr=asr, settings=settings)
    sessions = SessionManager(settings=settings)
    await sessions.start()

    app.state.settings = settings
    app.state.asr = asr
    app.state.engine = engine
    app.state.sessions = sessions

    log.info(
        "app_ready",
        extra={
            "device": asr.device,
            "dtype": asr.dtype,
            "sample_rate": asr.sample_rate,
        },
    )
    try:
        yield
    finally:
        log.info("app_shutdown_started")
        await sessions.stop()
        log.info("app_shutdown_complete")


app = FastAPI(
    title="Realtime ASR Backend (Sinhala-First)",
    version="0.1.0",
    description="WebSocket transcription gateway powered by SPEAK-ASR/whisper-si-exp-10-medium-all.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict:
    settings = get_settings()
    return {
        "service": "realtime-asr-backend",
        "model": settings.model_id,
        "ws_endpoint": "/ws/transcribe",
        "test_client": "/client",
    }


@app.get("/health/live")
async def health_live() -> dict:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    asr = getattr(app.state, "asr", None)
    sessions = getattr(app.state, "sessions", None)
    ready = asr is not None and sessions is not None
    payload = {
        "status": "ready" if ready else "not_ready",
        "model_id": asr.model_id if asr else None,
        "device": asr.device if asr else None,
        "active_sessions": sessions.active_count() if sessions else 0,
    }
    return JSONResponse(content=payload, status_code=200 if ready else 503)


app.include_router(ws_router)


# ---------------------------------------------------------------------------
# Static test client
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLIENT_DIR = _PROJECT_ROOT / "tests" / "manual"
_CLIENT_HTML = _CLIENT_DIR / "client.html"

if _CLIENT_DIR.exists():
    app.mount(
        "/client-static",
        StaticFiles(directory=str(_CLIENT_DIR)),
        name="client-static",
    )


@app.get("/client", include_in_schema=False)
async def client_page() -> FileResponse:
    if not _CLIENT_HTML.exists():
        return JSONResponse(
            content={"detail": "Test client not found. See tests/manual/client.html."},
            status_code=404,
        )
    return FileResponse(_CLIENT_HTML, media_type="text/html")
