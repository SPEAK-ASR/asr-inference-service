from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.http_transcribe import router as http_transcribe_router


class _DummyEngine:
    async def transcribe_audio_array(self, audio: np.ndarray, *, language_hint: str | None):
        assert audio.dtype == np.float32
        assert audio.size > 0
        return "hello world", 0.01


def _wav_bytes(sample_rate: int = 16000, duration_sec: float = 0.1) -> bytes:
    samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num=samples, endpoint=False)
    mono = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    pcm16 = (mono * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(http_transcribe_router)
    app.state.settings = SimpleNamespace(
        target_sample_rate=16000,
        language_hint="si",
        http_transcribe_max_upload_bytes=1024 * 1024,
        http_transcribe_allowed_mime_types=["audio/wav", "audio/webm"],
        http_transcribe_timeout_seconds=10,
    )
    app.state.engine = _DummyEngine()
    app.state.asr = SimpleNamespace(model_kind="peft")
    return TestClient(app)


def test_http_transcribe_happy_path() -> None:
    client = _make_client()
    files = {"audio_file": ("clip.wav", _wav_bytes(), "audio/wav")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["model_kind"] == "peft"


def test_http_transcribe_missing_audio_file() -> None:
    client = _make_client()
    response = client.post("/api/transcribe")
    assert response.status_code == 422


def test_http_transcribe_unsupported_type() -> None:
    client = _make_client()
    files = {"audio_file": ("clip.wav", _wav_bytes(), "audio/mp3")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 400


def test_http_transcribe_oversize() -> None:
    client = _make_client()
    client.app.state.settings.http_transcribe_max_upload_bytes = 32
    files = {"audio_file": ("clip.wav", _wav_bytes(), "audio/wav")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 413
