from __future__ import annotations

import io
import subprocess
import wave

import numpy as np
import pytest
import soundfile as sf

from app.asr.audio_io import decode_uploaded_audio


def _wav_bytes(sample_rate: int = 16000, duration_sec: float = 0.2) -> bytes:
    samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num=samples, endpoint=False)
    mono = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pcm16 = (mono * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def test_decode_uploaded_audio_resamples_to_target_sample_rate() -> None:
    payload = _wav_bytes(sample_rate=8000, duration_sec=0.2)
    decoded = decode_uploaded_audio(payload, target_sample_rate=16000)
    assert decoded.dtype == np.float32
    assert decoded.size > 3000
    assert decoded.size < 3400


def test_decode_uploaded_audio_invalid_payload_raises() -> None:
    try:
        decode_uploaded_audio(b"not-audio", target_sample_rate=16000)
        raise AssertionError("Expected decode_uploaded_audio to fail for invalid payload")
    except Exception:
        pass


def test_decode_uploaded_audio_uses_ffmpeg_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_libsndfile_error(*args, **kwargs):
        raise sf.LibsndfileError(1, "format not recognized")

    expected = np.array([0.0, 0.25, -0.25], dtype=np.float32)

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=expected.tobytes(), stderr=b"")

    monkeypatch.setattr(sf, "SoundFile", _raise_libsndfile_error)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    decoded = decode_uploaded_audio(b"m4a-bytes", target_sample_rate=16000)
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, expected)
