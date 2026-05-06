"""Audio decoding helpers for HTTP file uploads."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio with linear interpolation."""
    if src_rate == dst_rate or audio.size == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    duration = audio.shape[0] / float(src_rate)
    target_len = max(1, int(round(duration * dst_rate)))
    src_positions = np.arange(audio.shape[0], dtype=np.float64)
    dst_positions = np.linspace(0.0, max(0.0, audio.shape[0] - 1), num=target_len, dtype=np.float64)
    resampled = np.interp(dst_positions, src_positions, audio.astype(np.float64, copy=False))
    return np.ascontiguousarray(resampled.astype(np.float32), dtype=np.float32)


def decode_uploaded_audio(audio_bytes: bytes, target_sample_rate: int) -> np.ndarray:
    """Decode uploaded bytes into mono float32 audio at target sample rate."""
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)

    with sf.SoundFile(io.BytesIO(audio_bytes)) as snd:
        sample_rate = int(snd.samplerate)
        data = snd.read(dtype="float32", always_2d=True)

    if data.size == 0:
        return np.zeros(0, dtype=np.float32)

    mono = data.mean(axis=1, dtype=np.float32)
    mono = np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)
    return _resample_linear(mono, sample_rate, target_sample_rate)
