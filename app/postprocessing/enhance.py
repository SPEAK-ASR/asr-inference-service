"""Noise reduction / enhancement hooks (placeholder)."""

from __future__ import annotations

import numpy as np


def apply_noise_removal(audio: np.ndarray, *, enabled: bool) -> np.ndarray:
    """Return enhanced mono float32 audio.

    When ``enabled`` is True, this is currently a no-op; reserved for a future
    denoiser so the streaming path already branches correctly.
    """
    if not enabled or audio.size == 0:
        return audio
    return audio
