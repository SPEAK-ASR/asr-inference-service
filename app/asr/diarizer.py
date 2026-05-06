"""Speaker diarization wrapper around pyannote.audio.

Loaded lazily on first use after diarization is enabled via
``POST /admin/diarization``.  The model is cached for the lifetime of the
process, so re-enabling diarization after a disable does not re-download
or re-load the pipeline.

Usage::

    diarizer = load_diarizer(settings)
    spans   = diarizer.diarize_sync(audio_float32, sample_rate=16_000)
    # spans: list of SpeakerSpan(speaker, start_sec, end_sec)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import Settings

log = get_logger(__name__)


@dataclass
class SpeakerSpan:
    """A contiguous audio region attributed to a single speaker."""

    speaker: str
    start_sec: float
    end_sec: float


class Diarizer:
    """Wraps the pyannote speaker-diarization-3.1 pipeline.

    Inference is always blocking (``diarize_sync``).  The caller is
    responsible for off-loading to an executor so the event loop is not
    blocked.  A dedicated ``asyncio.Lock`` serialises concurrent callers
    that share a single ``Diarizer`` instance.
    """

    def __init__(self, hf_token: str, device: str) -> None:
        from pyannote.audio import Pipeline  # imported lazily to keep startup fast
        import torch

        log.info("diarizer_loading", extra={"device": device})
        t0 = time.perf_counter()

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        pipeline.to(torch.device(device))

        self._pipeline = pipeline
        self._device = device
        self.lock = asyncio.Lock()

        elapsed = time.perf_counter() - t0
        log.info("diarizer_ready", extra={"device": device, "load_seconds": round(elapsed, 2)})

    def diarize_sync(self, audio: np.ndarray, sample_rate: int) -> list[SpeakerSpan]:
        """Run diarization on a float32 mono waveform.

        Args:
            audio: 1-D float32 array in [-1, 1] at ``sample_rate``.
            sample_rate: Sampling rate in Hz (must match the ASR engine rate).

        Returns:
            Ordered list of :class:`SpeakerSpan` objects covering the audio.
        """
        import torch

        if audio.size == 0:
            return []

        waveform = torch.from_numpy(audio).unsqueeze(0).float()  # (1, samples)
        input_dict = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self._pipeline(input_dict)

        spans: list[SpeakerSpan] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            spans.append(
                SpeakerSpan(
                    speaker=speaker,
                    start_sec=turn.start,
                    end_sec=turn.end,
                )
            )
        return spans


def load_diarizer(settings: Settings) -> Diarizer:
    """Instantiate and return a :class:`Diarizer` from *settings*.

    Raises:
        RuntimeError: If ``settings.diarization_hf_token`` is not set.
    """
    if not settings.diarization_hf_token:
        raise RuntimeError(
            "ASR_DIARIZATION_HF_TOKEN is not set. "
            "A HuggingFace token is required to download the gated "
            "pyannote/speaker-diarization-3.1 model. "
            "Set it in your .env file or as an environment variable."
        )

    import torch

    if settings.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device

    return Diarizer(hf_token=settings.diarization_hf_token, device=device)
