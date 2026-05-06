"""Speaker diarization via pyannote (per finalized segment)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Very short clips are unreliable for clustering; skip diarization and save GPU/CPU.
_MIN_SECONDS = 0.45


def _dominant_speaker_exclusive(annotation: Any, *, t0: float, t1: float) -> str | None:
    """Pick the exclusive-diarization label with the most overlap in [t0, t1]."""
    best_label: str | None = None
    best_overlap = 0.0
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        overlap = max(0.0, min(segment.end, t1) - max(segment.start, t0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = speaker
    return best_label


class DiarizationService:
    """Lazy-loaded pyannote pipeline; calls are serialized for thread safety."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipeline: Any | None = None
        self._load_error: str | None = None
        self._lock = asyncio.Lock()

    def _sync_load(self) -> Any:
        from pyannote.audio import Pipeline

        token = self._settings.resolved_hf_token()
        if not token:
            raise RuntimeError(
                "Hugging Face token is not configured (set ASR_HF_TOKEN or HF_TOKEN)."
            )
        return Pipeline.from_pretrained(
            self._settings.diarization_model_id,
            token=token,
        )

    def _sync_label(self, pipeline: Any, audio_f32: np.ndarray, sample_rate: int) -> str | None:
        from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

        if audio_f32.size < int(_MIN_SECONDS * sample_rate):
            return None

        wave = torch.from_numpy(np.ascontiguousarray(audio_f32, dtype=np.float32))
        wave = wave.unsqueeze(0)
        file_dict: dict[str, Any] = {"waveform": wave, "sample_rate": sample_rate}

        with torch.inference_mode():
            output = pipeline(file_dict, min_speakers=1, max_speakers=8)

        if isinstance(output, DiarizeOutput):
            ann = output.exclusive_speaker_diarization
        else:
            ann = output

        duration = len(audio_f32) / float(sample_rate)
        return _dominant_speaker_exclusive(ann, t0=0.0, t1=duration)

    async def label_segment(self, audio_f32: np.ndarray, sample_rate: int) -> str | None:
        """Return a speaker label for this mono float32 segment, or None."""
        if audio_f32.size == 0:
            return None
        if self._load_error is not None:
            return None

        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._pipeline is None:
                try:
                    self._pipeline = await loop.run_in_executor(None, self._sync_load)
                    log.info(
                        "diarization_pipeline_ready",
                        extra={"model_id": self._settings.diarization_model_id},
                    )
                except Exception:  # noqa: BLE001
                    self._load_error = "load_failed"
                    log.exception(
                        "diarization_pipeline_load_failed",
                        extra={"model_id": self._settings.diarization_model_id},
                    )
                    return None

            try:
                return await loop.run_in_executor(
                    None,
                    self._sync_label,
                    self._pipeline,
                    audio_f32,
                    sample_rate,
                )
            except Exception:  # noqa: BLE001
                log.exception("diarization_inference_failed")
                return None
