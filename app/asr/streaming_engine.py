"""Streaming inference engine.

Per-session audio buffer + a single shared Whisper pipeline. Implements the
"buffered chunk + overlap" pattern from section 5 of `investigated_detail.md`.

The engine itself is stateless across sessions; per-session buffering lives on
`SessionState.engine_buffer` (held by `app.sessions.manager`). All concurrent
calls into the underlying Whisper pipeline are serialized via an asyncio Lock
because Whisper isn't safe to invoke in parallel on a single CUDA context.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.asr.model_loader import LoadedASR
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class EngineBuffer:
    """Mutable per-session audio buffer.

    Audio is stored as float32 mono in [-1, 1] at the engine sample rate.
    """

    sample_rate: int
    samples: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    utterance_start_ms: int = 0
    """Wall-clock-derived start timestamp (ms) of the active utterance."""
    samples_consumed: int = 0
    """Number of samples already finalized and dropped from `samples`."""

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def append(self, new: np.ndarray) -> None:
        if new.dtype != np.float32:
            new = new.astype(np.float32, copy=False)
        self.samples = np.concatenate([self.samples, new]) if self.samples.size else new

    def trim_to(self, max_seconds: float) -> None:
        max_samples = int(max_seconds * self.sample_rate)
        if len(self.samples) > max_samples:
            drop = len(self.samples) - max_samples
            self.samples = self.samples[drop:]
            self.samples_consumed += drop

    def reset_for_new_utterance(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.samples_consumed = 0


class StreamingEngine:
    """Wraps a `LoadedASR` (transformers or faster-whisper) and exposes async calls."""

    def __init__(self, asr: LoadedASR, settings: Settings | None = None) -> None:
        self.asr = asr
        self.settings = settings or get_settings()
        self._lock = asyncio.Lock()

    @property
    def sample_rate(self) -> int:
        return self.asr.sample_rate

    @staticmethod
    def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
        """Convert little-endian PCM16 bytes to float32 mono in [-1, 1]."""
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        if len(pcm) % 2:
            pcm = pcm[:-1]
        ints = np.frombuffer(pcm, dtype=np.int16)
        return (ints.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    async def transcribe_window(
        self,
        buffer: EngineBuffer,
        *,
        language_hint: str | None,
    ) -> tuple[str, float]:
        """Run inference on the trailing decode window.

        Returns ``(text, decode_seconds)``. Empty string if not enough audio.
        """
        if buffer.duration_seconds < self.settings.min_audio_for_partial_seconds:
            return "", 0.0

        window_samples = int(self.settings.decode_window_seconds * self.sample_rate)
        audio = buffer.samples[-window_samples:] if len(buffer.samples) > window_samples else buffer.samples
        audio_arr = np.ascontiguousarray(audio, dtype=np.float32)

        generate_kwargs: dict[str, Any] = {"task": self.settings.task}
        if language_hint:
            generate_kwargs["language"] = language_hint

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        async with self._lock:
            text = await loop.run_in_executor(
                None,
                self._run_pipeline_sync,
                audio_arr,
                generate_kwargs,
            )
        decode_seconds = time.perf_counter() - t0
        return (text or "").strip(), decode_seconds

    async def transcribe_full_utterance(
        self,
        buffer: EngineBuffer,
        *,
        language_hint: str | None,
    ) -> tuple[str, float]:
        """Final decode over the entire active utterance buffer."""
        if buffer.samples.size == 0:
            return "", 0.0

        audio_arr = np.ascontiguousarray(buffer.samples, dtype=np.float32)
        generate_kwargs: dict[str, Any] = {"task": self.settings.task}
        if language_hint:
            generate_kwargs["language"] = language_hint

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        async with self._lock:
            text = await loop.run_in_executor(
                None,
                self._run_pipeline_sync,
                audio_arr,
                generate_kwargs,
            )
        decode_seconds = time.perf_counter() - t0
        return (text or "").strip(), decode_seconds

    async def transcribe_audio_array(
        self,
        audio: np.ndarray,
        *,
        language_hint: str | None,
    ) -> tuple[str, float]:
        """Decode a standalone float32 mono segment (e.g. VAD output)."""
        if audio.size == 0:
            return "", 0.0

        audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
        generate_kwargs: dict[str, Any] = {"task": self.settings.task}
        if language_hint:
            generate_kwargs["language"] = language_hint

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        async with self._lock:
            text = await loop.run_in_executor(
                None,
                self._run_pipeline_sync,
                audio_arr,
                generate_kwargs,
            )
        decode_seconds = time.perf_counter() - t0
        return (text or "").strip(), decode_seconds

    def _run_pipeline_sync(
        self,
        audio: np.ndarray,
        generate_kwargs: dict[str, Any],
    ) -> str:
        """Blocking inference; runs in the default executor."""
        return self.asr.transcribe_sync(
            audio,
            self.sample_rate,
            generate_kwargs,
        )
