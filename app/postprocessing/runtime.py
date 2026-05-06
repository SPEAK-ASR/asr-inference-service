"""Process-wide toggles for optional postprocessing features."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class PostprocessingRuntime:
    """Mutable defaults applied to new WebSocket sessions (per server process)."""

    diarization_enabled: bool = False
    noise_removal_enabled: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def snapshot(self) -> dict[str, bool]:
        async with self._lock:
            return {
                "diarization_enabled": self.diarization_enabled,
                "noise_removal_enabled": self.noise_removal_enabled,
            }

    async def update(
        self,
        *,
        diarization_enabled: bool | None = None,
        noise_removal_enabled: bool | None = None,
    ) -> dict[str, bool]:
        async with self._lock:
            if diarization_enabled is not None:
                self.diarization_enabled = diarization_enabled
            if noise_removal_enabled is not None:
                self.noise_removal_enabled = noise_removal_enabled
            return {
                "diarization_enabled": self.diarization_enabled,
                "noise_removal_enabled": self.noise_removal_enabled,
            }
