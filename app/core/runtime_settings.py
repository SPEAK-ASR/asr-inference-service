"""Mutable runtime settings.

Unlike :class:`app.core.config.Settings` (immutable, env-driven, cached) the
``RuntimeSettings`` instance lives on ``app.state.runtime_settings`` and can be
toggled at runtime via the ``/settings`` REST endpoint.

Currently it only carries the noise-removal flag; future preprocessing /
postprocessing toggles (e.g. diarization) can be added as additional keys
without changing the public shape returned by ``get()``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


class RuntimeSettings:
    """Thread-safe container for runtime-mutable feature flags."""

    def __init__(self, *, use_noise_removal: bool) -> None:
        self._lock = threading.Lock()
        self._use_noise_removal = bool(use_noise_removal)

    @property
    def use_noise_removal(self) -> bool:
        with self._lock:
            return self._use_noise_removal

    def get(self) -> dict:
        """Return the current state as a JSON-serializable dict."""
        with self._lock:
            return {"use_noise_removal": self._use_noise_removal}

    def update(self, *, use_noise_removal: bool | None = None) -> dict:
        """Apply partial updates and return the new state."""
        with self._lock:
            if use_noise_removal is not None:
                self._use_noise_removal = bool(use_noise_removal)
            return {"use_noise_removal": self._use_noise_removal}


def get_runtime_settings(app: "FastAPI") -> RuntimeSettings:
    """Fetch the ``RuntimeSettings`` instance attached to a FastAPI app."""
    rt = getattr(app.state, "runtime_settings", None)
    if rt is None:
        raise RuntimeError(
            "RuntimeSettings not initialized; ensure lifespan ran before access."
        )
    return rt
