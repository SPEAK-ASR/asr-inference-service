"""Application configuration.

All tunables for the ASR backend live here. Values are loaded from environment
variables (with `ASR_` prefix) and an optional `.env` file at the project root.

Examples (PowerShell):
    $env:ASR_MODEL_ID = "SPEAK-ASR/whisper-si-exp-10-medium-all"
    $env:ASR_DEVICE   = "auto"  # auto | cuda | cpu
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings loaded from env / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASR_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model ---
    model_id: str = Field(
        default="SPEAK-ASR/whisper-si-exp-10-medium-all",
        description="Hugging Face model id for the primary ASR path.",
    )
    language_hint: str = Field(default="si", description="Whisper language hint, e.g. 'si' for Sinhala.")
    task: Literal["transcribe", "translate"] = "transcribe"

    # --- Device / precision ---
    device: Literal["auto", "cuda", "cpu"] = "auto"
    cuda_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    cpu_dtype: Literal["float32"] = "float32"

    # --- Audio / streaming ---
    target_sample_rate: int = 16_000
    """Internal sample rate the engine operates on. Inputs are validated/converted to this."""

    partial_interval_ms: int = 500
    """How often the gateway flushes a partial decode while audio is flowing."""

    max_buffer_seconds: float = 30.0
    """Hard cap on the per-session sliding audio buffer length."""

    decode_window_seconds: float = 6.0
    """Length of the trailing audio window fed to Whisper per partial decode."""

    min_audio_for_partial_seconds: float = 0.6
    """Don't run inference until at least this much audio is buffered."""

    # --- Session lifecycle ---
    idle_timeout_seconds: int = 60
    """Reap a session if no audio_chunk arrives for this long."""

    reaper_interval_seconds: int = 5

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    # --- Limits ---
    max_chunk_bytes: int = 256 * 1024
    """Reject incoming audio_chunk bigger than this (post-base64-decode)."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) global Settings instance."""
    return Settings()
