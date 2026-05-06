"""Application configuration.

All tunables for the ASR backend live here. Values are loaded from environment
variables (with `ASR_` prefix) and an optional `.env` file at the project root.

The model is selected via a single knob: ``ASR_MODEL_KIND``. There are three
supported model kinds, matching the three ways we ship Sinhala Whisper:

1. ``peft``           — PEFT/LoRA adapter on top of a base Whisper checkpoint.
                        Base model is detected from ``adapter_config.json``
                        (override with ``ASR_BASE_MODEL_ID`` if needed).
2. ``merged``         — A single, fully-merged Hugging Face checkpoint
                        (e.g. an exported merge of base + LoRA).
3. ``faster_whisper`` — A CTranslate2 export (``model.bin``) loaded via
                        ``faster_whisper.WhisperModel``.

Examples (PowerShell):

    # PEFT adapter (auto-detect base):
    $env:ASR_MODEL_KIND = "peft"
    $env:ASR_MODEL_ID   = "SPEAK-ASR/whisper-si-exp-10-medium-all"

    # PEFT adapter (explicit base override):
    $env:ASR_MODEL_KIND     = "peft"
    $env:ASR_MODEL_ID       = "SPEAK-ASR/whisper-si-exp-10-medium-all"
    $env:ASR_BASE_MODEL_ID  = "openai/whisper-medium"

    # Single merged checkpoint:
    $env:ASR_MODEL_KIND = "merged"
    $env:ASR_MODEL_ID   = "your-org/whisper-si-merged"

    # faster-whisper / CTranslate2:
    $env:ASR_MODEL_KIND                     = "faster_whisper"
    $env:ASR_MODEL_ID                       = "irudachirath/faster-whisper-medium-si-exp10-fp16"
    $env:ASR_FASTER_WHISPER_CUDA_COMPUTE_TYPE = "float16"
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelKind = Literal["peft", "merged", "faster_whisper"]


class Settings(BaseSettings):
    """Strongly-typed settings loaded from env / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASR_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_backend_env(cls, data: Any) -> Any:
        """Support deprecated ``ASR_BACKEND`` + ``ASR_TRANSFORMERS_LOAD_MODE`` env vars.

        If ``ASR_MODEL_KIND`` is unset, derive ``model_kind`` from the legacy pair
        exactly as older releases did:
        - ``ASR_BACKEND=faster_whisper`` → ``faster_whisper``
        - ``ASR_BACKEND=transformers`` + ``ASR_TRANSFORMERS_LOAD_MODE=full`` → ``merged``
        - otherwise → ``peft``
        """
        if not isinstance(data, dict):
            return data
        mk = data.get("model_kind")
        if mk not in (None, ""):
            return data

        b = (os.environ.get("ASR_BACKEND") or "").strip().lower()
        tm = (os.environ.get("ASR_TRANSFORMERS_LOAD_MODE") or "").strip().lower()
        if b == "faster_whisper":
            data["model_kind"] = "faster_whisper"
        elif b == "transformers":
            data["model_kind"] = "merged" if tm == "full" else "peft"
        return data

    # --- Model selection ---
    model_kind: ModelKind = Field(
        default="peft",
        description=(
            "How to load the model: "
            "'peft' = LoRA/PEFT adapter + base; "
            "'merged' = single full HF checkpoint; "
            "'faster_whisper' = CTranslate2 export."
        ),
    )
    model_id: str = Field(
        default="SPEAK-ASR/whisper-si-exp-10-medium-all",
        description="HF repo id or local path of the model_kind-specific artifact.",
    )
    base_model_id: str | None = Field(
        default=None,
        description=(
            "PEFT only: explicit base-model override. If None, the loader reads "
            "base_model_name_or_path from the adapter's config."
        ),
    )
    merge_peft_adapter: bool = Field(
        default=True,
        description="PEFT only: merge LoRA into base weights before inference.",
    )
    language_hint: str = Field(default="si", description="Whisper language hint, e.g. 'si' for Sinhala.")
    task: Literal["transcribe", "translate"] = "transcribe"

    # --- faster-whisper (CTranslate2) only; ignored for other kinds ---
    faster_whisper_cuda_compute_type: Literal[
        "float16", "int8_float16", "bfloat16", "int8", "float32"
    ] = Field(
        default="float16",
        description="CTranslate2 compute_type on GPU.",
    )
    faster_whisper_cpu_compute_type: Literal["int8", "int8_float32", "float32"] = Field(
        default="int8",
        description="CTranslate2 compute_type on CPU.",
    )
    faster_whisper_beam_size: int = Field(
        default=1,
        description=(
            "Beam size for faster-whisper. 1 (greedy) is fast and stable for "
            "streaming; raise for offline/quality-priority decoding."
        ),
    )

    # --- Device / precision ---
    device: Literal["auto", "cuda", "cpu"] = "auto"
    cuda_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    cpu_dtype: Literal["float32"] = "float32"

    # --- Audio / streaming ---
    target_sample_rate: int = 16_000
    """Internal sample rate the engine operates on. Inputs are validated/converted to this."""

    streaming_mode: Literal["vad", "sliding_window"] = Field(
        default="vad",
        description=(
            "vad = Silero VAD + silence-triggered segments (HF Space style); "
            "sliding_window = periodic partial decode over a trailing window."
        ),
    )

    partial_interval_ms: int = 500
    """How often the gateway flushes a partial decode (sliding_window mode only)."""

    max_buffer_seconds: float = 30.0
    """Hard cap on buffered audio (VAD segment cap; sliding window trim)."""

    silence_trigger_seconds: float = 1.0
    """VAD mode: seconds of silence after speech before a segment is finalized."""

    vad_threshold: float = 0.5
    """Silero VAD confidence threshold (0–1, higher = stricter)."""

    min_speech_seconds: float = 0.5
    """VAD mode: ignore segments shorter than this (noise blips)."""

    decode_window_seconds: float = 6.0
    """Length of the trailing audio window fed to Whisper per partial decode."""

    min_audio_for_partial_seconds: float = 0.6
    """sliding_window mode: minimum buffered seconds before running a partial decode."""

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

    # --- Postprocessing (diarization / future denoise) ---
    hf_token: str | None = Field(
        default=None,
        description=(
            "Hugging Face hub token for pyannote pipelines "
            "(falls back to HF_TOKEN / HUGGING_FACE_HUB_TOKEN)."
        ),
    )
    diarization_model_id: str = Field(
        default="pyannote/speaker-diarization-community-1",
        description="pyannote Pipeline.from_pretrained id for speaker diarization.",
    )
    postprocessing_diarization_default: bool = Field(
        default=False,
        description="Default on/off for diarization on new WebSocket sessions.",
    )
    postprocessing_noise_removal_default: bool = Field(
        default=False,
        description="Default on/off for noise removal on new sessions (not implemented yet).",
    )

    # --- Derived helpers -----------------------------------------------------

    def resolved_hf_token(self) -> str | None:
        """Token for Hugging Face downloads (pyannote), or None if unset."""
        for raw in (
            self.hf_token,
            os.environ.get("HF_TOKEN"),
            os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        ):
            if raw is None:
                continue
            tok = str(raw).strip()
            if tok:
                return tok
        return None

    @property
    def backend(self) -> Literal["transformers", "faster_whisper"]:
        """Inference dispatch backend implied by ``model_kind``.

        Both ``peft`` and ``merged`` run through the Hugging Face pipeline
        ('transformers' backend); ``faster_whisper`` is its own backend.
        """
        return "faster_whisper" if self.model_kind == "faster_whisper" else "transformers"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) global Settings instance."""
    return Settings()
