"""Whisper model loader with auto CUDA/CPU device selection.

Three model kinds are supported (selected by ``Settings.model_kind``):

1. ``peft``           — PEFT/LoRA adapter on top of a base Whisper checkpoint.
                        Base id is read from the adapter's config (or
                        ``Settings.base_model_id`` if explicitly set). When
                        ``Settings.merge_peft_adapter`` is True the adapter
                        is merged into the base weights for inference.
2. ``merged``         — A single, fully-merged HF checkpoint loaded directly.
3. ``faster_whisper`` — CTranslate2 export loaded via ``faster_whisper``.

The active model is loaded once at app startup and reused; concurrent calls
are serialized in the streaming engine via an asyncio lock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from app.core.config import ModelKind, Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

Backend = Literal["transformers", "faster_whisper"]


@dataclass
class LoadedASR:
    """Container for either a HF ASR pipeline or a faster-whisper model."""

    backend: Backend
    """Inference dispatch: 'transformers' for peft/merged, 'faster_whisper' for CT2."""

    model_kind: ModelKind
    """User-facing label: 'peft' | 'merged' | 'faster_whisper'."""

    pipe: Any | None
    """transformers Pipeline when backend == 'transformers'; else None."""

    faster_model: Any | None
    """faster_whisper.WhisperModel when backend == 'faster_whisper'; else None."""

    device: str
    """'cuda' or 'cpu'."""

    dtype: str
    """torch dtype name (transformers) or CTranslate2 compute_type (faster_whisper)."""

    model_id: str
    sample_rate: int

    def transcribe_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        generate_kwargs: dict[str, Any],
    ) -> str:
        """Run synchronous inference; called from a thread pool by StreamingEngine."""
        if self.backend == "transformers":
            return _transcribe_transformers_pipeline(
                self.pipe, audio, sample_rate, generate_kwargs
            )
        return _transcribe_faster_whisper(
            self.faster_model, audio, generate_kwargs
        )


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def _resolve_device(setting: str) -> str:
    if setting == "cpu":
        return "cpu"
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ASR_DEVICE=cuda requested but torch.cuda.is_available() is False."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str, settings: Settings) -> torch.dtype:
    if device == "cuda":
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[settings.cuda_dtype]
    return torch.float32


def _actual_dtype_str(model: Any) -> str:
    """Return the real dtype of the loaded model parameters as a short string."""
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return "unknown"
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# Transcribe paths
# ---------------------------------------------------------------------------

def _transcribe_transformers_pipeline(
    pipe: Any,
    audio: np.ndarray,
    sample_rate: int,
    generate_kwargs: dict[str, Any],
) -> str:
    result = pipe(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs,
    )
    if isinstance(result, dict):
        return result.get("text", "") or ""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0].get("text", "") or ""
    return str(result or "")


def _transcribe_faster_whisper(
    model: Any,
    audio: np.ndarray,
    generate_kwargs: dict[str, Any],
) -> str:
    # Streaming-friendly defaults:
    #   - vad_filter=False: we already do Silero VAD upstream.
    #   - condition_on_previous_text=False: avoids hallucination drift across
    #     short partial windows (a known faster-whisper streaming pitfall).
    settings = get_settings()
    kw: dict[str, Any] = {
        "task": generate_kwargs.get("task", "transcribe"),
        "beam_size": settings.faster_whisper_beam_size,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }
    language = generate_kwargs.get("language")
    if language:
        kw["language"] = language

    segments, _info = model.transcribe(audio, **kw)
    return "".join(s.text for s in segments).strip()


# ---------------------------------------------------------------------------
# Loaders — one per model_kind
# ---------------------------------------------------------------------------

def _detect_peft_base_model(adapter_id: str) -> str:
    """Read ``base_model_name_or_path`` from a PEFT adapter's config."""
    try:
        from peft import PeftConfig
    except ImportError as exc:  # pragma: no cover - dep listed in requirements
        raise RuntimeError(
            "model_kind='peft' requires the 'peft' package. Install requirements.txt."
        ) from exc

    try:
        peft_cfg = PeftConfig.from_pretrained(adapter_id)
    except Exception as exc:  # noqa: BLE001 - many possible HF/IO errors
        raise RuntimeError(
            f"Could not load PEFT config from '{adapter_id}'. "
            f"Is this actually a LoRA/PEFT adapter repo with adapter_config.json? "
            f"If it's a fully-merged checkpoint, set ASR_MODEL_KIND=merged. "
            f"Underlying error: {exc}"
        ) from exc

    base = getattr(peft_cfg, "base_model_name_or_path", None)
    if not base:
        raise RuntimeError(
            f"PEFT adapter '{adapter_id}' does not declare a base_model_name_or_path. "
            f"Set ASR_BASE_MODEL_ID explicitly."
        )
    return base


def _whisper_processor_needs_manual_assembly(exc: BaseException) -> bool:
    """Detect Hub layouts / tokenizer JSON that break stock ``AutoProcessor``."""
    if isinstance(exc, OSError):
        return "preprocessor_config.json" in str(exc).lower()
    if isinstance(exc, AttributeError):
        # e.g. tokenizer_config has ``extra_special_tokens`` as a list; fast
        # tokenizer calls ``.keys()`` on it (WhisperTokenizerFast).
        m = str(exc).lower()
        return "keys" in m and "list" in m
    return False


def _load_whisper_tokenizer_slow_fallback(model_id: str) -> Any:
    """Load tokenizer; fall back to slow tokenizer for odd ``tokenizer_config`` files."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except AttributeError:
        log.info(
            "asr_whisper_tokenizer_slow",
            extra={
                "model_id": model_id,
                "note": "WhisperTokenizerFast failed; using slow tokenizer",
            },
        )
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)


def _assemble_whisper_processor_manual(model_id: str, *, cause: BaseException) -> Any:
    """Build ``WhisperProcessor`` from ``processor_config.json`` + tokenizer files."""
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor, WhisperProcessor

    log.info(
        "asr_whisper_processor_manual_assembly",
        extra={"model_id": model_id, "trigger": type(cause).__name__},
    )

    try:
        proc_path = hf_hub_download(repo_id=model_id, filename="processor_config.json")
    except Exception as hub_exc:  # noqa: BLE001
        raise OSError(
            f"Could not load processor for '{model_id}': processor_config.json not available. "
            f"Original error: {cause}"
        ) from hub_exc

    with open(proc_path, encoding="utf-8") as f:
        proc_cfg = json.load(f)
    fe_cfg = proc_cfg.get("feature_extractor")
    if not isinstance(fe_cfg, dict):
        raise OSError(
            f"'{model_id}': processor_config.json has no 'feature_extractor' object; "
            f"cannot build WhisperFeatureExtractor."
        ) from cause

    feature_extractor = WhisperFeatureExtractor.from_dict(fe_cfg)
    tokenizer = _load_whisper_tokenizer_slow_fallback(model_id)
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def _load_whisper_processor(model_id: str) -> Any:
    """Load a ``WhisperProcessor`` from ``model_id``.

    Some community repos ship only ``processor_config.json`` (with a nested
    ``feature_extractor``) and no ``preprocessor_config.json``. Others ship a
    ``tokenizer_config.json`` where ``extra_special_tokens`` is a plain list,
    which breaks ``WhisperTokenizerFast``. In those cases we assemble the processor
    manually and prefer the slow tokenizer when needed.
    """
    try:
        return AutoProcessor.from_pretrained(model_id)
    except (OSError, AttributeError) as exc:
        if not _whisper_processor_needs_manual_assembly(exc):
            raise
        return _assemble_whisper_processor_manual(model_id, cause=exc)


def _build_pipeline(
    model: Any,
    processor: Any,
    device: str,
    torch_dtype: torch.dtype,
) -> Any:
    """Wrap an already-loaded HF model + processor into an ASR pipeline."""
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        return_timestamps=False,
    )
    return pipe


def _load_peft(
    settings: Settings,
    device: str,
    torch_dtype: torch.dtype,
) -> Any:
    from peft import PeftModel

    base_id = settings.base_model_id or _detect_peft_base_model(settings.model_id)
    log.info(
        "asr_peft_loading",
        extra={
            "adapter_id": settings.model_id,
            "base_model_id": base_id,
            "merge": settings.merge_peft_adapter,
        },
    )

    processor = AutoProcessor.from_pretrained(base_id)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, settings.model_id)

    if settings.merge_peft_adapter:
        log.info("asr_peft_merging", extra={"note": "merge_and_unload for inference"})
        model = model.merge_and_unload()

    model.to(device=device, dtype=torch_dtype)
    model.eval()

    return _build_pipeline(model, processor, device, torch_dtype)


def _load_merged(
    settings: Settings,
    device: str,
    torch_dtype: torch.dtype,
) -> Any:
    log.info(
        "asr_merged_loading",
        extra={"model_id": settings.model_id},
    )
    processor = _load_whisper_processor(settings.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        settings.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=torch_dtype)
    model.eval()

    return _build_pipeline(model, processor, device, torch_dtype)


def _load_faster_whisper(settings: Settings) -> tuple[Any, str, str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "model_kind='faster_whisper' requires the 'faster-whisper' package."
        ) from exc

    device = _resolve_device(settings.device)
    compute_type = (
        settings.faster_whisper_cuda_compute_type
        if device == "cuda"
        else settings.faster_whisper_cpu_compute_type
    )
    log.info(
        "asr_faster_whisper_loading",
        extra={
            "model_id": settings.model_id,
            "device": device,
            "compute_type": compute_type,
        },
    )
    model = WhisperModel(
        settings.model_id,
        device=device,
        compute_type=compute_type,
    )
    return model, device, compute_type


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_asr(settings: Settings | None = None) -> LoadedASR:
    """Load the configured ASR model and return it wrapped in ``LoadedASR``."""
    settings = settings or get_settings()

    if settings.model_kind == "faster_whisper":
        model, device, compute_type = _load_faster_whisper(settings)
        log.info(
            "asr_model_loaded",
            extra={
                "model_kind": "faster_whisper",
                "model_id": settings.model_id,
                "device": device,
                "compute_type": compute_type,
            },
        )
        return LoadedASR(
            backend="faster_whisper",
            model_kind="faster_whisper",
            pipe=None,
            faster_model=model,
            device=device,
            dtype=compute_type,
            model_id=settings.model_id,
            sample_rate=settings.target_sample_rate,
        )

    # transformers backend (peft or merged)
    device = _resolve_device(settings.device)
    torch_dtype = _resolve_dtype(device, settings)

    log.info(
        "asr_model_loading",
        extra={
            "model_kind": settings.model_kind,
            "model_id": settings.model_id,
            "device": device,
            "requested_dtype": str(torch_dtype).replace("torch.", ""),
        },
    )

    if settings.model_kind == "peft":
        pipe = _load_peft(settings, device, torch_dtype)
    elif settings.model_kind == "merged":
        pipe = _load_merged(settings, device, torch_dtype)
    else:  # pragma: no cover - exhaustively covered by Literal
        raise ValueError(f"Unsupported model_kind: {settings.model_kind!r}")

    actual_dtype = _actual_dtype_str(pipe.model)
    log.info(
        "asr_model_loaded",
        extra={
            "model_kind": settings.model_kind,
            "model_id": settings.model_id,
            "device": device,
            "dtype": actual_dtype,
        },
    )

    return LoadedASR(
        backend="transformers",
        model_kind=settings.model_kind,
        pipe=pipe,
        faster_model=None,
        device=device,
        dtype=actual_dtype,
        model_id=settings.model_id,
        sample_rate=settings.target_sample_rate,
    )
