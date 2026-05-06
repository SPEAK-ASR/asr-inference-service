"""Whisper model loader with CUDA/CPU selection.

``model_kind`` (see ``Settings``):

- ``peft`` — load base with ``WhisperProcessor`` + ``WhisperForConditionalGeneration``,
  attach the LoRA repo with ``PeftModel``, optionally ``merge_and_unload()``.
- ``merged`` — single checkpoint via ``AutoProcessor`` + ``AutoModelForSpeechSeq2Seq``.
- ``faster_whisper`` — CTranslate2 export via ``faster_whisper``.

The model loads once at startup; concurrent transcription is serialized upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
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
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return "unknown"
    return str(dtype).replace("torch.", "")


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


def _detect_peft_base_model(adapter_id: str) -> str:
    try:
        from peft import PeftConfig
    except ImportError as exc:
        raise RuntimeError(
            "model_kind='peft' requires the 'peft' package. Install requirements.txt."
        ) from exc

    try:
        peft_cfg = PeftConfig.from_pretrained(adapter_id)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load PEFT config from '{adapter_id}'. "
            f"If this is a fully merged checkpoint, set ASR_MODEL_KIND=merged. "
            f"Detail: {exc}"
        ) from exc

    base = getattr(peft_cfg, "base_model_name_or_path", None)
    if not base:
        raise RuntimeError(
            f"PEFT adapter '{adapter_id}' has no base_model_name_or_path; "
            f"set ASR_BASE_MODEL_ID."
        )
    return base


def _asr_pipeline(
    model: Any,
    processor: Any,
    device: str,
    torch_dtype: torch.dtype,
) -> Any:
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        return_timestamps=False,
    )


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

    processor = WhisperProcessor.from_pretrained(base_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        base_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, settings.model_id)

    if settings.merge_peft_adapter:
        log.info("asr_peft_merging", extra={"note": "merge_and_unload"})
        model = model.merge_and_unload()

    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    return _asr_pipeline(model, processor, device, torch_dtype)


def _load_merged(
    settings: Settings,
    device: str,
    torch_dtype: torch.dtype,
) -> Any:
    log.info("asr_merged_loading", extra={"model_id": settings.model_id})

    processor = AutoProcessor.from_pretrained(settings.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        settings.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    return _asr_pipeline(model, processor, device, torch_dtype)


def _load_faster_whisper(settings: Settings) -> tuple[Any, str, str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
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
    else:  # pragma: no cover
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
