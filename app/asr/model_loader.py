"""Whisper model loader with auto CUDA/CPU device selection.

Supports two backends (see ``Settings.backend``):

- **transformers**: Hugging Face ``AutomaticSpeechRecognition`` pipeline
  (PEFT adapter + base when detected, or a single merged/full checkpoint).
- **faster_whisper**: CTranslate2 models (e.g. ``model.bin`` on Hugging Face),
  loaded via ``faster_whisper.WhisperModel``.

The active model is loaded once at app startup and reused; concurrent calls
are serialized in the streaming engine via an asyncio lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class LoadedASR:
    """Container for either a HF ASR pipeline or a faster-whisper model."""

    backend: Literal["transformers", "faster_whisper"]
    pipe: Any | None  # transformers Pipeline when backend == transformers
    faster_model: Any | None  # faster_whisper.WhisperModel when backend == faster_whisper
    device: str  # 'cuda' or 'cpu'
    dtype: str  # torch dtype name or CTranslate2 compute_type string
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


def _transcribe_transformers_pipeline(
    pipe: Any,
    audio: np.ndarray,
    sample_rate: int,
    generate_kwargs: dict[str, Any],
) -> str:
    try:
        result = pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )
    except TypeError:
        result = pipe({"array": audio, "sampling_rate": sample_rate})
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
    language = generate_kwargs.get("language")
    task = generate_kwargs.get("task", "transcribe")
    kw: dict[str, Any] = {"task": task}
    if language:
        kw["language"] = language
    segments, _info = model.transcribe(audio, **kw)
    return "".join(s.text for s in segments).strip()


def _load_transformers_pipeline(settings: Settings) -> Any:
    device = _resolve_device(settings.device)
    torch_dtype = _resolve_dtype(device, settings)

    is_adapter_model = False
    base_model_id: str | None = None
    if settings.transformers_load_mode == "full":
        log.info(
            "asr_transformers_load_mode",
            extra={"mode": "full", "note": "single checkpoint (merged or native full weights)"},
        )
    else:
        try:
            cfg = AutoConfig.from_pretrained(settings.model_id)
            peft_cfg = getattr(cfg, "peft", None)
            if isinstance(peft_cfg, dict):
                base_model_id = peft_cfg.get("base_model_name_or_path")
                is_adapter_model = bool(base_model_id)
        except Exception:  # noqa: BLE001 - remote metadata can be incomplete
            is_adapter_model = False

    if (
        settings.transformers_load_mode == "auto"
        and is_adapter_model
        and base_model_id
    ):
        from peft import PeftModel  # imported lazily to keep non-PEFT path light

        log.info(
            "asr_adapter_detected",
            extra={
                "adapter_model_id": settings.model_id,
                "base_model_id": base_model_id,
            },
        )
        processor = AutoProcessor.from_pretrained(base_model_id)
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base_model, settings.model_id)

        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=False,
        )
    else:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=settings.model_id,
            device=device,
            dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=False,
        )

    if device == "cuda":
        try:
            pipe.model.eval()
        except Exception:  # noqa: BLE001 - eval() on some wrappers is a no-op
            pass

    return pipe


def _load_faster_whisper_model(settings: Settings) -> tuple[Any, str, str]:
    from faster_whisper import WhisperModel

    device = _resolve_device(settings.device)
    compute_type = (
        settings.faster_whisper_cuda_compute_type
        if device == "cuda"
        else settings.faster_whisper_cpu_compute_type
    )
    model = WhisperModel(
        settings.model_id,
        device=device,
        compute_type=compute_type,
    )
    return model, device, compute_type


def load_asr(settings: Settings | None = None) -> LoadedASR:
    """Load the configured ASR backend and return it wrapped in ``LoadedASR``."""
    settings = settings or get_settings()

    if settings.backend == "faster_whisper":
        log.info(
            "asr_model_loading",
            extra={
                "backend": "faster_whisper",
                "model_id": settings.model_id,
                "device": _resolve_device(settings.device),
            },
        )
        faster_model, device, compute_type = _load_faster_whisper_model(settings)
        log.info(
            "asr_model_loaded",
            extra={
                "backend": "faster_whisper",
                "model_id": settings.model_id,
                "device": device,
                "compute_type": compute_type,
            },
        )
        return LoadedASR(
            backend="faster_whisper",
            pipe=None,
            faster_model=faster_model,
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
            "backend": "transformers",
            "model_id": settings.model_id,
            "device": device,
            "dtype": str(torch_dtype).replace("torch.", ""),
        },
    )

    pipe = _load_transformers_pipeline(settings)

    log.info(
        "asr_model_loaded",
        extra={
            "backend": "transformers",
            "model_id": settings.model_id,
            "device": device,
            "dtype": str(torch_dtype).replace("torch.", ""),
        },
    )

    return LoadedASR(
        backend="transformers",
        pipe=pipe,
        faster_model=None,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        model_id=settings.model_id,
        sample_rate=settings.target_sample_rate,
    )
