"""Whisper model loader with auto CUDA/CPU device selection.

The loader is intentionally thin: it instantiates a Hugging Face
`AutomaticSpeechRecognition` pipeline so all the buffered-chunk plumbing in
the streaming engine just needs `pipe(audio_array, ...)`.

The pipeline is loaded once at app startup (`startup` event in `main.py`) and
reused across all sessions; concurrent calls into a single Whisper pipeline
are serialized inside the streaming engine via an asyncio lock to avoid
clobbering CUDA state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """Container holding a ready-to-use Whisper pipeline + metadata."""

    pipe: Any  # transformers Pipeline; typed loosely to avoid version drift
    device: str  # 'cuda' or 'cpu'
    dtype: str   # 'float16' | 'bfloat16' | 'float32'
    model_id: str
    sample_rate: int


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


def load_asr(settings: Settings | None = None) -> LoadedASR:
    """Load the Whisper pipeline and return it wrapped in `LoadedASR`."""
    settings = settings or get_settings()

    device = _resolve_device(settings.device)
    torch_dtype = _resolve_dtype(device, settings)

    log.info(
        "asr_model_loading",
        extra={
            "model_id": settings.model_id,
            "device": device,
            "dtype": str(torch_dtype).replace("torch.", ""),
        },
    )

    is_adapter_model = False
    base_model_id: str | None = None
    try:
        cfg = AutoConfig.from_pretrained(settings.model_id)
        peft_cfg = getattr(cfg, "peft", None)
        if isinstance(peft_cfg, dict):
            base_model_id = peft_cfg.get("base_model_name_or_path")
            is_adapter_model = bool(base_model_id)
    except Exception:  # noqa: BLE001 - remote metadata can be incomplete
        is_adapter_model = False

    if is_adapter_model and base_model_id:
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
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=False,
        )
    else:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=settings.model_id,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
            return_timestamps=False,
        )

    if device == "cuda":
        try:
            pipe.model.eval()
        except Exception:  # noqa: BLE001 - eval() on some wrappers is a no-op
            pass

    log.info(
        "asr_model_loaded",
        extra={
            "model_id": settings.model_id,
            "device": device,
            "dtype": str(torch_dtype).replace("torch.", ""),
        },
    )

    return LoadedASR(
        pipe=pipe,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        model_id=settings.model_id,
        sample_rate=settings.target_sample_rate,
    )
