"""DeepFilterNet noise-removal wrapper.

DeepFilterNet3 is a small, real-time-capable speech enhancer trained on 48 kHz
mono audio. Our pipeline operates at 16 kHz, so this module:

1. Lazily loads the DeepFilterNet model + state on first use (or eagerly via
   :func:`preload_denoiser`).
2. Resamples 16 kHz input → 48 kHz, runs ``df.enhance.enhance``, then
   resamples the cleaned audio back to 16 kHz.
3. Serializes calls with a ``threading.Lock`` because the underlying model
   shares a single torch context and isn't safe to invoke concurrently.

The toggle is checked upstream (in :class:`StreamingEngine`); this module
unconditionally denoises the audio it is given.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


_lock = threading.Lock()
_state: dict[str, Any] = {
    "model": None,
    "df_state": None,
    "df_sr": None,
    "device": None,
}


def _resolve_device(settings: Settings) -> str:
    """Match the ASR model loader's device choice, but degrade gracefully."""
    pref = settings.device
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def preload_denoiser(settings: Settings | None = None) -> None:
    """Eagerly load DeepFilterNet so first toggle-on incurs no cold start."""
    _ensure_loaded(settings or get_settings())


def _ensure_loaded(settings: Settings) -> None:
    if _state["model"] is not None:
        return

    with _lock:
        if _state["model"] is not None:
            return

        try:
            from df.enhance import init_df
        except ImportError as exc:  # pragma: no cover - dep should be installed
            raise RuntimeError(
                "DeepFilterNet is not installed; install 'deepfilternet' "
                "(see requirements.txt)."
            ) from exc

        device = _resolve_device(settings)
        log.info("denoiser_loading", extra={"device": device})

        model, df_state, _ = init_df()
        try:
            model = model.to(device)
        except Exception:  # noqa: BLE001 - some builds pin to CPU
            log.warning("denoiser_to_device_failed", extra={"device": device})
            device = "cpu"
        model.eval()

        _state["model"] = model
        _state["df_state"] = df_state
        _state["df_sr"] = int(df_state.sr())
        _state["device"] = device

        log.info(
            "denoiser_ready",
            extra={"device": device, "sample_rate": _state["df_sr"]},
        )


def _resample(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return audio
    import torchaudio.functional as AF

    return AF.resample(audio, orig_freq=src_sr, new_freq=dst_sr)


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run DeepFilterNet on ``audio`` and return float32 mono at ``sample_rate``.

    Resamples in/out as needed to match the DeepFilterNet model rate (48 kHz).
    Falls back to returning the input unchanged on unrecoverable errors so a
    misbehaving denoiser never blocks transcription.
    """
    if audio is None or audio.size == 0:
        return audio

    settings = get_settings()
    _ensure_loaded(settings)

    model = _state["model"]
    df_state = _state["df_state"]
    df_sr = int(_state["df_sr"])
    try:
        from df.enhance import enhance
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DeepFilterNet missing at runtime.") from exc

    arr = np.ascontiguousarray(audio, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0)

    with _lock:
        try:
            up = _resample(tensor, sample_rate, df_sr)
            # DeepFilterNet's feature path calls `.numpy()` on `audio`, so this
            # tensor must stay on host even when the model itself is on GPU.
            up = up.to("cpu")
            with torch.inference_mode():
                cleaned = enhance(model, df_state, up)
            cleaned = cleaned.detach().to("cpu")
            down = _resample(cleaned, df_sr, sample_rate)
        except Exception:  # noqa: BLE001
            log.exception("denoise_failed_passthrough")
            return arr

    out = down.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)
    if out.shape[0] != arr.shape[0]:
        if out.shape[0] > arr.shape[0]:
            out = out[: arr.shape[0]]
        else:
            pad = np.zeros(arr.shape[0] - out.shape[0], dtype=np.float32)
            out = np.concatenate([out, pad])
    return out
