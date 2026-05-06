"""Speaker diarization layer (opt-in, per session).

Wraps a single pyannote.audio diarization pipeline plus a separate
speaker-embedding model so we can:

1. Diarize a finalized VAD segment into local-label turns
   (``SPEAKER_00``, ``SPEAKER_01`` — these labels are *not* stable across calls).
2. Compute one embedding per local turn and match it against a
   session-scoped registry of speaker centroids to assign a *stable*
   ``spk_N`` label (consistent across multiple VAD segments inside the same
   WebSocket session).

The pipeline and embedding model are loaded lazily on first use and shared
across all sessions. Heavy work runs synchronously and is meant to be
dispatched from a thread pool by the caller, the same way Whisper does.

Configuration lives in :class:`app.core.config.Settings` (``diarization_*``
fields). The default pipeline is ``pyannote/speaker-diarization-3.1``, which
is the stable release for pyannote.audio 3.1.x.  It requires:

* A HuggingFace access token (``HF_TOKEN`` env var).
* Accepted user conditions for **both** gated hub models:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0  (sub-model loaded by the pipeline)
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

torch.backends.nnpack.set_flags(False)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Singletons (lazy)
# ---------------------------------------------------------------------------

_pipeline: Any | None = None
_embedder: Any | None = None
_pipeline_load_failed: bool = False
_embedder_load_failed: bool = False
_load_lock = threading.Lock()

# Serialize all calls into the diarization stack — both pipeline and embedder
# share GPU/CPU resources and pyannote's pipelines aren't safe for parallel
# invocation on a single CUDA context, just like Whisper.
_run_lock = asyncio.Lock()


def _resolve_torch_device(settings: Settings) -> torch.device:
    if settings.device == "cpu":
        return torch.device("cpu")
    if settings.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ASR_DEVICE=cuda requested but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_pipeline(settings: Settings) -> Any:
    """Build the pyannote SpeakerDiarization pipeline. Returns None on failure."""
    try:
        from pyannote.audio import Pipeline
    except Exception:  # noqa: BLE001
        log.exception("diarization_import_failed")
        return None

    try:
        log.info(
            "diarization_pipeline_loading",
            extra={"model_id": settings.diarization_model_id},
        )
        load_kwargs: dict[str, Any] = {}
        if settings.hf_token:
            load_kwargs["use_auth_token"] = settings.hf_token
        pipeline = Pipeline.from_pretrained(settings.diarization_model_id, **load_kwargs)
        device = _resolve_torch_device(settings)
        try:
            pipeline.to(device)
        except Exception:  # noqa: BLE001 - some pipelines accept torch.device, some don't
            log.exception("diarization_pipeline_device_move_failed")
        log.info(
            "diarization_pipeline_ready",
            extra={
                "model_id": settings.diarization_model_id,
                "device": str(device),
            },
        )
        return pipeline
    except Exception:  # noqa: BLE001
        log.exception(
            "diarization_pipeline_load_failed",
            extra={"model_id": settings.diarization_model_id},
        )
        return None


def _load_embedder(settings: Settings) -> Any:
    """Build the pyannote embedding ``Inference`` wrapper. Returns None on failure."""
    try:
        from pyannote.audio import Inference, Model
    except Exception:  # noqa: BLE001
        log.exception("diarization_embedder_import_failed")
        return None

    try:
        log.info(
            "diarization_embedder_loading",
            extra={"model_id": settings.diarization_embedding_model_id},
        )
        load_kwargs: dict[str, Any] = {}
        if settings.hf_token:
            load_kwargs["use_auth_token"] = settings.hf_token

        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            model = Model.from_pretrained(settings.diarization_embedding_model_id, **load_kwargs)
        finally:
            torch.load = _orig_load
        device = _resolve_torch_device(settings)
        try:
            model.to(device)
        except Exception:  # noqa: BLE001
            log.exception("diarization_embedder_device_move_failed")
        inference = Inference(model, window="whole")
        log.info(
            "diarization_embedder_ready",
            extra={
                "model_id": settings.diarization_embedding_model_id,
                "device": str(device),
            },
        )
        return inference
    except Exception:  # noqa: BLE001
        log.exception(
            "diarization_embedder_load_failed",
            extra={"model_id": settings.diarization_embedding_model_id},
        )
        return None


def get_diarizer(settings: Settings | None = None) -> Any | None:
    """Lazy singleton for the diarization pipeline.

    Returns ``None`` if loading failed (model missing, no HF auth, network etc.).
    Failures are sticky for the process lifetime to avoid hammering the loader.
    """
    global _pipeline, _pipeline_load_failed
    if _pipeline is not None:
        return _pipeline
    if _pipeline_load_failed:
        return None
    settings = settings or get_settings()
    if not settings.diarization_enabled_capability:
        return None
    with _load_lock:
        if _pipeline is not None:
            return _pipeline
        if _pipeline_load_failed:
            return None
        pipeline = _load_pipeline(settings)
        if pipeline is None:
            _pipeline_load_failed = True
            return None
        _pipeline = pipeline
        return _pipeline


def get_embedder(settings: Settings | None = None) -> Any | None:
    """Lazy singleton for the speaker-embedding inference wrapper."""
    global _embedder, _embedder_load_failed
    if _embedder is not None:
        return _embedder
    if _embedder_load_failed:
        return None
    settings = settings or get_settings()
    if not settings.diarization_enabled_capability:
        return None
    with _load_lock:
        if _embedder is not None:
            return _embedder
        if _embedder_load_failed:
            return None
        embedder = _load_embedder(settings)
        if embedder is None:
            _embedder_load_failed = True
            return None
        _embedder = embedder
        return _embedder


def preload(settings: Settings | None = None) -> bool:
    """Eager-load both models. Returns True iff both are usable."""
    settings = settings or get_settings()
    return get_diarizer(settings) is not None and get_embedder(settings) is not None


def is_loaded() -> bool:
    """Cheap probe used by ``/health/ready``."""
    return _pipeline is not None and _embedder is not None


def is_available(settings: Settings | None = None) -> bool:
    """True if diarization is enabled at the service level and not known-broken."""
    settings = settings or get_settings()
    if not settings.diarization_enabled_capability:
        return False
    return not (_pipeline_load_failed or _embedder_load_failed)


# ---------------------------------------------------------------------------
# Session-scoped speaker registry
# ---------------------------------------------------------------------------

@dataclass
class SessionSpeakerRegistry:
    """Maps speaker embeddings to stable ``spk_N`` IDs for one WebSocket session.

    Cosine similarity against running-average centroids; new speakers get a
    fresh sequential ID when no existing centroid is close enough.
    """

    match_threshold: float = 0.70
    ema: float = 0.3
    centroids: list[np.ndarray] = field(default_factory=list)

    def assign(self, embedding: np.ndarray) -> str:
        emb = _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if emb.size == 0:
            return self._new_speaker(emb)

        if not self.centroids:
            return self._new_speaker(emb)

        best_idx = -1
        best_sim = -1.0
        for i, c in enumerate(self.centroids):
            sim = float(np.dot(c, emb))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self.match_threshold and best_idx >= 0:
            updated = (1.0 - self.ema) * self.centroids[best_idx] + self.ema * emb
            self.centroids[best_idx] = _l2_normalize(updated)
            return _label(best_idx)

        return self._new_speaker(emb)

    def _new_speaker(self, normalized_emb: np.ndarray) -> str:
        if normalized_emb.size == 0:
            self.centroids.append(np.zeros(1, dtype=np.float32))
        else:
            self.centroids.append(normalized_emb.copy())
        return _label(len(self.centroids) - 1)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _label(idx: int) -> str:
    return f"spk_{idx + 1}"


# ---------------------------------------------------------------------------
# Diarization entry point
# ---------------------------------------------------------------------------

@dataclass
class SpeakerTurn:
    """Local representation; the schema-facing model lives in app.sessions.schemas."""

    speaker_id: str
    start_ms: int
    end_ms: int


def _audio_to_pyannote_input(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Convert float32 mono audio to the dict pyannote pipelines expect."""
    arr = np.ascontiguousarray(audio, dtype=np.float32)
    waveform = torch.from_numpy(arr).unsqueeze(0)  # (1, num_samples)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def diarize_segment_sync(
    audio: np.ndarray,
    sample_rate: int,
    registry: SessionSpeakerRegistry,
    settings: Settings | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote on ``audio`` and return turns labeled with stable ``spk_N`` IDs.

    Designed to be called from a thread executor (``loop.run_in_executor``) by the
    async gateway, similar to Whisper. The async caller should additionally hold
    :data:`_run_lock` to serialize concurrent diarization calls.

    Falls back to a single ``spk_1`` turn covering the whole segment when:

    - the segment is shorter than ``diarization_min_turn_seconds`` (warmup cost
      isn't worth it on tiny clips), or
    - the diarization pipeline is unavailable / errors out, or
    - the pipeline returns no turns at all.
    """
    settings = settings or get_settings()
    duration_seconds = float(audio.size) / float(sample_rate)

    if duration_seconds < settings.diarization_min_turn_seconds:
        return _single_turn_fallback(registry, audio, sample_rate)

    pipeline = get_diarizer(settings)
    if pipeline is None:
        return _single_turn_fallback(registry, audio, sample_rate)

    try:
        annotation = pipeline(_audio_to_pyannote_input(audio, sample_rate))
    except Exception:  # noqa: BLE001
        log.exception("diarization_pipeline_failed", extra={"duration_s": duration_seconds})
        return _single_turn_fallback(registry, audio, sample_rate)

    embedder = get_embedder(settings)

    raw_turns: list[tuple[float, float, str]] = []
    try:
        for segment, _track, local_label in annotation.itertracks(yield_label=True):
            start_s = max(0.0, float(segment.start))
            end_s = min(duration_seconds, float(segment.end))
            if end_s - start_s <= 0:
                continue
            raw_turns.append((start_s, end_s, str(local_label)))
    except Exception:  # noqa: BLE001
        log.exception("diarization_annotation_iter_failed")
        return _single_turn_fallback(registry, audio, sample_rate)

    if not raw_turns:
        return _single_turn_fallback(registry, audio, sample_rate)

    # Compute one centroid embedding per LOCAL label across this segment, not
    # per turn — speaker turns from the same person inside one segment must
    # collapse onto the same stable id.
    label_to_audio: dict[str, list[np.ndarray]] = {}
    for start_s, end_s, local_label in raw_turns:
        s_idx = max(0, int(start_s * sample_rate))
        e_idx = min(audio.size, int(end_s * sample_rate))
        if e_idx <= s_idx:
            continue
        label_to_audio.setdefault(local_label, []).append(audio[s_idx:e_idx])

    label_to_global: dict[str, str] = {}
    for local_label, slices in label_to_audio.items():
        concat = np.concatenate(slices) if len(slices) > 1 else slices[0]
        global_id: str | None = None
        if embedder is not None:
            try:
                emb = embedder(_audio_to_pyannote_input(concat, sample_rate))
                emb_np = _embedding_to_numpy(emb)
                global_id = registry.assign(emb_np)
            except Exception:  # noqa: BLE001
                log.exception(
                    "diarization_embedding_failed",
                    extra={"local_label": local_label},
                )
        if global_id is None:
            # Embedding failed or unavailable — treat each unique local label as
            # a new speaker so at least the local turn structure is preserved.
            global_id = registry.assign(np.zeros(1, dtype=np.float32))
        label_to_global[local_label] = global_id

    # Merge consecutive turns by the same global speaker (smoother UI).
    sorted_turns = sorted(raw_turns, key=lambda t: t[0])
    merged: list[SpeakerTurn] = []
    for start_s, end_s, local_label in sorted_turns:
        global_id = label_to_global[local_label]
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        if merged and merged[-1].speaker_id == global_id and start_ms <= merged[-1].end_ms + 50:
            merged[-1] = SpeakerTurn(
                speaker_id=global_id,
                start_ms=merged[-1].start_ms,
                end_ms=max(merged[-1].end_ms, end_ms),
            )
        else:
            merged.append(SpeakerTurn(speaker_id=global_id, start_ms=start_ms, end_ms=end_ms))

    return merged


async def diarize_segment(
    audio: np.ndarray,
    sample_rate: int,
    registry: SessionSpeakerRegistry,
    settings: Settings | None = None,
) -> list[SpeakerTurn]:
    """Async wrapper that serializes calls and dispatches to the executor."""
    loop = asyncio.get_running_loop()
    async with _run_lock:
        return await loop.run_in_executor(
            None,
            diarize_segment_sync,
            audio,
            sample_rate,
            registry,
            settings,
        )


def _embedding_to_numpy(emb: Any) -> np.ndarray:
    """Coerce pyannote ``Inference`` output (np.ndarray / SlidingWindowFeature / tensor) to 1D float32."""
    if isinstance(emb, np.ndarray):
        arr = emb
    elif hasattr(emb, "data"):  # SlidingWindowFeature
        arr = np.asarray(emb.data)
    elif torch.is_tensor(emb):
        arr = emb.detach().cpu().numpy()
    else:
        arr = np.asarray(emb)
    return arr.astype(np.float32, copy=False).reshape(-1)


def _single_turn_fallback(
    registry: SessionSpeakerRegistry,
    audio: np.ndarray,
    sample_rate: int,
) -> list[SpeakerTurn]:
    """One-speaker fallback: tag the whole segment as a single speaker turn.

    If the embedder is loaded, we still compute one embedding for the whole
    segment so stable ``spk_N`` labels carry across segments. If not, we
    reuse the last-seen speaker when the registry is non-empty (so a
    pipeline-less session collapses onto a single ``spk_1`` instead of
    creating a new id per segment).
    """
    duration_ms = int(round(audio.size / float(sample_rate) * 1000))
    if duration_ms <= 0:
        return []

    settings = get_settings()
    embedder = get_embedder(settings)
    speaker_id: str | None = None

    if embedder is not None:
        try:
            emb = embedder(_audio_to_pyannote_input(audio, sample_rate))
            speaker_id = registry.assign(_embedding_to_numpy(emb))
        except Exception:  # noqa: BLE001
            log.exception("diarization_fallback_embedding_failed")

    if speaker_id is None:
        if registry.centroids:
            speaker_id = _label(len(registry.centroids) - 1)
        else:
            speaker_id = registry.assign(np.zeros(1, dtype=np.float32))

    return [SpeakerTurn(speaker_id=speaker_id, start_ms=0, end_ms=duration_ms)]
