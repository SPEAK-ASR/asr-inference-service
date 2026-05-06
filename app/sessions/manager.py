"""Per-session state and lifecycle manager.

The manager owns:
- A dictionary of `SessionState` keyed by `session_id`.
- An asyncio reaper task that closes stale sessions on idle timeout.

It does NOT own the WebSocket itself; the WebSocket route registers a
``close_callback`` so the reaper can ask the route to send a final
``error`` / ``session_summary`` event and then close the socket.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.asr.decoder import IncrementalDecoder
from app.asr.streaming_engine import EngineBuffer
from app.asr.vad import VadSegmenterState, reset_segment
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


CloseCallback = Callable[[str], Awaitable[None]]


@dataclass
class SessionState:
    """Mutable runtime state for a single client connection."""

    session_id: str
    sample_rate: int
    language_hint: str | None
    utterance_id: str = field(default_factory=lambda: f"u{uuid.uuid4().hex[:8]}")
    engine_buffer: EngineBuffer = field(init=False)
    decoder: IncrementalDecoder = field(default_factory=IncrementalDecoder)
    vad_segmenter: VadSegmenterState = field(default_factory=VadSegmenterState)

    last_audio_seq: int = -1
    chunks_received: int = 0
    bytes_received: int = 0
    utterances_finalized: int = 0
    started_at: float = field(default_factory=time.time)
    last_chunk_at: float = field(default_factory=time.time)
    last_partial_emit_at: float = 0.0

    diarization_enabled: bool = False
    noise_removal_enabled: bool = False
    last_speaker: str | None = None

    close_callback: CloseCallback | None = None

    def __post_init__(self) -> None:
        self.engine_buffer = EngineBuffer(sample_rate=self.sample_rate)

    def touch(self) -> None:
        self.last_chunk_at = time.time()

    def new_utterance(self) -> str:
        self.utterance_id = f"u{uuid.uuid4().hex[:8]}"
        self.last_speaker = None
        self.engine_buffer.reset_for_new_utterance()
        reset_segment(self.vad_segmenter)
        self.decoder = IncrementalDecoder(
            min_partial_char_delta=self.decoder.min_partial_char_delta,
        )
        return self.utterance_id

    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)


class SessionManager:
    """Tracks active sessions and reaps stale ones."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            for state in list(self._sessions.values()):
                if state.close_callback:
                    try:
                        await state.close_callback("server_shutdown")
                    except Exception:  # noqa: BLE001
                        log.exception("session_close_failed", extra={"session_id": state.session_id})
            self._sessions.clear()

    # --- Registry ----------------------------------------------------------

    async def register(self, state: SessionState) -> None:
        async with self._lock:
            self._sessions[state.session_id] = state
        log.info(
            "session_registered",
            extra={
                "session_id": state.session_id,
                "active_sessions": len(self._sessions),
            },
        )

    async def unregister(self, session_id: str) -> SessionState | None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
        if state:
            log.info(
                "session_unregistered",
                extra={
                    "session_id": session_id,
                    "active_sessions": len(self._sessions),
                    "duration_ms": state.duration_ms(),
                    "chunks": state.chunks_received,
                    "utterances": state.utterances_finalized,
                },
            )
        return state

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def active_count(self) -> int:
        return len(self._sessions)

    # --- Reaper ------------------------------------------------------------

    async def _reaper_loop(self) -> None:
        log.info("session_reaper_started")
        try:
            while True:
                await asyncio.sleep(self.settings.reaper_interval_seconds)
                await self._reap_once()
        except asyncio.CancelledError:
            log.info("session_reaper_stopped")
            raise

    async def _reap_once(self) -> None:
        now = time.time()
        timeout = self.settings.idle_timeout_seconds
        stale: list[SessionState] = []
        async with self._lock:
            for state in list(self._sessions.values()):
                if (now - state.last_chunk_at) > timeout:
                    stale.append(state)

        for state in stale:
            log.warning(
                "session_idle_timeout",
                extra={
                    "session_id": state.session_id,
                    "idle_seconds": int(now - state.last_chunk_at),
                },
            )
            if state.close_callback:
                try:
                    await state.close_callback("idle_timeout")
                except Exception:  # noqa: BLE001
                    log.exception("session_close_failed", extra={"session_id": state.session_id})
            await self.unregister(state.session_id)
