"""Pydantic schemas for the WebSocket transcription protocol and admin API.

Implements the contract described in section 6 of `investigated_detail.md`.

All events carry an explicit `type` discriminator so we can dispatch with a
tagged union on the server (`ClientMessage`) and serialize cleanly on the
client side.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Shared / embedded models
# -----------------------------------------------------------------------------

class SpeakerSegment(BaseModel):
    """A contiguous audio span attributed to one speaker, embedded in FinalTranscript."""

    speaker: str
    start_ms: int
    end_ms: int


# -----------------------------------------------------------------------------
# Client -> Server messages
# -----------------------------------------------------------------------------

class _BaseClientMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StartMsg(_BaseClientMsg):
    """Initial message; opens an ASR session."""

    type: Literal["start"] = "start"
    session_id: str = Field(min_length=1, max_length=128)
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    channels: Literal[1] = 1
    language_hint: str | None = Field(default="si", max_length=8)


class AudioChunkMsg(_BaseClientMsg):
    """A base64-encoded PCM16 audio frame."""

    type: Literal["audio_chunk"] = "audio_chunk"
    seq: int = Field(ge=0)
    audio_b64: str
    duration_ms: int = Field(ge=1, le=2_000)


class EndUtteranceMsg(_BaseClientMsg):
    """Force finalization of the current utterance."""

    type: Literal["end_utterance"] = "end_utterance"
    seq: int | None = Field(default=None, ge=0)


class StopMsg(_BaseClientMsg):
    """End the streaming session gracefully."""

    type: Literal["stop"] = "stop"


class PingMsg(_BaseClientMsg):
    """Heartbeat from client; server replies with `ack` of message=`pong`."""

    type: Literal["ping"] = "ping"


ClientMessage = Annotated[
    Union[StartMsg, AudioChunkMsg, EndUtteranceMsg, StopMsg, PingMsg],
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Server -> Client messages
# -----------------------------------------------------------------------------

class _BaseServerMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ack(_BaseServerMsg):
    type: Literal["ack"] = "ack"
    session_id: str
    message: str


class PartialTranscript(_BaseServerMsg):
    type: Literal["partial_transcript"] = "partial_transcript"
    session_id: str
    utterance_id: str
    seq: int
    text: str
    start_ms: int
    end_ms: int
    is_stable: bool = False


class FinalTranscript(_BaseServerMsg):
    type: Literal["final_transcript"] = "final_transcript"
    session_id: str
    utterance_id: str
    text: str
    start_ms: int
    end_ms: int
    speaker: str | None = None
    speaker_segments: list[SpeakerSegment] | None = None


class Warning(_BaseServerMsg):
    type: Literal["warning"] = "warning"
    code: str
    message: str


class Error(_BaseServerMsg):
    type: Literal["error"] = "error"
    code: str
    message: str


class SessionSummary(_BaseServerMsg):
    type: Literal["session_summary"] = "session_summary"
    session_id: str
    utterances: int
    duration_ms: int
    reason: str


ServerMessage = Annotated[
    Union[Ack, PartialTranscript, FinalTranscript, Warning, Error, SessionSummary],
    Field(discriminator="type"),
]


# -----------------------------------------------------------------------------
# Admin API models
# -----------------------------------------------------------------------------

class DiarizationToggle(BaseModel):
    """Request body for POST /admin/diarization."""

    enabled: bool


class DiarizationStatus(BaseModel):
    """Response for GET and POST /admin/diarization."""

    enabled: bool
    model_loaded: bool


# Standard error codes (section 9 of investigated_detail.md)
class ErrorCode:
    INVALID_AUDIO_FORMAT = "INVALID_AUDIO_FORMAT"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
