"""HTTP API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranscribeHttpResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    language: str | None
    duration_ms: int
    model_kind: str
