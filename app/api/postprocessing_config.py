"""HTTP API to toggle optional postprocessing defaults (diarization, noise removal)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings

router = APIRouter(prefix="/api/postprocessing", tags=["postprocessing"])


class PostprocessingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diarization_enabled: bool | None = Field(
        default=None,
        description="If set, updates the server default for new sessions.",
    )
    noise_removal_enabled: bool | None = Field(
        default=None,
        description="If set, updates the server default for new sessions.",
    )


class PostprocessingView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diarization_enabled: bool
    noise_removal_enabled: bool
    diarization_model_id: str
    hf_token_configured: bool


def _runtime(request: Request):
    return request.app.state.postprocessing


@router.get("", response_model=PostprocessingView)
async def get_postprocessing(request: Request) -> PostprocessingView:
    settings = get_settings()
    snap = await _runtime(request).snapshot()
    return PostprocessingView(
        diarization_enabled=snap["diarization_enabled"],
        noise_removal_enabled=snap["noise_removal_enabled"],
        diarization_model_id=settings.diarization_model_id,
        hf_token_configured=bool(settings.resolved_hf_token()),
    )


@router.put("", response_model=PostprocessingView)
async def put_postprocessing(request: Request, body: PostprocessingBody) -> PostprocessingView:
    settings = get_settings()
    snap = await _runtime(request).update(
        diarization_enabled=body.diarization_enabled,
        noise_removal_enabled=body.noise_removal_enabled,
    )
    return PostprocessingView(
        diarization_enabled=snap["diarization_enabled"],
        noise_removal_enabled=snap["noise_removal_enabled"],
        diarization_model_id=settings.diarization_model_id,
        hf_token_configured=bool(settings.resolved_hf_token()),
    )
