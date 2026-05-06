"""REST endpoints for runtime-mutable feature toggles.

``GET  /settings`` returns the current state.
``PUT  /settings`` accepts a partial update and returns the new state.

Currently only the ``use_noise_removal`` flag is exposed; future toggles
(diarization, etc.) can be added as additional fields here without breaking
existing clients.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from app.core.runtime_settings import get_runtime_settings


router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    use_noise_removal: bool


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_noise_removal: bool | None = None


@router.get("", response_model=SettingsResponse)
async def get_settings_endpoint(request: Request) -> SettingsResponse:
    rt = get_runtime_settings(request.app)
    return SettingsResponse(**rt.get())


@router.put("", response_model=SettingsResponse)
async def update_settings_endpoint(
    payload: SettingsUpdate, request: Request
) -> SettingsResponse:
    rt = get_runtime_settings(request.app)
    new_state = rt.update(use_noise_removal=payload.use_noise_removal)
    return SettingsResponse(**new_state)
