"""Operational mode: "local" vs "brix".

A single named runtime mode for the assistant, independent of the generation
backend and of the query intent (find/summarize). The active mode is persisted
in the ``app_state`` table under the ``mode`` key, so a ``POST /mode`` survives
restarts. On first run (nothing stored yet) it falls back to
``settings.DEFAULT_MODE``.

Endpoints:
  * ``GET  /mode``  -> {"mode": "local"|"brix"}
  * ``POST /mode``  {"mode": "local"|"brix"} -> {"mode": ..., "previous": ...}
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from server import db

router = APIRouter()

_MODE_KEY = "mode"


def get_mode() -> str:
    """Return the persisted active mode, or the configured default if unset."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (_MODE_KEY,)
        ).fetchone()
    if row is None:
        return settings.DEFAULT_MODE
    return row["value"]


def set_mode(mode: str) -> str:
    """Validate and persist the active mode; return the normalized value."""
    normalized = mode.strip().lower()
    if normalized not in settings.ALLOWED_MODES:
        raise ValueError(
            f"mode must be one of {sorted(settings.ALLOWED_MODES)}, got {mode!r}"
        )
    now = datetime.now(timezone.utc).isoformat()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (_MODE_KEY, normalized, now),
        )
    return normalized


class ModeRequest(BaseModel):
    mode: str


@router.get("/mode")
def get_mode_endpoint():
    return {"mode": get_mode(), "allowed": sorted(settings.ALLOWED_MODES)}


@router.post("/mode")
def set_mode_endpoint(req: ModeRequest):
    previous = get_mode()
    try:
        active = set_mode(req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"mode": active, "previous": previous, "allowed": sorted(settings.ALLOWED_MODES)}
