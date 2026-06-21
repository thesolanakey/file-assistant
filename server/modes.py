"""Active profile: "assistant" vs "friend".

The profile is the user-facing toggle. Switching profiles changes the Ollama
model, the Qdrant collection, the personality (system prompt), and which slice
of conversation history is loaded — all defined in :data:`settings.PROFILES`.
The active profile is persisted in the ``app_state`` table under the ``mode``
key, so a ``POST /mode`` survives restarts. On first run (nothing stored yet, or
a stale value from an earlier scheme) it falls back to ``settings.DEFAULT_MODE``.

Endpoints:
  * ``GET  /mode``  -> {"mode": "assistant"|"friend"}
  * ``POST /mode``  {"mode": "assistant"|"friend"} -> {"mode": ..., "previous": ...}
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
    """Return the persisted active profile, or the default if unset/stale."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (_MODE_KEY,)
        ).fetchone()
    if row is None or row["value"] not in settings.ALLOWED_MODES:
        # Unset, or left over from the old local/brix scheme -> default profile.
        return settings.DEFAULT_MODE
    return row["value"]


def get_collection() -> str:
    """The active profile's Qdrant collection."""
    return settings.collection_for(get_mode())


def set_mode(mode: str) -> str:
    """Validate and persist the active profile; return the normalized value.

    Switching a profile also points generation at that profile's model (so the
    correct Ollama model is loaded for the new personality).
    """
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

    # Point generation at the new profile's model and best-effort warm it in
    # Ollama so the first message isn't slow. Both are non-fatal.
    from server import runtime, generate

    model = settings.profile(normalized)["model"]
    runtime.set_model(model)
    generate.warm_model(model)
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
