"""Runtime-adjustable generation config (backend / model / fallback).

These are knobs the UI flips at runtime via POST /config. They are persisted in
the SQLite ``app_state`` table (same store as the active mode) so they survive
restarts, and they override the static env defaults in :mod:`config.settings`.

  * backend  — "ollama" | "claude"   (default: settings.GENERATION_BACKEND)
  * model    — Ollama model name      (default: settings.OLLAMA_MODEL)
  * fallback — bool; when True an unreachable Ollama falls back to Claude
               (default: True)

The generation layer (server/generate.py) reads these getters rather than the
env settings directly.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from server import db

router = APIRouter()

_BACKEND_KEY = "backend"
_MODEL_KEY = "model"
_FALLBACK_KEY = "fallback"


def _get(key: str) -> str | None:
    with db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None


def _set(key: str, value: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db.connection() as conn:
        conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, value, now),
        )


def get_backend() -> str:
    return _get(_BACKEND_KEY) or settings.GENERATION_BACKEND


def get_model() -> str:
    return _get(_MODEL_KEY) or settings.OLLAMA_MODEL


def get_fallback() -> bool:
    val = _get(_FALLBACK_KEY)
    return True if val is None else val == "1"


def snapshot() -> dict:
    return {
        "backend": get_backend(),
        "model": get_model(),
        "fallback": get_fallback(),
    }


class ConfigRequest(BaseModel):
    backend: str | None = None
    model: str | None = None
    fallback: bool | None = None


@router.get("/config")
def get_config():
    return snapshot()


@router.post("/config")
def set_config(req: ConfigRequest):
    if req.backend is not None:
        backend = req.backend.strip().lower()
        if backend not in settings._ALLOWED_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=f"backend must be one of {sorted(settings._ALLOWED_BACKENDS)}",
            )
        _set(_BACKEND_KEY, backend)
    if req.model is not None:
        model = req.model.strip()
        if model:
            _set(_MODEL_KEY, model)
    if req.fallback is not None:
        _set(_FALLBACK_KEY, "1" if req.fallback else "0")
    return snapshot()
