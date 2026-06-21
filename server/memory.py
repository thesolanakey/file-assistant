"""Conversation memory.

Every message that flows through the assistant is persisted to the ``messages``
table and kept forever. A message records:

  * ``role``       — "user" or "assistant"
  * ``content``    — the message text
  * ``mode``       — the active operational mode ("local"/"brix") at the time
  * ``created_at`` — ISO-8601 UTC timestamp

A ``GET /messages`` endpoint is exposed so the stored history can be read back
(handy for verification and for any future UI). Storage itself is wired into the
query flow (see :mod:`server.query`).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from server import db, modes

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_message(role: str, content: str, mode: str) -> int:
    """Persist a single message and return its row id."""
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO messages (role, content, mode, created_at) VALUES (?, ?, ?, ?)",
            (role, content, mode, _now_iso()),
        )
        return int(cur.lastrowid)


def get_messages(limit: int | None = None, mode: str | None = None) -> list[dict]:
    """Return stored messages in chronological order (oldest first).

    ``limit`` returns the most recent N messages (still oldest-first). ``mode``
    filters to messages recorded under a given operational mode.
    """
    where = "WHERE mode = ?" if mode is not None else ""
    params: list = [mode] if mode is not None else []

    with db.connection() as conn:
        if limit is not None:
            # Grab the newest `limit` rows, then flip back to chronological order.
            inner = f"SELECT id, role, content, mode, created_at FROM messages {where} ORDER BY id DESC LIMIT ?"
            rows = conn.execute(
                f"SELECT * FROM ({inner}) ORDER BY id ASC",
                (*params, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT id, role, content, mode, created_at FROM messages {where} ORDER BY id ASC",
                params,
            ).fetchall()
    return [dict(row) for row in rows]


def count_messages(mode: str | None = None) -> int:
    """Count stored messages, optionally scoped to one profile."""
    where = "WHERE mode = ?" if mode is not None else ""
    params: tuple = (mode,) if mode is not None else ()
    with db.connection() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM messages {where}", params).fetchone()[0])


def clear_messages(mode: str | None = None) -> int:
    """Delete stored messages (all, or only one profile's). Returns count removed."""
    where = "WHERE mode = ?" if mode is not None else ""
    params: tuple = (mode,) if mode is not None else ()
    with db.connection() as conn:
        n = int(conn.execute(f"SELECT COUNT(*) FROM messages {where}", params).fetchone()[0])
        conn.execute(f"DELETE FROM messages {where}", params)
    return n


@router.get("/messages")
def messages_endpoint(
    limit: int | None = Query(default=None, ge=1, le=1000),
    mode: str | None = Query(default=None),
):
    """Return persisted conversation history (oldest first)."""
    return {
        "count": count_messages(),
        "messages": get_messages(limit=limit, mode=mode),
    }


@router.get("/history")
def history_endpoint(limit: int = Query(default=100, ge=1, le=1000)):
    """Conversation history for the chat UI (oldest first, most recent N).

    Scoped to the active profile — each profile has its own conversation.
    """
    mode = modes.get_mode()
    return {
        "count": count_messages(mode),
        "messages": get_messages(limit=limit, mode=mode),
    }


@router.delete("/history")
def clear_history_endpoint():
    """Clear the active profile's conversation history. Returns {"status": "ok"}.

    Called by the UI's "new chat" button — only the current profile is cleared.
    """
    clear_messages(modes.get_mode())
    return {"status": "ok"}
