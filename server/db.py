"""SQLite persistence layer.

Holds two things that must survive container restarts:

  * ``messages``  — the conversation memory: every message ever sent, kept
                    forever (role, content, mode, timestamp).
  * ``app_state`` — small key/value store; currently the active operational
                    mode (see :mod:`server.modes`).

The database file lives at ``settings.DB_PATH`` (under a mounted volume in
Docker). A fresh short-lived connection is opened per operation: SQLite handles
this cheaply, it sidesteps cross-thread connection-sharing issues (FastAPI runs
sync endpoints in a threadpool), and WAL mode keeps concurrent reads/writes
smooth.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config import settings


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL allows a writer and readers to proceed concurrently.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success and always closes."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the database file (and its parent dir) and tables if missing.

    Idempotent — safe to call on every startup.
    """
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    with connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                mode       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
