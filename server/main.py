"""FastAPI application entry point.

On startup: init SQLite, best-effort provision Qdrant collections, and start the
file watcher. Mounts the API routers, renders the chat UI (a single Jinja2
template) at "/", and exposes an extended /health used by the UI's live status.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import settings
from server import db, generate, ingest, memory, modes, query, runtime

_BASE_DIR = os.path.dirname(__file__)
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
_TEMPLATES_DIR = os.path.join(_BASE_DIR, "templates")

templates = Jinja2Templates(directory=_TEMPLATES_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite (conversation memory + persisted mode/config) first.
    db.init_db()
    # Best-effort: provision collections for the active mode's Qdrant. This must
    # NOT crash startup — if the active mode points at an unreachable instance
    # (e.g. hetzner remote is down), the app should still boot; collections are
    # ensured lazily when that Qdrant is first reached (see ingest.get_qdrant).
    try:
        ingest.ensure_collection()
        ingest.ensure_registry_collection()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] collection bootstrap skipped: {exc}", flush=True)
    ingest.start_watcher()
    yield


app = FastAPI(title="RAG File Assistant", lifespan=lifespan)

# API routes — registered before the catch-all static mount so they take
# precedence over the StaticFiles handler.
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(modes.router)
app.include_router(memory.router)
app.include_router(runtime.router)


def _ram_percent() -> float | None:
    """RAM in use as a percentage, parsed from /proc/meminfo (no deps)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0]] = int(parts[1].strip().split()[0])  # kB
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total and avail is not None:
            return round((1 - avail / total) * 100, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def _qdrant_stats() -> tuple[str, int | None]:
    """(status, doc_count) for the active mode's Qdrant — best-effort/fast."""
    try:
        client = ingest.get_qdrant()
        count = client.count(
            collection_name=settings.QDRANT_COLLECTION, exact=False
        ).count
        return "ok", int(count)
    except Exception:  # noqa: BLE001
        return "down", None


@app.get("/health")
def health():
    """Liveness + live status values polled by the UI every few seconds.

    Preserves the original fields (status, backend, mode) and adds the runtime
    config and live metrics the dashboard displays.
    """
    qdrant_status, doc_count = _qdrant_stats()
    cfg = runtime.snapshot()
    return {
        "status": "ok",
        "backend": cfg["backend"],
        "mode": modes.get_mode(),
        "model": cfg["model"],
        "fallback": cfg["fallback"],
        "qdrant": qdrant_status,
        "doc_count": doc_count,
        "msg_count": memory.count_messages(),
        "ram": _ram_percent(),
        "tps": generate.last_tps(),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Render the chat UI. Initial mode/backend/model are injected so the topbar
    paints correctly on first load (then kept live via /health polling)."""
    cfg = runtime.snapshot()
    return templates.TemplateResponse(
        "index.html.j2",
        {
            "request": request,
            "mode": modes.get_mode(),
            "backend": cfg["backend"],
            "model": cfg["model"],
            "fallback": cfg["fallback"],
        },
    )


# Static assets (and the preserved legacy file-manager UI at /files.html). Added
# LAST so it only handles paths the routes above didn't claim. The explicit "/"
# route above takes precedence over this mount's index.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
app.mount("/legacy", StaticFiles(directory=_STATIC_DIR, html=True), name="legacy")
