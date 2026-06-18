"""FastAPI application entry point.

On startup: ensure the Qdrant `files` collection exists and start the file
watcher. Mounts the ingest and query routers, serves the web UI from
server/static at "/", and exposes /health.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from server import ingest, query

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the collection up front so the watcher and queries have it ready.
    ingest.ensure_collection()
    ingest.start_watcher()
    yield


app = FastAPI(title="RAG File Assistant", lifespan=lifespan)

# API routes — registered before the catch-all static mount so they take
# precedence over the StaticFiles handler at "/".
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok", "backend": settings.GENERATION_BACKEND}


# Web UI: serve server/static/index.html at "/" (html=True). This mount is added
# LAST so it only handles paths the API routes above didn't claim.
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
