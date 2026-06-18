"""FastAPI application entry point.

On startup: ensure the Qdrant `files` collection exists and start the file
watcher. Mounts the ingest and query routers and exposes /health.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from server import ingest, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create the collection up front so the watcher and queries have it ready.
    ingest.ensure_collection()
    ingest.start_watcher()
    yield


app = FastAPI(title="RAG File Assistant", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok", "backend": settings.GENERATION_BACKEND}
