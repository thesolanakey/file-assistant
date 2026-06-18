"""Ingestion: file watching, parsing, embedding, and storage in Qdrant.

Exposes:
  - get_qdrant()            shared QdrantClient singleton
  - ensure_collection()     create the `files` collection if missing
  - ingest_path(path)       parse+embed+store a single file (with dedup)
  - start_watcher()         background watchdog observer over WATCH_DIR
  - router                  FastAPI router with POST /ingest
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from server import embed
from server.parsers import parse_file

router = APIRouter()

_qdrant: QdrantClient | None = None
_qdrant_lock = threading.Lock()

# Absolute paths currently being ingested via /upload. The file watcher skips
# these so it can't race the upload and ingest the file without its note.
_uploading: set[str] = set()

# Extensions / names we never ingest.
_SKIP_EXTENSIONS = {".pyc", ".db"}
_SKIP_NAMES = {".gitkeep"}


def get_qdrant() -> QdrantClient:
    """Return the shared QdrantClient, creating it on first use."""
    global _qdrant
    if _qdrant is None:
        with _qdrant_lock:
            if _qdrant is None:
                _qdrant = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
    return _qdrant


def ensure_collection() -> None:
    """Create the `files` collection (768-dim, cosine) if it does not exist."""
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=settings.EMBED_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _should_skip(path: str) -> bool:
    name = os.path.basename(path)
    if name.startswith("."):
        return True
    if name in _SKIP_NAMES:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in _SKIP_EXTENSIONS:
        return True
    return False


def _source_for(path: str) -> str:
    """Derive a human-readable source from the file's location under WATCH_DIR."""
    try:
        rel = os.path.relpath(path, settings.WATCH_DIR)
        top = rel.split(os.sep)[0]
        if top and top != "..":
            return top
    except ValueError:
        pass
    return "external"


_FOLDER_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_FOLDER = "documents"


def _sanitize_folder(name: str) -> str:
    """lowercase, spaces->hyphens, alphanumeric+hyphens only, max 20 chars."""
    name = (name or "").strip().lower()
    name = name.replace(" ", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)   # drop anything not alnum/hyphen
    name = re.sub(r"-+", "-", name).strip("-")
    name = name[:20].strip("-")
    return name or _DEFAULT_FOLDER


def detect_folder(note: str) -> str:
    """Derive a topic folder name from the user's note via Claude Haiku.

    Empty note or any failure -> the default `documents` folder. The chosen
    folder is created under WATCH_DIR if it does not already exist (existing
    folders are simply reused).
    """
    note = (note or "").strip()
    folder = _DEFAULT_FOLDER

    if note:
        try:
            from server import generate

            client = generate._get_claude_client()
            prompt = (
                "Given this note about a file, return only a single short folder "
                "name (lowercase, no spaces, use hyphens). Examples: 'quarterly "
                "report from accountant' → finance, 'notes from team standup' "
                "→ meetings, 'my workout plan' → health. Note: " + note
            )
            resp = client.messages.create(
                model=_FOLDER_MODEL,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text")
            folder = _sanitize_folder(raw)
        except Exception as exc:  # noqa: BLE001 - never fail an upload on this
            print(f"[detect_folder] error, defaulting to {_DEFAULT_FOLDER}: {exc}", flush=True)
            folder = _DEFAULT_FOLDER

    # Create (or reuse) the folder under WATCH_DIR. Never write to the root.
    os.makedirs(os.path.join(settings.WATCH_DIR, folder), exist_ok=True)
    return folder


def _already_ingested(filename: str, filesize: int) -> bool:
    """Dedup check: is a point with this filename+size already stored?"""
    client = get_qdrant()
    result = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="filename", match=qmodels.MatchValue(value=filename)
                ),
                qmodels.FieldCondition(
                    key="filesize", match=qmodels.MatchValue(value=filesize)
                ),
            ]
        ),
        limit=1,
    )
    points, _ = result
    return len(points) > 0


def ingest_path(path: str, note: str | None = None, folder: str | None = None) -> dict:
    """Parse, chunk, embed, and store a single file. Returns a status dict.

    If ``note`` is provided it is embedded as its own searchable chunk (so a
    query about the file's description retrieves it) and stored on every point's
    payload as ``note``. ``folder`` is stored on the payload; if omitted it is
    derived from the file's location under WATCH_DIR.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if _should_skip(path):
        return {"status": "skipped", "reason": "filtered file type", "path": path}

    filename = os.path.basename(path)
    filesize = os.path.getsize(path)

    ensure_collection()
    if _already_ingested(filename, filesize):
        return {"status": "skipped", "reason": "already ingested", "path": path}

    chunks = parse_file(path)

    note = (note or "").strip()
    if note:
        # Prepend the note as a self-describing, searchable chunk.
        chunks = [f"File note for {filename}: {note}"] + chunks

    if not chunks:
        return {"status": "skipped", "reason": "no extractable text", "path": path}

    vectors = embed.get_embeddings_batch(chunks)

    filetype = os.path.splitext(filename)[1].lower().lstrip(".") or "unknown"
    date_ingested = datetime.now(timezone.utc).isoformat()
    source = _source_for(path)
    # Folder defaults to the file's top-level location under WATCH_DIR, so the
    # watcher and uploads agree without the caller having to pass it.
    folder = folder or source

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "filepath": os.path.abspath(path),
                    "filetype": filetype,
                    "filesize": filesize,
                    "date_ingested": date_ingested,
                    "source": source,
                    "folder": folder,
                    "note": note,
                },
            )
        )

    get_qdrant().upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return {"status": "ingested", "path": path, "chunks": len(points), "note": note, "folder": folder}


# --- File watcher -----------------------------------------------------------

class _Handler:
    """watchdog event handler; defined via duck typing to keep imports lazy."""

    def __init__(self):
        from watchdog.events import FileSystemEventHandler

        # Build a concrete subclass instance.
        handler_self = self

        class _Inner(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    handler_self._safe_ingest(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    handler_self._safe_ingest(event.dest_path)

        self._inner = _Inner()

    @staticmethod
    def _safe_ingest(path: str) -> None:
        # An /upload is handling this file (with its note) — don't race it.
        if os.path.abspath(path) in _uploading:
            return
        try:
            result = ingest_path(path)
            print(f"[ingest] {result}", flush=True)
        except Exception as exc:  # noqa: BLE001 - watcher must never crash
            print(f"[ingest] error on {path}: {exc}", flush=True)


def _initial_scan() -> None:
    """Ingest anything already sitting in WATCH_DIR at startup."""
    for root, _dirs, files in os.walk(settings.WATCH_DIR):
        for name in files:
            path = os.path.join(root, name)
            if _should_skip(path):
                continue
            try:
                result = ingest_path(path)
                print(f"[ingest:startup] {result}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest:startup] error on {path}: {exc}", flush=True)


def start_watcher() -> None:
    """Run an initial scan, then start a background watchdog observer."""
    from watchdog.observers import Observer

    os.makedirs(settings.WATCH_DIR, exist_ok=True)

    # Warm the embedding model and do a first pass in a background thread so we
    # don't block FastAPI startup.
    def _bootstrap():
        embed.warmup()
        _initial_scan()

    threading.Thread(target=_bootstrap, daemon=True).start()

    handler = _Handler()
    observer = Observer()
    observer.schedule(handler._inner, settings.WATCH_DIR, recursive=True)
    observer.daemon = True
    observer.start()
    print(f"[watcher] watching {settings.WATCH_DIR}", flush=True)


# --- Manual ingest endpoint -------------------------------------------------

class IngestRequest(BaseModel):
    path: str


@router.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    """Manually trigger ingestion of a single file path."""
    try:
        return ingest_path(req.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    note: str = Form(""),
):
    """Accept a multipart upload (file + note), save it to watch/documents/,
    and ingest it with the note stored/searchable alongside the file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="no filename provided")

    filename = os.path.basename(file.filename)
    note_clean = (note or "").strip()

    # Derive the topic folder from the note (Claude call) — run off the loop.
    # detect_folder also creates the folder under WATCH_DIR. The watch root is
    # never written to directly; a subfolder is always used.
    folder = await run_in_threadpool(detect_folder, note_clean)

    dest_dir = os.path.join(settings.WATCH_DIR, folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    abspath = os.path.abspath(dest)

    # Register before writing so the watcher's on_created skips this file.
    _uploading.add(abspath)
    try:
        contents = await file.read()
        with open(dest, "wb") as f:
            f.write(contents)
        # Embedding is CPU-bound — run it off the event loop.
        result = await run_in_threadpool(ingest_path, dest, note_clean, folder)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _uploading.discard(abspath)

    return {
        "status": result.get("status", "ingested"),
        "filename": filename,
        "note": note_clean,
        "folder": folder,
        "detail": result,
    }
