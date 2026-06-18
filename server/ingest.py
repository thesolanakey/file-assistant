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


def detect_topic(note: str) -> str:
    """Derive a single short topic name from a note via Claude Haiku.

    Empty note or any failure -> the default `documents`. Pure string result;
    does not create any directories.
    """
    note = (note or "").strip()
    if not note:
        return _DEFAULT_FOLDER
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
        return _sanitize_folder(raw)
    except Exception as exc:  # noqa: BLE001 - never fail on topic detection
        print(f"[detect_topic] error, defaulting to {_DEFAULT_FOLDER}: {exc}", flush=True)
        return _DEFAULT_FOLDER


def detect_folder(note: str) -> str:
    """detect_topic + create (or reuse) the folder under WATCH_DIR.

    Never writes to the watch root — always a subfolder.
    """
    folder = detect_topic(note)
    os.makedirs(os.path.join(settings.WATCH_DIR, folder), exist_ok=True)
    return folder


def file_md5(path: str) -> str:
    """MD5 of a file's bytes, streamed."""
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
    fhash = file_md5(path)

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
                    "file_hash": fhash,
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


# --- Registry: lazy (on-demand) indexing -----------------------------------
#
# Folders/files are *registered* instantly into the `registry` collection as
# `pending`. They are only parsed + embedded into the `files` collection when a
# search actually needs them (embed_on_demand), then marked `indexed`.
#
# The registry stores metadata only, so its points carry a dummy 1-dim vector.

_REGISTRY_DIM = 1
_REGISTRY_DUMMY_VEC = [0.0]

# Directory/extension skip rules for folder scans.
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", "venv", ".venv", ".idea", ".mypy_cache"}
_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg",
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dll", ".exe", ".bin", ".o", ".a", ".dylib", ".class", ".jar",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
}


def ensure_registry_collection() -> None:
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.QDRANT_REGISTRY_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_REGISTRY_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=_REGISTRY_DIM, distance=qmodels.Distance.COSINE
            ),
        )


def _registry_skip(relpath: str) -> bool:
    """Skip junk for folder scans: skip dirs, binaries, db/pyc, hidden files."""
    relpath = relpath.replace("\\", "/")
    parts = [p for p in relpath.split("/") if p]
    for p in parts[:-1]:
        if p in _SKIP_DIRS or p.startswith("."):
            return True
    name = parts[-1] if parts else relpath
    if not name or name.startswith("."):
        return True
    if name in _SKIP_NAMES:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in _SKIP_EXTENSIONS or ext in _BINARY_EXTS:
        return True
    return False


def _registry_scroll(must):
    """Scroll all registry points matching the given filter conditions."""
    client = get_qdrant()
    out = []
    offset = None
    flt = qmodels.Filter(must=must)
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_REGISTRY_COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        out.extend(points)
        if offset is None:
            break
    return out


def check_duplicate(file_hash: str, filename: str) -> tuple[str, dict | None]:
    """Check registry + files collection for a matching file_hash / filename.

    Returns one of:
      ("exact", payload)        same content already known
      ("name_conflict", None)   same filename, different content
      ("new", None)             not seen before
    """
    ensure_registry_collection()
    ensure_collection()
    client = get_qdrant()

    for coll in (settings.QDRANT_REGISTRY_COLLECTION, settings.QDRANT_COLLECTION):
        pts, _ = client.scroll(
            collection_name=coll,
            scroll_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="file_hash", match=qmodels.MatchValue(value=file_hash))
            ]),
            with_payload=True, limit=1,
        )
        if pts:
            return "exact", (pts[0].payload or {})

    # same filename, different hash?
    for coll in (settings.QDRANT_REGISTRY_COLLECTION, settings.QDRANT_COLLECTION):
        pts, _ = client.scroll(
            collection_name=coll,
            scroll_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value=filename))
            ]),
            with_payload=True, limit=1,
        )
        if pts:
            return "name_conflict", None
    return "new", None


def register_folder(path: str, note: str, topic: str | None = None) -> dict:
    """Recursively scan a folder, register each file as `pending` in the
    registry, and create a folder entry. Embedding happens later, on demand."""
    if not os.path.isdir(path):
        raise NotADirectoryError(path)

    ensure_registry_collection()
    note = (note or "").strip()
    if topic is None:
        topic = detect_topic(note)

    date_registered = datetime.now(timezone.utc).date().isoformat()
    folder_id = str(uuid.uuid4())

    points = []
    registered = 0
    duplicates = 0
    for root, dirs, files in os.walk(path):
        # prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            fpath = os.path.join(root, name)
            rel = os.path.relpath(fpath, path)
            if _registry_skip(rel):
                continue
            try:
                fhash = file_md5(fpath)
            except OSError:
                continue
            kind, _existing = check_duplicate(fhash, name)
            if kind == "exact":
                duplicates += 1
                continue
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=_REGISTRY_DUMMY_VEC,
                    payload={
                        "type": "file",
                        "folder_id": folder_id,
                        "path": os.path.abspath(fpath),
                        "filename": name,
                        "note": note,
                        "topic": topic,
                        "status": "pending",
                        "file_hash": fhash,
                        "date_registered": date_registered,
                    },
                )
            )
            registered += 1

    # folder entry
    points.append(
        qmodels.PointStruct(
            id=folder_id,
            vector=_REGISTRY_DUMMY_VEC,
            payload={
                "id": folder_id,
                "type": "folder",
                "path": os.path.abspath(path),
                "note": note,
                "topic": topic,
                "status": "pending",
                "file_count": registered,
                "indexed_count": 0,
                "date_registered": date_registered,
            },
        )
    )
    if points:
        get_qdrant().upsert(collection_name=settings.QDRANT_REGISTRY_COLLECTION, points=points)

    return {
        "registered": registered,
        "topic": topic,
        "status": "pending",
        "folder_id": folder_id,
        "duplicates_skipped": duplicates,
    }


def _folder_topics(folder_ids: list[str]) -> set[str]:
    if not folder_ids:
        return set()
    pts = get_qdrant().retrieve(
        collection_name=settings.QDRANT_REGISTRY_COLLECTION,
        ids=folder_ids, with_payload=True,
    )
    topics = set()
    for p in pts:
        pl = p.payload or {}
        if pl.get("type") == "folder" and pl.get("topic"):
            topics.add(pl["topic"])
    return topics


def _pending_file_entries(folder_ids: list[str] | None = None):
    must = [
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="file")),
        qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="pending")),
    ]
    if folder_ids:
        must.append(qmodels.FieldCondition(key="folder_id", match=qmodels.MatchAny(any=folder_ids)))
    return _registry_scroll(must)


def _mark_indexed(point_id: str, payload: dict) -> None:
    client = get_qdrant()
    client.set_payload(
        collection_name=settings.QDRANT_REGISTRY_COLLECTION,
        payload={"status": "indexed"},
        points=[point_id],
    )
    # bump the parent folder's indexed_count / status
    folder_id = payload.get("folder_id")
    if not folder_id:
        return
    fpts = client.retrieve(
        collection_name=settings.QDRANT_REGISTRY_COLLECTION, ids=[folder_id], with_payload=True
    )
    if not fpts:
        return
    fpl = fpts[0].payload or {}
    indexed_count = int(fpl.get("indexed_count", 0)) + 1
    file_count = int(fpl.get("file_count", 0))
    status = "indexed" if file_count and indexed_count >= file_count else "partial"
    client.set_payload(
        collection_name=settings.QDRANT_REGISTRY_COLLECTION,
        payload={"indexed_count": indexed_count, "status": status},
        points=[folder_id],
    )


_MAX_ON_DEMAND = 25  # cap files embedded per on-demand pass


def embed_on_demand(query: str, folder_ids: list[str] | None = None) -> dict:
    """Match pending registry files against the query (by filename + note +
    topic), embed only the matches into the `files` collection, and mark them
    indexed. If folder_ids is given, all pending files in those folders are
    eligible (explicit scope); otherwise only token-overlapping files are."""
    pending = _pending_file_entries(folder_ids)
    if not pending:
        return {"indexed": 0, "candidates": 0}

    qtokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    scored = []
    for p in pending:
        pl = p.payload or {}
        hay = " ".join([pl.get("filename", ""), pl.get("note", ""), pl.get("topic", "")]).lower()
        overlap = len(qtokens & set(re.findall(r"[a-z0-9]+", hay)))
        if folder_ids or overlap > 0:
            scored.append((overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scored[:_MAX_ON_DEMAND]]

    indexed = 0
    for p in selected:
        pl = p.payload or {}
        path = pl.get("path")
        if path and os.path.isfile(path):
            try:
                res = ingest_path(path, note=pl.get("note", ""), folder=pl.get("topic"))
                if res.get("status") == "ingested":
                    _mark_indexed(p.id, pl)
                    indexed += 1
                elif res.get("status") == "skipped":
                    # already in files collection — still mark it done
                    _mark_indexed(p.id, pl)
            except Exception as exc:  # noqa: BLE001
                print(f"[embed_on_demand] error on {path}: {exc}", flush=True)

    return {"indexed": indexed, "candidates": len(scored)}


def list_registered_folders() -> list[dict]:
    """List folder registry entries with live indexed/pending status."""
    folders = _registry_scroll([
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="folder"))
    ])
    out = []
    for f in folders:
        pl = f.payload or {}
        fid = pl.get("id") or f.id
        files = _registry_scroll([
            qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="file")),
            qmodels.FieldCondition(key="folder_id", match=qmodels.MatchValue(value=fid)),
        ])
        total = len(files)
        indexed = sum(1 for x in files if (x.payload or {}).get("status") == "indexed")
        if total == 0 or indexed == 0:
            status = "pending"
        elif indexed >= total:
            status = "fully-indexed"
        else:
            status = "partially-indexed"
        out.append({
            "id": fid,
            "topic": pl.get("topic"),
            "note": pl.get("note"),
            "path": pl.get("path"),
            "file_count": total,
            "indexed_count": indexed,
            "status": status,
            "date_registered": pl.get("date_registered"),
        })
    out.sort(key=lambda x: (x.get("topic") or ""))
    return out


def remove_folder(folder_id: str) -> dict:
    """Remove a folder + its file entries from the registry. Does NOT delete the
    actual files on disk or any embeddings already in the files collection."""
    client = get_qdrant()
    client.delete(
        collection_name=settings.QDRANT_REGISTRY_COLLECTION,
        points_selector=qmodels.FilterSelector(filter=qmodels.Filter(should=[
            qmodels.FieldCondition(key="folder_id", match=qmodels.MatchValue(value=folder_id)),
            qmodels.HasIdCondition(has_id=[folder_id]),
        ])),
    )
    return {"removed": folder_id}


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

    import hashlib

    filename = os.path.basename(file.filename)
    note_clean = (note or "").strip()
    contents = await file.read()
    fhash = hashlib.md5(contents).hexdigest()

    # Duplicate check against registry + files collection BEFORE ingest.
    dup_kind, existing = await run_in_threadpool(check_duplicate, fhash, filename)
    if dup_kind == "exact":
        existing = existing or {}
        return {
            "status": "duplicate",
            "filename": filename,
            "note": note_clean,
            "folder": existing.get("folder") or existing.get("topic"),
            "message": "Exact duplicate — this file is already indexed, skipped.",
        }

    folder = await run_in_threadpool(detect_folder, note_clean)

    save_name = filename
    renamed = False
    if dup_kind == "name_conflict":
        # Same filename, different content — keep both by appending a timestamp.
        stem, ext = os.path.splitext(filename)
        save_name = f"{stem}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{ext}"
        renamed = True

    dest_dir = os.path.join(settings.WATCH_DIR, folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, save_name)
    abspath = os.path.abspath(dest)

    # Register before writing so the watcher's on_created skips this file.
    _uploading.add(abspath)
    try:
        with open(dest, "wb") as f:
            f.write(contents)
        result = await run_in_threadpool(ingest_path, dest, note_clean, folder)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _uploading.discard(abspath)

    resp = {
        "status": result.get("status", "ingested"),
        "filename": save_name,
        "note": note_clean,
        "folder": folder,
        "detail": result,
    }
    if renamed:
        resp["original_filename"] = filename
        resp["message"] = f"A different file named '{filename}' already exists — saved as '{save_name}'."
    return resp


@router.post("/register_folder")
async def register_folder_endpoint(
    files: list[UploadFile] = File(...),
    note: str = Form(""),
):
    """Register a folder for lazy indexing. The browser sends every file in the
    selected directory (webkitdirectory); we save them under REGISTERED_DIR (a
    non-watched area) and register each as pending — nothing is embedded yet."""
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")

    note_clean = (note or "").strip()
    topic = await run_in_threadpool(detect_topic, note_clean)

    # Derive the uploaded folder's top-level name from the relative paths.
    rels = [(f.filename or "").replace("\\", "/").lstrip("/") for f in files]
    top = "uploaded-folder"
    for r in rels:
        if "/" in r:
            top = r.split("/")[0]
            break

    base = os.path.join(settings.REGISTERED_DIR, topic, top)
    if os.path.exists(base):
        base = base + "-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    saved = 0
    for f in files:
        rel = (f.filename or "").replace("\\", "/").lstrip("/")
        data = await f.read()  # always drain
        if not rel or _registry_skip(rel):
            continue
        dest = os.path.join(base, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as out:
            out.write(data)
        saved += 1

    if saved == 0:
        return {"registered": 0, "topic": topic, "status": "pending",
                "note": note_clean, "message": "no indexable files in folder"}

    result = await run_in_threadpool(register_folder, base, note_clean, topic)
    result["note"] = note_clean
    return result


@router.get("/folders")
def folders_endpoint():
    """List all registered folders with their indexing status."""
    ensure_registry_collection()
    folders = list_registered_folders()
    return {"count": len(folders), "folders": folders}


@router.delete("/folders/{folder_id}")
def delete_folder_endpoint(folder_id: str):
    """Remove a folder from the registry (does not delete files or embeddings)."""
    return remove_folder(folder_id)
