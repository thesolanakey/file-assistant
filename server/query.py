"""Query API: vector search with find/summarize intent detection.

Endpoints:
  - POST /query   {"question": str, "filters": {}}  -> chunks or summary
  - GET  /files                                      -> indexed files + metadata
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.http import models as qmodels

from config import settings
from server import embed, generate, memory, modes
from server import ingest
from server.ingest import get_qdrant

router = APIRouter()

TOP_K = 5
# How many prior messages to inject into the generation prompt as history.
HISTORY_LIMIT = 20
# If the best already-indexed hit scores below this, we look for pending
# registry files that might be more relevant and embed them on demand.
_CONFIDENCE_THRESHOLD = 0.45
_SUMMARIZE_KEYWORDS = ("summarize", "explain", "overview", "tldr")


class QueryRequest(BaseModel):
    question: str
    filters: dict = Field(default_factory=dict)
    folder_ids: list[str] = Field(default_factory=list)


def _detect_mode(question: str) -> str:
    lowered = question.lower()
    if any(keyword in lowered for keyword in _SUMMARIZE_KEYWORDS):
        return "summarize"
    return "find"


def _build_filter(filters: dict, folder_topics: set[str] | None = None) -> qmodels.Filter | None:
    """Translate a simple {field: value} dict (+ optional folder scope) into a
    Qdrant filter."""
    conditions = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in (filters or {}).items()
    ]
    if folder_topics:
        conditions.append(
            qmodels.FieldCondition(key="folder", match=qmodels.MatchAny(any=list(folder_topics)))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def _search(question: str, filters: dict, folder_topics: set[str] | None = None):
    vector = embed.get_embedding(question, is_query=True)
    return get_qdrant().search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=_build_filter(filters, folder_topics),
        limit=TOP_K,
        with_payload=True,
    )


def _assistant_content(response: dict) -> str:
    """Text to persist as the assistant's turn for a /query response.

    summarize mode has a natural-language ``answer``; find mode does not, so we
    record a concise description of what was returned.
    """
    if "answer" in response:
        return response["answer"]
    sources = response.get("sources") or []
    n = len(response.get("chunks", []))
    return f"[find] {n} chunk(s) from: {', '.join(sources) if sources else 'none'}"


@router.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Active operational mode (local/hetzner) is recorded with every message.
    active_mode = modes.get_mode()
    # Snapshot prior history BEFORE logging the current question, so generation
    # sees "conversation so far" without the current turn duplicated in it.
    history = memory.get_messages(limit=HISTORY_LIMIT)
    memory.add_message("user", req.question, active_mode)

    mode = _detect_mode(req.question)
    folder_ids = req.folder_ids or []
    # Map requested folder ids -> their topics so we can scope the vector search.
    folder_topics = ingest._folder_topics(folder_ids) if folder_ids else set()

    embedded_now = 0
    try:
        # Step 1: search what's already indexed (scoped if folders requested).
        hits = _search(req.question, req.filters, folder_topics)
        top_score = hits[0].score if hits else 0.0

        # Step 2: if scope was requested, or confidence is low, pull in matching
        # pending files from the registry and embed them on demand.
        if folder_ids or top_score < _CONFIDENCE_THRESHOLD:
            info = ingest.embed_on_demand(req.question, folder_ids or None)
            embedded_now = info.get("indexed", 0)
            if embedded_now:
                # Step 6: re-search now that new content is indexed.
                hits = _search(req.question, req.filters, folder_topics)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"search failed: {exc}")

    chunks = []
    for hit in hits:
        payload = hit.payload or {}
        chunks.append(
            {
                "text": payload.get("text", ""),
                "filename": payload.get("filename"),
                "filepath": payload.get("filepath"),
                "source": payload.get("source"),
                "note": payload.get("note", ""),
                "score": hit.score,
            }
        )

    if mode == "find":
        response = {
            "mode": "find",
            "question": req.question,
            "sources": sorted({c["filename"] for c in chunks if c["filename"]}),
            "chunks": chunks,
            "embedded_on_demand": embedded_now,
            # find mode is pure retrieval — no LLM produced this answer.
            "backend": "retrieval",
        }
    elif not chunks:
        # summarize mode, but nothing relevant was retrieved.
        response = {
            "mode": "summarize",
            "question": req.question,
            "answer": "No relevant content was found in the indexed files.",
            "sources": [],
            "backend": "retrieval",
        }
    else:
        # summarize mode -> hand the retrieved context to the generation layer,
        # with the active personality (mode) and recent conversation history.
        context = "\n\n---\n\n".join(
            f"[{c['filename']}]\n{c['text']}" for c in chunks
        )
        try:
            result = generate.generate(
                req.question, context, mode=active_mode, history=history
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
        response = {
            "mode": "summarize",
            "question": req.question,
            "answer": result["answer"],
            "sources": sorted({c["filename"] for c in chunks if c["filename"]}),
            "embedded_on_demand": embedded_now,
            "backend": result["backend"],
        }

    memory.add_message("assistant", _assistant_content(response), active_mode)
    return response


@router.get("/files")
def files_endpoint():
    """List all indexed files with their metadata (deduplicated by filename)."""
    client = get_qdrant()
    files: dict[str, dict] = {}
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename")
            if not filename:
                continue
            entry = files.setdefault(
                filename,
                {
                    "filename": filename,
                    "filepath": payload.get("filepath"),
                    "filetype": payload.get("filetype"),
                    "filesize": payload.get("filesize"),
                    "source": payload.get("source"),
                    "folder": payload.get("folder", payload.get("source")),
                    "note": payload.get("note", ""),
                    "date_ingested": payload.get("date_ingested"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if next_offset is None:
            break

    return {"count": len(files), "files": sorted(files.values(), key=lambda f: f["filename"])}
