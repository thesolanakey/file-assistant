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
from server import embed, generate
from server.ingest import get_qdrant

router = APIRouter()

TOP_K = 5
_SUMMARIZE_KEYWORDS = ("summarize", "explain", "overview", "tldr")


class QueryRequest(BaseModel):
    question: str
    filters: dict = Field(default_factory=dict)


def _detect_mode(question: str) -> str:
    lowered = question.lower()
    if any(keyword in lowered for keyword in _SUMMARIZE_KEYWORDS):
        return "summarize"
    return "find"


def _build_filter(filters: dict) -> qmodels.Filter | None:
    """Translate a simple {field: value} dict into a Qdrant filter."""
    if not filters:
        return None
    conditions = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return qmodels.Filter(must=conditions)


def _search(question: str, filters: dict):
    vector = embed.get_embedding(question, is_query=True)
    return get_qdrant().search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=_build_filter(filters),
        limit=TOP_K,
        with_payload=True,
    )


@router.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    mode = _detect_mode(req.question)
    try:
        hits = _search(req.question, req.filters)
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
        return {
            "mode": "find",
            "question": req.question,
            "sources": sorted({c["filename"] for c in chunks if c["filename"]}),
            "chunks": chunks,
        }

    # summarize mode -> hand the retrieved context to the generation layer.
    if not chunks:
        return {
            "mode": "summarize",
            "question": req.question,
            "answer": "No relevant content was found in the indexed files.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[{c['filename']}]\n{c['text']}" for c in chunks
    )
    try:
        answer = generate.generate(req.question, context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    return {
        "mode": "summarize",
        "question": req.question,
        "answer": answer,
        "sources": sorted({c["filename"] for c in chunks if c["filename"]}),
    }


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
