"""Query API: tool-calling RAG.

Instead of always running a vector search, /query first asks the LLM whether the
question needs the stored documents. The model either answers normally (plain
chat, no Qdrant) or replies with {"action": "search", "query": "..."}, in which
case we run the search, inject the chunks, and ask the LLM for a final answer.

Endpoints:
  - POST /query   {"message": str, "filters": {}}  -> answer (chat or RAG)
  - GET  /files                                      -> indexed files + metadata
"""
from __future__ import annotations

import json
import re

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

# Appended to the system prompt on the first (decision) pass. The model picks one
# of three routes: search the indexed documents, search the live web, or just
# answer. We only run Qdrant / the web tool when it asks for them.
_SEARCH_INSTRUCTION = (
    "\n\nIf answering requires searching your stored documents, notes, or "
    'uploaded files, respond with only: {"action": "search", "query": "<search terms>"}\n'
    "If answering requires current/live information from the web (news, weather, "
    "prices, today's date, sports scores, recent events), respond with only: "
    '{"action": "web_search", "query": "<search terms>"}\n'
    "Otherwise respond normally.\n"
    "Examples:\n"
    "- 'hello' → respond normally\n"
    "- 'what is the capital of France?' → respond normally\n"
    "- 'what are today's football scores?' → "
    '{"action": "web_search", "query": "football scores today"}\n'
    "- 'find my notes on project zephyr' → "
    '{"action": "search", "query": "project zephyr"}\n'
    "- 'what did i upload about cold-start latency?' → "
    '{"action": "search", "query": "cold-start latency"}\n'
    "- 'what day is it?' → respond normally (use the current date/time provided)"
)


class QueryRequest(BaseModel):
    message: str
    filters: dict = Field(default_factory=dict)
    folder_ids: list[str] = Field(default_factory=list)


def _parse_action(text: str) -> dict | None:
    """If ``text`` is an action JSON, return ``{"action", "query"}`` (query may be
    empty); otherwise return None, meaning the model answered normally.

    Recognized actions are "search" (Qdrant) and "web_search" (live web).
    Tolerant of a surrounding ```json fence or extra prose around the object.
    """
    s = (text or "").strip()
    if not s:
        return None
    obj = None
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                obj = None
    if isinstance(obj, dict) and obj.get("action") in ("search", "web_search"):
        query = obj.get("query")
        return {
            "action": obj["action"],
            "query": query.strip() if isinstance(query, str) else "",
        }
    return None


# Signal phrases that strongly imply the user is asking about their own stored
# content. Easy to extend — just add to the tuple.
_DOC_SIGNALS = (
    "my files", "my notes", "my documents", "in my", "from my", "uploaded",
    "ingested", "stored", "you have", "i saved", "i uploaded", "find in",
    "look up", "search for", "what does", "what did i",
)


def _looks_like_doc_query(message: str) -> bool:
    """Heuristic fallback: does the message clearly reference the user's own
    files/notes/documents? Used to force the doc-search route when the decision
    pass answered directly despite an obvious document reference.

    Fires only on explicit document-reference language (the _DOC_SIGNALS
    phrases), not on arbitrary proper nouns.
    """
    lowered = message.lower()
    return any(signal in lowered for signal in _DOC_SIGNALS)


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
        collection_name=ingest.files_collection(),
        query_vector=vector,
        query_filter=_build_filter(filters, folder_topics),
        limit=TOP_K,
        with_payload=True,
    )


def _assistant_content(response: dict) -> str:
    """Text to persist as the assistant's turn for a /query response.

    Every response now carries a natural-language ``answer``; the fallback only
    matters if one is ever absent.
    """
    if response.get("answer"):
        return response["answer"]
    sources = response.get("sources") or []
    n = len(response.get("chunks", []))
    return f"[search] {n} chunk(s) from: {', '.join(sources) if sources else 'none'}"


def _run_search(query: str, filters: dict, folder_ids: list[str]):
    """Run the (intact) Qdrant search + on-demand embedding for ``query``.

    Returns ``(chunks, embedded_now)``. Raises on search failure.
    """
    folder_topics = ingest._folder_topics(folder_ids) if folder_ids else set()
    embedded_now = 0

    # Search what's already indexed (scoped if folders requested).
    hits = _search(query, filters, folder_topics)
    top_score = hits[0].score if hits else 0.0

    # If scope was requested, or confidence is low, pull in matching pending
    # files from the registry, embed them on demand, then re-search.
    if folder_ids or top_score < _CONFIDENCE_THRESHOLD:
        info = ingest.embed_on_demand(query, folder_ids or None)
        embedded_now = info.get("indexed", 0)
        if embedded_now:
            hits = _search(query, filters, folder_topics)

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
    return chunks, embedded_now


@router.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    # Active profile (assistant/friend) is recorded with every message.
    active_mode = modes.get_mode()
    # Snapshot prior history BEFORE logging the current question, so generation
    # sees "conversation so far" without the current turn duplicated in it.
    # History is per profile — only this profile's messages are loaded.
    history = memory.get_messages(limit=HISTORY_LIMIT, mode=active_mode)
    memory.add_message("user", req.message, active_mode)

    # Pass 1 — ask the LLM which route it needs. With tools off and the routing
    # instruction appended, it either answers normally or returns a
    # {"action": "search"|"web_search", "query": ...} JSON.
    try:
        decision = generate.generate(
            req.message,
            "",
            mode=active_mode,
            history=history,
            system_suffix=_SEARCH_INSTRUCTION,
            tools_enabled=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    parsed = _parse_action(decision["answer"])

    # Heuristic fallback: the decision model (esp. small local models) sometimes
    # answers directly even when the message clearly references the user's own
    # files. Force the doc-search route in that case, using the original message
    # as the search query.
    if parsed is None and _looks_like_doc_query(req.message):
        parsed = {"action": "search", "query": req.message}

    # No action -> the first response IS the answer. No retrieval involved.
    if parsed is None:
        response = {
            "mode": "chat",
            "active_mode": active_mode,
            "question": req.message,
            "answer": decision["answer"],
            "sources": [],
            "chunks": [],
            "embedded_on_demand": 0,
            "backend": decision["backend"],
            "tps": decision.get("tps"),
        }
        memory.add_message("assistant", _assistant_content(response), active_mode)
        return response

    # Web route -> run the web search directly (same call the /search endpoint
    # uses), inject the result as context, and ask for a final answer. Tools are
    # off on this pass — we already have the web result in hand.
    if parsed["action"] == "web_search":
        from server import tools  # lazy import to avoid an import cycle

        web_query = parsed["query"] or req.message
        results = tools.web_search(web_query)
        context = f"Web search results for '{web_query}':\n\n{results}"
        try:
            result = generate.generate(
                req.message,
                context,
                mode=active_mode,
                history=history,
                tools_enabled=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
        response = {
            "mode": "web_search",
            "active_mode": active_mode,
            "question": req.message,
            "answer": result["answer"],
            "sources": [],
            "chunks": [],
            "embedded_on_demand": 0,
            "backend": result["backend"],
            "tps": result.get("tps"),
        }
        memory.add_message("assistant", _assistant_content(response), active_mode)
        return response

    # Document route (action == "search") -> run retrieval with the model's
    # query (fall back to the original message if it didn't supply one).
    query = parsed["query"] or req.message
    folder_ids = req.folder_ids or []
    try:
        chunks, embedded_now = _run_search(query, req.filters, folder_ids)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"search failed: {exc}")

    if not chunks:
        response = {
            "mode": "summarize",
            "active_mode": active_mode,
            "question": req.message,
            "answer": "No relevant content was found in the indexed files.",
            "sources": [],
            "chunks": [],
            "embedded_on_demand": embedded_now,
            "backend": "retrieval",
            "tps": None,
        }
        memory.add_message("assistant", _assistant_content(response), active_mode)
        return response

    # Pass 2 — inject the retrieved chunks as context and ask for a final answer.
    context = "\n\n---\n\n".join(f"[{c['filename']}]\n{c['text']}" for c in chunks)
    try:
        result = generate.generate(
            req.message, context, mode=active_mode, history=history
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    response = {
        "mode": "summarize",
        "active_mode": active_mode,
        "question": req.message,
        "answer": result["answer"],
        "sources": sorted({c["filename"] for c in chunks if c["filename"]}),
        "chunks": chunks,
        "embedded_on_demand": embedded_now,
        "backend": result["backend"],
        "tps": result.get("tps"),
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
            collection_name=ingest.files_collection(),
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
                    "tag": payload.get("tag", ""),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if next_offset is None:
            break

    return {"count": len(files), "files": sorted(files.values(), key=lambda f: f["filename"])}


class SearchRequest(BaseModel):
    query: str


@router.post("/search")
def search_endpoint(req: SearchRequest):
    """Run a web search directly (used by the /search slash command)."""
    from server import tools  # lazy import to avoid an import cycle

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    return {"query": req.query, "results": tools.web_search(req.query)}


_ALLOWED_TAGS = {"remember", "storage", "project"}


class TagRequest(BaseModel):
    tag: str


@router.post("/files/{filename}/tag")
def tag_file_endpoint(filename: str, req: TagRequest):
    """Set a tag (remember/storage/project) on every chunk of a given file."""
    tag = req.tag.strip().lower()
    if tag not in _ALLOWED_TAGS:
        raise HTTPException(
            status_code=400, detail=f"tag must be one of {sorted(_ALLOWED_TAGS)}"
        )
    client = get_qdrant()
    client.set_payload(
        collection_name=ingest.files_collection(),
        payload={"tag": tag},
        points=qmodels.Filter(
            must=[qmodels.FieldCondition(key="filename", match=qmodels.MatchValue(value=filename))]
        ),
    )
    return {"filename": filename, "tag": tag}
