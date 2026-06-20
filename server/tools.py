"""Tools the Ollama model can call during generation (function calling).

Two tools are exposed, in Ollama's /api/chat tool schema (OpenAI-style):

  * ``self_ingest(title, content)`` — save a genuinely useful fact/summary as a
    markdown note under watch/notes/ and ingest it into Qdrant, so it can be
    recalled in future queries. Meant to be used sparingly.
  * ``web_search(query)`` — search the web via DuckDuckGo (no API key) for
    information not in the indexed files.

The generation layer (server/generate.py) advertises TOOL_SCHEMAS to the model,
intercepts any tool_calls, runs execute_tool(), feeds the result back, and lets
the model continue. Executors always return a short string (never raise) so a
failing tool can't break generation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from config import settings

log = logging.getLogger("tools")

# --- Schemas advertised to the model ----------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "self_ingest",
            "description": (
                "Save a genuinely useful fact, decision, or summary to long-term "
                "memory so it can be recalled in future conversations. The note is "
                "stored as a file and indexed for retrieval. Use this SPARINGLY — "
                "only for durable, reusable knowledge (a stable fact, a conclusion "
                "worth keeping, a useful summary). Do NOT call it for routine "
                "answers, chit-chat, or things already in the indexed files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A short descriptive title for the note.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The fact or summary to remember, in markdown.",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web via DuckDuckGo for information that is not "
                "in the indexed files or your training knowledge (e.g. current "
                "events, recent facts). Returns the top results with snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# --- self_ingest -------------------------------------------------------------
def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "note"


def self_ingest(title: str, content: str) -> str:
    """Write a markdown note into watch/notes/ and ingest it into Qdrant."""
    from server import ingest  # lazy import to avoid an import cycle

    title = (title or "").strip()
    content = (content or "").strip()
    if not title or not content:
        return "self_ingest failed: both title and content are required."

    notes_dir = os.path.join(settings.WATCH_DIR, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{_slugify(title)}-{stamp}.md"
    path = os.path.join(notes_dir, filename)

    abspath = os.path.abspath(path)
    # Claim the file so the directory watcher doesn't race us; we ingest it
    # explicitly below for a deterministic result (mirrors the /upload flow).
    ingest._uploading.add(abspath)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# {title}\n\n{content}\n")
        result = ingest.ingest_path(path, folder="notes")
    except Exception as exc:  # noqa: BLE001
        log.warning("self_ingest failed: %s", exc)
        return f"self_ingest failed: {exc}"
    finally:
        ingest._uploading.discard(abspath)

    chunks = result.get("chunks", result.get("status"))
    return (
        f"Saved to memory as '{filename}' and indexed into the knowledge base "
        f"({chunks} chunk(s)). It can be recalled in future queries."
    )


# --- web_search (DuckDuckGo, no API key) ------------------------------------
_DDG_API = "https://api.duckduckgo.com/"


def _flatten_related(topics: list, out: list, limit: int) -> None:
    """Collect (text, url) pairs from RelatedTopics, descending into the nested
    category groups DuckDuckGo sometimes returns (a topic with a 'Topics' list)."""
    for t in topics:
        if len(out) >= limit:
            return
        if isinstance(t, dict) and "Topics" in t:
            _flatten_related(t.get("Topics", []), out, limit)
        elif isinstance(t, dict):
            text = (t.get("Text") or "").strip()
            url = t.get("FirstURL") or ""
            if text:
                out.append((text, url))


def web_search(query: str, max_results: int = 5) -> str:
    """Search via DuckDuckGo's Instant Answer API (no API key required).

    GET https://api.duckduckgo.com/?q=<query>&format=json — returns instant
    answers (direct answers, topic abstracts, definitions, related topics). The
    relevant fields are parsed into a compact text block that is fed back into
    the generation context.
    """
    query = (query or "").strip()
    if not query:
        return "web_search failed: empty query."

    url = _DDG_API + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1", "t": "file-assistant"}
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "file-assistant/1.0"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("web_search failed: %s", exc)
        return f"web_search failed: could not reach DuckDuckGo ({exc})."
    except json.JSONDecodeError as exc:
        return f"web_search failed: invalid response from DuckDuckGo ({exc})."

    lines = [f"Web search results for '{query}' (DuckDuckGo Instant Answer):"]
    heading = (data.get("Heading") or "").strip()

    # Direct answer (calculations, conversions, simple facts).
    answer = (data.get("Answer") or "").strip()
    if answer:
        lines.append(f"Answer: {answer}")

    # Topic abstract / summary.
    abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
    if abstract:
        src = (data.get("AbstractSource") or "").strip()
        head = f"{heading}: " if heading else ""
        lines.append(f"{head}{abstract}" + (f" [{src}]" if src else ""))
        if data.get("AbstractURL"):
            lines.append(f"  {data['AbstractURL']}")

    # Dictionary-style definition.
    definition = (data.get("Definition") or "").strip()
    if definition:
        src = (data.get("DefinitionSource") or "").strip()
        lines.append(f"Definition: {definition}" + (f" [{src}]" if src else ""))
        if data.get("DefinitionURL"):
            lines.append(f"  {data['DefinitionURL']}")

    # Related topics.
    related: list = []
    _flatten_related(data.get("RelatedTopics", []), related, max_results)
    for text, u in related:
        lines.append(f"- {text}" + (f"\n  {u}" if u else ""))

    if len(lines) == 1:
        # No instant answer available. Be explicit so the model reports this
        # rather than fabricating an answer — the Instant Answer API does not
        # serve live/real-time results (e.g. today's sports scores).
        return (
            f"DuckDuckGo's Instant Answer API returned no results for '{query}'. "
            f"Note: this API does not provide live or real-time data such as "
            f"current sports scores, news, or stock prices."
        )
    return "\n".join(lines)


# --- dispatch ---------------------------------------------------------------
def execute_tool(name: str, arguments: dict) -> str:
    """Run a tool by name; always returns a string, never raises."""
    arguments = arguments or {}
    try:
        if name == "self_ingest":
            return self_ingest(arguments.get("title", ""), arguments.get("content", ""))
        if name == "web_search":
            return web_search(arguments.get("query", ""))
        return f"Unknown tool: {name!r}"
    except Exception as exc:  # noqa: BLE001
        log.warning("tool %s raised: %s", name, exc)
        return f"Tool {name} failed: {exc}"
