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

import html
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
_DDG_URL = "https://html.duckduckgo.com/html/"
# Pair each result link with its OWN snippet in one match, so titles and
# snippets can't drift out of alignment (ad blocks are filtered by href below).
_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]*)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _decode_ddg_href(href: str) -> str:
    """DDG result links are /l/?uddg=<encoded-real-url> redirects; decode them."""
    if "uddg=" in href:
        try:
            q = urllib.parse.urlparse(href).query
            uddg = urllib.parse.parse_qs(q).get("uddg", [""])[0]
            if uddg:
                return urllib.parse.unquote(uddg)
        except Exception:  # noqa: BLE001
            pass
    return href if href.startswith("http") else "https:" + href


def web_search(query: str, max_results: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "web_search failed: empty query."

    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        _DDG_URL,
        data=data,  # POST is the most reliable for the html endpoint
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("web_search failed: %s", exc)
        return f"web_search failed: could not reach DuckDuckGo ({exc})."

    # Organic results only — skip DuckDuckGo ad links (y.js / ad_domain).
    organic = [
        m for m in _RESULT_RE.finditer(page)
        if "y.js" not in m.group("href") and "ad_domain" not in m.group("href")
    ]
    if not organic:
        return f"No web results found for '{query}'."

    lines = [f"Web search results for '{query}':"]
    for i, m in enumerate(organic[:max_results]):
        title = _strip_tags(m.group("title"))
        url = _decode_ddg_href(m.group("href"))
        snippet = _strip_tags(m.group("snippet"))
        lines.append(f"{i + 1}. {title}\n   {url}\n   {snippet}")
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
