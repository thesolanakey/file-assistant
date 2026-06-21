"""Claude-powered dataset builder.

Extracts structured knowledge (facts, decisions, code, URLs, summaries) from a
conversation using Claude Haiku 4.5, then ingests each item into Qdrant by
reusing the existing self_ingest tool. Endpoints live in server/main.py
(POST /dataset/build, GET /dataset/status).

The extraction uses standard (synchronous) messages because the pipeline
processes a single conversation at a time. The Batch API
(anthropic.beta.messages.batches) is reserved for bulk runs of more than
:data:`BATCH_THRESHOLD` conversations — see :func:`extract_from_conversations`.
"""
from __future__ import annotations

import json
import logging
import re

from config import settings
from server import generate, memory, modes

log = logging.getLogger("dataset")

# Knowledge extraction always uses Claude Haiku 4.5 regardless of the active
# generation backend — this is a deliberate Claude pipeline.
EXTRACT_MODEL = "claude-haiku-4-5-20251001"

# Beyond this many conversations in one request, prefer the async Batch API.
BATCH_THRESHOLD = 5

_SYSTEM_PROMPT = (
    "You are a knowledge extraction assistant. Given a conversation, extract "
    "discrete facts, decisions, code snippets, URLs, and useful information "
    "worth remembering. Return ONLY a JSON array of objects, each with: "
    '{"title": "short title", "content": "the extracted fact or information", '
    '"type": "fact|decision|code|url|summary", "tags": ["tag1", "tag2"]}. '
    "Extract 3-10 items per conversation. No preamble, no markdown, just the "
    "JSON array."
)


def _format_conversation(messages: list[dict]) -> str:
    """Render messages as 'User: ...\\nAssistant: ...\\n' for the user prompt."""
    lines = []
    for m in messages:
        role = (m.get("role") or "").lower()
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {m.get('content', '')}")
    return "\n".join(lines)


def _parse_items(text: str) -> list[dict]:
    """Parse Claude's JSON array of items, tolerating a ```json fence or prose."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()

    data = None
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None

    if not isinstance(data, list):
        return []

    items: list[dict] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        title = (obj.get("title") or "").strip()
        content = (obj.get("content") or "").strip()
        if not title or not content:
            continue
        tags = [str(t).strip() for t in (obj.get("tags") or []) if str(t).strip()]
        items.append(
            {
                "title": title,
                "content": content,
                "type": (obj.get("type") or "fact").strip(),
                "tags": tags,
            }
        )
    return items


def _extract_one(messages: list[dict]) -> str:
    """Single standard-messages call to Claude Haiku; returns the raw text."""
    client = generate._get_claude_client()
    resp = client.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_conversation(messages)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_from_conversation(messages: list[dict]) -> list[dict]:
    """Extract structured knowledge items from a single conversation.

    Returns a list of ``{"title", "content", "type", "tags"}`` dicts (possibly
    empty if the model produced nothing parseable).
    """
    if not messages:
        return []
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot extract dataset")
    return _parse_items(_extract_one(messages))


def extract_from_conversations(conversations: list[list[dict]]) -> list[list[dict]]:
    """Extract from many conversations at once.

    Uses the async Batch API when there are more than :data:`BATCH_THRESHOLD`
    conversations (cheaper at scale), otherwise loops with standard messages.
    Returns one item-list per input conversation, in order.
    """
    if not conversations:
        return []
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot extract dataset")

    if len(conversations) <= BATCH_THRESHOLD:
        return [_parse_items(_extract_one(c)) for c in conversations]

    # Bulk path: submit one batch request per conversation and poll for results.
    import time

    client = generate._get_claude_client()
    requests = [
        {
            "custom_id": f"conv-{i}",
            "params": {
                "model": EXTRACT_MODEL,
                "max_tokens": 4096,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _format_conversation(c)}],
            },
        }
        for i, c in enumerate(conversations)
    ]
    batch = client.beta.messages.batches.create(requests=requests)
    while True:
        batch = client.beta.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(5)

    by_id: dict[str, list[dict]] = {}
    for result in client.beta.messages.batches.results(batch.id):
        text = ""
        if result.result.type == "succeeded":
            text = "".join(
                b.text for b in result.result.message.content if b.type == "text"
            )
        by_id[result.custom_id] = _parse_items(text)
    return [by_id.get(f"conv-{i}", []) for i in range(len(conversations))]


def _ingest_item(item: dict) -> bool:
    """Embed/store one extracted item via self_ingest. Returns True on success.

    The item's type and tags are appended to the note body so they are embedded
    alongside the content and recoverable in future queries.
    """
    from server import tools  # lazy import to avoid an import cycle

    body = item["content"]
    meta = []
    if item.get("type"):
        meta.append(f"Type: {item['type']}")
    if item.get("tags"):
        meta.append("Tags: " + ", ".join(item["tags"]))
    if meta:
        body = f"{body}\n\n" + "\n".join(meta)

    result = tools.self_ingest(item["title"], body)
    return not result.lower().startswith("self_ingest failed")


def build_dataset_from_history() -> dict:
    """Extract knowledge from the current conversation and ingest it into Qdrant.

    Returns ``{"extracted": N, "ingested": N, "items": [titles...]}``. Returns
    early (with a note) if there are fewer than 3 messages to extract from.
    """
    history = memory.get_messages(mode=modes.get_mode())
    if len(history) < 3:
        return {
            "extracted": 0,
            "ingested": 0,
            "items": [],
            "note": "not enough conversation to extract from (need at least 3 messages)",
        }

    items = extract_from_conversation(history)
    ingested = 0
    titles: list[str] = []
    for item in items:
        if _ingest_item(item):
            ingested += 1
            titles.append(item["title"])

    return {"extracted": len(items), "ingested": ingested, "items": titles}
