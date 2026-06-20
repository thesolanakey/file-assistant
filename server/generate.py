"""Swappable text-generation layer with per-mode personality + memory.

Public entry point::

    generate(question, context, mode, history) -> {"answer": str, "backend": str}

The active operational mode (local/hetzner) selects a personality (system
prompt). The last N messages of conversation history and the retrieved RAG
chunks are assembled into the prompt in this order::

    system prompt  ->  conversation history  ->  retrieved chunks  ->  question

``backend`` in the return value reports which backend actually produced the
answer (relevant once the Ollama backend can fall back to Claude).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from config import settings
from server import runtime

log = logging.getLogger("generate")

# Tokens/sec of the most recent Ollama generation, surfaced in /health and on
# /query responses. None until the first Ollama generation completes.
_last_tps: float | None = None


def last_tps() -> float | None:
    return _last_tps

# Per-mode personalities. Keyed by the operational mode (see server/modes.py).
MODE_SYSTEM_PROMPTS = {
    "local": (
        "You are the user's personal assistant. Your tone is warm, casual, and "
        "to the point — like a friend who knows them well. You have access to "
        "their files and to your past conversations with them; reference earlier "
        "messages naturally when they're relevant. Answer from the provided "
        "context and conversation history, and if something isn't there, just "
        "say so plainly."
    ),
    "hetzner": (
        "You are a technical research assistant. Your tone is precise, rigorous, "
        "and grounded in the source literature. Base every claim on the retrieved "
        "documents and cite source filenames in brackets, e.g. [paper.pdf], when "
        "you use them. Clearly distinguish what the sources state from your own "
        "inference. If the retrieved context does not support an answer, say so "
        "explicitly rather than speculating."
    ),
}

# Fallback personality if an unknown mode is ever passed.
_DEFAULT_MODE = "local"

# Appended to the system message for the Ollama (tool-capable) path only — tells
# the model the tools exist and the policy for using them. Claude does not get
# this (it is not offered these tools).
_TOOLS_GUIDANCE = (
    "\n\nYou have two tools. Use them deliberately, not in every reply:\n"
    "- web_search(query): when answering needs information that is not in the "
    "provided context and that you are unsure about or that may be recent or "
    "current, call web_search first and base your answer on the results.\n"
    "- self_ingest(title, content): when the conversation produces a genuinely "
    "useful, durable fact, decision, or summary worth recalling in future "
    "conversations, call self_ingest to save it to memory. Use this SPARINGLY — "
    "only for information clearly worth keeping, never for routine answers."
)


def system_prompt_for(mode: str | None) -> str:
    return MODE_SYSTEM_PROMPTS.get(mode or _DEFAULT_MODE, MODE_SYSTEM_PROMPTS[_DEFAULT_MODE])


def _format_history(history: list[dict] | None) -> str:
    """Render prior messages as a compact transcript (oldest first)."""
    if not history:
        return ""
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in history)


def build_prompt(question: str, context: str, history: list[dict] | None) -> str:
    """Assemble the linear user-facing prompt body.

    Order: conversation history -> retrieved chunks -> current question. The
    system prompt is supplied separately (Claude's system slot / Ollama's
    ``system`` field).
    """
    parts: list[str] = []
    transcript = _format_history(history)
    if transcript:
        parts.append("Conversation so far:\n" + transcript)
    parts.append("Retrieved context from the indexed files:\n" + (context or "(none)"))
    parts.append("Current question: " + question)
    return "\n\n---\n\n".join(parts)


# --- Claude backend ---------------------------------------------------------
_claude_client = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic

        _claude_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _claude_client


def claude_generate(question: str, context: str, mode: str | None, history: list[dict] | None) -> str:
    client = _get_claude_client()
    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt_for(mode),
        messages=[{"role": "user", "content": build_prompt(question, context, history)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# --- Ollama backend ---------------------------------------------------------
class OllamaUnavailable(Exception):
    """Raised when the Ollama server cannot be reached or returns an error.

    This is the signal the dispatcher uses to fall back to Claude.
    """


# Safety cap on tool-call rounds so a misbehaving model can't loop forever.
_MAX_TOOL_ROUNDS = 5


def _ollama_chat(messages: list[dict], tools: list[dict] | None) -> dict:
    """POST one round to Ollama's /api/chat and return the parsed response.

    Raises :class:`OllamaUnavailable` on any connection/transport problem so the
    dispatcher can fall back to Claude.
    """
    url = settings.OLLAMA_HOST.rstrip("/") + "/api/chat"
    body: dict = {"model": runtime.get_model(), "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.OLLAMA_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise OllamaUnavailable(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable(f"invalid response from Ollama: {exc}") from exc


def ollama_generate(question: str, context: str, mode: str | None, history: list[dict] | None) -> str:
    """Generate via Ollama's /api/chat endpoint, with tool calling.

    The model is offered the tools in :data:`server.tools.TOOL_SCHEMAS`. When it
    emits ``tool_calls`` we execute each one, feed the results back as ``tool``
    messages, and let the model continue — looping until it returns a plain text
    answer (or the safety cap is hit). Any connection problem raises
    :class:`OllamaUnavailable`, and the dispatcher falls back to Claude.
    """
    from server import tools  # lazy import to avoid an import cycle

    messages: list[dict] = [
        {"role": "system", "content": system_prompt_for(mode) + _TOOLS_GUIDANCE},
        {"role": "user", "content": build_prompt(question, context, history)},
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        data = _ollama_chat(messages, tools.TOOL_SCHEMAS)
        msg = data.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            _record_tps(data)
            return msg.get("content", "") or ""

        # Record the assistant's tool-call turn, then execute each tool and
        # append its result so the model can use it on the next round.
        messages.append(msg)
        for call in tool_calls:
            fn = call.get("function", {}) or {}
            name = fn.get("name", "")
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"[tool] {name}({', '.join(args.keys())})", flush=True)
            result = tools.execute_tool(name, args)
            messages.append({"role": "tool", "tool_name": name, "content": result})

    # Tool budget exhausted — ask once more with tools disabled to force a reply.
    data = _ollama_chat(messages, None)
    _record_tps(data)
    return (data.get("message", {}) or {}).get("content", "") or ""


def _record_tps(data: dict) -> None:
    """Stash tokens/sec from an Ollama response (eval_count / eval_duration)."""
    global _last_tps
    count = data.get("eval_count")
    dur_ns = data.get("eval_duration")
    if count and dur_ns:
        _last_tps = round(count / (dur_ns / 1e9), 1)


# --- Dispatch ---------------------------------------------------------------
def generate(
    question: str,
    context: str,
    mode: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Generate an answer with the configured backend.

    Returns ``{"answer": str, "backend": str, "tps": float|None}`` where
    ``backend`` is the backend that *actually* produced the text. The backend,
    model and fallback policy are read from the runtime config (server/runtime),
    so the UI's POST /config takes effect immediately. When backend is "ollama"
    and the auto-fallback toggle is on, an unreachable Ollama falls back to
    Claude.
    """
    backend = runtime.get_backend()
    if backend == "ollama":
        try:
            answer = ollama_generate(question, context, mode, history)
            return {"answer": answer, "backend": "ollama", "tps": _last_tps}
        except OllamaUnavailable as exc:
            if not runtime.get_fallback():
                raise
            log.warning("falling back to Claude (Ollama unavailable: %s)", exc)
            answer = claude_generate(question, context, mode, history)
            return {"answer": answer, "backend": "claude", "tps": None}
    if backend == "claude":
        return {
            "answer": claude_generate(question, context, mode, history),
            "backend": "claude",
            "tps": None,
        }
    raise ValueError(f"Unknown backend: {backend!r}")
