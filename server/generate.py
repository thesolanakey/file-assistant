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

log = logging.getLogger("generate")

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


def ollama_generate(question: str, context: str, mode: str | None, history: list[dict] | None) -> str:
    """Generate via Ollama's /api/generate endpoint.

    This is a complete implementation, not a no-op: the moment an Ollama server
    is reachable at ``OLLAMA_HOST`` serving ``OLLAMA_MODEL``, switching
    ``GENERATION_BACKEND=ollama`` works with no further code changes. Until then
    the call raises :class:`OllamaUnavailable` and the dispatcher falls back.
    """
    url = settings.OLLAMA_HOST.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": settings.OLLAMA_MODEL,
            "system": system_prompt_for(mode),
            "prompt": build_prompt(question, context, history),
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        # Connection refused, DNS failure, timeout, server down, etc.
        raise OllamaUnavailable(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable(f"invalid response from Ollama: {exc}") from exc

    answer = data.get("response", "")
    if not answer:
        raise OllamaUnavailable("Ollama returned an empty response")
    return answer


# --- Dispatch ---------------------------------------------------------------
def generate(
    question: str,
    context: str,
    mode: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Generate an answer with the configured backend.

    Returns ``{"answer": str, "backend": str}`` where ``backend`` is the backend
    that *actually* produced the text. With ``GENERATION_BACKEND=ollama``, an
    unreachable Ollama transparently falls back to Claude.
    """
    backend = settings.GENERATION_BACKEND
    if backend == "ollama":
        try:
            return {
                "answer": ollama_generate(question, context, mode, history),
                "backend": "ollama",
            }
        except OllamaUnavailable as exc:
            log.warning("falling back to Claude (Ollama unavailable: %s)", exc)
            return {
                "answer": claude_generate(question, context, mode, history),
                "backend": "claude",
            }
    if backend == "claude":
        return {"answer": claude_generate(question, context, mode, history), "backend": "claude"}
    raise ValueError(f"Unknown GENERATION_BACKEND: {backend!r}")
