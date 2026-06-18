"""Swappable text-generation layer.

The public entry point is ``generate(question, context)``. It dispatches to the
backend named by GENERATION_BACKEND. Today only the Claude backend is real; the
Ollama backend is a stub so the swap is a one-line config change later.
"""
from __future__ import annotations

from config import settings

SYSTEM_PROMPT = (
    "You are a file assistant. Answer only from the provided context. "
    "If the answer is not in the context say so clearly."
)

# Cache the Anthropic client so we build it once.
_claude_client = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic

        _claude_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _claude_client


def claude_generate(question: str, context: str) -> str:
    """Summarize/answer using the Claude Messages API."""
    client = _get_claude_client()
    user_content = (
        f"Context from the indexed files:\n\n{context}\n\n"
        f"Question: {question}"
    )
    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def ollama_generate(question: str, context: str) -> str:
    """Placeholder for a future local-model backend."""
    raise NotImplementedError(
        "Switch GENERATION_BACKEND to claude until Ollama is configured"
    )


def generate(question: str, context: str) -> str:
    if settings.GENERATION_BACKEND == "claude":
        return claude_generate(question, context)
    elif settings.GENERATION_BACKEND == "ollama":
        return ollama_generate(question, context)
    raise ValueError(f"Unknown GENERATION_BACKEND: {settings.GENERATION_BACKEND!r}")
