"""Plain-text / Markdown parser (.txt, .md) and the generic text fallback."""
from __future__ import annotations

from server.parsers import chunk_text


def parse(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return chunk_text(content)
