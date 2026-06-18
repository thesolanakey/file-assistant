"""PDF parser using pdfplumber. Extracts all text, page by page."""
from __future__ import annotations

import pdfplumber

from server.parsers import chunk_text


def parse(filepath: str) -> list[str]:
    pages_text: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                pages_text.append(extracted)

    full_text = "\n\n".join(pages_text)
    return chunk_text(full_text)
