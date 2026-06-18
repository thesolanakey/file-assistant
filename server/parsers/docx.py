"""DOCX parser using python-docx. Extracts paragraph text."""
from __future__ import annotations

from docx import Document

from server.parsers import chunk_text


def parse(filepath: str) -> list[str]:
    document = Document(filepath)
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    full_text = "\n".join(paragraphs)
    return chunk_text(full_text)
