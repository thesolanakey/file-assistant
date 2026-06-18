"""Document parsers.

Each parser module exposes ``parse(filepath: str) -> list[str]`` returning a
list of text chunks ready for embedding. A shared ``chunk_text`` helper applies
word-based chunking with overlap, configured via settings.
"""
from __future__ import annotations

import os

from config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping word-based chunks.

    chunk_size/overlap are measured in words and default to the configured
    CHUNK_SIZE / CHUNK_OVERLAP.
    """
    if not text or not text.strip():
        return []

    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def parse_file(filepath: str) -> list[str]:
    """Dispatch a file to the correct parser based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()

    # Imported lazily to avoid importing heavy deps (pandas, pdfplumber) unless
    # the corresponding file type is actually encountered.
    if ext in {".txt", ".md"}:
        from server.parsers import text

        return text.parse(filepath)
    if ext == ".pdf":
        from server.parsers import pdf

        return pdf.parse(filepath)
    if ext == ".docx":
        from server.parsers import docx

        return docx.parse(filepath)
    if ext == ".csv":
        from server.parsers import csv as csv_parser

        return csv_parser.parse(filepath)
    if ext == ".json":
        from server.parsers import json_parser

        return json_parser.parse(filepath)

    # Fallback: treat unknown but text-like files (e.g. source code) as plain text.
    from server.parsers import text

    return text.parse(filepath)
