"""Claude knowledge-extraction pipeline.

POST /process  — given a filename already under watch/, send its content to
                 Claude and split the response into a structured summary, a list
                 of Q&A pairs, and a list of key facts. Each is saved as its own
                 markdown file in watch/notes/ (auto-ingested by the watcher).
POST /dataset  — run a single extraction step (qa | facts | summary) only.

Both endpoints use the Claude API directly regardless of the active generation
backend (this is a deliberate Claude pipeline).
"""
from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from server import generate
from server.parsers import parse_file

router = APIRouter()

# The full three-part extraction instruction (verbatim spec).
PROCESS_INSTRUCTION = (
    "You are a knowledge extraction assistant. Given the following document, "
    "produce three outputs separated by ---SPLIT---: "
    "(1) a structured markdown summary with headers and bullet points "
    "(2) a list of Q&A pairs in the format Q: ... A: ... "
    "(3) a list of key facts, one per line starting with FACT:"
)

# Per-type instructions for the single-step dataset builder.
DATASET_INSTRUCTIONS = {
    "summary": (
        "You are a knowledge extraction assistant. Given the following document, "
        "produce a structured markdown summary with headers and bullet points."
    ),
    "qa": (
        "You are a knowledge extraction assistant. Given the following document, "
        "produce a list of Q&A pairs in the format Q: ... A: ..."
    ),
    "facts": (
        "You are a knowledge extraction assistant. Given the following document, "
        "produce a list of key facts, one per line starting with FACT:"
    ),
}

_SPLIT = "---SPLIT---"
_TEXT_EXT = {".md", ".txt", ".csv", ".json", ".py", ".js", ".ts", ".html", ".log"}


def _locate(filename: str) -> str:
    """Find a file by basename anywhere under WATCH_DIR. Raises 404 if missing."""
    target = os.path.basename(filename)
    for root, _dirs, files in os.walk(settings.WATCH_DIR):
        if target in files:
            return os.path.join(root, target)
    raise HTTPException(status_code=404, detail=f"file not found under watch/: {target}")


def _read_doc(path: str) -> str:
    """Return the document's text. Reads text files directly; parses others."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return "\n".join(parse_file(path))


def _slug(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "doc"


def _claude(instruction: str, document: str) -> str:
    client = generate._get_claude_client()
    resp = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=4096,
        system=instruction,
        messages=[{"role": "user", "content": document}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _write_note(slug: str, kind: str, body: str) -> str:
    """Write a note into watch/notes/ (watcher auto-ingests). Returns filename."""
    notes_dir = os.path.join(settings.WATCH_DIR, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    filename = f"{slug}-{kind}.md"
    with open(os.path.join(notes_dir, filename), "w", encoding="utf-8") as fh:
        fh.write(body.strip() + "\n")
    return filename


class ProcessRequest(BaseModel):
    filename: str


class DatasetRequest(BaseModel):
    filename: str
    type: str


@router.post("/process")
def process_endpoint(req: ProcessRequest):
    """Run the full three-part Claude extraction and save three notes."""
    path = _locate(req.filename)
    document = _read_doc(path)
    if not document.strip():
        raise HTTPException(status_code=400, detail="file has no extractable text")

    response = _claude(PROCESS_INSTRUCTION, document)
    parts = [p.strip() for p in response.split(_SPLIT)]
    # Map the three parts in order; tolerate Claude returning fewer than three.
    kinds = ["summary", "qa", "facts"]
    slug = _slug(req.filename)
    created = []
    for kind, body in zip(kinds, parts):
        if body:
            created.append(_write_note(slug, kind, body))

    if not created:
        raise HTTPException(status_code=502, detail="Claude returned no usable output")
    return {
        "status": "processed",
        "source": os.path.basename(path),
        "created": created,
        "note": "files saved to watch/notes/ and will be auto-ingested by the watcher",
    }


@router.post("/dataset")
def dataset_endpoint(req: DatasetRequest):
    """Run a single extraction step (qa | facts | summary)."""
    dtype = req.type.strip().lower()
    if dtype not in DATASET_INSTRUCTIONS:
        raise HTTPException(
            status_code=400, detail=f"type must be one of {sorted(DATASET_INSTRUCTIONS)}"
        )
    path = _locate(req.filename)
    document = _read_doc(path)
    if not document.strip():
        raise HTTPException(status_code=400, detail="file has no extractable text")

    body = _claude(DATASET_INSTRUCTIONS[dtype], document)
    if not body:
        raise HTTPException(status_code=502, detail="Claude returned no usable output")
    filename = _write_note(_slug(req.filename), dtype, body)
    return {
        "status": "built",
        "source": os.path.basename(path),
        "type": dtype,
        "created": filename,
        "note": "saved to watch/notes/ and will be auto-ingested by the watcher",
    }
