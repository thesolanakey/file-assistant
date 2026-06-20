"""URL ingestion.

POST /ingest-url — fetch a web page with httpx, strip HTML to readable text,
save it as a markdown file under watch/documents/ (auto-ingested by the
watcher), and report the filename + estimated chunk count.
"""
from __future__ import annotations

import html as html_mod
import os
import re
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from server.parsers import chunk_text

router = APIRouter()

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _html_to_text(page: str) -> str:
    """Strip HTML to readable text, preserving rough line/paragraph breaks."""
    page = _SCRIPT_STYLE.sub(" ", page)
    page = re.sub(r"<br\s*/?>", "\n", page, flags=re.IGNORECASE)
    page = re.sub(r"</(p|div|h[1-6]|li|tr|section|article)>", "\n", page, flags=re.IGNORECASE)
    text = html_mod.unescape(_TAG.sub("", page))
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _slug_from_url(url: str, title: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = parsed.path.rstrip("/").split("/")[-1] or parsed.netloc or "page"
    base = os.path.splitext(base)[0]
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if not slug and title:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (slug or "page")[:60]


class IngestUrlRequest(BaseModel):
    url: str


@router.post("/ingest-url")
def ingest_url_endpoint(req: IngestUrlRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url must start with http:// or https://")

    try:
        resp = httpx.get(
            url, follow_redirects=True, timeout=20,
            headers={"User-Agent": "file-assistant/1.0"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"failed to fetch URL: {exc}")

    page = resp.text
    m = _TITLE.search(page)
    title = html_mod.unescape(_TAG.sub("", m.group(1))).strip() if m else ""
    text = _html_to_text(page)
    if not text:
        raise HTTPException(status_code=422, detail="no readable text extracted from page")

    slug = _slug_from_url(url, title)
    docs_dir = os.path.join(settings.WATCH_DIR, "documents")
    os.makedirs(docs_dir, exist_ok=True)
    filename = f"{slug}.md"
    header = (f"# {title}\n\n" if title else "") + f"Source: {url}\n\n"
    body = header + text + "\n"
    with open(os.path.join(docs_dir, filename), "w", encoding="utf-8") as fh:
        fh.write(body)

    return {
        "status": "ingested",
        "url": url,
        "filename": filename,
        "estimated_chunks": len(chunk_text(body)),
        "note": "saved to watch/documents/ and will be auto-ingested by the watcher",
    }
