"""JSON parser. Flattens arbitrarily nested JSON into readable
``key.path: value`` lines, then chunks the result for embedding.
"""
from __future__ import annotations

import json

from server.parsers import chunk_text


def _flatten(obj, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]"
            lines.extend(_flatten(value, new_prefix))
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def parse(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    lines = _flatten(data)
    full_text = "\n".join(lines)
    return chunk_text(full_text)
