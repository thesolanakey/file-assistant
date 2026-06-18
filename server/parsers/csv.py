"""CSV parser using pandas. Emits one chunk per row with column headers as
context, so each chunk is self-describing for retrieval.
"""
from __future__ import annotations

import pandas as pd


def parse(filepath: str) -> list[str]:
    df = pd.read_csv(filepath)
    df = df.fillna("")

    chunks: list[str] = []
    columns = [str(c) for c in df.columns]
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in columns]
        chunks.append(" | ".join(parts))
    return chunks
