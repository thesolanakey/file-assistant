"""Central configuration for the RAG file assistant.

Loads every variable from the environment (populated from .env via
python-dotenv) and exposes them as typed module-level constants.
"""
import os

from dotenv import load_dotenv

# Load .env if present. In Docker the values also arrive via env_file, in which
# case load_dotenv is a harmless no-op for already-set variables.
load_dotenv()


def _get(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


# --- Generation backend -----------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GENERATION_BACKEND: str = _get("GENERATION_BACKEND", "claude").strip().lower()
CLAUDE_MODEL: str = _get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Ollama (local model) backend. Used when GENERATION_BACKEND=ollama; if the
# Ollama server is unreachable, generation falls back to Claude automatically.
# In Docker, host.docker.internal resolves to the host (see docker-compose
# extra_hosts), so an Ollama installed on the host works without code changes.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))

_ALLOWED_BACKENDS = {"claude", "ollama"}
if GENERATION_BACKEND not in _ALLOWED_BACKENDS:
    raise ValueError(
        f"GENERATION_BACKEND must be one of {sorted(_ALLOWED_BACKENDS)}, "
        f"got {GENERATION_BACKEND!r}"
    )

# --- Qdrant -----------------------------------------------------------------
QDRANT_HOST: str = _get("QDRANT_HOST", "qdrant")
QDRANT_PORT: int = int(_get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "files")
# Registry of registered folders/files for lazy (on-demand) indexing.
QDRANT_REGISTRY_COLLECTION: str = os.getenv("QDRANT_REGISTRY_COLLECTION", "registry")

# Lazily-registered folder uploads land here — deliberately NOT under WATCH_DIR
# so the eager file watcher never embeds them. They are embedded on demand.
REGISTERED_DIR: str = os.getenv("REGISTERED_DIR", "/app/registered")

# --- Embeddings -------------------------------------------------------------
EMBED_MODEL: str = _get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))

# --- Operational mode -------------------------------------------------------
# Named runtime modes for the assistant. These are independent of the
# generation backend and of the query intent (find/summarize). The active mode
# is persisted to SQLite (see server/modes.py) so it survives restarts; this
# value is only the fallback used the very first time, before anything is saved.
ALLOWED_MODES = {"local", "hetzner"}
DEFAULT_MODE: str = _get("DEFAULT_MODE", "local").strip().lower()
if DEFAULT_MODE not in ALLOWED_MODES:
    raise ValueError(
        f"DEFAULT_MODE must be one of {sorted(ALLOWED_MODES)}, got {DEFAULT_MODE!r}"
    )

# --- Persistence (SQLite: conversation memory + persisted mode) -------------
DATA_DIR: str = os.getenv("DATA_DIR", "/app/data")
DB_PATH: str = os.getenv("DB_PATH", os.path.join(DATA_DIR, "conversation.db"))

# --- Server -----------------------------------------------------------------
SERVER_HOST: str = _get("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(_get("SERVER_PORT", "8000"))

# --- Basic auth (consumed by Caddy, surfaced here for completeness) ---------
BASIC_AUTH_USER: str = os.getenv("BASIC_AUTH_USER", "admin")
BASIC_AUTH_PASSWORD: str = os.getenv("BASIC_AUTH_PASSWORD", "")

# --- Ingestion --------------------------------------------------------------
WATCH_DIR: str = _get("WATCH_DIR", "/app/watch")
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "50"))
