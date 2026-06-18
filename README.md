# file-assistant

A self-hosted RAG (retrieval-augmented generation) file assistant. Drop files
into watched folders, and it parses, embeds, and indexes them into a vector
database. Ask questions and it either returns the most relevant chunks (find
mode) or summarizes them with an LLM (summarize mode).

Everything runs on a single Linux host via Docker.

## Stack

| Component | Role |
|-----------|------|
| **Qdrant** | Vector database (Docker) |
| **FastAPI** | API server (Docker) |
| **Caddy** | Reverse proxy + HTTPS + basic auth (Docker) |
| **sentence-transformers** | Embeddings via `nomic-ai/nomic-embed-text-v1` (768-dim) |
| **Claude API** | Summarization (`claude-sonnet-4-6`) |

The generation layer is **swappable**: set `GENERATION_BACKEND=ollama` in `.env`
to switch off Claude later (Ollama backend is currently a stub).

## Layout

```
server/        FastAPI app, ingestion, query, embedding, generation, parsers
config/        settings loaded from environment
watch/         drop files here — subfolders: documents, notes, exports, pdfs, code
qdrant-storage/ Qdrant data (gitignored)
Dockerfile, docker-compose.yml, Caddyfile, deploy.sh
```

## Setup

1. **Create your env file**

   ```bash
   cp .env.template .env
   ```

   Edit `.env` and set:
   - `ANTHROPIC_API_KEY` — your Claude API key
   - `BASIC_AUTH_USER` — defaults to `admin`
   - `BASIC_AUTH_PASSWORD` — a **bcrypt hash**, generated with:

     ```bash
     docker run --rm caddy caddy hash-password --plaintext 'your-password'
     ```

2. **Start everything**

   ```bash
   docker compose up --build -d
   ```

   First boot downloads the embedding model (a few hundred MB), so the watcher
   takes a minute before it begins indexing.

## API

All routes sit behind Caddy basic auth. Direct (no auth) access is available on
`localhost:8000` on the host.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | `{"status":"ok","backend":"claude"}` |
| `GET`  | `/files` | List indexed files + metadata |
| `POST` | `/ingest` | `{"path": "/app/watch/..."}` — manually ingest a file |
| `POST` | `/query` | `{"question": "...", "filters": {}}` — find or summarize |

### Query modes

Intent is auto-detected. If the question contains `summarize`, `explain`,
`overview`, or `tldr`, it runs **summarize mode** (chunks → Claude). Otherwise
**find mode** returns the top chunks with their source filenames.

```bash
# find mode
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is mindcast"}'

# summarize mode
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "summarize the mindcast architecture"}'
```

## Ingestion

Files copied into `watch/` (and its subfolders) are automatically parsed,
chunked, embedded, and stored. Supported types: `.txt`, `.md`, `.pdf`, `.docx`,
`.csv`, `.json`, plus source code (treated as text).

Skipped: hidden files, `.gitkeep`, `.pyc`, `.db`. Files already indexed (matched
by filename + size) are skipped to avoid duplicates.

Each stored chunk carries: `text`, `filename`, `filepath`, `filetype`,
`filesize`, `date_ingested`, `source`.

## HTTPS note

The site is addressed by raw IP, so Caddy uses an internal self-signed CA
(`tls internal`). Use `curl -k` or trust Caddy's root CA when hitting it over
HTTPS.

## Deploy / update

```bash
./deploy.sh
```

Pulls `main`, rebuilds, and restarts the stack.
