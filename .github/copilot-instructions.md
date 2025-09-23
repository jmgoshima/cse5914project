## Copilot instructions — quick start for this repo

This file is a concise, actionable guide for AI coding agents working on this project. It focuses on the discoverable structure, integration points, and project-specific quirks so you can get productive quickly.

### Big picture

- Repo layout: top-level `backend/` (API, LLM orchestration, search), `frontend/` (UI). Expect the runtime entry in `backend/app.py`.
- Major responsibilities:
  - `backend/langchain/` — conversation orchestration, explanation logic, and data schemas for LLM flows. Edit here to change prompt/response behavior.
  - `backend/search/` — Elasticsearch client, document loader, and query logic. This is the source of truth for retrieval.
  - `backend/utils/` — small utilities: caching and logging used across modules.

### Typical dataflow and integration points

1. Frontend -> Backend HTTP API (entry: `backend/app.py`). If you add endpoints, wire them into `app.py`.
2. Backend handler calls `backend/langchain/conversation.py` (or `explain.py`) to orchestrate the LLM flow.
3. Langchain orchestration uses `backend/search/loader.py` and `backend/search/query.py` to retrieve relevant docs via `backend/search/es_client.py`.
4. Results may be cached via `backend/utils/ cache.py` and written to logs via `backend/utils/logging.py`.

### Where to change LLM provider or prompts

- Look in `backend/langchain/` first. `conversation.py` is where conversational flows live; `explain.py` contains explanation-specific logic. `schemas.py` contains request/response schemas (pydantic-like expected).
- Provider wiring (OpenAI/Anthropic/etc.) is expected to be injected/configured in the langchain modules. If you need global keys, use environment variables (see below).

### Where to change search behavior

- `backend/search/es_client.py` — low-level ES client and connection setup. Update this to change ES connection options, auth, or client library.
- `backend/search/loader.py` — document ingestion/transform rules.
- `backend/search/query.py` — how queries are constructed and how results are scored/filtered. Edit these files to adjust retrieval precision/recall.

### Project-specific quirks and conventions

- File naming: there is a file with a leading space in the filename: `backend/utils/ cache.py` (note the space). This is brittle for many tools and CI; consider renaming to `cache.py` and updating imports. Until renamed, reference the exact path when editing.
- The repo groups code by domain (langchain / search / utils) rather than framework-specific folders. Keep changes within these domains where possible.
- Expect Pydantic-style schemas in `backend/langchain/schemas.py` — validate inputs there rather than in handlers.

### Environment variables (expected / typical)

- ELASTICSEARCH_URL — Elasticsearch endpoint (http[s]://host:port)
- ELASTICSEARCH_API_KEY or ELASTICSEARCH_USER/ELASTICSEARCH_PASS — optional auth
- OPENAI_API_KEY (or other provider keys) — used by LLM integrations
- ENV (development|production) — when present, may toggle logging/behavior

If no `requirements.txt` is present, assume common Python libs: `fastapi` or `flask` (API), `uvicorn` (dev server), `elasticsearch` or `opensearch-py`, `langchain`, `openai` (or provider SDK), `python-dotenv` (optional), `pytest`.

### Quick developer workflows (discoverable, conservative suggestions)

- Run server (if using FastAPI — inspect `backend/app.py` and adapt):
  - `pip install -r requirements.txt` (or install the packages above)
  - `uvicorn backend.app:app --reload --port 8000`
- Run tests: `pytest` (add `tests/` if none exist)

### Recommended safe edits for AI agents

- Small bugfixes and additions: modify `backend/search/query.py` to change query construction; update `backend/langchain/conversation.py` to tune prompts or add streaming outputs.
- When adding secrets or keys do not write them into code: reference env vars or a `.env` file and call out that secrets must be stored in the repo's secret store.
- If you rename `backend/utils/ cache.py`, update all imports across the repo; search/replace for the leading-space filename to avoid broken imports.

### Examples to look at when implementing features

- To change retrieval ranking: edit `backend/search/query.py` and `backend/search/es_client.py` (connection + scoring).
- To add a new API route returning LLM output: add handler in `backend/app.py` and call `backend/langchain/conversation.py`.

### If something is missing or empty

- Several files appear as placeholders (empty). If an implementation is missing, follow these rules:
  1. Add minimal, well-tested changes — small PRs.
  2. Document assumptions in the CHANGELOG or PR description.
  3. Add unit tests for behavior in `tests/`.

Please review and tell me if you'd like me to:

- create a top-level `README.md` and `requirements.txt` (I can infer dependencies and add a smoke script),
- rename the `backend/utils/ cache.py` file to remove the leading space, or
- scaffold a minimal FastAPI app in `backend/app.py` that wires the components together.
