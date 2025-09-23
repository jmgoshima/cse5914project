Project: Places + LLM search backend

## Overview

This repo contains a small backend that combines a retrieval layer (Elasticsearch) with LLM helpers (explain / conversation). The important folders are:

- `backend/` - Python backend code (app, langchain orchestration, search loader & query)
- `backend/langchain/` - LLM orchestration and schemas
- `backend/search/` - Elasticsearch loader & query builder
- `frontend/` - UI (if present)

## Quick start (PowerShell)

These are the basic steps to get a developer environment running on Windows PowerShell.

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install pinned dependencies

```powershell
pip install -r requirements.txt
```

3. Environment variables (examples)

```powershell
$env:ELASTICSEARCH_URL = 'http://localhost:9200'
$env:ES_INDEX = 'us_cities'
$env:OPENAI_API_KEY = 'sk-xxxx'    # only if you want LLM features
$env:ENV = 'development'
```

4. Run the backend

This repo's backend uses Flask in a factory pattern. Run using Python directly (development):

```powershell
# run via flask if FLASK_APP is configured or invoke the module directly
python -m backend.app
# or, if you prefer uvicorn for ASGI wrappers, adapt as needed
```

5. Run tests

```powershell
pytest -q
```

## Important files

- `backend/app.py` - Flask application entry points (health, conversation endpoints).
- `backend/langchain/schemas.py` - canonical `Profile` model used between conversation and search.
- `backend/langchain/explain.py` - LLM helper (OpenAI-backed with stub fallback).
- `backend/search/loader.py` - CSV loader that normalizes `places.csv` into canonical document fields for ES.
- `backend/search/query.py` - query builder (may be under active development by the query team).
- `backend/utils/ cache.py` - NOTE: filename contains a leading space; consider renaming to `backend/utils/cache.py` to avoid import issues.

## Notes & troubleshooting

- Pydantic version: The codebase uses Pydantic v2-style APIs (e.g. `model_dump_json()`); ensure `pydantic>=2.x` is installed.
- Elasticsearch client: This repo expects an Elasticsearch 8+ client API; if you use OpenSearch please update the client import and pins.
- If LLM requests fail with `OPENAI_API_KEY` not set, the code falls back to deterministic stubs for local development.
- If you run into import or version errors, try creating a fresh virtualenv and reinstalling pinned deps.

## Next steps I can help with

- Rename `backend/utils/ cache.py` to `backend/utils/cache.py` and update imports.
- Add a small smoke script to run an end-to-end query through the loader -> query -> explain flow (requires an ES instance).
- Add CI workflow to run tests on push.
