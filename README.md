Project: Places + LLM search backend

## Overview

This repo contains a small backend that combines a retrieval layer (Elasticsearch) with LLM helpers (explain / conversation). The important folders are:

- `backend/` - Python backend code (app, langchain orchestration, search loader & query)
- `backend/langchain/` - LLM orchestration and schemas
- `backend/search/` - Elasticsearch loader & query builder
- `frontend/` - UI (if present)

## Quick start (PowerShell)

These are the basic steps to get a developer environment running on Windows PowerShell.

Follow the steps below to run Elasticsearch, the backend, and the frontend simultaneously using three separate terminals.

***Terminal 1 – Start Elasticsearch***

Start Docker Desktop.

Open a terminal and navigate to the project root:
cd <project-root>

Remove any existing Elasticsearch container:
docker rm -f places-es

Start Elasticsearch:
docker run -d --name places-es -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.11.1

Monitor logs to confirm it started correctly:
docker logs -f places-es

***Terminal 2 – Backend Setup and Launch***

Open a new terminal and navigate to the project root:
cd <project-root>

Deactivate any existing Python environment:
if (Get-Command deactivate -ErrorAction SilentlyContinue) { deactivate }

Remove the existing virtual environment:
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

Create a fresh virtual environment:
py -3.11 -m venv .venv

Allow script execution for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Activate the environment:
..venv\Scripts\Activate.ps1

Upgrade pip and install dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt

Set backend environment variables:
$env:MPLBACKEND = 'Agg'
$env:ELASTICSEARCH_URL = 'http://localhost:9200
'
$env:ES_INDEX = 'cities'
$env:ES_LOCAL_PASSWORD = '<your-elasticsearch-password>'
$env:OPENAI_API_KEY = '<your-openai-api-key>'

Load the Elasticsearch index:
python -m backend.search.loader

Start the backend service:
python -m backend.app

***Terminal 3 – Frontend Setup and Launch***

Open a third terminal and navigate to the frontend folder:
cd <project-root>/frontend/app

Allow script execution for this session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Install Node dependencies:
npm install

Start the frontend development server:
npm start

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
