"""Elasticsearch client factory used across the backend.

We avoid creating the client at import time so that modules can be imported
without requiring Elasticsearch to be available immediately (critical for unit
tests and CLI tools). Credentials are pulled from environment variables to keep
local development flexible.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from elasticsearch import Elasticsearch


def _auth_kwargs() -> Dict[str, Any]:
    """Build authentication keyword arguments based on environment variables."""
    api_key = os.getenv("ES_API_KEY") or os.getenv("ELASTIC_API_KEY")
    if api_key:
        return {"api_key": api_key}

    username = (
        os.getenv("ES_USERNAME")
        or os.getenv("ES_USER")
        or os.getenv("ES_LOCAL_USER")
        or "elastic"
    )
    password = (
        os.getenv("ES_PASSWORD")
        or os.getenv("ES_PASS")
        or os.getenv("ES_LOCAL_PASSWORD")
    )

    if password:
        return {"basic_auth": (username, password)}

    return {}


@lru_cache()
def get_client() -> Elasticsearch:
    """Return a cached Elasticsearch client instance."""
    host = (
        os.getenv("ES_HOST")
        or os.getenv("ELASTIC_HOST")
        or os.getenv("ES_LOCAL_HOST")
        or "http://localhost:9200"
    )
    timeout = int(os.getenv("ES_TIMEOUT", "30"))
    verify = os.getenv("ES_VERIFY_CERTS", "false").lower() in {"1", "true", "yes"}

    return Elasticsearch(
        host,
        request_timeout=timeout,
        verify_certs=verify,
        **_auth_kwargs(),
    )

