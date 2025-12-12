from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

_cache_lock = threading.Lock()
_cache_instance: Optional["BaseCache"] = None


class BaseCache:
    """Minimal cache interface required by the app."""

    def get(self, key: str) -> Any:  # pragma: no cover - interface definition
        raise NotImplementedError

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:  # pragma: no cover
        raise NotImplementedError


class InMemoryCache(BaseCache):
    """Thread-safe in-memory cache with optional TTL support."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expires_at = item
            if expires_at is not None and expires_at <= time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        expires_at = time.time() + ex if ex else None
        with self._lock:
            self._store[key] = (value, expires_at)


class RedisCache(BaseCache):
    """Thin wrapper around redis-py to match the BaseCache API."""

    def __init__(self, client: "redis.Redis") -> None:
        self._client = client

    def get(self, key: str) -> Any:
        value = self._client.get(key)
        if value is None:
            return None
        try:
            return value.decode("utf-8")
        except Exception:
            return value

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        self._client.set(name=key, value=value, ex=ex)


def _build_cache() -> BaseCache:
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_CONNECTION_URL")
    if url and redis:
        try:
            client = redis.from_url(url)
            client.ping()
            return RedisCache(client)
        except Exception:
            pass
    return InMemoryCache()


def get_cache() -> BaseCache:
    """Return a singleton cache instance."""
    global _cache_instance
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = _build_cache()
        return _cache_instance
