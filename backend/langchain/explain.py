"""Lightweight LLM explain helper used by the app.

Provides a synchronous `explain(prompt, ...) -> str` function. If an OpenAI
API key is present in the environment, the function will call OpenAI's
ChatCompletion API to generate a short explanation. Otherwise, it returns a
deterministic stub so the app remains runnable during development and CI.

This module avoids performing network calls at import time.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _stub_explain(prompt: str) -> str:
    # deterministic, short stub for offline/dev environments
    snippet = prompt.replace("\n", " ")[:200]
    return f"stub-explain: {snippet}" if snippet else "stub-explain: (empty prompt)"


def explain(prompt: str, *, model: Optional[str] = None, max_tokens: int = 150, temperature: float = 0.2) -> str:
    """Return a short textual explanation for `prompt`.

    If `OPENAI_API_KEY` is present, call OpenAI ChatCompletion; otherwise return
    a safe stub. This function is synchronous on purpose to match the calling
    code in `backend/app.py`.

    Raises:
      ValueError: when prompt is empty or only whitespace.
      RuntimeError: when provider call fails in production mode.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    api_key = os.getenv("OPENAI_API_KEY")
    env = os.getenv("ENV", "development").lower()
    model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    if not api_key:
        logger.debug("OPENAI_API_KEY not set, returning stub explain")
        return _stub_explain(prompt)

    try:
        # Import openai lazily to avoid import-time dependency when stubbed
        import openai

        openai.api_key = api_key
        # Build a single-user message payload
        messages = [{"role": "user", "content": prompt}]

        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract text from response. This matches classic ChatCompletion.
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError("empty response from LLM")
        text = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
        return text.strip()

    except Exception as exc:  # keep this broad to surface provider errors cleanly
        logger.exception("explain() LLM call failed")
        if env == "development":
            # helpful message for developers, still deterministic-ish
            return f"llm-error: {type(exc).__name__}: {str(exc)[:200]}"
        raise RuntimeError("LLM provider call failed") from exc
