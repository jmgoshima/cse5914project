"""Lightweight LLM explain helper used by the app.

Provides a synchronous `explain(prompt, ...) -> str` function. If a Gemini
API key is present in the environment, the function will call Google's
Generative AI API to generate a short explanation. Otherwise, it returns a
deterministic stub so the app remains runnable during development and CI.

This module avoids performing network calls at import time.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

# Load .env if present so GOOGLE_API_KEY is available locally
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

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

    api_key = os.getenv("GOOGLE_API_KEY")
    env = os.getenv("ENV", "development").lower()
    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    if not api_key:
        logger.debug("GOOGLE_API_KEY not set, returning stub explain")
        return _stub_explain(prompt)

    try:
        # Import and configure Gemini lazily
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)

        resp = gen_model.generate_content(prompt)
        # Prefer the convenience text property when available
        text = getattr(resp, "text", None)
        if not text:
            # Fallback: attempt to compose text from candidates
            try:
                parts = []
                for cand in getattr(resp, "candidates", []) or []:
                    for part in getattr(cand, "content", {}).get("parts", []) or []:
                        val = getattr(part, "text", None)
                        if val:
                            parts.append(val)
                text = "\n".join(parts)
            except Exception:
                text = ""
        if not text:
            raise RuntimeError("empty response from LLM")
        return text.strip()

    except Exception as exc:  # keep this broad to surface provider errors cleanly
        logger.exception("explain() LLM call failed")
        if env == "development":
            # helpful message for developers, still deterministic-ish
            return f"llm-error: {type(exc).__name__}: {str(exc)[:200]}"
        raise RuntimeError("LLM provider call failed") from exc
