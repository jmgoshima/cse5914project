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
from pathlib import Path
from typing import Optional

# Load .env if present so GOOGLE_API_KEY is available locally
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except Exception:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


def _coerce_message_text(msg) -> str:
    """Extract text content from a LangChain message object."""
    if hasattr(msg, 'content'):
        return str(msg.content)
    elif isinstance(msg, str):
        return msg
    else:
        return str(msg)


def _stub_explain(prompt: str) -> str:
    # deterministic, short stub for offline/dev environments
    snippet = prompt.replace("\n", " ")[:200]
    return f"stub-explain: {snippet}" if snippet else "stub-explain: (empty prompt)"


def explain(prompt: str, *, model: Optional[str] = None, max_tokens: int = 150, temperature: float = 0.2) -> str:
    """Return a short textual explanation for `prompt`.

    If `OPENAI_API_KEY` is present, call OpenAI; otherwise return
    a safe stub. This function is synchronous on purpose to match the calling
    code in `backend/app.py`.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    api_key = os.getenv("OPENAI_API_KEY")
    env = os.getenv("ENV", "development").lower()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        logger.debug("OPENAI_API_KEY not set, returning stub explain")
        return _stub_explain(prompt)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        text = _coerce_message_text(response)
        
        if not text:
            raise RuntimeError("empty response from LLM")
        return text.strip()

    except Exception as exc:
        logger.exception("explain() LLM call failed")
        if env == "development":
            return f"llm-error: {type(exc).__name__}: {str(exc)[:200]}"
        raise RuntimeError("LLM provider call failed") from exc
