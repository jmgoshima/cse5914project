# /backend/langchain/conversation.py
"""
LangChain conversation step for building a relocation Profile.

Design goals
------------
- Stateless server: the caller (frontend) sends the current Profile and the latest
  user message; we return an updated Profile.
- Structured output: the LLM returns a full `Profile` object (Pydantic) so we
  avoid brittle string parsing.
- Safe merge: fields the model leaves as `null`/None keep their previous values.
- Pluggable: model and temperature come from env vars; if LangChain/OpenAI
  aren't available the code falls back to a tiny heuristic updater so local dev
  still works.
"""
from __future__ import annotations

import os
import re
import json
from typing import Optional

from .schemas import Profile

# --- Optional LangChain imports (graceful fallback if missing) ------------
_LC_AVAILABLE = True
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
except Exception:  # pragma: no cover
    _LC_AVAILABLE = False


# -------------------- Utility: merge profiles -----------------------------

def _merge_profiles(old: Profile, new: Profile) -> Profile:
    """Merge `new` into `old`, preferring `new` when it's not None/empty.

    Lists are merged (de-duplicated, preserving order), scalars use `new` if set.
    """
    data_old = old.model_dump()
    data_new = new.model_dump()

    def _merge_value(k: str, v_old, v_new):
        if isinstance(v_old, list) and isinstance(v_new, list):
            seen = set()
            result = []
            for item in list(v_old) + list(v_new):
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        # Prefer v_new if it is truthy or explicitly False/0 (valid values)
        if v_new is not None:
            return v_new
        return v_old

    merged = {k: _merge_value(k, data_old.get(k), data_new.get(k)) for k in data_old.keys()}
    return Profile(**merged)


# -------------------- Heuristic fallback (no LLM) -------------------------

def _heuristic_update(profile: Profile, message: str) -> Profile:
    """Very small rule-based updater used when LLM isn't configured.
    This keeps local dev unblocked.
    """
    m = message.lower()

    # remote-friendly
    if any(tok in m for tok in ["remote", "work from home", "wfh"]):
        profile.wants_remote_friendly = True

    # climates
    climate_keywords = {
        "warm": "warm",
        "sunny": "warm",
        "beach": "mediterranean",
        "mediterranean": "mediterranean",
        "cold": "cold",
        "snow": "cold",
        "temperate": "temperate",
        "tropical": "tropical",
    }
    for k, label in climate_keywords.items():
        if k in m and label not in profile.preferred_climates:
            profile.preferred_climates.append(label)

    # industry
    if "tech" in m or "software" in m:
        profile.industry = profile.industry or "technology"
    if "finance" in m and profile.industry is None:
        profile.industry = "finance"

    # commute preference
    if any(w in m for w in ["walkable", "walkability", "walk"]):
        profile.commute_preference = "walkable"
    elif "transit" in m or "subway" in m:
        profile.commute_preference = "transit"
    elif "car" in m or "drive" in m:
        profile.commute_preference = "car"

    # budget extraction like: "$2,500", "2500", "budget 1800"
    money = re.search(r"\$?\s*([0-9]{3,5}(?:,[0-9]{3})?)", m)
    if money and profile.budget_monthly_usd is None:
        try:
            amt = int(money.group(1).replace(",", ""))
            if 300 <= amt <= 20000:
                profile.budget_monthly_usd = amt
        except Exception:
            pass

    # track message for traceability
    profile.notes.setdefault("turns", []).append(message)
    return profile


# -------------------- LLM-powered updater ---------------------------------

# Build LLM objects at import time (once per process). If unavailable, we keep None.
_LLM = None
_PARSER = None
_PROMPT = None

if _LC_AVAILABLE:
    try:  # lazy, tolerate missing keys
        _LLM = ChatOpenAI(
            model=os.getenv("LC_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            temperature=float(os.getenv("LC_TEMPERATURE", "0")),
            timeout=30,
        )
        _PARSER = PydanticOutputParser(pydantic_object=Profile)
        _PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a relocation profile builder. Given the CURRENT_PROFILE (a JSON object) and the latest USER_MESSAGE, produce a COMPLETE Profile JSON that best reflects the user's preferences.
Ask clarifying questions across turns until you are at least 85% confident in the profile. When you are confident, do not ask more questions—just update the JSON.
Rules:
- Only change fields you are confident about based on the USER_MESSAGE.
- Keep unspecified fields the same as CURRENT_PROFILE.
- For list fields, append new values and avoid duplicates.
- Be concise: do not invent details, use only what the user implies.
- Output JSON only, no extra text.
{format_instructions}
                """.strip(),
            ),
            ("human", "CURRENT_PROFILE:\n{current_profile}\n\nUSER_MESSAGE:\n{user_message}"),
        ])
    except Exception:
        _LLM = None
        _PARSER = None
        _PROMPT = None


def stepAgent(profile: Profile, message: str) -> Profile:
    """Single turn of the conversation: update and return the Profile.

    If LangChain + an LLM are configured, we use structured output to produce a
    full Profile and then merge it with the incoming one. Otherwise, we fall
    back to a small heuristic updater.
    """
    # If LLM isn't available, use heuristics
    if not (_LLM and _PARSER and _PROMPT):
        return _heuristic_update(profile, message)

    # Prepare inputs
    current_profile_json = profile.model_dump_json()
    format_instructions = _PARSER.get_format_instructions()

    # Run chain: prompt → llm → parse
    try:
        chain = _PROMPT | _LLM | _PARSER
        new_profile: Profile = chain.invoke({
            "current_profile": current_profile_json,
            "user_message": message,
            "format_instructions": format_instructions,
        })
    except Exception:
        # If anything goes wrong, gracefully fall back
        return _heuristic_update(profile, message)

    # Merge and return
    merged = _merge_profiles(profile, new_profile)
    merged.notes.setdefault("turns", []).append(message)
    # annotate readiness for callers (frontend or app.py can read this)
    try:
        ready = is_profile_ready(merged)
    except Exception:
        ready = False
    merged.notes["ready"] = ready
    if ready:
        # Suggest next action for the orchestrator. Do not perform the search here.
        merged.notes["next_action"] = {
            "type": "search_places",   # app.py can use this to call /recommend or /search/places
            "params": {
                "topK": 20,
                "take": 5
            }
        }
    return merged


# Optional readiness check the frontend can use (imported from here)

def is_profile_ready(profile: Profile) -> bool:
    """Define your own readiness criteria for moving on to search/recommend.
    Example heuristic: budget, at least one climate preference, and either
    industry or remote preference.
    """
    has_budget = profile.budget_monthly_usd is not None
    has_climate = bool(profile.preferred_climates)
    has_work = bool(profile.industry) or bool(profile.wants_remote_friendly)
    return has_budget and has_climate and has_work


# Optional helper: allow caller to inject a callback that runs search/recommend
# without this module importing ES directly.
from typing import Callable, Dict, Any

def stepAgent_with_callback(profile: Profile, message: str, on_ready: Callable[[Profile], Dict[str, Any]]):
    """Run stepAgent; if the profile becomes ready, call `on_ready(profile)` and
    return a dict with both the profile and the callback result. This preserves a
    clean separation (no ES imports here) while enabling automatic orchestration.
    """
    updated = stepAgent(profile, message)
    result: Dict[str, Any] = {"profile": updated}
    try:
        if updated.notes.get("ready"):
            result["on_ready_result"] = on_ready(updated)
    except Exception:
        # If callback fails, still return the profile
        pass
    return result