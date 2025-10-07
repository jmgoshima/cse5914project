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

# Load .env variables if present for local dev (e.g., GOOGLE_API_KEY)
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .schemas import Profile

# --- Optional LangChain imports (graceful fallback if missing) ------------
_LC_AVAILABLE = True
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
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
        # Shallow-merge dictionaries (e.g., notes) so we don't lose history
        if isinstance(v_old, dict) and isinstance(v_new, dict):
            merged = dict(v_old)
            merged.update(v_new)
            return merged
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

    # ensure a clarifying question when not ready
    try:
        if not is_profile_ready(profile):
            if not profile.notes.get("next_question"):
                nq = _default_next_question(profile)
                if nq:
                    profile.notes["next_question"] = nq
    except Exception:
        pass
    return profile


# -------------------- LLM-powered updater ---------------------------------

# Build LLM objects at import time (once per process). If unavailable, we keep None.
_LLM = None
_PARSER = None
_PROMPT = None

if _LC_AVAILABLE:
    try:  # lazy, tolerate missing keys
        _LLM = ChatGoogleGenerativeAI(
            model=os.getenv("LC_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
            temperature=float(os.getenv("LC_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("LC_MAX_TOKENS", "1024")),
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
- When the profile is NOT ready, include a single clear clarifying question at notes.next_question.
- When the profile IS ready, set notes.ready=true and omit notes.next_question.
- Location rules: set hard_filters.country only when the user names a country (e.g., "United States"). Set hard_filters.state only when the user explicitly mentions a state by name; otherwise leave it null. Never guess states.
 - Default scope: unless the user names a different country, assume the search is within the United States and set hard_filters.country = "United States".
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
    # Drop any stale follow-up question so each turn can propose a fresh one
    try:
        if isinstance(profile.notes, dict) and "next_question" in profile.notes:
            profile.notes.pop("next_question", None)
    except Exception:
        pass

    # Capture whether this is the first observed turn based on the incoming profile
    has_prev_turns = False
    try:
        if isinstance(profile.notes, dict):
            has_prev_turns = bool(profile.notes.get("turns"))
    except Exception:
        pass

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
    # Ensure notes is a dict
    if not isinstance(merged.notes, dict):
        merged.notes = {}
    # Apply lightweight heuristics to fill obvious fields the model may omit
    _post_update_heuristics(merged, message, has_prev_turns)
    turns = merged.notes.setdefault("turns", [])
    if not turns or turns[-1] != message:
        turns.append(message)
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
        # Remove any leftover clarifying question when ready
        merged.notes.pop("next_question", None)
    else:
        # Ensure there is a clear next question; if the model omitted it, add a simple default.
        if not merged.notes.get("next_question"):
            nq = _default_next_question(merged)
            if nq:
                merged.notes["next_question"] = nq
    return merged


def _post_update_heuristics(profile: Profile, message: str, has_prev_turns: bool) -> None:
    """Best-effort fillers that do not overwrite confident values.

    Runs after the LLM merge to improve robustness in local/dev runs.
    """
    m = (message or "").lower()

    # Remote preference
    if profile.wants_remote_friendly is None and any(tok in m for tok in ["remote", "work from home", "wfh"]):
        profile.wants_remote_friendly = True

    # Budget
    if profile.budget_monthly_usd is None:
        money = re.search(r"\$?\s*([0-9]{3,5}(?:,[0-9]{3})?)", m)
        if money:
            try:
                amt = int(money.group(1).replace(",", ""))
                if 300 <= amt <= 20000:
                    profile.budget_monthly_usd = amt
            except Exception:
                pass

    # Climate hints and normalization
    if isinstance(profile.preferred_climates, list):
        existing = {str(x).lower() for x in profile.preferred_climates}
        def add_climate(label: str):
            label = label.lower()
            synonyms = {"warmer": "warm", "hot": "warm", "mild": "temperate"}
            norm = synonyms.get(label, label)
            if norm not in existing:
                profile.preferred_climates.append(norm)
                existing.add(norm)
        if any(k in m for k in ["warm", "warmer", "hot", "sunny"]):
            add_climate("warm")
        if any(k in m for k in ["cold", "snow", "wintry"]):
            add_climate("cold")
        if any(k in m for k in ["mild", "temperate", "constant weather", "doesn't change", "doesnt change", "stable weather"]):
            add_climate("temperate")

    # Country / state extraction
    hf = profile.hard_filters
    # If hard_filters missing, initialize a dict-like via model default
    if hf is None:
        try:
            from .schemas import HardFilters  # local import to avoid cycle
            profile.hard_filters = HardFilters()
            hf = profile.hard_filters
        except Exception:
            pass
    try:
        if hf is not None:
            # Default to US unless user named a different country
            if hf.country is None:
                if any(k in m for k in ["united states", "usa", "u.s.", "us", "america", "u.s.a."]):
                    hf.country = "United States"
                else:
                    # Default domain assumption per product: US cities
                    hf.country = "United States"
            # Simple state mapping
            state_map = {
                "ca": "California",
                "california": "California",
                "tx": "Texas",
                "texas": "Texas",
                "wa": "Washington",
                "washington": "Washington",
            }
            for key, val in state_map.items():
                if key in m:
                    hf.state = hf.state or val
                    break
            # Avoid first-turn hallucinated state if user didn't mention any state tokens
            if not has_prev_turns:
                any_state_token = any(k in m for k in state_map.keys())
                if hf.state and not any_state_token:
                    hf.state = None
    except Exception:
        pass

def _default_next_question(profile: Profile) -> Optional[str]:
    """Fallback clarifying question when the model doesn't provide one."""
    if profile.budget_monthly_usd is None:
        return "What is your approximate monthly housing budget (in USD)?"
    if not profile.preferred_climates:
        return "Do you prefer a warm, mild/temperate, or cold climate?"
    if not (profile.hard_filters and (profile.hard_filters.country or profile.hard_filters.state)):
        return "Which country or state do you want to focus on?"
    if not profile.commute_preference:
        return "Do you prefer walkable areas, good transit, or driving?"
    return "Any other must-haves (e.g., safety, schools, healthcare)?"


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
