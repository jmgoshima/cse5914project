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
- Configurable: model and temperature come from environment variables; the
  conversation logic always routes through the configured LLM.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Callable, List

# Load .env variables if present for local dev (e.g., GOOGLE_API_KEY)
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .schemas import Profile

METRIC_ORDER = [
    "Climate",
    "HousingCost",
    "HlthCare",
    "Crime",
    "Transp",
    "Educ",
    "Arts",
    "Recreat",
    "Econ",
    "Pop",
]

METRIC_QUESTIONS = {
    "Climate": "Tell me about the climates you enjoy - are you picturing somewhere warm like Austin, mild like San Diego, or cooler like Seattle?",
    "HousingCost": "How flexible is your housing budget? Are you hoping for very affordable places or okay paying more if everything else fits?",
    "HlthCare": "How important is it for you to have strong healthcare and hospitals nearby?",
    "Crime": "What level of day-to-day safety feels right for you - very quiet neighborhoods or is some urban energy okay?",
    "Transp": "How do you like to get around in a new city - reliable transit, walkable streets, or mostly driving yourself?",
    "Educ": "Do good schools or education resources play a big role in your move?",
    "Arts": "How much do arts and cultural scenes (museums, music, festivals) matter to you?",
    "Recreat": "How important are outdoor activities or recreation options like parks, hiking, or beaches?",
    "Econ": "Tell me about the job market or economic strength you're looking for - does it need to be booming or just stable?",
    "Pop": "Do you gravitate toward small towns, midsize cities, or big metros? Any specific places come to mind?",
}

def _all_metrics_set(profile: Profile) -> bool:
    return all(getattr(profile, metric) is not None for metric in METRIC_ORDER)


def _record_turn(notes: Dict[str, Any], message: str) -> None:
    turns = notes.setdefault("turns", []) if isinstance(notes, dict) else None
    if isinstance(turns, list) and (not turns or turns[-1] != message):
        turns.append(message)



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
    """Prefer values from `new`, falling back to `old` when empty."""
    merged = Profile(**old.model_dump())

    for metric in METRIC_ORDER:
        value = getattr(new, metric, None)
        if value is not None:
            setattr(merged, metric, value)

    old_notes = old.notes if isinstance(old.notes, dict) else {}
    new_notes = new.notes if isinstance(new.notes, dict) else {}
    merged.notes = {**old_notes, **new_notes}
    return merged


# -------------------- LLM-powered updater ---------------------------------

# Build LLM objects at import time (once per process). If unavailable, we keep None.
_LLM = None
_BASE_PARSER = None
_PARSER = None
_PROMPT = None

if _LC_AVAILABLE:
    try:  # lazy, tolerate missing keys
        _LLM = ChatGoogleGenerativeAI(
            model=os.getenv("LC_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
            temperature=float(os.getenv("LC_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("LC_MAX_TOKENS", "1024")),
        )
        from langchain.output_parsers import OutputFixingParser

        _BASE_PARSER = PydanticOutputParser(pydantic_object=Profile)
        _PARSER = OutputFixingParser.from_llm(llm=_LLM, parser=_BASE_PARSER)
        _PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You score relocation preferences using a 10-value profile vector with these keys in order:
Climate, HousingCost, HlthCare, Crime, Transp, Educ, Arts, Recreat, Econ, Pop.
Every value must be a float between 0 and 10 (inclusive). Higher numbers mean a stronger preference or better fit for that attribute.

Guidance:
- Output JSON only, matching the Profile schema. Do not add prose around it.
- Carry forward existing scores from CURRENT_PROFILE unless the USER_MESSAGE gives new evidence.
- Ask clarifying questions in natural language (e.g., "Which cities have the vibe you like?" or "Do you lean toward warmer weather?"). Prefer stories, examples, and qualitative answers.
- Infer 0-10 scores from the conversation: translate qualitative cues into numbers (e.g., "very important" ~ 9, "I don't care" ~ 2, "I love hot weather" ~ 8). Only request an explicit number if the user volunteers it or it is absolutely necessary.
- Maintain notes.turns as an array and append the latest USER_MESSAGE if it is missing.
- When any metric is still uncertain, set notes.ready=false and place ONE conversational follow-up question about the most uncertain metric into notes.next_question. The question should sound natural (no scale references).
- When every metric is confidently filled, set notes.ready=true and remove notes.next_question.
- Never mention external tools or searching; you only build the profile.

Be precise and deterministic.
{format_instructions}
                """.strip(),
            ),
            ("human", "CURRENT_PROFILE:\n{current_profile}\n\nUSER_MESSAGE:\n{user_message}"),
        ])
    except Exception:
        _LLM = None
        _BASE_PARSER = None
        _PARSER = None
        _PROMPT = None


def stepAgent(profile: Profile, message: str) -> Profile:
    """Single turn of the conversation driven entirely by the configured LLM."""
    if not (_LLM and _PARSER and _PROMPT):
        raise RuntimeError(
            "LangChain conversation agent requires a configured LLM. "
            "Set GOOGLE_API_KEY / GEMINI credentials before running the CLI."
        )

    # Prepare inputs
    # Drop any stale follow-up question so each turn can propose a fresh one
    try:
        if isinstance(profile.notes, dict) and "next_question" in profile.notes:
            profile.notes.pop("next_question", None)
    except Exception:
        pass

    current_profile_json = profile.model_dump_json()
    if not _BASE_PARSER:
        raise RuntimeError("LLM parser is not initialized")

    format_instructions = _BASE_PARSER.get_format_instructions()

    # Run chain: prompt → llm → parse
    chain = _PROMPT | _LLM | _PARSER
    new_profile: Profile = chain.invoke({
        "current_profile": current_profile_json,
        "user_message": message,
        "format_instructions": format_instructions,
    })

    # Merge and return
    merged = _merge_profiles(profile, new_profile)
    # Ensure notes is a dict
    if not isinstance(merged.notes, dict):
        merged.notes = {}
    _record_turn(merged.notes, message)

    ready = _all_metrics_set(merged)
    merged.notes["ready"] = ready
    if ready:
        merged.notes.pop("next_question", None)
    else:
        if not merged.notes.get("next_question"):
            nq = _default_next_question(merged)
            if nq:
                merged.notes["next_question"] = nq
    return merged


def _default_next_question(profile: Profile) -> Optional[str]:
    """Ask for the first metric that is still unset."""
    for metric in METRIC_ORDER:
        if getattr(profile, metric) is None:
            return METRIC_QUESTIONS.get(metric)
    return None


# Optional readiness check the frontend can use (imported from here)

def is_profile_ready(profile: Profile) -> bool:
    """Ready when every metric has a numeric value."""
    return _all_metrics_set(profile)


# Optional helper: allow caller to inject a callback that runs search/recommend
# without this module importing ES directly.

def stepAgent_with_callback(profile: Profile, message: str, on_ready: Callable[[Profile], Dict[str, Any]]):
    """Run stepAgent; if the profile becomes ready, call `on_ready(profile)` and
    return a dict with both the profile and the callback result. This preserves a
    clean separation (no ES imports here) while enabling automatic orchestration.
    """
    updated = stepAgent(profile, message)
    result: Dict[str, Any] = {"profile": updated}
    try:
        if isinstance(updated.notes, dict) and updated.notes.get("ready"):
            result["on_ready_result"] = on_ready(updated)
    except Exception:
        # If callback fails, still return the profile
        pass
    return result
