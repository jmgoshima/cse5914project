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
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

from pydantic import BaseModel, Field

# Load .env variables if present for local dev (e.g., GOOGLE_API_KEY)
# try:  # pragma: no cover
#     from dotenv import load_dotenv
#     load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
# except Exception:  # pragma: no cover
#     pass



# Temporary hardcoded Gemini creds to guarantee the agent initializes locally.
HARDCODED_GOOGLE_API_KEY = "AIzaSyBxNcTpTuCM7OyG_uuRGOXeuBZhTbwFn38"
# Old default model kept here for reference:
# HARDCODED_GEMINI_MODEL = "gemini-2.5-flash"
HARDCODED_GEMINI_MODEL = "gemini-1.5-flash"
os.environ.setdefault("GOOGLE_API_KEY", HARDCODED_GOOGLE_API_KEY)
os.environ.setdefault("GEMINI_MODEL", HARDCODED_GEMINI_MODEL)

from .schemas import Profile
from backend.search.qualitative import qualitative_to_numeric

# --- Optional LangChain imports (graceful fallback if missing) ------------
_LC_AVAILABLE = True
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.exceptions import OutputParserException
except Exception as exc:  # pragma: no cover
    print("LangChain Gemini imports failed:", type(exc).__name__, exc)
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


QUAL_FIELD_ORDER = [
    "climate",
    "transit",
    "safety",
    "healthcare",
    "education",
    "arts",
    "recreation",
    "economy",
    "population",
]

REMOTE_PREFERENCE_PROMPT = "Do you prefer remote-friendly work options? Answer yes or no."
MUST_HAVE_PROMPT = "Any other must-haves (e.g., safety, schools, healthcare)?"

_AFFIRMATIVE_WORDS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "absolutely",
    "affirmative",
    "of course",
    "definitely",
    "true",
}

_NEGATIVE_WORDS = {
    "no",
    "n",
    "nah",
    "nope",
    "none",
    "nothing",
    "no more",
    "no thanks",
    "no thank you",
    "no other must haves",
    "no other must have",
    "no other must-haves",
    "no other must-have",
    "none other",
    "done",
    "we are done here",
    "we're done",
    "that is all",
    "that's all",
}

_CONVERSATIONAL_FIELD_PROMPTS = {
    "climate": "What kind of climate feels best to you—are you chasing sunshine, four distinct seasons, or something in between?",
    "transit": "Describe your ideal transportation setup. Should it feel super walkable, have great transit, or is driving totally fine?",
    "safety": "Tell me about the level of safety that would make you feel at ease day to day.",
    "healthcare": "How close or high-quality do you need healthcare options to be?",
    "education": "Are nearby schools, universities, or learning hubs important for you or your household?",
    "arts": "Paint me a picture of the arts and culture vibe you’d love—live music, galleries, theaters, or something else entirely?",
    "recreation": "What kind of outdoor or recreation energy fits you best—endless trails, beaches, parks, or low-key green spaces?",
    "economy": "When you think about the local economy, are you drawn to booming job markets, steady stability, or a relaxed pace?",
    "population": "Awesome! For city size, do you imagine a massive metropolis, something mid-sized, or a more intimate community?",
}

FIELD_GUIDELINES = "\n".join([
    "• climate – understand the temperatures or seasons the user enjoys.",
    "• transit – learn if they need walkability, transit, or are fine driving.",
    "• safety – capture how safe they want to feel in daily life.",
    "• healthcare – ask about proximity/quality of care they expect.",
    "• education – note whether schools or learning hubs matter to them.",
    "• arts – uncover the culture/arts scene that energizes them.",
    "• recreation – find out their preferred outdoor or leisure vibe.",
    "• economy – learn whether they prefer booming job markets or calmer stability.",
    "• population – gauge whether they want a huge metropolis, something mid-sized, or intimate.",
])

STRUCTURE_INSTRUCTIONS = (
    "Respond with exactly one compact, single-line JSON object (no markdown, no prose before/after, no newlines) containing:\n"
    '{ "assistant_reply": string,\n'
    '  "ready": boolean,\n'
    '  "next_question": string or null,\n'
    '  "pending_fields": array_of_strings,\n'
    '  "profile": ProfileObject }\n'
    "assistant_reply and next_question must be short (1-2 sentences, each <= 100 characters), plain strings (no quotes that break JSON). "
    "Always include all keys above, even if values are null/empty. Total JSON < 400 characters; never truncate with ellipses; no extra keys. "
    "ProfileObject: use only these fields and types: preferred_climates (array: hot, warm, temperate, mild, cool, cold, tropical), "
    "budget_monthly_usd (number), housing_cost_target_max/min (number), healthcare_min_score/safety_min_score/transit_min_score/education_min_score/economy_score_min (numbers 0-10), "
    "population_min (number), hard_filters.country/state (string), hard_filters.min_population (number), notes (object), weights (object). "
    "Do NOT introduce other profile keys. Numeric fields must be numbers (not strings); scores must be 0-10. "
    "pending_fields must be a subset of: climate, transit, safety, healthcare, education, arts, recreation, economy, population. "
    "When ready=true, pending_fields must be empty and next_question null."
)


def _repair_json_text(text: str) -> str:
    """Best-effort trim/close JSON to avoid trivial truncation issues."""
    if not text or "{" not in text:
        return text
    candidate = text.strip()
    # Drop trailing ellipses or stray commas
    candidate = candidate.rstrip(".… ,")
    # Balance braces/brackets if obviously short
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if close_braces < open_braces:
        candidate += "}" * (open_braces - close_braces)
    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if close_brackets < open_brackets:
        candidate += "]" * (open_brackets - close_brackets)
    return candidate


def _extract_assistant_reply(text: str) -> Optional[str]:
    """Pull assistant_reply from a malformed JSON string if possible."""
    if not text:
        return None
    try:
        import re
        match = re.search(r'"assistant_reply"\s*:\s*"([^"]*)"', text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _coerce_partial_payload(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction when JSON is truncated/invalid."""
    if not text:
        return None
    payload: Dict[str, Any] = {}
    try:
        import re
        reply = _extract_assistant_reply(text)
        if reply:
            payload["assistant_reply"] = reply

        m_ready = re.search(r'"ready"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m_ready:
            payload["ready"] = m_ready.group(1).lower() == "true"

        m_next = re.search(r'"next_question"\s*:\s*"([^"]*)', text)
        if m_next:
            payload["next_question"] = m_next.group(1)

        m_pending = re.search(r'"pending_fields"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m_pending:
            raw_list = m_pending.group(1)
            items = []
            for part in raw_list.split(","):
                val = part.strip().strip('"').strip()
                if val:
                    items.append(val)
            payload["pending_fields"] = items

        if '"profile"' in text:
            payload["profile"] = {}
    except Exception:
        return None

    if payload:
        # ensure required keys exist
        payload.setdefault("assistant_reply", "")
        payload.setdefault("ready", False)
        payload.setdefault("next_question", None)
        payload.setdefault("pending_fields", [])
        payload.setdefault("profile", {})
        return payload
    return None

_QUAL_FIELD_INFO = {
    "climate": {
        "dataset_field": "Climate",
        "label": "climate",
        "keywords": ["climate", "weather", "temperature"],
        "note_key": None,
        "score_attr": None,
        "descriptions": {
            "arctic": "Extremely cold year round with long, dark winters.",
            "cold": "Long winters with frequent snow and freezing temperatures.",
            "continental": "Distinct seasons: hot summers and cold winters.",
            "cool": "Generally cool temperatures without severe cold.",
            "hot": "Very high temperatures most of the year.",
            "mild": "Gentle seasons with few temperature extremes.",
            "moderate": "Balanced temperatures with limited extremes.",
            "temperate": "Noticeable seasons with moderate summers and winters.",
            "tropical": "Year-round heat and humidity with wet and dry seasons.",
            "warm": "Mild winters paired with warm or hot summers.",
        },
        "synonyms": {
            "hot weather": "hot",
            "warm weather": "warm",
            "warm climate": "warm",
            "cold weather": "cold",
            "cold climate": "cold",
            "mild weather": "mild",
            "temperate climate": "temperate",
            "tropical weather": "tropical",
        },
    },
    "transit": {
        "dataset_field": "Transp",
        "label": "transit access",
        "keywords": ["transit", "transport", "transportation", "commute", "mobility"],
        "note_key": "transit",
        "score_attr": "transit_min_score",
        "descriptions": {
            "adequate": "Basic transit coverage that works for most essential trips.",
            "car dependent": "Driving is required for most errands or commuting.",
            "excellent": "Extensive, high-frequency transit network with broad coverage.",
            "good": "Reliable transit that covers many neighborhoods and job centers.",
            "limited": "Sparse routes or hours; transit useful only in specific cases.",
            "poor": "Transit rarely practical; expect long waits or gaps in service.",
            "transit friendly": "Designed around public transit with easy access to lines.",
            "walkable": "Daily needs reachable on foot without relying on vehicles.",
        },
        "synonyms": {
            "good transit": "good",
            "public transit": "transit friendly",
            "walkability": "walkable",
            "walkable": "walkable",
            "walkable city": "walkable",
            "need a car": "car dependent",
            "car required": "car dependent",
            "rely on car": "car dependent",
        },
    },
    "safety": {
        "dataset_field": "Crime",
        "label": "safety",
        "keywords": ["safety", "crime", "safe", "danger"],
        "note_key": "safety",
        "score_attr": "safety_min_score",
        "descriptions": {
            "average": "Crime levels similar to the national average.",
            "dangerous": "High crime; residents often express safety concerns.",
            "high": "Notably higher crime than average.",
            "low": "Lower crime than average; generally comfortable.",
            "medium": "Moderate crime levels between low and high.",
            "moderate": "Comparable to medium—some risk but manageable.",
            "safe": "Perceived as safe with low day-to-day risk.",
            "very high": "Significantly elevated crime, requiring caution.",
            "very low": "Exceptionally safe with minimal crime.",
        },
        "synonyms": {
            "good safety": "safe",
            "high safety": "high",
            "very high safety": "very high",
            "very safe": "very low",
            "extremely safe": "very low",
            "low crime": "very low",
            "no crime": "very low",
            "unsafe": "dangerous",
            "safe neighborhood": "safe",
        },
    },
    "healthcare": {
        "dataset_field": "HlthCare",
        "label": "healthcare",
        "keywords": ["healthcare", "health care", "medical", "hospital"],
        "note_key": "healthcare",
        "score_attr": "healthcare_min_score",
        "descriptions": {
            "adequate": "Basic access with acceptable quality of care.",
            "average": "Comparable to national norms for access and outcomes.",
            "excellent": "Top-tier hospitals and specialists widely available.",
            "good": "Reliable care with multiple hospital and clinic options.",
            "limited": "Few providers or services; longer travel for specialty care.",
            "poor": "Challenging access or quality concerns in local system.",
        },
        "synonyms": {
            "good hospitals": "good",
            "hospital": "good",
            "nearby hospital": "good",
            "top hospitals": "excellent",
            "average hospitals": "average",
            "limited healthcare": "limited",
        },
    },
    "education": {
        "dataset_field": "Educ",
        "label": "education",
        "keywords": ["education", "schools", "school", "students"],
        "note_key": "education",
        "score_attr": "education_min_score",
        "descriptions": {
            "not important": "Education is not a priority for this move.",
            "average": "Schools on par with national averages.",
            "excellent": "Top-performing schools with strong outcomes.",
            "good": "Above-average schools with solid reputations.",
            "poor": "Under-resourced schools or lower performance metrics.",
            "top": "Elite programs and consistently exceptional results.",
        },
        "synonyms": {
            "not important": "not important",
            "not a priority": "not important",
            "no school": "not important",
            "none needed": "not important",
            "good schools": "good",
            "great schools": "excellent",
            "top schools": "top",
            "average schools": "average",
            "poor schools": "poor",
        },
    },
    "arts": {
        "dataset_field": "Arts",
        "label": "arts & culture",
        "keywords": ["arts", "art", "culture", "arts & culture"],
        "note_key": "arts",
        "score_attr": None,
        "descriptions": {
            "average": "Some venues and events, comparable to mid-size cities.",
            "excellent": "Rich and renowned arts scene with many institutions.",
            "few": "Very limited arts venues or events.",
            "limited": "Occasional offerings but not a major focus.",
            "moderate": "Steady mix of arts experiences without being expansive.",
            "rich": "Diverse, high-quality arts organizations and venues.",
            "vibrant": "Frequent events and lively creative community.",
        },
        "synonyms": {
            "arts scene": "rich",
            "strong arts": "rich",
            "vibrant arts": "vibrant",
        },
    },
    "recreation": {
        "dataset_field": "Recreat",
        "label": "recreation & outdoors",
        "keywords": ["recreation", "outdoors", "outdoor", "parks"],
        "note_key": "recreation",
        "score_attr": None,
        "descriptions": {
            "abundant": "Plentiful outdoor spaces, trails, and recreation options.",
            "average": "A mix of parks and recreational programs.",
            "few": "Limited recreation infrastructure or green space.",
            "good": "Above-average variety of outdoor and recreation amenities.",
            "limited": "Some options, but availability can be sporadic.",
            "moderate": "Steady access to parks and activities without being expansive.",
        },
        "synonyms": {
            "outdoor activities": "abundant",
            "abundant outdoors": "abundant",
            "limited outdoors": "limited",
        },
    },
    "economy": {
        "dataset_field": "Econ",
        "label": "local economy",
        "keywords": ["economy", "economic", "jobs", "job market", "employment"],
        "note_key": "economy",
        "score_attr": "economy_score_min",
        "descriptions": {
            "average": "Job market aligned with national averages.",
            "booming": "Rapid growth and plentiful new opportunities.",
            "robust": "Stable economy with resilience to downturns.",
            "steady": "Consistent performance without major swings.",
            "strong": "Healthy job market with diverse industries.",
            "weak": "Limited growth or constrained job opportunities.",
        },
        "synonyms": {
            "strong job market": "strong",
            "booming jobs": "booming",
            "weak economy": "weak",
            "steady jobs": "steady",
        },
    },
    "population": {
        "dataset_field": "Pop",
        "label": "population size",
        "keywords": ["population", "size", "city size", "community size"],
        "note_key": "population",
        "score_attr": "population_min",
        "descriptions": {
            "huge": "Very large metro areas or major cities.",
            "large": "Large cities with substantial urban footprint.",
            "major": "Major metropolitan areas with significant influence.",
            "mid": "Mid-sized cities balancing urban and suburban feel.",
            "medium": "Medium population with moderate density.",
            "small": "Smaller towns or communities with quieter pace.",
        },
        "synonyms": {
            "small town": "small",
            "mid sized": "mid",
            "mid-sized": "mid",
            "big city": "large",
            "major city": "major",
        },
    },
}

for _field_key, _info in _QUAL_FIELD_INFO.items():
    _info["options"] = []
    _info["options_lower"] = []
    _info["synonyms"] = {str(k).lower(): v for k, v in _info.get("synonyms", {}).items()}

_FIELD_KEYWORD_ENTRIES: List[tuple[str, str]] = []
for _field_key, _info in _QUAL_FIELD_INFO.items():
    for _kw in _info.get("keywords", []):
        if isinstance(_kw, str) and _kw:
            _FIELD_KEYWORD_ENTRIES.append((_kw.lower(), _field_key))
_FIELD_KEYWORD_ENTRIES.sort(key=lambda item: len(item[0]), reverse=True)

_FIELD_NAME_LOOKUP: Dict[str, str] = {}
for _field_key, _info in _QUAL_FIELD_INFO.items():
    aliases = [
        _field_key,
        _info.get("dataset_field"),
        _info.get("label"),
        _info.get("note_key"),
    ]
    for alias in aliases:
        if isinstance(alias, str) and alias:
            alias_key = alias.lower().strip()
            if alias_key:
                _FIELD_NAME_LOOKUP[alias_key] = _field_key

_CLIMATE_OPTIONS: List[str] = []


def _set_qualitative_options_note(profile: Profile) -> None:
    """Legacy hook removed: no qualitative options shared in notes."""
    if isinstance(profile.notes, dict):
        profile.notes.pop("qualitative_options", None)


_EXPLANATION_TRIGGERS = (
    "what do",
    "what does",
    "what is",
    "what are",
    "meaning of",
    "mean",
    "explain",
    "difference",
)


def _user_requested_explanation(message: str) -> bool:
    text = (message or "").lower()
    return any(trigger in text for trigger in _EXPLANATION_TRIGGERS)


class AgentTurnOutput(BaseModel):
    """Structured response returned by the Gemini agent each turn."""

    profile: Profile
    assistant_reply: str = Field(
        ..., description="Natural language response to show the user."
    )
    ready: bool = False
    next_question: Optional[str] = Field(
        default=None,
        description="Follow-up question to keep refining the profile when not ready.",
    )
    pending_fields: List[str] = Field(
        default_factory=list,
        description="Short list of profile concepts that still require clarification.",
    )


# Keep prompts lean to avoid token overruns.
# Keep prompts lean to avoid token overruns, but retain enough context (last 6 messages).
_CHAT_HISTORY_LIMIT = 1


def _format_history_for_prompt(history: Any) -> str:
    """Return only the most recent assistant + user messages to keep prompts lean."""
    if not isinstance(history, list) or not history:
        return "No prior conversation."
    recent = history[-_CHAT_HISTORY_LIMIT:]
    lines: List[str] = []
    for entry in recent:
        if isinstance(entry, dict):
            role = entry.get("role") or "assistant"
            content = entry.get("content") or ""
        else:
            role = "user"
            content = str(entry)
        content = content.strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _append_history_entry(history: List[Dict[str, str]], role: str, content: str) -> None:
    text = (content or "").strip()
    if not text:
        return
    history.append({"role": role, "content": text})
    if len(history) > _CHAT_HISTORY_LIMIT:
        del history[: len(history) - _CHAT_HISTORY_LIMIT]


def _coerce_message_text(output: Any) -> str:
    if isinstance(output, BaseMessage):
        return _coerce_message_text(output.content)
    if isinstance(output, list):
        parts = [_coerce_message_text(item) for item in output]
        return "\n".join(part for part in parts if part)
    if output is None:
        return ""
    return str(output)


def _build_retry_feedback(error: Exception, raw_text: str) -> str:
    """Generate additional system guidance when a retry is needed."""
    reason = str(error).strip() or type(error).__name__
    snippet = (raw_text or "").strip()
    if snippet:
        if len(snippet) > 600:
            snippet = f"{snippet[:600]}…"
    else:
        snippet = "No JSON output was produced."
    return (
        "Reminder: respond with a single JSON object that matches the schema. "
        f"The previous attempt failed because: {reason}. "
        f"Previous output snippet: {snippet}"
    )


def _build_fallback_turn_output(profile: Profile, notes: Dict[str, Any]) -> AgentTurnOutput:
    """Create a safe fallback response when Gemini keeps failing."""
    turns = notes.get("turns") if isinstance(notes.get("turns"), list) else []
    if turns:
        message = (
            "I hit a snag reaching my language model, but I saved what you just shared. "
            "Let's keep going while I reconnect."
        )
    else:
        message = (
            "Hello there! I'm Compass, your friendly guide to finding a new home in the United States. "
            "Tell me your name and what you're dreaming about for this move so I can get started."
        )
    pending = notes.get("pending_fields") if isinstance(notes.get("pending_fields"), list) else []
    next_question = (
        notes.get("next_question")
        or notes.get("_last_question")
        or notes.get("_fallback_question")
        or _default_next_question(profile)
        or "Could you tell me a bit more about your ideal city?"
    )
    fallback_profile = Profile()
    return AgentTurnOutput(
        profile=fallback_profile,
        assistant_reply=message,
        ready=False,
        next_question=next_question,
        pending_fields=pending,
    )


def _sanitize_structured_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    profile = data.get("profile")
    if isinstance(profile, str):
        try:
            profile = json.loads(profile) if profile.strip() else {}
        except Exception:
            profile = {}
    if not isinstance(profile, dict):
        profile = {}
    notes = profile.get("notes")
    if isinstance(notes, str):
        try:
            profile["notes"] = json.loads(notes) if notes.strip() else {}
        except Exception:
            profile["notes"] = {}
    elif not isinstance(notes, dict):
        profile["notes"] = {}
    for key in ("hard_filters", "weights"):
        value = profile.get(key)
        if isinstance(value, str):
            profile[key] = None
    data["profile"] = profile

    pending = data.get("pending_fields")
    if isinstance(pending, str):
        data["pending_fields"] = [pending] if pending.strip() else []
    elif not isinstance(pending, list):
        data["pending_fields"] = []

    reply = data.get("assistant_reply")
    data["assistant_reply"] = reply or ""
    data["ready"] = bool(data.get("ready"))
    return data


def _profile_json_for_prompt(profile: Profile) -> str:
    """Trim bulky fields before sending profile context to the LLM."""
    data = profile.model_dump()
    notes = data.get("notes")
    if isinstance(notes, dict):
        for key in (
            "qualitative_options",
            "chat_history",
            "turns",
            "response",
            "next_action",
        ):
            notes.pop(key, None)
        for key in list(notes.keys()):
            if key.startswith("qualitative_options"):
                notes.pop(key, None)
    return json.dumps(data, ensure_ascii=False)




def _infer_fields_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    lower = text.lower()
    matched: List[str] = []
    for keyword, field_key in _FIELD_KEYWORD_ENTRIES:
        if keyword in lower and field_key not in matched:
            matched.append(field_key)
    for field_key, info in _QUAL_FIELD_INFO.items():
        if field_key in matched:
            continue
        hits = sum(1 for option in info.get("options_lower", []) if option in lower)
        if hits >= 2:
            matched.append(field_key)
    return matched


def _canonicalize_option(field_key: str, value: str) -> Optional[str]:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info or not value:
        return None
    value_lower = value.lower()
    for option, opt_lower in zip(info.get("options", []), info.get("options_lower", [])):
        if opt_lower == value_lower:
            return option
    return None


def _build_field_explanation(field_key: str) -> Optional[str]:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info:
        return None
    options = info.get("options", [])
    descriptions = info.get("descriptions", {})
    lines: List[str] = []
    for option in options:
        desc = descriptions.get(option.lower()) or descriptions.get(option) or "Description unavailable."
        lines.append(f"{option}: {desc}")
    label = info.get("label", field_key)
    label_title = label[:1].upper() + label[1:]
    options_list = ", ".join(options)
    section = f"{label_title} options explained:\n" + "\n".join(lines)
    question = f"Which {label} option do you prefer? Choose from: {options_list}."
    return f"{section}\n{question}"


def _build_clarification_text(fields: List[str]) -> Optional[str]:
    sections: List[str] = []
    for field_key in fields:
        section = _build_field_explanation(field_key)
        if section:
            sections.append(section)
    if not sections:
        return None
    return "\n\n".join(sections)


def _maybe_handle_clarification(profile: Profile, message: str) -> bool:
    """Detect clarification requests and populate the next question with explanations."""
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    if not _user_requested_explanation(message):
        return False

    pending: List[str] = []
    try:
        if isinstance(profile.notes.get("pending_fields"), list):
            pending = [f for f in profile.notes["pending_fields"] if f in _QUAL_FIELD_INFO]
    except Exception:
        pending = []

    last_question = profile.notes.get("next_question") or profile.notes.get("_last_question")
    fields = pending or _infer_fields_from_text(last_question) or _infer_fields_from_text(message)
    if not fields:
        return False

    clarification_text = _build_clarification_text(fields)
    if not clarification_text:
        return False

    profile.notes["next_question"] = clarification_text
    profile.notes["pending_fields"] = fields
    profile.notes["_last_question"] = clarification_text
    profile.notes.pop("next_action", None)
    profile.notes["ready"] = False
    return True


def _collect_invalid_values(field_name: str, raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    values: List[str] = []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [v for v in raw_value if isinstance(v, str)]
    else:
        return []
    return [v for v in values if qualitative_to_numeric(field_name, v) is None]


def _match_field_keyword(raw_key: str) -> Optional[str]:
    text = (raw_key or "").lower()
    for keyword, field_key in _FIELD_KEYWORD_ENTRIES:
        if keyword in text:
            return field_key
    return None


def _match_option(field_key: str, text: str) -> Optional[str]:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info or not text:
        return None
    cleaned = re.sub(r"[^a-z0-9\s&-]", " ", text.lower())
    for synonym, target in info.get("synonyms", {}).items():
        if synonym in cleaned:
            canonical = _canonicalize_option(field_key, target)
            return canonical or target
    for option, opt_lower in zip(info.get("options", []), info.get("options_lower", [])):
        if opt_lower and opt_lower in cleaned:
            return option
    return None


def _normalize_for_match(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _interpret_binary_response(normalized_message: str, positives: set[str], negatives: set[str]) -> Optional[bool]:
    if not normalized_message:
        return None
    first_token = normalized_message.split(" ", 1)[0]
    if normalized_message in negatives or first_token in negatives:
        return False
    if normalized_message in positives or first_token in positives:
        return True
    return None


def _capture_qualitative_answers(profile: Profile, message: str) -> bool:
    if not message or not isinstance(profile.notes, dict):
        return False
    text = message.lower()
    found: Dict[str, str] = {}

    pending_fields: List[str] = []
    if isinstance(profile.notes.get("pending_fields"), list):
        pending_fields = [f for f in profile.notes["pending_fields"] if f in _QUAL_FIELD_INFO]
    candidate_fields: List[str] = list(pending_fields)
    if not candidate_fields:
        awaiting_question = profile.notes.get("_awaiting_question")
        last_question = profile.notes.get("_last_question")
        next_question = profile.notes.get("next_question")
        question_text = awaiting_question or last_question or next_question
        candidate_fields = _infer_fields_from_text(question_text) or []
    if not candidate_fields:
        candidate_fields = list(_QUAL_FIELD_INFO.keys())

    for raw_key, raw_value in re.findall(r"([a-z &]+?)\s*[:\-]\s*([a-z0-9\s&-]+)", text):
        field_key = _match_field_keyword(raw_key.strip())
        if not field_key:
            continue
        if candidate_fields and field_key not in candidate_fields:
            continue
        option = _match_option(field_key, raw_value.strip())
        if option:
            found[field_key] = option

    for field_key in pending_fields:
        if field_key in found:
            continue
        option = _match_option(field_key, text)
        if option:
            found[field_key] = option

    if candidate_fields:
        for field_key in candidate_fields:
            if field_key in pending_fields or field_key in found:
                continue
            option = _match_option(field_key, text)
            if option:
                found[field_key] = option

    if not found:
        return False

    if profile.notes.get("_awaiting_question"):
        profile.notes.pop("_awaiting_question", None)

    profile.notes.setdefault("qual_answers", {})
    for field_key, option in found.items():
        _apply_field_value(profile, field_key, option)

    if "pending_fields" in profile.notes:
        remaining = [f for f in profile.notes["pending_fields"] if f not in found]
        if remaining:
            profile.notes["pending_fields"] = remaining
        else:
            profile.notes.pop("pending_fields", None)
    return True


def _handle_binary_questions(profile: Profile, message: str) -> bool:
    if not message or not isinstance(profile.notes, dict):
        return False
    last_question = profile.notes.get("next_question") or profile.notes.get("_last_question")
    normalized_last = _normalize_for_match(last_question)
    if not normalized_last:
        return False
    normalized_message = _normalize_for_match(message)
    if not normalized_message:
        return False

    handled = False
    if normalized_last == _normalize_for_match(REMOTE_PREFERENCE_PROMPT):
        answer = _interpret_binary_response(normalized_message, _AFFIRMATIVE_WORDS, _NEGATIVE_WORDS)
        if answer is not None:
            profile.wants_remote_friendly = answer
            handled = True
    elif normalized_last == _normalize_for_match(MUST_HAVE_PROMPT):
        answer = _interpret_binary_response(normalized_message, _AFFIRMATIVE_WORDS, _NEGATIVE_WORDS)
        if answer is False:
            profile.notes["declined_additional_must_haves"] = True
            handled = True

    if handled:
        profile.notes.pop("pending_fields", None)
        profile.notes.pop("next_question", None)
    return handled


def _apply_field_value(profile: Profile, field_key: str, option: str) -> None:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info:
        return
    profile.notes.setdefault("qual_answers", {})
    profile.notes["qual_answers"][field_key] = option
    note_key = info.get("note_key")
    if note_key:
        profile.notes[note_key] = option

    if field_key == "climate":
        if profile.preferred_climates is None:
            profile.preferred_climates = []
        existing = [str(x).lower() for x in profile.preferred_climates if isinstance(x, str)]
        option_lower = option.lower()
        if option_lower not in existing:
            profile.preferred_climates.append(option)
        return

    score_attr = info.get("score_attr")
    if score_attr:
        numeric = qualitative_to_numeric(info["dataset_field"], option)
        if numeric is not None:
            setattr(profile, score_attr, numeric)


def _normalize_qualitative_answers(profile: Profile) -> None:
    """Ensure qualitative answers use canonical field keys and option values."""
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    qual_answers = profile.notes.get("qual_answers")
    if not isinstance(qual_answers, dict):
        return

    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in list(qual_answers.items()):
        if not isinstance(raw_key, str):
            continue
        lookup_key = raw_key.lower().strip()
        field_key = _FIELD_NAME_LOOKUP.get(lookup_key, raw_key)
        value = raw_value
        if isinstance(raw_value, str):
            canonical_option = _canonicalize_option(field_key, raw_value)
            if canonical_option and canonical_option != raw_value:
                value = canonical_option
        normalized[field_key] = value

    if not normalized:
        return

    profile.notes["qual_answers"] = normalized

    for field_key, value in normalized.items():
        if isinstance(value, str):
            _apply_field_value(profile, field_key, value)

    # Also normalize any alias keys the LLM may have added directly onto notes
    for field_key, info in _QUAL_FIELD_INFO.items():
        note_key = info.get("note_key")
        if not note_key:
            continue
        if isinstance(profile.notes.get(note_key), str):
            continue
        label = info.get("label") or ""
        dataset_field = info.get("dataset_field") or ""
        candidate_keys = {
            label,
            label.lower(),
            dataset_field,
            dataset_field.lower(),
            re.sub(r"[^a-z0-9]+", "_", label.lower()),
            re.sub(r"[^a-z0-9]+", "", label.lower()),
            re.sub(r"[^a-z0-9]+", " ", label.lower()).strip(),
            label.replace("&", "and"),
            label.replace("&", "").strip(),
        }
        candidate_keys = {str(k).strip() for k in candidate_keys if isinstance(k, str) and k}
        for alias in candidate_keys:
            if alias == note_key:
                continue
            value = profile.notes.get(alias)
            if not isinstance(value, str):
                continue
            canonical = _canonicalize_option(field_key, value) or _canonicalize_option(field_key, value.lower())
            if canonical:
                _apply_field_value(profile, field_key, canonical)
                if alias != note_key:
                    try:
                        profile.notes.pop(alias, None)
                    except Exception:
                        pass
                break


def _is_field_answered(profile: Profile, field_key: str) -> bool:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info or not isinstance(profile.notes, dict):
        return False
    answers = profile.notes.get("qual_answers", {})

    if field_key == "climate":
        return bool(profile.preferred_climates)

    note_key = info.get("note_key")
    if note_key and profile.notes.get(note_key):
        return True

    score_attr = info.get("score_attr")
    if score_attr and getattr(profile, score_attr, None) is not None:
        return True

    return bool(answers.get(field_key))


def _question_for_field(field_key: str) -> Optional[str]:
    info = _QUAL_FIELD_INFO.get(field_key)
    if not info:
        return None
    conversational = _CONVERSATIONAL_FIELD_PROMPTS.get(field_key)
    if conversational:
        return conversational
    label = info.get("label", field_key)
    return f"Tell me about your preferences for {label} in your next city."


def _sanitize_question_text(question: str, profile: Profile) -> Optional[str]:
    text = (question or "").strip()
    if not text:
        return None
    if text and text[-1] not in ".?!":
        text = text + "?"
    return text


def _cleanup_pending_fields(profile: Profile) -> None:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    if "pending_fields" not in profile.notes:
        return
    pending = profile.notes.get("pending_fields")
    if not pending:
        profile.notes.pop("pending_fields", None)
        return
    if not isinstance(pending, list):
        profile.notes.pop("pending_fields", None)
        return
    remaining: List[str] = []
    for field_key in pending:
        if field_key in _QUAL_FIELD_INFO and not _is_field_answered(profile, field_key):
            remaining.append(field_key)
    if remaining:
        profile.notes["pending_fields"] = remaining
    else:
        profile.notes.pop("pending_fields", None)


def _refresh_questionnaire_state(profile: Profile) -> None:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    _cleanup_pending_fields(profile)
    notes = profile.notes
    if notes.get("pending_fields"):
        return
    next_question = notes.get("next_question")
    if not next_question:
        return
    related_fields = _infer_fields_from_text(next_question)
    if related_fields and all(_is_field_answered(profile, f) for f in related_fields):
        notes.pop("next_question", None)


def _determine_next_question(profile: Profile) -> Tuple[Optional[str], List[str]]:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    for field_key in QUAL_FIELD_ORDER:
        if _is_field_answered(profile, field_key):
            continue
        question = _question_for_field(field_key)
        if question:
            return question, [field_key]
    return None, []


def _advance_questionnaire(profile: Profile) -> None:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    if profile.notes.get("pending_fields"):
        return
    question, fields = _determine_next_question(profile)
    if fields:
        profile.notes["pending_fields"] = fields
    else:
        profile.notes.pop("pending_fields", None)


def _ensure_initial_question(profile: Profile) -> None:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    question, fields = _determine_next_question(profile)
    if fields:
        profile.notes["pending_fields"] = fields
    profile.notes["ready"] = False


def _ensure_qualitative_alignment(profile: Profile) -> List[str]:
    """Validate that qualitative responses map to known descriptors.

    Returns a list of guidance sentences if re-asking is required.
    """
    guidance: List[str] = []

    if isinstance(profile.preferred_climates, list) and profile.preferred_climates:
        valid_climates: List[str] = []
        invalid_climates: List[str] = []
        for value in profile.preferred_climates:
            if isinstance(value, str) and qualitative_to_numeric("Climate", value) is None:
                invalid_climates.append(value)
            else:
                valid_climates.append(value)
        if invalid_climates:
            profile.preferred_climates = valid_climates
            climate_options = ", ".join(_CLIMATE_OPTIONS)
            guidance.append(
                f"Please choose a climate preference from: {climate_options}."
            )

    if isinstance(profile.notes, dict):
        qual_answers = profile.notes.get("qual_answers", {})
        for field_key, info in _QUAL_FIELD_INFO.items():
            note_key = info.get("note_key")
            if not note_key:
                continue
            invalid_values = _collect_invalid_values(info["dataset_field"], profile.notes.get(note_key))
            if invalid_values:
                profile.notes.pop(note_key, None)
                if isinstance(qual_answers, dict) and field_key in qual_answers:
                    qual_answers.pop(field_key, None)
                options = ", ".join(info.get("options", []))
                human_label = info.get("label", field_key)
                guidance.append(
                    f"For {human_label}, respond using: {options}."
                )

    return guidance


# -------------------- Heuristic fallback (no LLM) -------------------------

def _heuristic_update(profile: Profile, message: str) -> Profile:
    """Very small rule-based updater used when LLM isn't configured.
    This keeps local dev unblocked.
    """
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    notes = profile.notes
    _normalize_qualitative_answers(profile)
    _refresh_questionnaire_state(profile)
    previous_question = notes.get("next_question")
    if previous_question:
        notes["_last_question"] = previous_question

    incoming_message = message or ""
    if not notes.get("turns") and not incoming_message.strip():
        _ensure_initial_question(profile)
        notes.setdefault("turns", [])
        notes["ready"] = False
        return profile

    m = incoming_message.lower()

    # remote-friendly
    if any(tok in m for tok in ["remote", "work from home", "wfh"]):
        profile.wants_remote_friendly = True

    # climates
    climate_keywords = {
        "warm": "warm",
        "sunny": "warm",
        "hot": "hot",
        "beach": "warm",
        "cold": "cold",
        "snow": "cold",
        "temperate": "temperate",
        "mild": "mild",
        "cool": "cool",
        "tropical": "tropical",
    }
    for k, label in climate_keywords.items():
        if k in m:
            _apply_field_value(profile, "climate", label)

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

    clarification_handled = _maybe_handle_clarification(profile, incoming_message)
    captured_answers = False
    handled_binary = False
    if not clarification_handled:
        captured_answers = _capture_qualitative_answers(profile, incoming_message)
        handled_binary = _handle_binary_questions(profile, incoming_message)
        if (captured_answers or handled_binary) and "next_question" in notes and not notes.get("pending_fields"):
            notes.pop("next_question", None)
        if captured_answers or handled_binary:
            _refresh_questionnaire_state(profile)

    # track message for traceability
    notes.setdefault("turns", []).append(incoming_message)

    if not clarification_handled:
        if notes.get("pending_fields"):
            pass
        else:
            _advance_questionnaire(profile)

    # Compute missing fields deterministically to avoid repeats.
    missing_fields = _missing_fields(profile)
    # Always overwrite pending_fields based on current profile state
    if missing_fields:
        pending = [f for f in missing_fields if f != "budget"]
        notes["pending_fields"] = pending
    else:
        notes.pop("pending_fields", None)

    guidance = [] if clarification_handled else _ensure_qualitative_alignment(profile)
    if guidance:
        notes["ready"] = False
        notes["next_question"] = " ".join(guidance)
        notes.pop("pending_fields", None)
    elif clarification_handled:
        notes["ready"] = False
    else:
        try:
            ready = is_profile_ready(profile)
        except Exception:
            ready = False
        notes["ready"] = ready
        if ready:
            notes.pop("next_question", None)
            notes.pop("pending_fields", None)
        else:
            # Always drive the next question from the first missing field (budget first if missing)
            field_for_question = None
            if "budget" in missing_fields:
                field_for_question = "budget"
            elif notes.get("pending_fields"):
                field_for_question = notes["pending_fields"][0]
            if field_for_question:
                qtext = _question_for_key(field_for_question)
                if qtext:
                    notes["next_question"] = qtext
                    notes["_last_question"] = qtext
                    notes["response"] = qtext
            else:
                notes.pop("next_question", None)

    _set_qualitative_options_note(profile)
    return profile


# -------------------- LLM parsing helpers --------------------

def _parse_llm_profile_block(raw_text: str) -> tuple[str, Dict[str, str]]:
    """Split assistant reply and fenced profile block."""
    import re
    if not raw_text:
        return "", {}
    match = re.search(r"```profile(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return raw_text.strip(), {}
    block = match.group(1)
    # Remove the fenced block from assistant reply
    assistant_reply = (raw_text[: match.start()] + raw_text[match.end() :]).strip()
    parsed: Dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower()
        v = val.strip()
        parsed[k] = v
    return assistant_reply, parsed


def _apply_parsed_profile(profile: Profile, data: Dict[str, str]) -> None:
    """Apply parsed LLM block into the Profile object."""
    if not isinstance(data, dict):
        return
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    notes = profile.notes

    def to_float(value: str) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value.replace(",", ""))
        except Exception:
            return None

    # Key normalization
    key_map = {
        "climate": "climate",
        "transit": "transit",
        "safety": "safety",
        "healthcare": "healthcare",
        "education": "education",
        "arts": "arts",
        "recreation": "recreation",
        "economy": "economy",
        "population": "population",
        "pop": "population",
        "budget": "budget",
        "next_question": "next_question",
    }

    for raw_key, raw_val in data.items():
        key = key_map.get(raw_key.lower().strip())
        if not key:
            continue
        val = raw_val.strip()
        if not val:
            continue
        if key == "climate":
            if profile.preferred_climates is None:
                profile.preferred_climates = []
            if val.lower() not in [str(x).lower() for x in profile.preferred_climates]:
                profile.preferred_climates.append(val)
        elif key == "transit":
            num = to_float(val)
            if num is not None:
                profile.transit_min_score = num
        elif key == "safety":
            num = to_float(val)
            if num is not None:
                profile.safety_min_score = num
        elif key == "healthcare":
            num = to_float(val)
            if num is not None:
                profile.healthcare_min_score = num
        elif key == "education":
            num = to_float(val)
            if num is not None:
                profile.education_min_score = num
        elif key == "economy":
            num = to_float(val)
            if num is not None:
                profile.economy_score_min = num
        elif key == "population":
            num = to_float(val)
            if num is not None and num > 0:
                profile.population_min = num
                notes["population"] = val
        elif key == "budget":
            num = to_float(val)
            if num is not None and num > 0:
                profile.budget_monthly_usd = int(num)
                profile.housing_cost_target_max = float(num)
                notes["budget"] = val
        elif key == "arts":
            notes["arts"] = val
        elif key == "recreation":
            notes["recreation"] = val
        elif key == "next_question":
            notes["next_question"] = val
            notes["_last_question"] = val



# -------------------- LLM-powered updater ---------------------------------

# Build LLM objects at import time (once per process). If unavailable, we keep None.
_LLM = None
if _LC_AVAILABLE:
    try:  # lazy, tolerate missing keys
        api_key = os.getenv("GOOGLE_API_KEY")
        print("LLM init: API key present?", bool(api_key))
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set; cannot configure Gemini.")
        genai.configure(api_key=api_key)
        _LLM = ChatGoogleGenerativeAI(
            model=os.getenv("LC_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")),
            temperature=float(os.getenv("LC_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("LC_MAX_TOKENS", "256")),
        )
        # Legacy structured prompt retained for easy rollback:
        # _PROMPT = ChatPromptTemplate.from_messages([
        #     (
        #         "system",
        #         (
        #             "You are Compass, a relocation assistant. Respond with exactly one JSON object "
        #             "that matches the structured schema you have been given."
        #         ),
        #     ),
        #     (
        #         "system",
        #         (
        #             "Schema definition and formatting rules:\n{format_instructions}\n"
        #             "Do not echo these instructions in fields other than assistant_reply."
        #         ),
        #     ),
        #     (
        #         "system",
        #         (
        #             "{system_feedback}"
        #         ),
        #     ),
        #     ("human", "CONVERSATION_HISTORY:\n{conversation_history}\n\nCURRENT_PROFILE_JSON:\n{current_profile}\n\nLATEST_USER_MESSAGE:\n{user_message}\n\nRespond only with JSON."),
        # ]).partial(
        #     system_feedback="",
        # )
        _PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are Compass, a US relocation assistant. Output ONE JSON line that matches the schema. "
                    "Only collect: climate, transit, safety, healthcare, education, arts, recreation, economy, population/size, housing budget, optional state/country. "
                    "If the user message contains ANY of these, update the corresponding profile fields immediately (do not wait to ask again). "
                    "For scores, infer a 0–10 numeric value when possible; for climate, use hot/warm/temperate/mild/cool/cold/tropical; for population, set a numeric minimum if a size is implied (e.g., 'big city' => 1_000_000). "
                    "If nothing new is provided, ask the next missing field. Short, complete sentences; no ellipses; no extra keys. Stay concise and friendly."
                ),
            ),
            (
                "system",
                "Structured output rules:\n{structure_instructions}\nDo not restate these instructions; put only conversational text in assistant_reply.",
            ),
            (
                "system",
                "{system_feedback}",
            ),
            (
                "human",
                "HISTORY:\n{conversation_history}\nPROFILE:\n{current_profile}\nUSER:\n{user_message}\nReturn only JSON.",
            ),
        ]).partial(
            system_feedback="",
            structure_instructions=STRUCTURE_INSTRUCTIONS,
        )
        print("LLM init complete.")
    except Exception as exc:
        print("LLM INIT FAILED:", type(exc).__name__, exc)
        _LLM = None
        _PROMPT = None


def stepAgent(profile: Profile, message: str) -> Profile:
    """Single conversational turn handled purely by the Gemini agent."""
    incoming_message = message or ""
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    notes = profile.notes
    has_prev_turns = bool(notes.get("turns"))
    previous_question = notes.get("_last_question")
    state_before = profile.model_dump()

    if not _LLM:
        print("stepAgent aborting: LLM stack not ready.")
        raise RuntimeError(
            "Gemini chat agent is not configured. "
            "Set GOOGLE_API_KEY / GEMINI_MODEL and restart the backend."
        )

    if not has_prev_turns and not incoming_message.strip():
        _ensure_initial_question(profile)
        notes.setdefault("turns", [])
        notes["ready"] = False
        notes.setdefault(
            "response",
            "Hi! Tell me a bit about your ideal city so I can start building your profile.",
        )
        return profile

    # Build prompt focusing on LLM-driven flow (assistant_reply + fenced profile block)
    history_text = _format_history_for_prompt(notes.get("chat_history"))
    current_profile_json = _profile_json_for_prompt(profile)
    user_msg = incoming_message.strip()
    prompt = (
        "You are Compass, a professional yet warm US relocation assistant.\n"
        "On each turn you must:\n"
        "1) Give a short assistant reply to the user (keep it concise and friendly).\n"
        "2) Update any fields you can infer and output them inside a fenced block exactly like:\n"
        "```profile\n"
        "climate: <hot|warm|temperate|mild|cool|cold|tropical or blank>\n"
        "transit: <0-10 or blank>\n"
        "safety: <0-10 or blank>\n"
        "healthcare: <0-10 or blank>\n"
        "education: <0-10 or blank>\n"
        "arts: <short text or blank>\n"
        "recreation: <short text or blank>\n"
        "economy: <0-10 or blank>\n"
        "population: <numeric minimum or blank>\n"
        "budget: <monthly housing USD number or blank>\n"
        "next_question: <one concise follow-up for the most important missing field>\n"
        "```\n"
        "Rules:\n"
        "- Keep numbers 0–10 where applicable; you may infer decimals up to 15 places but keep the string short.\n"
        "- Population is numeric (e.g., 1000000 for a big city). Budget is numeric USD (accept $ or commas in user text, output plain number).\n"
        "- If you cannot infer a field, leave it blank. Do not invent extra keys. Always include next_question.\n"
        "- Stay concise; no extra fences or commentary. The user-facing reply must stay outside the fenced block. Always end with a question to the user, never end with a statement and only about the fields stated above nothing else, stick to the script.\n\n"
        f"HISTORY (recent):\n{history_text}\n\n"
        f"CURRENT_PROFILE_JSON:\n{current_profile_json}\n\n"
        f"LATEST_USER_MESSAGE:\n{user_msg or 'None'}\n"
        "Produce the assistant reply, then the fenced profile block."
    )

    # Call LLM
    try:
        llm_output_text = _coerce_message_text(_LLM.invoke([HumanMessage(content=prompt)])).strip()
    except Exception:
        # Fall back to deterministic flow on error
        _finalize_turn(profile, notes, has_prev_turns, incoming_message)
        return profile

    assistant_reply, parsed_block = _parse_llm_profile_block(llm_output_text)
    _apply_parsed_profile(profile, parsed_block)

    # Lightweight deterministic inference as a safety net
    _capture_qualitative_answers(profile, user_msg)
    _post_update_heuristics(profile, user_msg or "", has_prev_turns)
    _normalize_qualitative_answers(profile)
    _refresh_questionnaire_state(profile)

    # Track history
    history = notes.get("chat_history")
    if not isinstance(history, list):
        history = []
        notes["chat_history"] = history
    if user_msg:
        _append_history_entry(history, "user", user_msg)
    if assistant_reply:
        _append_history_entry(history, "assistant", assistant_reply)

    # Determine readiness and next question
    missing = _missing_fields(profile)
    ready = False
    try:
        ready = is_profile_ready(profile)
    except Exception:
        ready = False
    notes["ready"] = ready

    next_q = parsed_block.get("next_question") if isinstance(parsed_block, dict) else None
    if not next_q and not ready and missing:
        field = "budget" if "budget" in missing else missing[0]
        next_q = _question_for_key(field) or "Can you tell me a bit more?"

    notes["next_question"] = None if ready else next_q
    notes["_last_question"] = None if ready else next_q

    # response shown to user
    if ready:
        notes["response"] = assistant_reply or "Thanks! I have what I need. Let me fetch recommendations."
        notes.pop("pending_fields", None)
        notes.pop("next_action", None)
    else:
        # Prefer to show both the assistant reply and the next question so the user knows what to answer.
        if assistant_reply and next_q:
            notes["response"] = f"{assistant_reply} {next_q}"
        else:
            notes["response"] = assistant_reply or next_q or "Can you share a bit more?"
        if missing:
            notes["pending_fields"] = [f for f in missing if f != "budget"]

    # Track turns
    turns = notes.setdefault("turns", [])
    if user_msg:
        turns.append(user_msg)
    if assistant_reply:
        turns.append(assistant_reply)

    return profile


def _post_update_heuristics(profile: Profile, message: str, has_prev_turns: bool) -> None:
    """Best-effort fillers that do not overwrite confident values.

    Runs after the LLM merge to improve robustness in local/dev runs.
    """
    m = (message or "").lower()
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    last_q = (notes.get("_last_question") or "").lower()

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

    # Population from qualitative hints
    if profile.population_min is None:
        pop_val = None
        if "big city" in m or "metropolis" in m or "large city" in m:
            pop_val = 1_000_000
        elif "mid" in m or "medium city" in m or "mid-size" in m:
            pop_val = 200_000
        elif "small town" in m or "small city" in m:
            pop_val = 50_000
        pop_note = None
        try:
            pop_note = notes.get("population")
        except Exception:
            pop_note = None
        if isinstance(pop_note, str):
            pn = pop_note.lower()
            if "large" in pn or "big" in pn:
                pop_val = pop_val or 1_000_000
            elif "mid" in pn:
                pop_val = pop_val or 200_000
            elif "small" in pn:
                pop_val = pop_val or 50_000
        if pop_val:
            profile.population_min = float(pop_val)

    # Normalize improbable population values (discard tiny numbers that look like scores)
    if profile.population_min is not None and profile.population_min < 1000:
        profile.population_min = None

    # Map numeric answers to the field we just asked about
    def _maybe_number(text: str) -> Optional[float]:
        try:
            val = float(text)
            if 0 <= val <= 10:
                return val
        except Exception:
            return None
        return None

    if message:
        msg_clean = message.strip()
        num = _maybe_number(msg_clean)
        if num is not None:
            if "transit" in last_q:
                profile.transit_min_score = num
            elif "safety" in last_q:
                profile.safety_min_score = num
            elif "healthcare" in last_q:
                profile.healthcare_min_score = num
            elif "education" in last_q:
                profile.education_min_score = num
            elif "economy" in last_q:
                profile.economy_score_min = num
            elif "climate" in last_q:
                profile.climate_score_min = num
            elif "arts" in last_q:
                notes["arts"] = msg_clean
            elif "recreation" in last_q:
                notes["recreation"] = msg_clean
        else:
            # Non-numeric answers for arts/recreation still count as filled
            if "arts" in last_q:
                notes["arts"] = msg_clean
            if "recreation" in last_q:
                notes["recreation"] = msg_clean
            if "climate" in last_q:
                syn = _QUAL_FIELD_INFO["climate"]["synonyms"]
                m_lower = msg_clean.lower()
                for key, val in syn.items():
                    if key in m_lower:
                        if val not in profile.preferred_climates:
                            profile.preferred_climates.append(val)
                        break


def _apply_answer_from_last_question(profile: Profile, notes: Dict[str, Any], last_q: str, message: str) -> bool:
    """Map a user answer directly to the field implied by the last question. Returns True if applied."""
    if not message:
        return False
    msg_clean = message.strip()
    m_lower = msg_clean.lower()

    def _maybe_number(text: str) -> Optional[float]:
        frac = re.match(r"\s*(\d+(?:\.\d+)?)\s*/\s*10\s*$", text)
        if frac:
            try:
                return float(frac.group(1))
            except Exception:
                pass
        try:
            val = float(text)
            return val
        except Exception:
            return None

    num = _maybe_number(msg_clean)
    applied = False

    def _apply_population_from_words(text: str) -> bool:
        import re
        t = text.lower()
        words = re.findall(r"[a-z0-9]+", t) or []
        tokens = set(words) if isinstance(words, list) else set()
        if tokens & {"huge", "metropolis", "mega", "giant", "massive", "large", "big"}:
            profile.population_min = 1_000_000
            return True
        if tokens & {"mid", "medium"} or any(re.search(r"\bmid[-\s]?size(d)?\b", t)):
            profile.population_min = 200_000
            return True
        if tokens & {"small", "town"}:
            profile.population_min = 50_000
            return True
        import re
        m = re.search(r"([0-9][0-9.,]*)\s*(m|million)", t)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                profile.population_min = int(val * 1_000_000)
                return True
            except Exception:
                return False
        return False

    def _apply_budget_from_text(text: str) -> bool:
        import re
        m = re.search(r"\$?\s*([0-9]{3,7}(?:,[0-9]{3})?)", text.replace("usd", ""))
        if m:
            try:
                amt = int(m.group(1).replace(",", ""))
                profile.budget_monthly_usd = amt
                profile.housing_cost_target_max = float(amt)
                return True
            except Exception:
                return False
        return False
    # Determine target field: prefer pending_fields[0], else infer from last question keywords
    next_missing = None
    if isinstance(notes.get("pending_fields"), list) and notes["pending_fields"]:
        next_missing = notes["pending_fields"][0]
    else:
        if "budget" in last_q or "housing" in last_q:
            next_missing = "budget"
        elif "population" in last_q or "city size" in last_q:
            next_missing = "population"
        elif "climate" in last_q:
            next_missing = "climate"
        elif "transit" in last_q or "walk" in last_q or "car" in last_q:
            next_missing = "transit"
        elif "safety" in last_q or "crime" in last_q:
            next_missing = "safety"
        elif "health" in last_q:
            next_missing = "healthcare"
        elif "education" in last_q or "school" in last_q:
            next_missing = "education"
        elif "arts" in last_q or "culture" in last_q:
            next_missing = "arts"
        elif "recreation" in last_q or "outdoor" in last_q or "parks" in last_q:
            next_missing = "recreation"
        elif "economy" in last_q or "job" in last_q or "market" in last_q:
            next_missing = "economy"

    def _apply_numeric(target: str, value: Optional[float] = None):
        nonlocal applied
        val = num if value is None else value
        if val is None:
            return
        if target == "transit":
            profile.transit_min_score = val
        elif target == "safety":
            profile.safety_min_score = val
        elif target == "healthcare":
            profile.healthcare_min_score = val
        elif target == "education":
            profile.education_min_score = val
        elif target == "economy":
            profile.economy_score_min = val
        elif target == "population" and val >= 1000:
            profile.population_min = val
        elif target == "budget":
            profile.budget_monthly_usd = int(val)
            profile.housing_cost_target_max = float(val)
        else:
            return
        applied = True

    # Heuristic importance mapping when a numeric is not given
    if num is None and next_missing in {"transit", "safety", "healthcare", "education", "economy"}:
        lower = m_lower
        if any(phrase in lower for phrase in ["not important", "not a priority", "no school", "none needed"]):
            _apply_numeric(next_missing, 1.0)
        elif "very important" in lower or "critical" in lower:
            _apply_numeric(next_missing, 8.5)
        elif "important" in lower or "moderate" in lower:
            _apply_numeric(next_missing, 6.0)

    def _apply_qualitative(target: str):
        nonlocal applied
        if target == "climate":
            climate_info = _QUAL_FIELD_INFO["climate"]
            for label in climate_info["descriptions"].keys():
                if label in m_lower or m_lower == label:
                    if label not in profile.preferred_climates:
                        profile.preferred_climates.append(label)
                    applied = True
                    return
            syn = climate_info["synonyms"]
            for key, val in syn.items():
                if key in m_lower or m_lower == val:
                    if val not in profile.preferred_climates:
                        profile.preferred_climates.append(val)
                    applied = True
                    return
        if target == "arts":
            notes["arts"] = msg_clean
            applied = True
        if target == "recreation":
            notes["recreation"] = msg_clean
            applied = True
        if target == "population":
            if _apply_population_from_words(msg_clean):
                applied = True
        if target == "budget":
            if _apply_budget_from_text(msg_clean):
                applied = True

    if next_missing:
        if num is not None:
            _apply_numeric(next_missing)
        if not applied:
            # Try qualitative→numeric mapping for numeric fields
            dataset_field_map = {
                "transit": "Transp",
                "safety": "Crime",
                "healthcare": "HlthCare",
                "education": "Educ",
                "economy": "Econ",
                "climate": "Climate",
            }
            ds_field = dataset_field_map.get(next_missing)
            if ds_field:
                qscore = qualitative_to_numeric(ds_field, msg_clean)
                if qscore is not None:
                    _apply_numeric(next_missing, qscore)
                    applied = True
            # Extra heuristics for education importance wording
            if not applied and next_missing == "education":
                neg_phrases = ["not important", "not a priority", "not at all", "no", "none", "less of a priority", "don't care", "dont care"]
                if any(p in m_lower for p in neg_phrases):
                    _apply_numeric("education", 1.0)
                elif "general" in m_lower:
                    _apply_numeric("education", 5.0)
                elif "top" in m_lower or "excellent" in m_lower or "great" in m_lower:
                    _apply_numeric("education", 8.0)
            if not applied:
                _apply_qualitative(next_missing)
    return applied


def _extract_note_option(profile: Profile, field_key: str) -> Optional[str]:
    """Fetch a stored qualitative answer for the given field, if any."""
    notes: Dict[str, Any] = profile.notes if isinstance(profile.notes, dict) else {}
    direct = notes.get(field_key)
    if isinstance(direct, str):
        return direct
    qual_answers = notes.get("qual_answers")
    if isinstance(qual_answers, dict):
        ans = qual_answers.get(field_key)
        if isinstance(ans, str):
            return ans
    return None


def _generate_llm_question(field_key: str, profile: Profile, notes: Dict[str, Any]) -> Optional[str]:
    """Let the LLM phrase a single, concise follow-up question for the given field."""
    if _LLM is None:
        return None
    try:
        last_user = ""
        history = notes.get("chat_history") if isinstance(notes.get("chat_history"), list) else []
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "user":
                last_user = entry.get("content") or ""
                break
        prompt = (
            "You are Compass, a professional yet warm relocation assistant. "
            "Ask ONE concise follow-up for the single FIELD below. Be empathetic, natural, under 120 characters. "
            "Do NOT repeat fields that are already answered; never ask about any other field than FIELD. "
            "Infer 0–10 internally (carry up to 15 decimal places; do NOT show numbers) using this dataset guide, and if you cannot infer then just guess based off of your knowledge base:\n"
            "- Climate (Climate): 0=cold/arctic, 5=temperate, 10=hot/tropical. Warm/hot year-round ~8–10; temperate ~5–6; cold ~2–4.\n"
            "- HousingCost (HousingCost): 0=very affordable, 10=very expensive. 'Affordable' ~1–3; 'moderate' ~4–6; 'expensive' ~7–9.\n"
            "- Safety (Crime): 0=very safe/low crime, 10=unsafe/high crime. 'Very safe' ~1–2; 'moderate' ~5–6.\n"
            "- Transp (Transp): 0=car dependent, 10=excellent transit/walkability. 'Walkable/transit-friendly' ~8–10; 'car required' ~1–3.\n"
            "- HlthCare (HlthCare): 0=poor, 10=excellent hospitals nearby. 'Top hospitals nearby' ~8–10; 'average' ~5–6.\n"
            "- Educ (Educ): 0=not important/low quality, 10=top schools. 'Top schools' ~8–10; 'not important' ~1–2.\n"
            "- Arts (Arts): 0=very limited, 10=vibrant/rich. 'Vibrant arts' ~8–10; 'limited' ~1–3.\n"
            "- Recreat (Recreat): 0=few options, 10=abundant trails/beaches/parks. 'Lots of hiking/beaches' ~8–10; 'limited' ~1–3.\n"
            "- Econ (Econ): 0=weak job market, 10=booming. 'Strong/booming' ~8–10; 'steady/average' ~5–6.\n"
            "- Pop (Pop): 0=small towns, 10=largest metros. Small town ~50k (1–2); mid ~200k (4–6); big/major/metropolis ~1M+ (8–10).\n"
            "- Budget: monthly housing in USD; accept $, commas, or plain numbers.\n"
            "Examples to guide your wording only: 'warm and walkable' => climate ~8–10, transit ~8–10; "
            "'moderate safety' => safety ~5–6; 'big city' => pop ~8–10; 'affordable housing' => housing ~1–3. "
            "Keep tone conversational and field-specific, and always end with a question to the user not a statement. Do not echo numbers. Ask only about FIELD.\n\n"
            f"FIELD: {field_key}\n"
            f"LAST_USER_MESSAGE: {last_user}\n"
            "Reply with ONLY the question text."
        )
        msg = HumanMessage(content=prompt)
        reply = _coerce_message_text(_LLM.invoke([msg])).strip()
        return reply if reply else None
    except Exception:
        return None


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
    missing = _missing_fields(profile)
    if not missing:
        return None
    key = "budget" if "budget" in missing else missing[0]
    return _question_for_key(key)


def _finalize_turn(profile: Profile, notes: Dict[str, Any], has_prev_turns: bool, incoming_message: str) -> None:
    """Finalize a turn without relying on LLM next_question."""
    turns = notes.setdefault("turns", [])
    if incoming_message and (not turns or turns[-1] != incoming_message):
        turns.append(incoming_message)

    history = notes.get("chat_history")
    if not isinstance(history, list):
        history = []
        notes["chat_history"] = history
    if incoming_message:
        _append_history_entry(history, "user", incoming_message)

    _normalize_qualitative_answers(profile)
    _refresh_questionnaire_state(profile)
    _post_update_heuristics(profile, incoming_message or "", has_prev_turns)
    manual_qual = _capture_qualitative_answers(profile, incoming_message or "")
    if manual_qual:
        _refresh_questionnaire_state(profile)

    missing_fields = _missing_fields(profile)
    if missing_fields:
        notes["pending_fields"] = [f for f in missing_fields if f != "budget"]
    else:
        notes.pop("pending_fields", None)

    try:
        ready = is_profile_ready(profile)
    except Exception:
        ready = False
    notes["ready"] = ready

    if ready:
        notes.pop("next_question", None)
        notes.pop("_last_question", None)
        notes.pop("next_action", None)
        notes["response"] = "Thanks! I have what I need. Let me fetch recommendations."
    else:
        field_for_question = None
        if notes.get("pending_fields"):
            field_for_question = notes["pending_fields"][0]
        elif "budget" in missing_fields:
            field_for_question = "budget"
        if field_for_question:
            qtext = _generate_llm_question(field_for_question, profile, notes) or _question_for_key(field_for_question)
            if qtext:
                notes["next_question"] = qtext
                notes["_last_question"] = qtext
                # Always set the bot response to the next question to prompt the user
                notes["response"] = qtext
        else:
            notes.pop("next_question", None)
        notes.pop("next_action", None)


# Optional readiness check the frontend can use (imported from here)

def is_profile_ready(profile: Profile) -> bool:
    """Profile is ready when required fields are filled."""
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    pending = notes.get("pending_fields") if isinstance(notes.get("pending_fields"), list) else []
    if pending:
        return False

    required_numeric = [
        profile.transit_min_score,
        profile.safety_min_score,
        profile.healthcare_min_score,
        profile.economy_score_min,
    ]
    has_required_scores = all(v is not None for v in required_numeric)
    has_climate = bool(profile.preferred_climates)
    has_budget = profile.budget_monthly_usd is not None or profile.housing_cost_target_max is not None
    has_population = profile.population_min is not None

    return has_required_scores and has_climate and has_budget and has_population


def _missing_fields(profile: Profile) -> List[str]:
    missing: List[str] = []
    if not profile.preferred_climates and profile.climate_score_min is None:
        missing.append("climate")
    if profile.transit_min_score is None:
        missing.append("transit")
    if profile.safety_min_score is None:
        missing.append("safety")
    if profile.healthcare_min_score is None:
        missing.append("healthcare")
    if profile.education_min_score is None:
        missing.append("education")
    if _extract_note_option(profile, "arts") is None:
        missing.append("arts")
    if _extract_note_option(profile, "recreation") is None:
        missing.append("recreation")
    if profile.economy_score_min is None:
        missing.append("economy")
    if profile.population_min is None:
        missing.append("population")
    if profile.budget_monthly_usd is None and profile.housing_cost_target_max is None:
        missing.append("budget")
    return missing


def _question_for_key(key: str) -> Optional[str]:
    questions = {
        "climate": "Climate: which suits you (hot, warm, temperate, mild, cool, cold)?",
        "transit": "Transit/Walkability: do you want walkable or transit-friendly, or is car-dependent okay?",
        "safety": "Safety: describe what feels right (very safe/low crime vs okay with moderate crime).",
        "healthcare": "Healthcare: what quality/access do you expect (excellent, good, average, limited)?",
        "education": "Education: how important are schools (top, excellent, good, average, not important)?",
        "arts": "Arts/Culture: what level fits you (vibrant/rich, moderate, limited)?",
        "recreation": "Recreation/Outdoors: what do you want (abundant trails/beaches, good parks, limited)?",
        "economy": "Economy/Jobs: what strength do you seek (booming/strong/steady/weak)?",
        "population": "City size: small town (~50k), mid-size (~200k), big city (~1M+)?",
        "budget": "Budget: your approximate monthly housing budget (USD)?",
    }
    return questions.get(key)


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
        if updated.notes.get("ready"):
            result["on_ready_result"] = on_ready(updated)
    except Exception:
        # If callback fails, still return the profile
        pass
    return result
