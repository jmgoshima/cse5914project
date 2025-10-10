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
from typing import Optional, List, Dict, Any, Callable, Tuple

# Load .env variables if present for local dev (e.g., GOOGLE_API_KEY)
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .schemas import Profile
from backend.search.qualitative import get_qualitative_options, qualitative_to_numeric

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
            "average": "Schools on par with national averages.",
            "excellent": "Top-performing schools with strong outcomes.",
            "good": "Above-average schools with solid reputations.",
            "poor": "Under-resourced schools or lower performance metrics.",
            "top": "Elite programs and consistently exceptional results.",
        },
        "synonyms": {
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
        "score_attr": None,
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
    _info["options"] = list(get_qualitative_options(_info["dataset_field"]))
    _info["options_lower"] = [opt.lower() for opt in _info["options"]]
    _info["synonyms"] = {str(k).lower(): v for k, v in _info.get("synonyms", {}).items()}

_FIELD_KEYWORD_ENTRIES: List[tuple[str, str]] = []
for _field_key, _info in _QUAL_FIELD_INFO.items():
    for _kw in _info.get("keywords", []):
        if isinstance(_kw, str) and _kw:
            _FIELD_KEYWORD_ENTRIES.append((_kw.lower(), _field_key))
_FIELD_KEYWORD_ENTRIES.sort(key=lambda item: len(item[0]), reverse=True)

_CLIMATE_OPTIONS = list(_QUAL_FIELD_INFO["climate"]["options"])


def _build_qualitative_guidance_text() -> str:
    lines: List[str] = []
    for field_key in QUAL_FIELD_ORDER:
        info = _QUAL_FIELD_INFO.get(field_key)
        if not info:
            continue
        options = info.get("options") or []
        if options:
            lines.append(f"- {info['label']}: {', '.join(options)}")
    return "\n".join(lines)


QUALITATIVE_GUIDANCE_TEXT = _build_qualitative_guidance_text()


def _set_qualitative_options_note(profile: Profile) -> None:
    """Expose acceptable qualitative descriptors to calling layers via notes."""
    if not isinstance(profile.notes, dict):
        return
    options_note = profile.notes.setdefault("qualitative_options", {})
    for field_key in QUAL_FIELD_ORDER:
        info = _QUAL_FIELD_INFO.get(field_key)
        if not info:
            continue
        options_note[field_key] = list(info.get("options", []))


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

    for raw_key, raw_value in re.findall(r"([a-z &]+?)\s*[:\-]\s*([a-z0-9\s&-]+)", text):
        field_key = _match_field_keyword(raw_key.strip())
        if not field_key:
            continue
        option = _match_option(field_key, raw_value.strip())
        if option:
            found[field_key] = option

    pending_fields: List[str] = []
    if isinstance(profile.notes.get("pending_fields"), list):
        pending_fields = [f for f in profile.notes["pending_fields"] if f in _QUAL_FIELD_INFO]
    for field_key in pending_fields:
        if field_key in found:
            continue
        option = _match_option(field_key, text)
        if option:
            found[field_key] = option

    if not found:
        return False

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
    options = info.get("options") or []
    if not options:
        return None
    options_list = ", ".join(options)
    if field_key == "climate":
        return f"Which climate do you prefer? Choose from: {options_list}."
    label = info.get("label", field_key)
    return f"What are your preferences for {label}? Choose from: {options_list}."


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
    if question:
        profile.notes["next_question"] = question
        profile.notes["pending_fields"] = fields
        profile.notes["_last_question"] = question
    else:
        profile.notes.pop("next_question", None)
        profile.notes.pop("pending_fields", None)


def _ensure_initial_question(profile: Profile) -> None:
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    if profile.notes.get("next_question"):
        return
    question, fields = _determine_next_question(profile)
    if question:
        profile.notes["next_question"] = question
        profile.notes["pending_fields"] = fields
        profile.notes["_last_question"] = question
        profile.notes["ready"] = False
        return
    fallback = _default_next_question(profile)
    if fallback:
        profile.notes["next_question"] = fallback
        profile.notes["_last_question"] = fallback
        profile.notes.pop("pending_fields", None)
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

    # track message for traceability
    notes.setdefault("turns", []).append(incoming_message)

    if not clarification_handled:
        if notes.get("pending_fields"):
            pass
        elif profile.budget_monthly_usd is None and not notes.get("next_question"):
            notes.pop("pending_fields", None)
            notes["next_question"] = "What is your approximate monthly housing budget (in USD)?"
            notes["_last_question"] = notes["next_question"]
        else:
            _advance_questionnaire(profile)

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
        elif not notes.get("next_question"):
            _advance_questionnaire(profile)
            if not notes.get("next_question"):
                nq = _default_next_question(profile)
                if nq:
                    notes.pop("pending_fields", None)
                    notes["next_question"] = nq
                    notes["_last_question"] = nq

    _set_qualitative_options_note(profile)
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
                f"""
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
- When gathering qualitative ratings (climate, transit, safety, healthcare, education, arts, recreation, economy, population), list the acceptable descriptors, require the user to choose from them, and if they answer with something else, ask again and repeat the acceptable options.
- Acceptable qualitative descriptors:\n{QUALITATIVE_GUIDANCE_TEXT}
- Location rules: set hard_filters.country only when the user names a country (e.g., "United States"). Set hard_filters.state only when the user explicitly mentions a state by name; otherwise leave it null. Never guess states.
 - Default scope: unless the user names a different country, assume the search is within the United States and set hard_filters.country = "United States".
{{format_instructions}}
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
    incoming_message = message or ""
    if not isinstance(profile.notes, dict):
        profile.notes = {}
    notes = profile.notes
    has_prev_turns = bool(notes.get("turns"))

    if not (_LLM and _PARSER and _PROMPT):
        return _heuristic_update(profile, incoming_message)

    if not has_prev_turns and not incoming_message.strip():
        _ensure_initial_question(profile)
        notes.setdefault("turns", [])
        notes["ready"] = False
        return profile

    if "next_question" in notes:
        notes["_last_question"] = notes.get("next_question")
        notes.pop("next_question", None)

    current_profile_json = profile.model_dump_json()
    format_instructions = _PARSER.get_format_instructions()

    # Run chain: prompt → llm → parse
    try:
        chain = _PROMPT | _LLM | _PARSER
        new_profile: Profile = chain.invoke({
            "current_profile": current_profile_json,
            "user_message": incoming_message,
            "format_instructions": format_instructions,
        })
    except Exception:
        # If anything goes wrong, gracefully fall back
        return _heuristic_update(profile, incoming_message)

    # Merge and return
    merged = _merge_profiles(profile, new_profile)
    # Ensure notes is a dict
    if not isinstance(merged.notes, dict):
        merged.notes = {}
    # Apply lightweight heuristics to fill obvious fields the model may omit
    _post_update_heuristics(merged, incoming_message, has_prev_turns)
    _set_qualitative_options_note(merged)
    turns = merged.notes.setdefault("turns", [])
    if incoming_message:
        if not turns or turns[-1] != incoming_message:
            turns.append(incoming_message)
    elif not turns:
        turns.append(incoming_message)
    clarification_handled = _maybe_handle_clarification(merged, incoming_message)
    handled_binary = False
    if not clarification_handled:
        handled_binary = _handle_binary_questions(merged, incoming_message)
        if handled_binary and "next_question" in merged.notes and not merged.notes.get("pending_fields"):
            merged.notes.pop("next_question", None)
    qualitative_guidance = [] if clarification_handled else _ensure_qualitative_alignment(merged)
    if qualitative_guidance:
        merged.notes["next_question"] = " ".join(qualitative_guidance)
        merged.notes["_last_question"] = merged.notes["next_question"]
        merged.notes.pop("pending_fields", None)
        merged.notes["ready"] = False
    elif clarification_handled:
        merged.notes["ready"] = False
    else:
        if not merged.notes.get("next_question"):
            _advance_questionnaire(merged)
        try:
            ready = is_profile_ready(merged)
        except Exception:
            ready = False
        merged.notes["ready"] = ready
        if ready:
            merged.notes["next_action"] = {
                "type": "search_places",
                "params": {
                    "topK": 20,
                    "take": 5
                }
            }
            merged.notes.pop("next_question", None)
            merged.notes.pop("pending_fields", None)
        else:
            merged.notes.pop("next_action", None)
            if not merged.notes.get("next_question"):
                question, fields = _determine_next_question(merged)
                if question:
                    merged.notes["next_question"] = question
                    merged.notes["pending_fields"] = fields
                    merged.notes["_last_question"] = question
                else:
                    nq = _default_next_question(merged)
                    if nq:
                        merged.notes.pop("pending_fields", None)
                        merged.notes["next_question"] = nq
                        merged.notes["_last_question"] = nq
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
    question, _fields = _determine_next_question(profile)
    if question:
        return question
    if not (profile.hard_filters and (profile.hard_filters.country or profile.hard_filters.state)):
        return "Which country or state do you want to focus on?"
    if not profile.commute_preference:
        return "Do you prefer walkable areas, good transit, or driving?"
    if profile.wants_remote_friendly is None:
        return REMOTE_PREFERENCE_PROMPT
    notes = profile.notes if isinstance(profile.notes, dict) else {}
    if not notes.get("declined_additional_must_haves"):
        return MUST_HAVE_PROMPT
    return None


# Optional readiness check the frontend can use (imported from here)

def is_profile_ready(profile: Profile) -> bool:
    """Define your own readiness criteria for moving on to search/recommend.
    Example heuristic: budget, at least one climate preference, and either
    industry or remote preference.
    """
    has_budget = profile.budget_monthly_usd is not None
    has_climate = bool(profile.preferred_climates)
    has_work = bool(profile.industry) or (profile.wants_remote_friendly is not None)
    return has_budget and has_climate and has_work


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
