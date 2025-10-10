from __future__ import annotations

import re
from statistics import mean
from typing import Dict, Iterable, List, Optional

QUALITATIVE_FIELD_KEYWORDS: Dict[str, Dict[str, float]] = {
    "climate": {
        "tropical": 9.0,
        "hot": 8.0,
        "warm": 7.0,
        "temperate": 6.0,
        "mild": 6.0,
        "continental": 5.0,
        "moderate": 5.0,
        "cool": 4.0,
        "cold": 3.0,
        "arctic": 2.0,
    },
    "housingcost": {
        "very low": 1.5,
        "low": 3.0,
        "affordable": 3.5,
        "moderate": 5.0,
        "average": 5.0,
        "medium": 5.0,
        "high": 7.0,
        "very high": 8.5,
        "expensive": 8.0,
        "cheap": 2.5,
    },
    "crime": {
        "very low": 2.0,
        "low": 3.5,
        "safe": 3.5,
        "moderate": 5.0,
        "average": 5.0,
        "medium": 5.5,
        "high": 7.5,
        "very high": 9.0,
        "dangerous": 8.5,
    },
    "transp": {
        "limited": 3.0,
        "poor": 3.0,
        "adequate": 5.0,
        "good": 7.0,
        "excellent": 9.0,
        "car dependent": 2.5,
        "transit friendly": 7.5,
        "walkable": 7.5,
    },
    "hlthcare": {
        "poor": 2.5,
        "limited": 3.5,
        "average": 5.0,
        "adequate": 5.5,
        "good": 7.0,
        "excellent": 8.5,
    },
    "educ": {
        "poor": 3.0,
        "average": 5.0,
        "good": 7.0,
        "excellent": 8.5,
        "top": 9.0,
    },
    "arts": {
        "limited": 3.0,
        "few": 3.0,
        "moderate": 5.0,
        "average": 5.0,
        "vibrant": 7.5,
        "rich": 7.5,
        "excellent": 8.5,
    },
    "recreat": {
        "limited": 3.0,
        "few": 3.5,
        "moderate": 5.0,
        "average": 5.0,
        "good": 7.0,
        "abundant": 8.5,
    },
    "econ": {
        "weak": 3.0,
        "average": 5.0,
        "steady": 6.0,
        "strong": 7.5,
        "robust": 8.5,
        "booming": 9.0,
    },
    "pop": {
        "small": 2.5,
        "mid": 5.0,
        "medium": 5.0,
        "large": 7.5,
        "huge": 8.5,
        "major": 8.5,
    },
}


def _extract_numeric_from_text(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def qualitative_to_numeric(field_name: str, raw: str) -> Optional[float]:
    """Translate qualitative text into a numeric score aligned with loader ranges."""
    if not raw:
        return None

    text = raw.lower()
    keywords = QUALITATIVE_FIELD_KEYWORDS.get(field_name.lower())

    if keywords:
        scores = [score for keyword, score in keywords.items() if keyword in text]
        if scores:
            return float(mean(scores))

    numeric = _extract_numeric_from_text(text)
    if numeric is not None:
        return numeric

    return None


def get_qualitative_options(field_name: str) -> List[str]:
    """Return the acceptable qualitative keywords for a field."""
    keywords = QUALITATIVE_FIELD_KEYWORDS.get(field_name.lower(), {})
    return sorted(keywords.keys())


def summarize_qualitative_options(fields: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    """Return a mapping of field -> sorted acceptable options."""
    if fields is None:
        fields = QUALITATIVE_FIELD_KEYWORDS.keys()
    summary: Dict[str, List[str]] = {}
    for field in fields:
        summary[field] = get_qualitative_options(field)
    return summary


__all__ = [
    "QUALITATIVE_FIELD_KEYWORDS",
    "qualitative_to_numeric",
    "get_qualitative_options",
    "summarize_qualitative_options",
]
