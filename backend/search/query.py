from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from elasticsearch import Elasticsearch
from dotenv import load_dotenv

try:  # pragma: no cover - allow running as module or script
    from .qualitative import qualitative_to_numeric, get_qualitative_options
except ImportError:  # pragma: no cover
    from qualitative import qualitative_to_numeric, get_qualitative_options  # type: ignore

try:  # pragma: no cover
    from backend.langchain.schemas import Profile
except Exception:  # pragma: no cover
    Profile = None  # type: ignore


env_path_relative = Path(__file__).parent.parent.parent / "elastic-start-local" / ".env"
load_dotenv(dotenv_path=env_path_relative)

# Build Elasticsearch client without auth if no local password is provided.
es_password = os.getenv("ES_LOCAL_PASSWORD")
es_client_kwargs: Dict[str, Any] = {"verify_certs": False}
if es_password:
    es_client_kwargs["basic_auth"] = ("elastic", es_password)
else:
    print(
        "Warning: ES_LOCAL_PASSWORD not set; attempting unauthenticated connection to Elasticsearch."
    )

es = Elasticsearch("http://localhost:9200", **es_client_kwargs)
index_name = "cities"

DATA_FIELDS: Tuple[str, ...] = (
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
)


def _load_min_max() -> Dict[str, Tuple[float, float]]:
    """Compute min/max for continuous fields once so we can scale inputs."""
    dataset_path = Path(__file__).parent / "data" / "places.csv"
    target_fields = {"HousingCost", "Pop"}
    min_max: Dict[str, Tuple[float, float]] = {field: (float("inf"), float("-inf")) for field in target_fields}
    if not dataset_path.exists():
        return min_max
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in target_fields:
                raw = row.get(field)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                current_min, current_max = min_max[field]
                if value < current_min:
                    current_min = value
                if value > current_max:
                    current_max = value
                min_max[field] = (current_min, current_max)
    return min_max


FIELD_MIN_MAX: Dict[str, Tuple[float, float]] = _load_min_max()

EXPECTED_VECTOR_KEYS: Sequence[str] = DATA_FIELDS


def _coerce_payload_record(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list):
        if len(payload) == 1:
            payload = payload[0]
        else:
            return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_city_label(payload: Any) -> Optional[str]:
    record = _coerce_payload_record(payload)
    if not record:
        return None
    value = record.get("City") or record.get("city")
    if isinstance(value, str):
        value = value.strip()
    return value or None


def _normalize_query_values(values: Sequence[Union[int, float]]) -> List[float]:
    return [float(v) for v in values]


def _dict_to_vector(data: Dict[str, Any]) -> List[float]:
    vector: List[float] = []
    for key in EXPECTED_VECTOR_KEYS:
        value = data.get(key)
        if value is None:
            vector.append(0.0)
            continue
        numeric_value: Any = value
        if isinstance(value, str):
            translated = qualitative_to_numeric(key, value)
            if translated is not None:
                numeric_value = translated
        try:
            vector.append(float(numeric_value))
        except (TypeError, ValueError) as exc:
            message = f"Value for '{key}' must be numeric or null, got {value!r}."
            if isinstance(value, str):
                options = get_qualitative_options(key)
                if options:
                    message += f" Try one of: {', '.join(options)}."
            raise ValueError(message) from exc
    return vector


def get_query_vector_from_payload(payload: str) -> List[float]:
    data = json.loads(payload)

    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                "JSON payload must contain exactly one record when provided as a list."
            )
        data = data[0]

    if isinstance(data, dict):
        return _dict_to_vector(data)

    if isinstance(data, (list, tuple)):
        return _normalize_query_values(data)

    raise TypeError(
        "Query payload must be a JSON object, single-element list, or list of numbers."
    )


def get_query_vector_from_file(json_path: Path) -> List[float]:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    return get_query_vector_from_payload(raw)


def resolve_query_vector(argv: Sequence[str]) -> List[float]:
    if len(argv) >= 2:
        payload = argv[1]
        return get_query_vector_from_payload(payload)

    default_path = Path(__file__).parent / "data" / "test_city.json"
    if not default_path.exists():
        raise FileNotFoundError(
            "No query payload provided and default vector file is missing."
        )
    print(f"No query payload supplied; falling back to {default_path}.")
    return get_query_vector_from_file(default_path)


def main(argv: Sequence[str]) -> None:
    query_vector = resolve_query_vector(argv)

    city_label: Optional[str] = None
    payload_data: Any = None
    if len(argv) >= 2:
        try:
            payload_data = json.loads(argv[1])
        except json.JSONDecodeError:
            payload_data = None
    else:
        default_path = Path(__file__).parent / "data" / "test_city.json"
        if default_path.exists():
            try:
                payload_data = json.loads(default_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload_data = None
    city_label = _extract_city_label(payload_data) if payload_data is not None else None

    # Run kNN to find similar cities
    response = es.search(
        index=index_name,
        knn={
            "field": "review_vector",
            "query_vector": query_vector,
            "k": 5,  # number of nearest neighbors
            "num_candidates": 50,  # search depth
        },
    )

    # Print results (excluding the city itself when we know it)
    display_label = city_label or "input vector"
    print(f"Nearest neighbors to {display_label}:")
    results = response.get("hits", {}).get("hits", [])
    rank = 1
    for hit in results:
        source = hit.get("_source", {}) if isinstance(hit, dict) else {}
        candidate_city = source.get("city") or source.get("City") or "Unknown City"
        score = hit.get("_score")
        if city_label and isinstance(candidate_city, str) and candidate_city.lower() == city_label.lower():
            continue
        if isinstance(score, (int, float)):
            print(f"  {rank}. {candidate_city} (score={float(score):.4f})")
        else:
            print(f"  {rank}. {candidate_city}")
        rank += 1


if __name__ == "__main__":
    main(sys.argv)

# ---------------------------------------------------------------------------
# Profile-driven query helpers
# ---------------------------------------------------------------------------


def _normalize_score(value: Optional[Union[int, float]]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= numeric <= 1.0:
        numeric *= 10.0
    return numeric


def _scale_continuous(field: str, value: Optional[Union[int, float]]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    bounds = FIELD_MIN_MAX.get(field)
    if not bounds:
        return None
    field_min, field_max = bounds
    if not (field_min < field_max):
        return None
    scaled = (numeric - field_min) / (field_max - field_min) * 10.0
    return max(0.0, min(10.0, scaled))


def _extract_note_option(profile: "Profile", field_key: str) -> Optional[str]:
    notes: Dict[str, Any] = profile.notes if isinstance(profile.notes, dict) else {}
    direct = notes.get(field_key)
    if isinstance(direct, str):
        return direct
    qual_answers = notes.get("qual_answers")
    if isinstance(qual_answers, dict):
        answer = qual_answers.get(field_key)
        if isinstance(answer, str):
            return answer
    return None


def _score_from_option(dataset_field: str, option: Optional[str]) -> Optional[float]:
    if not option:
        return None
    score = qualitative_to_numeric(dataset_field, option)
    if score is None:
        return None
    return float(score)


def _profile_to_vector(profile: "Profile") -> List[float]:
    if Profile is None:
        raise RuntimeError("Profile model not available; ensure backend.langchain is installed.")

    vector: List[float] = []

    # Climate
    climate_scores = [
        qualitative_to_numeric("Climate", value)
        for value in profile.preferred_climates
        if isinstance(value, str) and qualitative_to_numeric("Climate", value) is not None
    ]
    climate_score = None
    if climate_scores:
        climate_score = sum(climate_scores) / len(climate_scores)
    else:
        climate_score = _score_from_option("Climate", _extract_note_option(profile, "climate"))
    vector.append(climate_score if climate_score is not None else 5.0)

    # Housing cost (scaled budget)
    housing_value = (
        profile.housing_cost_target_max
        or profile.housing_cost_target_min
        or profile.budget_monthly_usd
    )
    housing_score = _scale_continuous("HousingCost", housing_value)
    vector.append(housing_score if housing_score is not None else 5.0)

    # Healthcare
    healthcare_score = _normalize_score(profile.healthcare_min_score)
    if healthcare_score is None:
        healthcare_score = _score_from_option("HlthCare", _extract_note_option(profile, "healthcare"))
    vector.append(healthcare_score if healthcare_score is not None else 5.0)

    # Crime (lower is safer)
    crime_score = _normalize_score(profile.safety_min_score)
    if crime_score is None:
        crime_score = _score_from_option("Crime", _extract_note_option(profile, "safety"))
    vector.append(crime_score if crime_score is not None else 5.0)

    # Transit
    transit_score = _normalize_score(profile.transit_min_score)
    if transit_score is None:
        transit_score = _score_from_option("Transp", _extract_note_option(profile, "transit"))
    vector.append(transit_score if transit_score is not None else 5.0)

    # Education
    education_score = _normalize_score(profile.education_min_score)
    if education_score is None:
        education_score = _score_from_option("Educ", _extract_note_option(profile, "education"))
    vector.append(education_score if education_score is not None else 5.0)

    # Arts
    arts_score = _score_from_option("Arts", _extract_note_option(profile, "arts"))
    vector.append(arts_score if arts_score is not None else 5.0)

    # Recreation
    recreation_score = _score_from_option("Recreat", _extract_note_option(profile, "recreation"))
    vector.append(recreation_score if recreation_score is not None else 5.0)

    # Economy
    economy_score = _normalize_score(profile.economy_score_min)
    if economy_score is None:
        economy_score = _score_from_option("Econ", _extract_note_option(profile, "economy"))
    vector.append(economy_score if economy_score is not None else 5.0)

    # Population
    population_score: Optional[float] = None
    if profile.population_min is not None:
        population_score = _scale_continuous("Pop", profile.population_min)
    if population_score is None:
        population_score = _score_from_option("Pop", _extract_note_option(profile, "population"))
    vector.append(population_score if population_score is not None else 5.0)

    return [float(value) for value in vector]


def buildQuery(profile: "Profile", topN: int) -> Dict[str, Any]:
    """Build an Elasticsearch kNN query tailored to the user's profile."""
    if topN <= 0:
        topN = 5
    vector = _profile_to_vector(profile)
    k = int(topN)
    return {
        "size": k,
        "knn": {
            "field": "review_vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(50, k * 4),
        },
    }
