from __future__ import annotations

from typing import Dict

from backend.langchain.cli_demo import _profile_to_query_payload
from backend.langchain.schemas import Profile
from backend.search.qualitative import qualitative_to_numeric
from backend.search.query import buildQuery


def _notes_from_options(**options: str) -> Dict[str, str]:
    notes = {"qual_answers": {}}
    for key, value in options.items():
        notes[key] = value
        notes["qual_answers"][key] = value
    return notes


def _profile_high_service() -> Profile:
    profile = Profile(
        preferred_climates=["temperate"],
        budget_monthly_usd=6000,
    )
    profile.notes = _notes_from_options(
        transit="walkable",
        safety="safe",
        healthcare="excellent",
        education="top",
        arts="vibrant",
        recreation="abundant",
        economy="booming",
        population="major",
    )
    return profile


def _profile_low_service() -> Profile:
    profile = Profile(
        preferred_climates=["tropical"],
        budget_monthly_usd=2000,
    )
    profile.notes = _notes_from_options(
        transit="car dependent",
        safety="dangerous",
        healthcare="limited",
        education="poor",
        arts="few",
        recreation="limited",
        economy="weak",
        population="small",
    )
    return profile


def test_profile_to_query_payload_maps_qualitative_answers():
    profile = _profile_low_service()
    payload = _profile_to_query_payload(profile)

    assert payload["HlthCare"] == qualitative_to_numeric("hlthcare", "limited")
    assert payload["Crime"] == qualitative_to_numeric("crime", "dangerous")
    assert payload["Transp"] == qualitative_to_numeric("transp", "car dependent")
    assert payload["Educ"] == qualitative_to_numeric("educ", "poor")
    assert payload["Econ"] == qualitative_to_numeric("econ", "weak")
    assert payload["Pop"] == qualitative_to_numeric("pop", "small")
    assert payload["HousingCost"] == 2000


def test_build_query_returns_distinct_vectors_for_distinct_profiles():
    profile_a = _profile_low_service()
    profile_b = _profile_high_service()

    query_a = buildQuery(profile_a, topN=5)
    query_b = buildQuery(profile_b, topN=5)

    vector_a = query_a["knn"]["query_vector"]
    vector_b = query_b["knn"]["query_vector"]

    assert vector_a != vector_b
    assert vector_a[0] == qualitative_to_numeric("climate", "tropical")
    assert vector_b[0] == qualitative_to_numeric("climate", "temperate")
    assert vector_a[6] == qualitative_to_numeric("arts", "few")
    assert vector_b[6] == qualitative_to_numeric("arts", "vibrant")
    assert vector_a[9] == qualitative_to_numeric("pop", "small")
    assert vector_b[9] == qualitative_to_numeric("pop", "major")
    assert vector_a[1] < vector_b[1]
