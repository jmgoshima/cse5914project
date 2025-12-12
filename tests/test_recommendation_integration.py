from __future__ import annotations

from typing import Dict, List

import pytest

from backend.langchain.schemas import Profile
from backend.search.query import buildQuery, es, index_name


def _notes_from_options(**options: str) -> Dict[str, str]:
    notes = {"qual_answers": {}}
    for key, value in options.items():
        notes[key] = value
        notes["qual_answers"][key] = value
    return notes


def _complete_profile(climate: str, budget: int, **options: str) -> Profile:
    profile = Profile(preferred_climates=[climate], budget_monthly_usd=budget)
    profile.notes = _notes_from_options(**options)
    return profile


def _search_cities(profile: Profile, top_n: int) -> List[str]:
    query = buildQuery(profile, topN=top_n)
    response = es.search(index=index_name, **query)
    return [
        hit["_source"].get("city") or hit["_source"].get("name") or hit.get("_id")
        for hit in response["hits"]["hits"]
    ]


@pytest.fixture(scope="module")
def _ensure_elasticsearch():
    if not es.ping():
        pytest.skip("Elasticsearch not available; skip integration tests.")


@pytest.mark.usefixtures("_ensure_elasticsearch")
def test_distinct_profiles_yield_different_cities():
    high_service = _complete_profile(
        climate="temperate",
        budget=6000,
        transit="walkable",
        safety="safe",
        healthcare="excellent",
        education="top",
        arts="vibrant",
        recreation="abundant",
        economy="booming",
        population="major",
    )
    low_service = _complete_profile(
        climate="tropical",
        budget=2000,
        transit="car dependent",
        safety="dangerous",
        healthcare="limited",
        education="poor",
        arts="few",
        recreation="limited",
        economy="weak",
        population="small",
    )

    cities_high = _search_cities(high_service, top_n=5)
    cities_low = _search_cities(low_service, top_n=5)

    print("High-service profile cities:", cities_high)
    print("Low-service profile cities:", cities_low)

    assert cities_high, "Expected results for high-service profile."
    assert cities_low, "Expected results for low-service profile."
    assert set(cities_high) != set(cities_low), (
        f"Expected different city recommendations but both profiles returned "
        f"{set(cities_high) & set(cities_low)} in common."
    )
