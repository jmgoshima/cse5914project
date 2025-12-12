from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Weights(BaseModel):
    # Optional weights for scoring components (0..1)
    housing_cost: Optional[float] = None
    economy: Optional[float] = None
    safety: Optional[float] = None
    healthcare: Optional[float] = None


class HardFilters(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    min_population: Optional[float] = None
    visa_required: Optional[bool] = None


class Profile(BaseModel):
    # Canonical search fields aligned to loader
    case_num: Optional[int] = None
    name: Optional[str] = None

    # Cost preferences (either absolute or target max)
    housing_cost_target_max: Optional[float] = None
    housing_cost_target_min: Optional[float] = None

    # Preferred climates as categories (warm/temperate/cold) OR numeric scores
    preferred_climates: List[str] = Field(default_factory=list)
    climate_score_min: Optional[int] = None

    # Economy / job market
    economy_score_min: Optional[float] = None

    # Safety: we treat lower crime as better; callers can set a minimum safety metric
    safety_min_score: Optional[float] = None

    # Other dataset-aligned signals
    healthcare_min_score: Optional[float] = None
    transit_min_score: Optional[float] = None
    education_min_score: Optional[float] = None

    # Location and population
    location: Optional[Dict[str, float]] = None  # {'lat': float, 'lon': float}
    population_min: Optional[float] = None

    # Legacy / conversational fields (kept for backward compatibility)
    budget_monthly_usd: Optional[int] = None
    industry: Optional[str] = None
    wants_remote_friendly: Optional[bool] = None
    commute_preference: Optional[str] = None

    # Optional weighting and hard filters
    weights: Optional[Weights] = None
    hard_filters: Optional[HardFilters] = None

    # Free-form notes & metadata
    notes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


class ReasoningEntry(BaseModel):
    """Single reasoning entry for a recommended city."""

    model_config = ConfigDict(extra="allow")

    city: Optional[str] = None
    reason: Optional[str] = None
    score: Optional[float] = None
    id: Optional[str] = None
    rank: Optional[int] = None


class RecommendationResults(BaseModel):
    """Container for nearest-neighbor / recommendation responses."""

    model_config = ConfigDict(extra="allow")

    header: Optional[str] = None
    raw_output: Optional[str] = None
    profile_payload: Optional[Dict[str, Any]] = None
    reasoning: List[ReasoningEntry] = Field(default_factory=list)
    command: Optional[List[str]] = None
    stderr: Optional[str] = None
    error: Optional[str] = None
