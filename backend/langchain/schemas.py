from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Weights(BaseModel):
    # Optional weights for scoring components (0..1)
    housing_cost: Optional[float] = None
    economy: Optional[float] = None
    safety: Optional[float] = None
    healthcare: Optional[float] = None


class HardFilters(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    min_population: Optional[int] = None
    visa_required: Optional[bool] = None


class Profile(BaseModel):
    Climate: Optional[float] = Field(default=None, ge=0, le=10)
    HousingCost: Optional[float] = Field(default=None, ge=0, le=10)
    HlthCare: Optional[float] = Field(default=None, ge=0, le=10)
    Crime: Optional[float] = Field(default=None, ge=0, le=10)
    Transp: Optional[float] = Field(default=None, ge=0, le=10)
    Educ: Optional[float] = Field(default=None, ge=0, le=10)
    Arts: Optional[float] = Field(default=None, ge=0, le=10)
    Recreat: Optional[float] = Field(default=None, ge=0, le=10)
    Econ: Optional[float] = Field(default=None, ge=0, le=10)
    Pop: Optional[float] = Field(default=None, ge=0, le=10)

    # Metadata the agent uses for clarifying questions, readiness, etc.
    notes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "ignore"
