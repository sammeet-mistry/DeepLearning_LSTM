from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    location: str = Field(default="New_Delhi", min_length=1)
    history: Optional[list[float]] = Field(default=None)
    forecast_date: Optional[str] = None

    @field_validator("history")
    @classmethod
    def validate_history(cls, value):
        if value is not None and len(value) == 0:
            raise ValueError("history must contain at least one temperature value when provided.")
        return value


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_path: str


class PredictionResponse(BaseModel):
    location: str
    forecast_date: str
    predicted_temperature_c: float
    model_id: str
    generated_at_utc: str
    input_sequence_length: int
    warnings: list[str] = []


class MetricSummary(BaseModel):
    model: dict
    baseline: dict
    lookback: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    location: str
    evaluated_at_utc: str


class ModelInfo(BaseModel):
    location: str
    lookback: int
    model_path: str
    train_start: str
    train_end: str
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    training_timestamp_utc: str | None = None
    architecture: list[str] = []
