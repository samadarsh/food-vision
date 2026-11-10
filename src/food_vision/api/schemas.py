"""Response models for the food-vision API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NutritionOut(BaseModel):
    serving: str = Field(description="Typical serving the values refer to")
    calories: float = Field(description="kcal, approximate")
    protein: float = Field(description="g, approximate")
    carbs: float = Field(description="g, approximate")
    fat: float = Field(description="g, approximate")


class Prediction(BaseModel):
    rank: int
    label: str = Field(description="Food-101 class name, e.g. chicken_curry")
    display_name: str = Field(description="Human-readable name, e.g. Chicken curry")
    confidence: float = Field(ge=0, le=1, description="Softmax probability")
    nutrition: NutritionOut | None = Field(description="Approximate per-serving values, if available")


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    nutrition_disclaimer: str
    nutrition_units: dict[str, str]
    model: str
    inference_ms: float = Field(description="Preprocessing + model forward time on the server")


class HealthResponse(BaseModel):
    status: str
    model: str
    num_classes: int


class ClassInfo(BaseModel):
    label: str
    display_name: str
    nutrition: NutritionOut | None
