"""Approximate nutrition lookup per food class (reference values, not measurements)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Nutrition:
    """Approximate values for one typical serving."""

    serving: str
    calories: float  # kcal
    protein: float   # g
    carbs: float     # g
    fat: float       # g

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NutritionTable:
    """All entries plus the disclaimer that must accompany them."""

    foods: dict[str, Nutrition]
    disclaimer: str
    units: dict[str, str]

    def lookup(self, class_name: str) -> Nutrition | None:
        """Entry for a class, or None if the class has no entry."""
        return self.foods.get(class_name)

    def missing(self, class_names: list[str]) -> list[str]:
        """Classes without an entry (used to validate config/model coverage)."""
        return [c for c in class_names if c not in self.foods]


def load_nutrition(path: str | Path | None = None) -> NutritionTable:
    """Load nutrition data from `path`, or the copy bundled with the package."""
    if path is None:
        text = resources.files("food_vision").joinpath("resources/nutrition.json").read_text()
    else:
        text = Path(path).read_text()
    raw = json.loads(text)
    foods = {name: Nutrition(**entry) for name, entry in raw["foods"].items()}
    for name, n in foods.items():
        if min(n.calories, n.protein, n.carbs, n.fat) < 0:
            raise ValueError(f"nutrition.json: negative value for {name}")
    return NutritionTable(foods=foods, disclaimer=raw["disclaimer"], units=raw["units"])


@lru_cache(maxsize=1)
def default_nutrition() -> NutritionTable:
    """Cached bundled table (what the API uses)."""
    return load_nutrition()
