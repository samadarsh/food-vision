"""Nutrition lookup: coverage, sanity of values, and the disclaimer."""

from __future__ import annotations

import json

import pytest

from food_vision.config import load_config
from food_vision.nutrition import default_nutrition, load_nutrition


def test_every_configured_class_has_an_entry() -> None:
    assert default_nutrition().missing(load_config().data.class_names) == []


def test_lookup_known_and_unknown() -> None:
    table = default_nutrition()
    samosa = table.lookup("samosa")
    assert samosa is not None and samosa.calories > 0 and samosa.serving
    assert table.lookup("not_a_dish") is None


@pytest.mark.parametrize("name", sorted(default_nutrition().foods))
def test_calories_consistent_with_macros(name: str) -> None:
    """Atwater factors: kcal ~= 4*protein + 4*carbs + 9*fat (within 10%)."""
    n = default_nutrition().foods[name]
    estimate = 4 * n.protein + 4 * n.carbs + 9 * n.fat
    assert abs(estimate - n.calories) / n.calories < 0.10, (name, n.calories, estimate)


def test_disclaimer_says_approximate() -> None:
    table = default_nutrition()
    assert "approximate" in table.disclaimer.lower()
    assert table.units == {"calories": "kcal", "protein": "g", "carbs": "g", "fat": "g"}


def test_load_custom_file_and_reject_negative(tmp_path) -> None:
    entry = {"serving": "1 bowl", "calories": 100, "protein": 5, "carbs": 10, "fat": 4}
    good = {"disclaimer": "approximate", "units": {}, "foods": {"soup": entry}}
    (tmp_path / "ok.json").write_text(json.dumps(good))
    assert load_nutrition(tmp_path / "ok.json").lookup("soup").fat == 4

    bad = {**good, "foods": {"soup": {**entry, "fat": -1}}}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="negative"):
        load_nutrition(tmp_path / "bad.json")
