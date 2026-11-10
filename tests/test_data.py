"""Subset sampling, split integrity and reproducibility."""

from __future__ import annotations

import pytest

from food_vision.data import SPLITS, build_datasets, get_splits


def test_split_sizes(make_config) -> None:
    splits = get_splits(make_config())
    assert [len(splits[s]) for s in SPLITS] == [3 * 10, 3 * 4, 3 * 5]


def test_splits_do_not_overlap(make_config) -> None:
    sets = {s: {x.rel_path for x in v} for s, v in get_splits(make_config()).items()}
    assert not sets["train"] & sets["val"]
    assert not (sets["train"] | sets["val"]) & sets["test"]


def test_test_split_comes_from_official_test_list(make_config) -> None:
    # In the fake dataset, official test ids are >= 100.
    assert all(int(x.rel_path.split("/")[1].split(".")[0]) >= 100 for x in get_splits(make_config())["test"])


def test_labels_follow_sorted_class_order(make_config) -> None:
    cfg = make_config(classes=["samosa", "pizza", "chicken_curry"])
    assert cfg.data.class_names == ["chicken_curry", "pizza", "samosa"]
    for x in get_splits(cfg)["train"]:
        assert cfg.data.class_names[x.label] == x.class_name


def test_rebuild_is_deterministic(make_config) -> None:
    cfg = make_config()
    assert get_splits(cfg) == get_splits(cfg, rebuild=True)


def test_exclude_list_removes_images(make_config) -> None:
    before = get_splits(make_config())
    victim = before["train"][0].rel_path.removesuffix(".jpg")
    after = get_splits(make_config(exclude=[victim]))
    all_after = {x.rel_path for s in SPLITS for x in after[s]}
    assert f"{victim}.jpg" not in all_after
    assert [len(after[s]) for s in SPLITS] == [len(before[s]) for s in SPLITS]  # refilled from the pool


def test_config_change_changes_fingerprint(make_config) -> None:
    a, b = make_config(), make_config(train_per_class=8)
    assert a.data.split_fingerprint(42) != b.data.split_fingerprint(42)
    assert len(get_splits(b)["train"]) == 3 * 8


def test_unknown_class_is_rejected(make_config) -> None:
    with pytest.raises(ValueError, match="Not Food-101 classes"):
        get_splits(make_config(classes=["pizza", "not_a_dish"]))


def test_requesting_too_many_images_fails(make_config) -> None:
    with pytest.raises(ValueError, match="only"):
        get_splits(make_config(train_per_class=50))


def test_dataset_returns_tensor_and_label(make_config) -> None:
    ds = build_datasets(make_config())
    x, y = ds["val"][0]
    assert tuple(x.shape) == (3, 224, 224) and isinstance(y, int)
    assert ds["val"].path(0).exists()
