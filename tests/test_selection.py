import pytest

from training_platform.selection import select_best


def test_select_best_maximize_picks_highest_metric():
    results = [({"n_estimators": 10}, 0.7), ({"n_estimators": 50}, 0.9), ({"n_estimators": 100}, 0.85)]
    assert select_best(results, "maximize") == {"n_estimators": 50}


def test_select_best_minimize_picks_lowest_metric():
    results = [({"alpha": 0.1}, 12.0), ({"alpha": 1.0}, 8.5), ({"alpha": 10.0}, 15.0)]
    assert select_best(results, "minimize") == {"alpha": 1.0}


def test_select_best_rejects_empty_results():
    with pytest.raises(ValueError, match="must not be empty"):
        select_best([], "maximize")


def test_select_best_rejects_unknown_direction():
    with pytest.raises(ValueError, match="unknown metric_direction"):
        select_best([({"a": 1}, 0.5)], "sideways")
