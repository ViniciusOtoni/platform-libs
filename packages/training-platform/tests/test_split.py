from datetime import date

import pytest

from training_platform.split import assign_split, compute_split_dates


def _dates(n):
    return [date(2026, 1, 1 + i) for i in range(n)]


def test_compute_split_dates_60_20_20_over_ten_dates():
    train_end, val_end = compute_split_dates(_dates(10), train_pct=0.6, val_pct=0.2, test_pct=0.2)
    assert train_end == date(2026, 1, 6)
    assert val_end == date(2026, 1, 8)


def test_compute_split_dates_rejects_empty_list():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_split_dates([], train_pct=0.6, val_pct=0.2, test_pct=0.2)


def test_compute_split_dates_deduplicates_input():
    dates_with_dupes = _dates(10) + [date(2026, 1, 1)]
    train_end, val_end = compute_split_dates(dates_with_dupes, train_pct=0.6, val_pct=0.2, test_pct=0.2)
    assert train_end == date(2026, 1, 6)
    assert val_end == date(2026, 1, 8)


def test_assign_split_train():
    assert assign_split(date(2026, 1, 3), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "train"


def test_assign_split_val():
    assert assign_split(date(2026, 1, 7), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "val"


def test_assign_split_test():
    assert assign_split(date(2026, 1, 9), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "test"


def test_assign_split_boundary_belongs_to_earlier_split():
    assert assign_split(date(2026, 1, 6), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "train"
    assert assign_split(date(2026, 1, 8), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "val"
