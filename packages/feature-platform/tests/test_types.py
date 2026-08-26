from datetime import date

import pytest

from feature_platform.types import DateRange


def test_date_range_holds_start_and_end():
    r = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 31))
    assert r.start == date(2026, 1, 1)
    assert r.end == date(2026, 1, 31)


def test_date_range_rejects_start_after_end():
    with pytest.raises(ValueError, match="must not be after"):
        DateRange(start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_date_range_allows_start_equal_end():
    r = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 1))
    assert r.start == r.end
