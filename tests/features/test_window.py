from datetime import date

import pytest

from mlplatform.features.window import (
    NoCheckpointError,
    parse_backfill_window,
    resolve_incremental_window,
)


def test_incremental_window_starts_at_last_checkpoint():
    w = resolve_incremental_window(last_success_end=date(2026, 1, 10), today=date(2026, 1, 15))
    assert w.start == date(2026, 1, 10)
    assert w.end == date(2026, 1, 15)


def test_incremental_window_raises_without_checkpoint():
    with pytest.raises(NoCheckpointError, match="run a backfill first"):
        resolve_incremental_window(last_success_end=None, today=date(2026, 1, 15))


def test_incremental_window_raises_when_nothing_to_process():
    with pytest.raises(ValueError, match="nothing to process"):
        resolve_incremental_window(last_success_end=date(2026, 1, 15), today=date(2026, 1, 15))


def test_parse_backfill_window():
    w = parse_backfill_window("2026-01-01", "2026-01-31")
    assert w.start == date(2026, 1, 1)
    assert w.end == date(2026, 1, 31)
