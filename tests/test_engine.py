from datetime import date

import pytest

from feature_platform.contract import FeatureTableSpec
from feature_platform.writer import WriteMode
from feature_platform.window import NoCheckpointError
from feature_platform import engine


def _spec(**overrides):
    defaults = dict(
        name="minha_feature",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.transactions"],
        compute_fn=lambda sources, window: None,
        domain="exemplo",
    )
    defaults.update(overrides)
    return FeatureTableSpec(**defaults)


def test_resolve_window_backfill_uses_explicit_range():
    window = engine.resolve_window(
        spec=_spec(),
        mode=WriteMode.BACKFILL,
        today=date(2026, 1, 20),
        backfill_start="2026-01-01",
        backfill_end="2026-01-15",
        spark=None,
    )
    assert window.start == date(2026, 1, 1)
    assert window.end == date(2026, 1, 15)


def test_resolve_window_backfill_requires_dates():
    with pytest.raises(ValueError, match="backfill mode requires"):
        engine.resolve_window(
            spec=_spec(),
            mode=WriteMode.BACKFILL,
            today=date(2026, 1, 20),
            backfill_start=None,
            backfill_end=None,
            spark=None,
        )


def test_resolve_window_incremental_uses_checkpoint(monkeypatch):
    monkeypatch.setattr(engine, "get_last_success_checkpoint", lambda spark, component, entity: date(2026, 1, 10))

    window = engine.resolve_window(
        spec=_spec(),
        mode=WriteMode.INCREMENTAL,
        today=date(2026, 1, 15),
        backfill_start=None,
        backfill_end=None,
        spark=None,
    )
    assert window.start == date(2026, 1, 10)
    assert window.end == date(2026, 1, 15)


def test_resolve_window_incremental_without_checkpoint_raises(monkeypatch):
    monkeypatch.setattr(engine, "get_last_success_checkpoint", lambda spark, component, entity: None)

    with pytest.raises(NoCheckpointError):
        engine.resolve_window(
            spec=_spec(),
            mode=WriteMode.INCREMENTAL,
            today=date(2026, 1, 15),
            backfill_start=None,
            backfill_end=None,
            spark=None,
        )
