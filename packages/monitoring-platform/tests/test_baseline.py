from datetime import date, datetime

import pytest

from monitoring_platform.baseline import NoTrainingRunError, TrainingRun, resolve_baseline_window

MODEL = "workspace.exemplo_models.propensao_exemplo"
OTHER_MODEL = "workspace.exemplo_models.outro_modelo"


def _run(entity_name, status, window_start, window_end, run_ts):
    return TrainingRun(
        entity_name=entity_name, status=status, window_start=window_start, window_end=window_end, run_ts=run_ts
    )


def test_resolve_baseline_window_picks_most_recent_success():
    runs = [
        _run(MODEL, "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1)),
        _run(MODEL, "SUCCESS", date(2026, 2, 1), date(2026, 7, 31), datetime(2026, 8, 1)),
    ]
    start, end = resolve_baseline_window(runs, MODEL)
    assert start == date(2026, 2, 1)
    assert end == date(2026, 7, 31)


def test_resolve_baseline_window_ignores_failed_runs():
    runs = [
        _run(MODEL, "FAILED", date(2026, 3, 1), date(2026, 8, 31), datetime(2026, 9, 1)),
        _run(MODEL, "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1)),
    ]
    start, end = resolve_baseline_window(runs, MODEL)
    assert start == date(2026, 1, 1)
    assert end == date(2026, 6, 30)


def test_resolve_baseline_window_ignores_other_models():
    runs = [_run(OTHER_MODEL, "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1))]

    with pytest.raises(NoTrainingRunError, match="propensao_exemplo"):
        resolve_baseline_window(runs, MODEL)


def test_resolve_baseline_window_raises_when_no_runs():
    with pytest.raises(NoTrainingRunError):
        resolve_baseline_window([], MODEL)
