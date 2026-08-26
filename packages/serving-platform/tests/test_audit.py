from datetime import date, datetime

from serving_platform.audit import AUDIT_TABLE, RunRecord, to_row


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="serving",
        entity_name="workspace.credito_predictions.propensao_default",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="batch",
        status="SUCCESS",
        window_start=date(2026, 8, 23),
        window_end=date(2026, 8, 23),
        run_ts=datetime(2026, 8, 23, 6, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "serving",
        "entity_name": "workspace.credito_predictions.propensao_default",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "batch",
        "status": "SUCCESS",
        "window_start": date(2026, 8, 23),
        "window_end": date(2026, 8, 23),
        "run_ts": datetime(2026, 8, 23, 6, 0, 0),
    }
