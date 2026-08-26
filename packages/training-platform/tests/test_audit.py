from datetime import date, datetime

from training_platform.audit import AUDIT_TABLE, RunRecord, to_row


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="training",
        entity_name="workspace.credito_models.propensao_default",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="train",
        status="SUCCESS",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        run_ts=datetime(2026, 8, 23, 3, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "training",
        "entity_name": "workspace.credito_models.propensao_default",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "train",
        "status": "SUCCESS",
        "window_start": date(2026, 1, 1),
        "window_end": date(2026, 6, 30),
        "run_ts": datetime(2026, 8, 23, 3, 0, 0),
    }
