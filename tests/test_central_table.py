from datetime import date, datetime

from monitoring_platform.evaluation import DriftResult
from monitoring_platform.central_table import build_drift_metric_row, DRIFT_METRICS_TABLE


def test_drift_metrics_table_name():
    assert DRIFT_METRICS_TABLE == "platform_monitoring.drift_metrics"


def test_build_drift_metric_row_maps_all_fields():
    result = DriftResult(
        column_name="txn_count",
        drift_metric_name="js_distance",
        drift_metric_value=0.35,
        threshold=0.2,
        status="DRIFT_DETECTED",
    )

    row = build_drift_metric_row(
        domain="exemplo",
        model_name="propensao_exemplo",
        entity_name="workspace.exemplo_features.customer_transaction_features",
        target_type="feature_table",
        result=result,
        window_start=date(2026, 8, 23),
        window_end=date(2026, 8, 23),
        run_ts=datetime(2026, 8, 23, 7, 0, 0),
    )

    assert row == {
        "domain": "exemplo",
        "model_name": "propensao_exemplo",
        "entity_name": "workspace.exemplo_features.customer_transaction_features",
        "target_type": "feature_table",
        "column_name": "txn_count",
        "drift_metric_name": "js_distance",
        "drift_metric_value": 0.35,
        "threshold": 0.2,
        "status": "DRIFT_DETECTED",
        "window_start": date(2026, 8, 23),
        "window_end": date(2026, 8, 23),
        "run_ts": datetime(2026, 8, 23, 7, 0, 0),
    }
