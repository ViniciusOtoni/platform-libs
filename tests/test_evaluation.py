from monitoring_platform.evaluation import DriftResult, evaluate_drift


def test_evaluate_drift_passes_below_threshold():
    result = evaluate_drift("txn_count", "js_distance", 0.05, threshold=0.2)
    assert result.status == "PASS"


def test_evaluate_drift_detects_above_threshold():
    result = evaluate_drift("txn_count", "js_distance", 0.35, threshold=0.2)
    assert result.status == "DRIFT_DETECTED"


def test_evaluate_drift_boundary_value_passes():
    result = evaluate_drift("txn_count", "js_distance", 0.2, threshold=0.2)
    assert result.status == "PASS"


def test_evaluate_drift_carries_input_fields():
    result = evaluate_drift("avg_ticket", "ks_test_pvalue", 0.01, threshold=0.05)
    assert result == DriftResult(
        column_name="avg_ticket",
        drift_metric_name="ks_test_pvalue",
        drift_metric_value=0.01,
        threshold=0.05,
        status="DRIFT_DETECTED",
    )
