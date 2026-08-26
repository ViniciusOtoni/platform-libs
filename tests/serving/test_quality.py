import pandas as pd

from mlplatform.serving.quality import (
    Finding,
    check_no_nulls_in_joined_columns,
    check_no_nulls_in_predictions,
    check_row_count_matches,
    gate_passed,
    run_predictions_gate,
)


def test_check_no_nulls_in_predictions_passes():
    df = pd.DataFrame({"prediction": [0.1, 0.9, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "PASS"


def test_check_no_nulls_in_predictions_fails():
    df = pd.DataFrame({"prediction": [0.1, None, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "FAIL"


def test_check_no_nulls_in_joined_columns_passes():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "txn_count": [3, 5], "prediction": [0.1, 0.9]})
    assert check_no_nulls_in_joined_columns(df, "prediction").status == "PASS"


def test_check_no_nulls_in_joined_columns_fails_on_unmatched_feature_lookup():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "txn_count": [3, None], "prediction": [0.1, 0.9]})
    finding = check_no_nulls_in_joined_columns(df, "prediction")
    assert finding.status == "FAIL"
    assert "txn_count" in finding.detail


def test_check_row_count_matches_passes_when_equal():
    assert check_row_count_matches(100, 100).status == "PASS"


def test_check_row_count_matches_fails_when_different():
    finding = check_row_count_matches(100, 87)
    assert finding.status == "FAIL"
    assert "input=100" in finding.detail


def test_run_predictions_gate_returns_all_checks():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "prediction": [0.1, 0.9]})
    findings = run_predictions_gate(df, "prediction", input_row_count=2)
    assert {f.check for f in findings} == {
        "no_nulls_in_predictions",
        "no_nulls_in_joined_columns",
        "row_count_matches",
    }


def test_gate_passed_true_when_all_pass():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "PASS")]) is True


def test_gate_passed_false_when_any_fails():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "FAIL")]) is False
