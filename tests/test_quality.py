import pandas as pd

from serving_platform.quality import (
    Finding,
    check_no_nulls_in_predictions,
    check_row_count_matches,
    run_predictions_gate,
    gate_passed,
)


def test_check_no_nulls_in_predictions_passes():
    df = pd.DataFrame({"prediction": [0.1, 0.9, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "PASS"


def test_check_no_nulls_in_predictions_fails():
    df = pd.DataFrame({"prediction": [0.1, None, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "FAIL"


def test_check_row_count_matches_passes_when_equal():
    assert check_row_count_matches(100, 100).status == "PASS"


def test_check_row_count_matches_fails_when_different():
    finding = check_row_count_matches(100, 87)
    assert finding.status == "FAIL"
    assert "input=100" in finding.detail


def test_run_predictions_gate_returns_both_checks():
    df = pd.DataFrame({"prediction": [0.1, 0.9]})
    findings = run_predictions_gate(df, "prediction", input_row_count=2)
    assert {f.check for f in findings} == {"no_nulls_in_predictions", "row_count_matches"}


def test_gate_passed_true_when_all_pass():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "PASS")]) is True


def test_gate_passed_false_when_any_fails():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "FAIL")]) is False
