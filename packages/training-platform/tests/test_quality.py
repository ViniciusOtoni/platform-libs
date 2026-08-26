import math

from training_platform.quality import (
    Finding,
    check_metric_is_finite,
    check_predictions_not_empty,
    gate_passed,
    run_sanity_gate,
)


def test_check_metric_is_finite_passes_on_valid_float():
    finding = check_metric_is_finite(0.87)
    assert finding.status == "PASS"


def test_check_metric_is_finite_fails_on_nan():
    finding = check_metric_is_finite(float("nan"))
    assert finding.status == "FAIL"


def test_check_metric_is_finite_fails_on_infinite():
    finding = check_metric_is_finite(math.inf)
    assert finding.status == "FAIL"


def test_check_predictions_not_empty_passes_when_positive():
    assert check_predictions_not_empty(100).status == "PASS"


def test_check_predictions_not_empty_fails_when_zero():
    assert check_predictions_not_empty(0).status == "FAIL"


def test_run_sanity_gate_returns_both_checks():
    findings = run_sanity_gate(0.87, num_predictions=100)
    assert {f.check for f in findings} == {"metric_is_finite", "predictions_not_empty"}


def test_gate_passed_true_when_all_pass():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "PASS")]) is True


def test_gate_passed_false_when_any_fails():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "FAIL")]) is False
