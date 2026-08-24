from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    detail: str = ""


def check_metric_is_finite(metric_value: float) -> Finding:
    is_finite = isinstance(metric_value, (int, float)) and math.isfinite(metric_value)
    return Finding(
        check="metric_is_finite",
        status="PASS" if is_finite else "FAIL",
        detail=f"metric={metric_value}",
    )


def check_predictions_not_empty(num_predictions: int) -> Finding:
    return Finding(
        check="predictions_not_empty",
        status="PASS" if num_predictions > 0 else "FAIL",
        detail=f"num_predictions={num_predictions}",
    )


def run_sanity_gate(metric_value: float, num_predictions: int) -> list[Finding]:
    return [
        check_metric_is_finite(metric_value),
        check_predictions_not_empty(num_predictions),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
