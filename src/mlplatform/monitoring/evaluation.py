from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    column_name: str
    drift_metric_name: str
    drift_metric_value: float
    threshold: float
    status: str  # "PASS" ou "DRIFT_DETECTED"


def evaluate_drift(
    column_name: str, drift_metric_name: str, drift_metric_value: float, threshold: float
) -> DriftResult:
    status = "DRIFT_DETECTED" if drift_metric_value > threshold else "PASS"
    return DriftResult(
        column_name=column_name,
        drift_metric_name=drift_metric_name,
        drift_metric_value=drift_metric_value,
        threshold=threshold,
        status=status,
    )
