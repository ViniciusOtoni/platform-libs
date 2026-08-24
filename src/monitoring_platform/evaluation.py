from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    column_name: str
    drift_metric_name: str
    drift_metric_value: float
    threshold: float
    status: str  # "PASS" ou "DRIFT_DETECTED"


def evaluate_drift(column_name: str, drift_metric_name: str, drift_metric_value: float, threshold: float) -> DriftResult:
    # Metricas do tipo p-value (ex.: teste KS) sinalizam drift quando o valor
    # e BAIXO (rejeicao da hipotese nula de "sem drift"). Metricas de
    # distancia (ex.: JS distance, PSI) sinalizam drift quando o valor e ALTO.
    if "pvalue" in drift_metric_name.lower():
        status = "DRIFT_DETECTED" if drift_metric_value < threshold else "PASS"
    else:
        status = "DRIFT_DETECTED" if drift_metric_value > threshold else "PASS"
    return DriftResult(
        column_name=column_name,
        drift_metric_name=drift_metric_name,
        drift_metric_value=drift_metric_value,
        threshold=threshold,
        status=status,
    )
