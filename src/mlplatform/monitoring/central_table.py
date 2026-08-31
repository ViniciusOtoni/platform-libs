from datetime import date, datetime

from .evaluation import DriftResult

DRIFT_METRICS_TABLE = "platform_monitoring.drift_metrics"


def build_drift_metric_row(
    domain: str,
    model_name: str,
    entity_name: str,
    target_type: str,
    result: DriftResult,
    window_start: date,
    window_end: date,
    run_ts: datetime,
) -> dict:
    return {
        "domain": domain,
        "model_name": model_name,
        "entity_name": entity_name,
        "target_type": target_type,
        "column_name": result.column_name,
        "drift_metric_name": result.drift_metric_name,
        "drift_metric_value": result.drift_metric_value,
        "threshold": result.threshold,
        "status": result.status,
        "window_start": window_start,
        "window_end": window_end,
        "run_ts": run_ts,
    }
