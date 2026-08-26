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


def write_drift_metrics(spark, rows: list[dict]) -> None:
    """Requer SparkSession — exercitado via notebook (Task 9), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — platform_monitoring
    # é um schema novo, nunca criado por nenhum componente anterior (mesmo bug já
    # confirmado nos três componentes anteriores).
    schema = DRIFT_METRICS_TABLE.split(".")[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    df = spark.createDataFrame(rows)
    if spark.catalog.tableExists(DRIFT_METRICS_TABLE):
        df.write.format("delta").mode("append").saveAsTable(DRIFT_METRICS_TABLE)
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(DRIFT_METRICS_TABLE)
