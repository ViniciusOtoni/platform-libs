from dataclasses import dataclass
from datetime import date, datetime

AUDIT_TABLE = "platform_audit.pipeline_runs"


@dataclass(frozen=True)
class RunRecord:
    component: str
    entity_name: str
    git_commit: str
    git_branch: str
    run_id: str
    mode: str
    status: str
    window_start: date
    window_end: date
    run_ts: datetime


def to_row(record: RunRecord) -> dict:
    return {
        "component": record.component,
        "entity_name": record.entity_name,
        "git_commit": record.git_commit,
        "git_branch": record.git_branch,
        "run_id": record.run_id,
        "mode": record.mode,
        "status": record.status,
        "window_start": record.window_start,
        "window_end": record.window_end,
        "run_ts": record.run_ts,
    }


def write_run(spark, record: RunRecord) -> None:
    """Escreve um registro na tabela de auditoria central. Requer SparkSession —
    exercitado via notebook (Task 12), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — sem isso,
    # a primeira escrita falha com SCHEMA_NOT_FOUND.
    schema = AUDIT_TABLE.split(".")[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    df = spark.createDataFrame([to_row(record)])
    if spark.catalog.tableExists(AUDIT_TABLE):
        df.write.format("delta").mode("append").saveAsTable(AUDIT_TABLE)
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(AUDIT_TABLE)


def get_last_success_checkpoint(spark, component: str, entity_name: str):
    """Retorna o window_end do último run SUCCESS, ou None se não houver nenhum.
    Requer SparkSession — exercitado via notebook (Task 12), não via pytest."""
    import pyspark.sql.functions as F

    if not spark.catalog.tableExists(AUDIT_TABLE):
        return None

    result = (
        spark.table(AUDIT_TABLE)
        .filter(
            (F.col("component") == component)
            & (F.col("entity_name") == entity_name)
            & (F.col("status") == "SUCCESS")
        )
        .orderBy(F.col("window_end").desc())
        .limit(1)
        .collect()
    )
    return result[0]["window_end"] if result else None
