from datetime import date, datetime

from mlplatform.core.audit import RunRecord, get_last_success_checkpoint, write_run
from mlplatform.core.quality import Finding, gate_passed

from .contract import FeatureTableSpec
from .naming import resolve_table_name
from .quality import run_quality_gate
from .types import DateRange
from .window import parse_backfill_window, resolve_incremental_window
from .writer import WriteMode, write_feature_table

COMPONENT = "feature_generation"


class QualityGateFailure(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        failed = [f.check for f in findings if f.status == "FAIL"]
        super().__init__(f"quality gate failed: {failed}")


def resolve_window(
    spec: FeatureTableSpec,
    mode: WriteMode,
    today: date,
    backfill_start: str | None,
    backfill_end: str | None,
    spark,
) -> DateRange:
    if mode == WriteMode.BACKFILL:
        if not backfill_start or not backfill_end:
            raise ValueError("backfill mode requires start_date and end_date")
        return parse_backfill_window(backfill_start, backfill_end)

    checkpoint = get_last_success_checkpoint(spark, COMPONENT, spec.name)
    return resolve_incremental_window(checkpoint, today)


def run_feature_table(
    spec: FeatureTableSpec,
    spark,
    catalog: str,
    mode: WriteMode,
    today: date,
    run_id: str,
    git_commit: str,
    git_branch: str,
    backfill_start: str | None = None,
    backfill_end: str | None = None,
    database_instance_name: str = "",
) -> None:
    """Orquestra uma execução completa: resolve janela, computa, aplica o gate de
    qualidade, escreve (se passar), audita e sincroniza online (se configurado).
    Requer SparkSession real — exercitado via notebook (Task 12), não via pytest.
    database_instance_name só é usado (e obrigatório) quando spec.online=True."""
    window = resolve_window(spec, mode, today, backfill_start, backfill_end, spark)
    sources = {name: spark.table(name) for name in spec.sources}
    result_df = spec.compute_fn(sources, window)
    pandas_df = result_df.toPandas()

    findings = run_quality_gate(pandas_df, spec.entity_keys, spec.timestamp_key, window.end)
    table_name = resolve_table_name(catalog, spec.domain, spec.name, spec.table_name)

    if not gate_passed(findings):
        write_run(
            spark,
            RunRecord(
                component=COMPONENT,
                entity_name=spec.name,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode=mode.value,
                status="FAILED",
                window_start=window.start,
                window_end=window.end,
                run_ts=datetime.utcnow(),
            ),
        )
        raise QualityGateFailure(findings)

    write_feature_table(
        spark,
        result_df,
        table_name,
        spec.entity_keys,
        spec.timestamp_key,
        mode,
        partition_cols=spec.entity_keys[:1],
        enable_cdf=spec.online,
    )
    spark.sql(
        f"ALTER TABLE {table_name} SET TBLPROPERTIES "
        f"('git_commit' = '{git_commit}', 'git_branch' = '{git_branch}')"
    )

    write_run(
        spark,
        RunRecord(
            component=COMPONENT,
            entity_name=spec.name,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id,
            mode=mode.value,
            status="SUCCESS",
            window_start=window.start,
            window_end=window.end,
            run_ts=datetime.utcnow(),
        ),
    )

    if spec.online:
        from .online_sync import sync_online_table

        sync_online_table(spark, table_name, spec.entity_keys, database_instance_name)
