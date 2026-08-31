"""Implementações reais dos ports de monitoring.

Imports de infraestrutura ficam dentro dos métodos — ver a nota em
`core/adapters.py` sobre o container do endpoint de serving.
"""

import time
from typing import Any

import pandas as pd

from mlplatform.core.audit import AUDIT_TABLE

from .baseline import TrainingRun

# O refresh de um monitor recém-criado varre a tabela inteira; 20 minutos é o
# teto herdado do notebook original, mantido porque foi calibrado contra o
# workspace real.
REFRESH_TIMEOUT_SECONDS = 1200
REFRESH_POLL_SECONDS = 15


class DeltaTableReader:
    def __init__(self, spark):
        self._spark = spark

    def to_pandas(self, table_name: str) -> pd.DataFrame:
        return self._spark.table(table_name).toPandas()


class AuditTrainingRunReader:
    """Traduz linhas da tabela de auditoria em `TrainingRun`.

    A tradução é o anticorruption layer: monitoring não importa `RunRecord`, e
    por isso mudar o schema da auditoria não quebra o drift em silêncio.
    """

    def __init__(self, spark, table: str = AUDIT_TABLE):
        self._spark = spark
        self._table = table

    def training_runs(self) -> list[TrainingRun]:
        if not self._spark.catalog.tableExists(self._table):
            return []

        frame = self._spark.table(self._table).filter("component = 'training'").toPandas()
        return [
            TrainingRun(
                entity_name=row["entity_name"],
                status=row["status"],
                window_start=row["window_start"],
                window_end=row["window_end"],
                run_ts=row["run_ts"],
            )
            for _, row in frame.iterrows()
        ]


class DatabricksQualityMonitor:
    """Cria (se preciso), atualiza e espera o monitor de qualidade.

    Os três passos ficam aqui, e não no caso de uso, porque juntos são uma
    máquina de estados com espera de até 20 minutos — infraestrutura pura, que
    nenhum teste sem workspace exercita de verdade.
    """

    def __init__(self, spark):
        self._spark = spark

    def refreshed_drift_table(self, target_table: str, assets_dir: str, output_schema: str) -> str:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import MonitorSnapshot

        client = WorkspaceClient()
        monitor = self._existing(client, target_table)

        if monitor is not None:
            refresh = client.quality_monitors.run_refresh(table_name=target_table)
        else:
            # saveAsTable não cria o schema em Unity Catalog, e o monitor grava
            # as tabelas de saída nele.
            self._spark.sql(f"CREATE SCHEMA IF NOT EXISTS {output_schema}")
            monitor = client.quality_monitors.create(
                table_name=target_table,
                assets_dir=assets_dir,
                output_schema_name=output_schema,
                snapshot=MonitorSnapshot(),
            )
            # `create` já dispara um refresh, mas não devolve qual: pegamos o
            # mais recente em vez de disparar um segundo, que ficaria enfileirado
            # atrás do primeiro e dobraria a espera.
            refreshes = client.quality_monitors.list_refreshes(table_name=target_table).refreshes
            refresh = max(refreshes, key=lambda r: r.start_time_ms)

        self._wait(client, target_table, refresh)
        return monitor.drift_metrics_table_name

    @staticmethod
    def _existing(client, target_table: str) -> Any:
        from databricks.sdk.errors import NotFound

        try:
            return client.quality_monitors.get(table_name=target_table)
        except NotFound:
            return None

    @staticmethod
    def _wait(client, target_table: str, refresh) -> None:
        from databricks.sdk.service.catalog import MonitorRefreshInfoState

        deadline = time.time() + REFRESH_TIMEOUT_SECONDS
        while refresh.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
            if time.time() > deadline:
                raise TimeoutError(
                    f"refresh do monitor de '{target_table}' não terminou em "
                    f"{REFRESH_TIMEOUT_SECONDS}s"
                )
            time.sleep(REFRESH_POLL_SECONDS)
            refresh = client.quality_monitors.get_refresh(
                table_name=target_table, refresh_id=refresh.refresh_id
            )

        if refresh.state != MonitorRefreshInfoState.SUCCESS:
            raise RuntimeError(
                f"refresh do monitor de '{target_table}' terminou em {refresh.state}: {refresh.message}"
            )


class DeltaDriftMetricsWriter:
    def __init__(self, spark):
        self._spark = spark

    def append(self, rows: list[dict], table_name: str) -> None:
        schema = table_name.rsplit(".", 1)[0]
        self._spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        frame = self._spark.createDataFrame(rows)
        mode = "append" if self._spark.catalog.tableExists(table_name) else "overwrite"
        # mergeSchema pelo mesmo motivo da tabela de predições: o conjunto de
        # colunas cresce quando o framework passa a registrar mais sobre a
        # medição, e sem isso o append falha até alguém alterar a tabela à mão.
        frame.write.format("delta").mode(mode).option("mergeSchema", "true").saveAsTable(table_name)
