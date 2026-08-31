"""Implementações reais dos ports de monitoring.

Imports de infraestrutura ficam dentro dos métodos — ver a nota em
`core/adapters.py` sobre o container do endpoint de serving.
"""

import time
from datetime import date
from typing import Any

import pandas as pd

from mlplatform.core.audit import AUDIT_TABLE
from mlplatform.core.uc import ensure_schema

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


class DeltaBaselineBuilder:
    """Recorta a tabela observada na janela de treino e grava a baseline."""

    def __init__(self, spark, reader_group: str | None = None):
        self._spark = spark
        self._reader_group = reader_group

    def materialise(
        self,
        source_table: str,
        timestamp_column: str,
        start: date,
        end: date,
        target_table: str,
    ) -> int:
        ensure_schema(self._spark, target_table.rsplit(".", 1)[0], self._reader_group)
        # CREATE OR REPLACE, e não append: a baseline É a janela de treino do
        # modelo vigente. Depois de um retreino ela precisa passar a ser a nova
        # janela, senão o monitor continuaria comparando contra o que um modelo
        # aposentado aprendeu.
        self._spark.sql(
            f"CREATE OR REPLACE TABLE {target_table} AS "
            f"SELECT * FROM {source_table} "
            f"WHERE {timestamp_column} >= DATE'{start}' AND {timestamp_column} <= DATE'{end}'"
        )
        return self._spark.table(target_table).count()


class DatabricksQualityMonitor:
    """Cria (se preciso), atualiza e espera o monitor de qualidade.

    Os três passos ficam aqui, e não no caso de uso, porque juntos são uma
    máquina de estados com espera de até 20 minutos — infraestrutura pura, que
    nenhum teste sem workspace exercita de verdade.
    """

    def __init__(self, spark, reader_group: str | None = None):
        self._spark = spark
        # Grupo do domínio que recebe leitura nos schemas criados aqui. `None`
        # não concede — é o certo em workspace pessoal, onde não há grupo.
        self._reader_group = reader_group

    def refreshed_drift_table(
        self,
        target_table: str,
        assets_dir: str,
        output_schema: str,
        baseline_table: str | None = None,
    ) -> str:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import MonitorSnapshot

        client = WorkspaceClient()
        monitor = self._existing(client, target_table)

        if monitor is not None and monitor.baseline_table_name != baseline_table:
            # `create` é idempotente mas NÃO reconfigura. Sem este update, um
            # monitor criado antes de a baseline existir seguiria emitindo só
            # linhas CONSECUTIVE — e o caso de uso, esperando linhas BASELINE,
            # não acharia medição nenhuma.
            monitor = client.quality_monitors.update(
                table_name=target_table,
                output_schema_name=output_schema,
                baseline_table_name=baseline_table,
                snapshot=MonitorSnapshot(),
            )
            refresh = client.quality_monitors.run_refresh(table_name=target_table)
        elif monitor is not None:
            refresh = client.quality_monitors.run_refresh(table_name=target_table)
        else:
            # saveAsTable não cria o schema em Unity Catalog, e o monitor grava
            # as tabelas de saída nele.
            ensure_schema(self._spark, output_schema, self._reader_group)
            monitor = client.quality_monitors.create(
                table_name=target_table,
                assets_dir=assets_dir,
                output_schema_name=output_schema,
                baseline_table_name=baseline_table,
                snapshot=MonitorSnapshot(),
            )
            # Um monitor recem-criado fica MONITOR_STATUS_PENDING, e listar
            # refreshes nesse estado e recusado com BadRequest. Com um monitor
            # que ja existia o caminho nem passa por aqui — foi por isso que so
            # apareceu ao criar o primeiro monitor de um dominio novo.
            self._esperar_ativo(client, target_table)
            # `create` ja dispara um refresh, mas nao devolve qual: pegamos o
            # mais recente em vez de disparar um segundo, que ficaria enfileirado
            # atras do primeiro e dobraria a espera.
            refreshes = client.quality_monitors.list_refreshes(table_name=target_table).refreshes
            refresh = max(refreshes, key=lambda r: r.start_time_ms)

        self._wait(client, target_table, refresh)
        return monitor.drift_metrics_table_name

    @staticmethod
    def _esperar_ativo(client, target_table: str) -> None:
        """Espera o monitor sair de PENDING.

        Recem-criado, ele nao aceita `list_refreshes` — a API responde
        BadRequest em vez de uma lista vazia, entao nao da para tratar como
        "ainda nao ha refresh".
        """
        from databricks.sdk.service.catalog import MonitorInfoStatus

        limite = time.time() + REFRESH_TIMEOUT_SECONDS
        while True:
            info = client.quality_monitors.get(table_name=target_table)
            if info.status != MonitorInfoStatus.MONITOR_STATUS_PENDING:
                return
            if time.time() > limite:
                raise TimeoutError(
                    f"monitor de '{target_table}' seguiu PENDING por "
                    f"{REFRESH_TIMEOUT_SECONDS}s apos a criacao"
                )
            time.sleep(REFRESH_POLL_SECONDS)

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
        # `platform_monitoring` é da plataforma, não de um domínio.
        ensure_schema(self._spark, table_name.rsplit(".", 1)[0])
        frame = self._spark.createDataFrame(rows)
        mode = "append" if self._spark.catalog.tableExists(table_name) else "overwrite"
        # mergeSchema pelo mesmo motivo da tabela de predições: o conjunto de
        # colunas cresce quando o framework passa a registrar mais sobre a
        # medição, e sem isso o append falha até alguém alterar a tabela à mão.
        frame.write.format("delta").mode(mode).option("mergeSchema", "true").saveAsTable(table_name)


class GitHubRepositoryDispatch:
    """Dispara um `repository_dispatch` no repositório da esteira.

    `repository_dispatch` e não `workflow_dispatch`: o primeiro carrega um
    payload livre (qual modelo, quais colunas driftaram) que o workflow lê, e
    não exige que o workflow exista numa branch específica.

    O token precisa de escopo `contents: write` no repositório alvo. Falhar aqui
    é falha do job: drift detectado sem retreino pedido é pior que não medir —
    alguém olharia a tabela, veria DRIFT_DETECTED, e assumiria que a esteira
    reagiu.
    """

    def __init__(self, repository: str, token: str, event_type: str = "mlplatform-drift"):
        self._repository = repository
        self._token = token
        self._event_type = event_type

    def request_retrain(self, domain: str, model_name: str, drifted_columns: list[str]) -> str:
        import json
        import urllib.request

        payload = {
            "event_type": self._event_type,
            "client_payload": {
                "domain": domain,
                "model_name": model_name,
                "drifted_columns": drifted_columns,
            },
        }
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self._repository}/dispatches",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            # 204 é a resposta de sucesso do endpoint de dispatches: sem corpo.
            if response.status != 204:
                raise RuntimeError(f"repository_dispatch devolveu {response.status}")
        return f"{self._repository}#{self._event_type}"
