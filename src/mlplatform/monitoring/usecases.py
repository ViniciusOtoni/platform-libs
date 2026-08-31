"""Caso de uso de verificação de drift.

Era ~120 linhas de notebook no repositório de domínio, copiadas por domínio e
sem teste — incluindo a máquina de estados de refresh do monitor, que é a parte
mais cara de acertar e a que ninguém revisava.
"""

from datetime import date

from mlplatform.core.audit import RunRecord
from mlplatform.core.naming import derive_model_name
from mlplatform.core.ports import AuditStore, Clock

from .baseline import resolve_baseline_window
from .central_table import DRIFT_METRICS_TABLE, build_drift_metric_row
from .contract import MonitoringConfig
from .evaluation import DriftResult, evaluate_drift
from .ports import DriftMetricsWriter, QualityMonitor, TableReader, TrainingRunReader

COMPONENT = "monitoring"
MODE = "drift_check"

# Colunas da tabela de métricas do monitor do Databricks que consumimos.
_COLUMN_KEY = "column_name"
_METRIC_NAME_KEY = "drift_type"
_METRIC_VALUE_KEY = "statistic"


def assets_dir(domain: str, model_name: str, target_type: str) -> str:
    return f"/Shared/mlplatform/{domain}/{model_name}/{target_type}"


def monitoring_schema(catalog: str, domain: str) -> str:
    return f"{catalog}.{domain}_monitoring"


class EvaluateDrift:
    """Compara a distribuição corrente contra a do monitor e registra o veredito."""

    def __init__(
        self,
        runs: TrainingRunReader,
        monitor: QualityMonitor,
        reader: TableReader,
        writer: DriftMetricsWriter,
        audit: AuditStore,
        clock: Clock,
    ):
        self._runs = runs
        self._monitor = monitor
        self._reader = reader
        self._writer = writer
        self._audit = audit
        self._clock = clock

    def execute(
        self,
        config: MonitoringConfig,
        catalog: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
        today: date | None = None,
    ) -> list[DriftResult]:
        today = today or self._clock.now().date()
        full_model_name = derive_model_name(catalog, config.domain, config.model_name)

        # Porteiro, não janela: exigir um treino bem-sucedido evita medir drift
        # de um modelo que nunca chegou a produção. A janela devolvida não é
        # usada — o monitor do Databricks faz a comparação por conta própria, a
        # partir do snapshot dele.
        try:
            resolve_baseline_window(self._runs.training_runs(), full_model_name)
        except Exception:
            self._record(config, "FAILED", today, run_id, git_commit, git_branch)
            raise

        drift_table = self._monitor.refreshed_drift_table(
            target_table=config.target_table,
            assets_dir=assets_dir(config.domain, config.model_name, config.target_type),
            output_schema=monitoring_schema(catalog, config.domain),
        )

        metrics = self._reader.to_pandas(drift_table)
        results = [r for r in (self._latest_for(metrics, c, config) for c in config.columns) if r]

        rows = [
            build_drift_metric_row(
                domain=config.domain,
                model_name=config.model_name,
                entity_name=config.target_table,
                target_type=config.target_type,
                result=result,
                window_start=today,
                window_end=today,
                run_ts=self._clock.now(),
            )
            for result in results
        ]
        if rows:
            self._writer.append(rows, DRIFT_METRICS_TABLE)

        self._record(config, "SUCCESS", today, run_id, git_commit, git_branch)
        return results

    def _latest_for(self, metrics, column: str, config: MonitoringConfig) -> DriftResult | None:
        """A linha mais recente da coluna, ou nada.

        Uma coluna sem linha não é erro: o monitor pode ainda não ter material
        para comparar — no primeiro refresh não existe janela anterior nenhuma.
        Tratar isso como falha reprovaria toda primeira execução.
        """
        rows = metrics[metrics[_COLUMN_KEY] == column]
        if rows.empty:
            return None

        latest = rows.iloc[-1]
        return evaluate_drift(
            column_name=column,
            drift_metric_name=str(latest.get(_METRIC_NAME_KEY, "unknown_metric")),
            drift_metric_value=float(latest.get(_METRIC_VALUE_KEY, 0.0)),
            threshold=config.threshold,
        )

    def _record(
        self,
        config: MonitoringConfig,
        status: str,
        day: date,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> None:
        self._audit.append(
            RunRecord(
                component=COMPONENT,
                entity_name=config.target_table,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode=MODE,
                status=status,
                window_start=day,
                window_end=day,
                run_ts=self._clock.now(),
            )
        )
