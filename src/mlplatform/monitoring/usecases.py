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
from .evaluation import DRIFT_DETECTED, DriftResult, evaluate_drift
from .metrics import BASELINE, CONSECUTIVE, resolve
from .ports import (
    BaselineBuilder,
    DriftMetricsWriter,
    QualityMonitor,
    RetrainTrigger,
    TableReader,
    TrainingRunReader,
)

COMPONENT = "monitoring"
MODE = "drift_check"

# Colunas da tabela de métricas do monitor do Databricks que consumimos.
_COLUMN_KEY = "column_name"
_COMPARISON_KEY = "drift_type"
_SLICE_KEY = "slice_key"
_WINDOW_KEY = "window"


class DriftMetricUnavailable(Exception):
    """A métrica escolhida não existe na tabela do monitor.

    Falha alta de propósito. O código anterior fazia `.get(coluna, 0.0)` sobre
    uma coluna inexistente e registrava zero em toda medição — os jobs passavam
    verdes, a tabela enchia, e nada daquilo significava nada. Um gate que nunca
    pode disparar é pior que gate nenhum, porque parece proteção.
    """


class NoColumnMeasured(Exception):
    """Nenhuma das colunas declaradas rendeu medição.

    Uma coluna sem linha é normal — no primeiro refresh não há janela anterior.
    TODAS sem linha não é: ou os nomes não existem na tabela observada, ou a
    métrica escolhida não se aplica ao tipo delas (as limitadas a [0,1] vêm
    nulas em colunas numéricas). Os dois casos são erro de configuração, e os
    dois se disfarçam de "sem drift".
    """


def assets_dir(domain: str, model_name: str, target_type: str) -> str:
    return f"/Shared/mlplatform/{domain}/{model_name}/{target_type}"


def monitoring_schema(catalog: str, domain: str) -> str:
    return f"{catalog}.{domain}_monitoring"


def baseline_table_name(catalog: str, domain: str, target_table: str) -> str:
    """Uma baseline por tabela observada, no schema de monitoring do domínio."""
    return f"{monitoring_schema(catalog, domain)}.{target_table.rsplit('.', 1)[-1]}_baseline"


class EmptyBaseline(Exception):
    """A janela de treino não recortou linha nenhuma da tabela observada.

    Acontece quando a coluna de tempo declarada não é a que a tabela usa, ou
    quando o histórico já não cobre a janela em que o modelo foi treinado. Nos
    dois casos o monitor compararia contra o vazio e reportaria drift em tudo —
    ou em nada, dependendo da métrica. Falhar aqui é mais barato.
    """


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
        retrain: RetrainTrigger | None = None,
        baseline: BaselineBuilder | None = None,
    ):
        self._runs = runs
        self._monitor = monitor
        self._reader = reader
        self._writer = writer
        self._audit = audit
        self._clock = clock
        # Opcional: sem gatilho o componente só mede e registra, que é o
        # comportamento de quem ainda não quer retreino automático.
        self._retrain = retrain
        self._baseline = baseline

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

        # Duas funções ao mesmo tempo. Porteiro: exigir um treino bem-sucedido
        # evita medir drift de um modelo que nunca chegou a produção. E janela:
        # quando o domínio declara `baseline_timestamp_column`, é este intervalo
        # que define a baseline — a fatia da tabela observada sobre a qual o
        # modelo vigente aprendeu.
        try:
            window_start, window_end = resolve_baseline_window(
                self._runs.training_runs(), full_model_name
            )
        except Exception:
            self._record(config, "FAILED", today, run_id, git_commit, git_branch)
            raise

        baseline_table = self._materialise_baseline(config, catalog, window_start, window_end)
        comparison = CONSECUTIVE if baseline_table is None else BASELINE

        drift_table = self._monitor.refreshed_drift_table(
            target_table=config.target_table,
            assets_dir=assets_dir(config.domain, config.model_name, config.target_type),
            output_schema=monitoring_schema(catalog, config.domain),
            baseline_table=baseline_table,
        )

        metric = resolve(config.drift_metric)
        metrics = self._reader.to_pandas(drift_table)

        if not metrics.empty and metric.column not in metrics.columns:
            self._record(config, "FAILED", today, run_id, git_commit, git_branch)
            raise DriftMetricUnavailable(
                f"'{metric.column}' não existe em {drift_table}. "
                f"Colunas disponíveis: {sorted(metrics.columns)}"
            )

        comparable = self._comparable(metrics, comparison)
        results = [r for r in (self._latest_for(comparable, c, config, metric) for c in config.columns) if r]

        if config.columns and not results:
            self._record(config, "FAILED", today, run_id, git_commit, git_branch)
            raise NoColumnMeasured(
                f"nenhuma de {config.columns} rendeu medição de '{metric.column}' em "
                f"{drift_table} — os nomes existem na tabela observada, e a métrica "
                f"se aplica ao tipo delas? ({metric.note or 'sem restrição conhecida'})"
            )

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

        # O gatilho vem DEPOIS da escrita: a medição é o produto do job e não
        # pode se perder porque o GitHub estava fora do ar. E vem ANTES da
        # auditoria de sucesso, porque drift detectado sem retreino pedido não é
        # sucesso — alguém veria DRIFT_DETECTED na tabela e assumiria que a
        # esteira reagiu.
        drifted = [r.column_name for r in results if r.status == DRIFT_DETECTED]
        if drifted and self._retrain is not None:
            try:
                self._retrain.request_retrain(config.domain, config.model_name, drifted)
            except Exception:
                self._record(config, "FAILED", today, run_id, git_commit, git_branch)
                raise

        self._record(config, "SUCCESS", today, run_id, git_commit, git_branch)
        return results

    def _materialise_baseline(
        self, config: MonitoringConfig, catalog: str, start: date, end: date
    ) -> str | None:
        """A baseline, quando o domínio pediu comparação contra a janela de treino."""
        if not config.baseline_timestamp_column or self._baseline is None:
            return None

        table = baseline_table_name(catalog, config.domain, config.target_table)
        rows = self._baseline.materialise(
            source_table=config.target_table,
            timestamp_column=config.baseline_timestamp_column,
            start=start,
            end=end,
            target_table=table,
        )
        if rows == 0:
            raise EmptyBaseline(
                f"a janela de treino {start}..{end} não recortou linha nenhuma de "
                f"{config.target_table} por '{config.baseline_timestamp_column}' — "
                f"a coluna de tempo é essa mesma, e o histórico ainda cobre a janela?"
            )
        return table

    @staticmethod
    def _comparable(metrics, comparison: str):
        """Só as linhas que representam a tabela inteira, na comparação pedida.

        O monitor emite uma linha por FATIA quando `slicing_exprs` está
        configurado, e uma linha por tipo de comparação. Com baseline
        configurada convivem `BASELINE` e `CONSECUTIVE` na mesma tabela: ler as
        duas juntas responderia uma pergunta diferente da que o domínio fez.
        """
        if metrics.empty:
            return metrics
        rows = metrics
        if _COMPARISON_KEY in rows.columns:
            rows = rows[rows[_COMPARISON_KEY] == comparison]
        if _SLICE_KEY in rows.columns:
            rows = rows[rows[_SLICE_KEY].isna()]
        return rows

    @staticmethod
    def _window_end(row):
        """Fim da janela da linha, para ordenar sem depender da ordem física."""
        window = row.get(_WINDOW_KEY)
        if isinstance(window, dict):
            return window.get("end")
        return getattr(window, "end", None)

    def _latest_for(self, metrics, column: str, config: MonitoringConfig, metric) -> DriftResult | None:
        """A medição mais recente da coluna, ou nada.

        Uma coluna sem linha não é erro: no primeiro refresh não existe janela
        anterior para comparar. Tratar isso como falha reprovaria toda primeira
        execução — mas TODAS sem linha é erro, e quem cobra isso é o chamador.
        """
        rows = metrics[metrics[_COLUMN_KEY] == column] if not metrics.empty else metrics
        if rows.empty:
            return None

        ends = [self._window_end(row) for _, row in rows.iterrows()]
        latest = rows.iloc[ends.index(max(ends))] if all(e is not None for e in ends) else rows.iloc[-1]

        raw = latest[metric.column]
        if metric.nested_field is not None and raw is not None:
            raw = raw.get(metric.nested_field) if isinstance(raw, dict) else getattr(raw, metric.nested_field, None)

        # Nulo significa "não se aplica a esta coluna", não "sem drift". Zerar
        # aqui reproduziria o bug original numa forma nova.
        if raw is None or (isinstance(raw, float) and raw != raw):
            return None

        return evaluate_drift(
            column_name=column,
            drift_metric_name=config.drift_metric,
            drift_metric_value=float(raw),
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
