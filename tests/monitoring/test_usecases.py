"""Testes do caso de uso de drift.

Toda esta lógica vivia num notebook de ~120 linhas no repositório de domínio,
copiado por domínio e sem teste nenhum.
"""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from mlplatform.monitoring.baseline import NoTrainingRunError, TrainingRun
from mlplatform.monitoring.central_table import DRIFT_METRICS_TABLE
from mlplatform.monitoring.contract import MonitoringConfig
from mlplatform.monitoring.usecases import (
    DriftMetricUnavailable,
    EvaluateDrift,
    NoColumnMeasured,
)
from mlplatform.testing import (
    FakeDriftMetricsWriter,
    FakeQualityMonitor,
    FakeTableReader,
    FakeTrainingRunReader,
    FixedClock,
    InMemoryAuditStore,
)

INSTANT = datetime(2026, 8, 30, 7, 0, 0, tzinfo=UTC)
DRIFT_TABLE = "workspace.exemplo_monitoring.drift"
MODEL = "workspace.exemplo_models.propensao_exemplo"

JANELA_NOVA = {"start": "2026-08-30", "end": "2026-08-31"}
JANELA_VELHA = {"start": "2026-08-28", "end": "2026-08-29"}


def _config(**o) -> MonitoringConfig:
    return MonitoringConfig(
        **{
            "domain": "exemplo",
            "model_name": "propensao_exemplo",
            "target_type": "feature_table",
            "target_table": "workspace.exemplo_features.customer_transaction_features",
            "columns": ["txn_count", "avg_ticket"],
            "threshold": 0.2,
            "schedule_cron": "0 0 7 * * ?",
            **o,
        }
    )


def _training_run(status: str = "SUCCESS") -> TrainingRun:
    return TrainingRun(
        entity_name=MODEL,
        status=status,
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        run_ts=datetime(2026, 7, 1, tzinfo=UTC),
    )


def _metrics(**o) -> pd.DataFrame:
    """Reproduz a tabela de drift que o Lakehouse Monitoring gera de verdade.

    O fixture antigo tinha uma coluna `statistic`, que NÃO existe: no topo da
    tabela as métricas se chamam `population_stability_index`, `js_distance`,
    `wasserstein_distance` e afins. `statistic` é um campo ANINHADO dentro dos
    structs `ks_test` e `chi_squared_test` — era daí que vinha a confusão.

    Foi o fixture confortável que deixou o bug passar: o código lia `statistic`,
    não achava, e registrava 0.0 em toda medição.

    `js_distance` vem nula aqui de propósito: o monitor só a calcula para
    colunas categóricas, e estas são numéricas.
    """
    base = {
        "column_name": ["txn_count", "avg_ticket"],
        "drift_type": ["CONSECUTIVE", "CONSECUTIVE"],
        "slice_key": [None, None],
        "window": [JANELA_NOVA, JANELA_NOVA],
        "population_stability_index": [0.05, 0.05],
        "js_distance": [None, None],
        "ks_test": [{"statistic": 0.05, "pvalue": 0.9}] * 2,
    }
    base.update(o)
    return pd.DataFrame(base)


def _run(metrics=None, runs=None, config=None, writer=None, audit=None, monitor=None):
    writer = writer or FakeDriftMetricsWriter()
    audit = audit or InMemoryAuditStore()
    monitor = monitor or FakeQualityMonitor(drift_table=DRIFT_TABLE)
    results = EvaluateDrift(
        runs=FakeTrainingRunReader(runs if runs is not None else [_training_run()]),
        monitor=monitor,
        reader=FakeTableReader({DRIFT_TABLE: metrics if metrics is not None else _metrics()}),
        writer=writer,
        audit=audit,
        clock=FixedClock(INSTANT),
    ).execute(
        config=config or _config(),
        catalog="workspace",
        run_id="run-1",
        git_commit="abc",
        git_branch="main",
    )
    return results, writer, audit, monitor


# -- veredito ---------------------------------------------------------------


def test_a_metric_under_the_threshold_passes():
    results, _, _, _ = _run()

    assert {r.status for r in results} == {"PASS"}
    assert {r.drift_metric_name for r in results} == {"population_stability_index"}


def test_a_metric_over_the_threshold_is_flagged():
    results, _, _, _ = _run(metrics=_metrics(population_stability_index=[0.35, 0.05]))

    flagged = {r.column_name for r in results if r.status == "DRIFT_DETECTED"}
    assert flagged == {"txn_count"}


def test_only_the_declared_columns_are_evaluated():
    """A tabela do monitor traz uma linha por coluna da tabela observada; o
    domínio escolhe quais importam."""
    results, _, _, _ = _run(config=_config(columns=["txn_count"]))

    assert [r.column_name for r in results] == ["txn_count"]


def test_a_column_without_measurement_is_skipped_not_failed():
    """No primeiro refresh o monitor pode não ter janela anterior para comparar.
    Tratar ausência como erro reprovaria toda primeira execução."""
    results, _, _, _ = _run(metrics=_metrics(column_name=["txn_count", "outra"]))

    assert [r.column_name for r in results] == ["txn_count"]


def test_a_struct_metric_is_read_from_its_nested_field():
    """`ks_test` e `chi_squared_test` são structs com `statistic` e `pvalue`."""
    results, _, _, _ = _run(config=_config(drift_metric="ks_test"))

    assert {r.drift_metric_name for r in results} == {"ks_test"}
    assert results[0].drift_metric_value == 0.05


# -- de que linha o veredito sai ---------------------------------------------


def test_the_most_recent_window_wins_regardless_of_row_order():
    """A tabela acumula uma linha por refresh, e a ordem física não é garantida.
    Pegar a última linha faria o veredito depender de como o Delta as devolveu."""
    metrics = pd.DataFrame(
        {
            "column_name": ["txn_count", "txn_count"],
            "drift_type": ["CONSECUTIVE", "CONSECUTIVE"],
            "slice_key": [None, None],
            # a janela mais recente vem PRIMEIRO de propósito
            "window": [JANELA_NOVA, JANELA_VELHA],
            "population_stability_index": [0.01, 0.9],
        }
    )
    results, _, _, _ = _run(metrics=metrics, config=_config(columns=["txn_count"]))

    assert results[0].drift_metric_value == 0.01


def test_sliced_rows_do_not_decide_the_verdict():
    """O monitor emite uma linha por fatia quando `slicing_exprs` existe. Ler
    tudo junto deixaria uma fatia decidir pelo total."""
    metrics = pd.DataFrame(
        {
            "column_name": ["txn_count", "txn_count"],
            "drift_type": ["CONSECUTIVE", "CONSECUTIVE"],
            "slice_key": [None, "regiao"],
            "window": [JANELA_NOVA, JANELA_NOVA],
            "population_stability_index": [0.01, 0.9],
        }
    )
    results, _, _, _ = _run(metrics=metrics, config=_config(columns=["txn_count"]))

    assert results[0].drift_metric_value == 0.01


def test_the_baseline_comparison_does_not_mix_with_the_consecutive_one():
    """Se um dia `baseline_table_name` for configurado, os dois tipos de
    comparação convivem na mesma tabela."""
    metrics = pd.DataFrame(
        {
            "column_name": ["txn_count", "txn_count"],
            "drift_type": ["CONSECUTIVE", "BASELINE"],
            "slice_key": [None, None],
            "window": [JANELA_NOVA, JANELA_NOVA],
            "population_stability_index": [0.01, 0.9],
        }
    )
    results, _, _, _ = _run(metrics=metrics, config=_config(columns=["txn_count"]))

    assert results[0].drift_metric_value == 0.01


# -- o que antes falhava em silêncio ----------------------------------------


def test_a_metric_column_that_does_not_exist_fails_loudly():
    """O bug original: `.get(coluna, 0.0)` sobre coluna inexistente registrava
    zero em toda medição, e o gate nunca podia disparar."""
    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()
    metrics = _metrics().drop(columns=["population_stability_index"])

    with pytest.raises(DriftMetricUnavailable, match="population_stability_index"):
        _run(metrics=metrics, writer=writer, audit=audit)

    assert writer.written == []
    assert audit.statuses() == ["FAILED"]


def test_a_metric_null_for_every_column_fails_loudly():
    """`js_distance` vem nula em colunas numéricas. Sem esta checagem, escolher
    a métrica errada para o tipo de dado produziria "sem drift" para sempre."""
    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()

    with pytest.raises(NoColumnMeasured, match="js_distance"):
        _run(config=_config(drift_metric="js_distance"), writer=writer, audit=audit)

    assert writer.written == []
    assert audit.statuses() == ["FAILED"]


def test_when_no_declared_column_is_measured_it_audits_and_stops():
    """Uma coluna sem medição é normal; TODAS sem medição é erro de
    configuração — nomes que não existem na tabela observada."""
    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()

    with pytest.raises(NoColumnMeasured):
        _run(metrics=_metrics(column_name=["outra", "mais_outra"]), writer=writer, audit=audit)

    assert audit.statuses() == ["FAILED"]


# -- escrita e auditoria ----------------------------------------------------


def test_the_measurements_land_in_the_central_table():
    _, writer, _, _ = _run()

    table, rows = writer.written[0]
    assert table == DRIFT_METRICS_TABLE
    assert {r["column_name"] for r in rows} == {"txn_count", "avg_ticket"}
    assert {r["entity_name"] for r in rows} == {_config().target_table}


def test_the_monitor_is_scoped_to_the_domain_and_target():
    _, _, _, monitor = _run()

    call = monitor.calls[0]
    assert call["target_table"] == _config().target_table
    assert call["assets_dir"] == "/Shared/mlplatform/exemplo/propensao_exemplo/feature_table"
    assert call["output_schema"] == "workspace.exemplo_monitoring"


# -- o porteiro -------------------------------------------------------------


def test_without_a_successful_training_run_it_audits_and_stops():
    """Medir drift de um modelo que nunca chegou a produção não significa nada —
    e o monitor seria criado à toa, cobrando um refresh de 20 minutos."""
    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()

    with pytest.raises(NoTrainingRunError):
        _run(runs=[_training_run(status="FAILED")], writer=writer, audit=audit)

    assert writer.written == []
    assert audit.statuses() == ["FAILED"]


def test_a_training_run_of_another_model_does_not_count():
    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()
    outro = TrainingRun(
        entity_name="workspace.exemplo_models.outro",
        status="SUCCESS",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        run_ts=datetime(2026, 7, 1, tzinfo=UTC),
    )

    with pytest.raises(NoTrainingRunError):
        _run(runs=[outro], writer=writer, audit=audit)

    assert audit.statuses() == ["FAILED"]


# -- retreino disparado por drift -------------------------------------------


def test_no_drift_asks_for_no_retrain():
    from mlplatform.testing import FakeRetrainTrigger

    trigger = FakeRetrainTrigger()
    EvaluateDrift(
        runs=FakeTrainingRunReader([_training_run()]),
        monitor=FakeQualityMonitor(drift_table=DRIFT_TABLE),
        reader=FakeTableReader({DRIFT_TABLE: _metrics()}),
        writer=FakeDriftMetricsWriter(),
        audit=InMemoryAuditStore(),
        clock=FixedClock(INSTANT),
        retrain=trigger,
    ).execute(config=_config(), catalog="workspace", run_id="r", git_commit="a", git_branch="main")

    assert trigger.requests == []


def test_drift_asks_for_a_retrain_naming_the_columns():
    """O payload carrega quais colunas driftaram: quem for revisar o modelo
    retreinado precisa saber o que mudou para julgar se ele faz sentido."""
    from mlplatform.testing import FakeRetrainTrigger

    trigger = FakeRetrainTrigger()
    EvaluateDrift(
        runs=FakeTrainingRunReader([_training_run()]),
        monitor=FakeQualityMonitor(drift_table=DRIFT_TABLE),
        reader=FakeTableReader({DRIFT_TABLE: _metrics(population_stability_index=[0.9, 0.05])}),
        writer=FakeDriftMetricsWriter(),
        audit=InMemoryAuditStore(),
        clock=FixedClock(INSTANT),
        retrain=trigger,
    ).execute(config=_config(), catalog="workspace", run_id="r", git_commit="a", git_branch="main")

    assert trigger.requests == [("exemplo", "propensao_exemplo", ["txn_count"])]


def test_the_measurement_survives_a_trigger_failure():
    """A medição é o produto do job e não pode se perder porque o GitHub estava
    fora do ar — mas o job não pode reportar sucesso, senão alguém veria
    DRIFT_DETECTED na tabela e assumiria que a esteira reagiu."""
    from mlplatform.testing import FakeRetrainTrigger

    writer, audit = FakeDriftMetricsWriter(), InMemoryAuditStore()

    with pytest.raises(RuntimeError):
        EvaluateDrift(
            runs=FakeTrainingRunReader([_training_run()]),
            monitor=FakeQualityMonitor(drift_table=DRIFT_TABLE),
            reader=FakeTableReader({DRIFT_TABLE: _metrics(population_stability_index=[0.9, 0.9])}),
            writer=writer,
            audit=audit,
            clock=FixedClock(INSTANT),
            retrain=FakeRetrainTrigger(fails=True),
        ).execute(
            config=_config(), catalog="workspace", run_id="r", git_commit="a", git_branch="main"
        )

    assert writer.written, "as métricas medidas foram gravadas"
    assert audit.statuses() == ["FAILED"], "mas o job não reporta sucesso"


def test_without_a_trigger_it_only_measures():
    """Quem ainda não quer retreino automático usa o componente sem gatilho."""
    results, _, audit, _ = _run(metrics=_metrics(population_stability_index=[0.9, 0.9]))

    assert {r.status for r in results} == {"DRIFT_DETECTED"}
    assert audit.statuses() == ["SUCCESS"]
