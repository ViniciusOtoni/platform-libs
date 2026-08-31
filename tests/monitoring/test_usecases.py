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
from mlplatform.monitoring.usecases import EvaluateDrift
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
    base = {
        "column_name": ["txn_count", "avg_ticket"],
        "drift_type": ["JENSEN_SHANNON", "JENSEN_SHANNON"],
        "statistic": [0.05, 0.05],
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


def test_a_metric_over_the_threshold_is_flagged():
    results, _, _, _ = _run(metrics=_metrics(statistic=[0.35, 0.05]))

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


def test_the_most_recent_measurement_wins():
    """A tabela acumula uma linha por refresh; drift é sobre a medição atual."""
    metrics = pd.DataFrame(
        {
            "column_name": ["txn_count", "txn_count"],
            "drift_type": ["JENSEN_SHANNON", "JENSEN_SHANNON"],
            "statistic": [0.9, 0.01],
        }
    )
    results, _, _, _ = _run(metrics=metrics, config=_config(columns=["txn_count"]))

    assert results[0].drift_metric_value == 0.01
    assert results[0].status == "PASS"


# -- escrita e auditoria ----------------------------------------------------


def test_the_measurements_land_in_the_central_table():
    _, writer, _, _ = _run()

    table, rows = writer.written[0]
    assert table == DRIFT_METRICS_TABLE
    assert {r["column_name"] for r in rows} == {"txn_count", "avg_ticket"}
    assert {r["entity_name"] for r in rows} == {_config().target_table}


def test_nothing_is_written_when_there_is_nothing_to_measure():
    """Gravar zero linha criaria uma escrita vazia por execução, e o histórico
    passaria a sugerir medições que não aconteceram."""
    _, writer, audit, _ = _run(metrics=_metrics(column_name=["outra", "mais_outra"]))

    assert writer.written == []
    assert audit.statuses() == ["SUCCESS"]


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
