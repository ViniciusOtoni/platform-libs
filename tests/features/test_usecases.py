"""Testes do pipeline completo de feature table.

Antes dos ports isso era impossível: `run_feature_table` recebia uma
SparkSession e o único teste que existia cobria a resolução de janela com
monkeypatch. O corpo do pipeline — gate, escrita, auditoria, sync online —
carregava o comentário "exercitado via notebook, não via pytest". Aqui ele roda
inteiro com fakes.
"""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from mlplatform.core.audit import RunRecord
from mlplatform.features.contract import FeatureTableSpec
from mlplatform.features.modes import WriteMode
from mlplatform.features.usecases import COMPONENT, QualityGateFailure, RunFeatureTable
from mlplatform.features.window import NoCheckpointError
from mlplatform.testing import (
    FakeFeatureWriter,
    FakeOnlineStore,
    FakeSourceReader,
    FixedClock,
    InMemoryAuditStore,
)

INSTANT = datetime(2026, 1, 15, 3, 0, 0, tzinfo=UTC)


def _valid_frame() -> pd.DataFrame:
    """Um resultado que passa nos quatro checks do gate."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2],
            "feature_ts": [date(2026, 1, 15), date(2026, 1, 15)],
            "txn_count": [3, 5],
        }
    )


def _spec(**overrides) -> FeatureTableSpec:
    defaults = dict(
        name="minha_feature",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.transactions"],
        compute_fn=lambda sources, window: _valid_frame(),
        domain="exemplo",
    )
    defaults.update(overrides)
    return FeatureTableSpec(**defaults)


def _usecase(audit=None, online=None, reader=None, writer=None):
    return (
        RunFeatureTable(
            reader=reader or FakeSourceReader({"raw.transactions": _valid_frame()}),
            writer=writer or FakeFeatureWriter(),
            audit=audit or InMemoryAuditStore(),
            clock=FixedClock(INSTANT),
            online=online,
        ),
    )[0]


def _execute(usecase, spec=None, **overrides):
    kwargs = dict(
        spec=spec or _spec(),
        catalog="workspace",
        mode=WriteMode.BACKFILL,
        today=date(2026, 1, 20),
        run_id="run-1",
        git_commit="abc123",
        git_branch="main",
        backfill_start="2026-01-01",
        backfill_end="2026-01-15",
    )
    kwargs.update(overrides)
    return usecase.execute(**kwargs)


# --------------------------------------------------------------------------
# Resolução de janela — antes feita com monkeypatch, agora com o fake de audit
# --------------------------------------------------------------------------


def test_backfill_uses_the_explicit_range():
    writer = FakeFeatureWriter()
    _execute(_usecase(writer=writer))

    # a janela chega ao gate; a prova indireta é a escrita ter acontecido
    assert len(writer.writes) == 1


def test_backfill_requires_both_dates():
    with pytest.raises(ValueError, match="backfill mode requires"):
        _execute(_usecase(), backfill_start=None, backfill_end=None)


def test_incremental_reads_the_checkpoint_from_the_audit_store():
    audit = InMemoryAuditStore(checkpoints={(COMPONENT, "minha_feature"): date(2026, 1, 10)})
    writer = FakeFeatureWriter()

    _execute(
        _usecase(audit=audit, writer=writer),
        mode=WriteMode.INCREMENTAL,
        today=date(2026, 1, 15),
        backfill_start=None,
        backfill_end=None,
    )

    assert writer.writes[0]["mode"] == WriteMode.INCREMENTAL


def test_incremental_without_checkpoint_refuses_to_run():
    """Sem run SUCCESS anterior não há de onde continuar — tem que falhar alto,
    não processar a janela inteira por acidente."""
    with pytest.raises(NoCheckpointError):
        _execute(
            _usecase(audit=InMemoryAuditStore()),
            mode=WriteMode.INCREMENTAL,
            today=date(2026, 1, 15),
            backfill_start=None,
            backfill_end=None,
        )


# --------------------------------------------------------------------------
# O pipeline em si — o que antes só rodava em notebook
# --------------------------------------------------------------------------


def test_happy_path_writes_tags_and_audits_success():
    audit, writer = InMemoryAuditStore(), FakeFeatureWriter()

    _execute(_usecase(audit=audit, writer=writer))

    write = writer.writes[0]
    assert write["table_name"] == "workspace.exemplo_features.minha_feature"
    assert write["entity_keys"] == ["customer_id"]
    assert writer.provenance == [("workspace.exemplo_features.minha_feature", "abc123", "main")]
    assert audit.statuses() == ["SUCCESS"]


def test_reads_every_declared_source():
    reader = FakeSourceReader({"raw.a": _valid_frame(), "raw.b": _valid_frame()})

    _execute(_usecase(reader=reader), spec=_spec(sources=["raw.a", "raw.b"]))

    assert reader.read_tables == ["raw.a", "raw.b"]


def test_quality_gate_failure_audits_failed_and_does_not_write():
    """A ordem importa: auditar a falha ANTES de levantar, senão a execução
    desaparece do registro e o próximo run incremental não sabe que houve falha."""
    audit, writer = InMemoryAuditStore(), FakeFeatureWriter()
    broken = pd.DataFrame({"customer_id": [1, 1], "feature_ts": [date(2026, 1, 15)] * 2})

    with pytest.raises(QualityGateFailure) as exc:
        _execute(
            _usecase(audit=audit, writer=writer),
            spec=_spec(compute_fn=lambda sources, window: broken),
        )

    assert [f.check for f in exc.value.findings if f.status == "FAIL"] == ["unique_keys"]
    assert audit.statuses() == ["FAILED"]
    assert writer.writes == []


def test_online_table_syncs_and_enables_cdf():
    online, writer = FakeOnlineStore(), FakeFeatureWriter()

    _execute(_usecase(online=online, writer=writer), spec=_spec(online=True), database_instance_name="lakebase-1")

    assert writer.writes[0]["enable_cdf"] is True
    assert online.syncs == [("workspace.exemplo_features.minha_feature", ["customer_id"], "lakebase-1")]


def test_offline_table_does_not_sync():
    online, writer = FakeOnlineStore(), FakeFeatureWriter()

    _execute(_usecase(online=online, writer=writer))

    assert writer.writes[0]["enable_cdf"] is False
    assert online.syncs == []


def test_online_true_without_an_online_store_fails_loudly():
    """Falhar aqui é melhor que gravar a tabela e sair sem sincronizar — o
    domínio pediu online e ficaria sem, silenciosamente."""
    with pytest.raises(ValueError, match="online=True"):
        _execute(_usecase(online=None), spec=_spec(online=True))


def test_partition_cols_default_to_the_first_entity_key():
    writer = FakeFeatureWriter()
    # chave composta: o frame precisa carregar as duas colunas, senão o gate
    # reprova por schema antes de a escrita acontecer
    frame = _valid_frame().assign(region=["sul", "norte"])

    _execute(
        _usecase(writer=writer),
        spec=_spec(entity_keys=["customer_id", "region"], compute_fn=lambda sources, window: frame),
    )

    assert writer.writes[0]["partition_cols"] == ["customer_id"]


def test_partition_cols_can_be_declared_on_the_spec():
    writer = FakeFeatureWriter()

    _execute(_usecase(writer=writer), spec=_spec(partition_by=["region"]))

    assert writer.writes[0]["partition_cols"] == ["region"]


def test_the_whole_run_carries_a_single_timestamp():
    """Antes cada ponto de escrita chamava utcnow() por conta própria, então
    caminho de falha e de sucesso carimbavam instantes diferentes."""
    audit = InMemoryAuditStore()

    _execute(_usecase(audit=audit))

    assert [r.run_ts for r in audit.records] == [INSTANT]


def test_audit_record_carries_the_provenance_of_the_run():
    audit = InMemoryAuditStore()

    _execute(_usecase(audit=audit))

    record = audit.records[0]
    assert isinstance(record, RunRecord)
    assert (record.component, record.entity_name) == (COMPONENT, "minha_feature")
    assert (record.git_commit, record.git_branch, record.run_id) == ("abc123", "main", "run-1")
    assert (record.window_start, record.window_end) == (date(2026, 1, 1), date(2026, 1, 15))
