"""Testes dos use cases de serving.

Antes esta lógica vivia em ~60 linhas de notebook no repositório de domínio,
copiadas por domínio e sem teste nenhum.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from mlplatform.serving.contract import BatchServingConfig, OnlineServingConfig
from mlplatform.serving.usecases import PredictionsGateFailure, RefreshEndpoint, ScoreBatch
from mlplatform.testing import (
    FakeBatchScorer,
    FakeEndpointGateway,
    FakePredictionWriter,
    FixedClock,
    InMemoryAuditStore,
)

INSTANT = datetime(2026, 8, 26, 6, 0, 0, tzinfo=UTC)
SPINE = pd.DataFrame({"customer_id": [1, 2], "reference_date": ["2026-08-25"] * 2})


def _batch_config(**o) -> BatchServingConfig:
    return BatchServingConfig(
        **{
            "domain": "exemplo",
            "model_name": "propensao",
            "spine_inference_table": "workspace.exemplo.spine_inference",
            "schedule_cron": "0 0 6 * * ?",
            **o,
        }
    )


def _good_predictions() -> pd.DataFrame:
    return SPINE.assign(prediction=[0.1, 0.9])


def _run(scorer=None, writer=None, audit=None, config=None):
    audit = audit or InMemoryAuditStore()
    writer = writer or FakePredictionWriter()
    scorer = scorer or FakeBatchScorer(
        tables={"workspace.exemplo.spine_inference": SPINE}, predictions=_good_predictions()
    )
    ScoreBatch(scorer=scorer, writer=writer, audit=audit, clock=FixedClock(INSTANT)).execute(
        config=config or _batch_config(),
        catalog="workspace",
        run_id="run-1",
        git_commit="abc",
        git_branch="main",
    )
    return scorer, writer, audit


def test_scores_against_the_alias_of_the_registered_model():
    scorer, _, _ = _run()

    assert scorer.scored[0][0] == "models:/workspace.exemplo_models.propensao@champion"


def test_writes_predictions_to_the_derived_table_and_audits_success():
    _, writer, audit = _run()

    assert writer.appends[0][1] == "workspace.exemplo_predictions.propensao"
    assert audit.statuses() == ["SUCCESS"]


def test_a_failing_gate_audits_and_does_not_write():
    """Uma entidade sem correspondência no FeatureLookup recebe features nulas, e
    o modelo pode ainda assim devolver predição não-nula — por isso o gate olha as
    colunas juntadas, não só a de predição."""
    broken = SPINE.assign(prediction=[0.5, 0.5], txn_count=[1.0, None])
    writer, audit = FakePredictionWriter(), InMemoryAuditStore()

    with pytest.raises(PredictionsGateFailure) as exc:
        _run(
            scorer=FakeBatchScorer(
                tables={"workspace.exemplo.spine_inference": SPINE}, predictions=broken
            ),
            writer=writer,
            audit=audit,
        )

    assert "no_nulls_in_joined_columns" in [f.check for f in exc.value.findings if f.status == "FAIL"]
    assert writer.appends == []
    assert audit.statuses() == ["FAILED"]


def test_row_count_mismatch_fails_the_gate():
    """Perder linhas na junção é silencioso sem essa checagem."""
    with pytest.raises(PredictionsGateFailure):
        _run(
            scorer=FakeBatchScorer(
                tables={"workspace.exemplo.spine_inference": SPINE},
                predictions=_good_predictions().head(1),
            )
        )


def test_refresh_points_the_endpoint_at_the_current_alias():
    gateway = FakeEndpointGateway()
    config = OnlineServingConfig(domain="exemplo", model_name="propensao")

    name = RefreshEndpoint(gateway=gateway).execute(
        config=config, catalog="workspace", endpoint_name="dev_alguem_exemplo-propensao-serving"
    )

    # o nome usado é o que veio de fora, com o prefixo do target — não o derivado
    assert name == "dev_alguem_exemplo-propensao-serving"
    assert gateway.updates[0]["endpoint_name"] == "dev_alguem_exemplo-propensao-serving"
    assert gateway.updates[0]["full_model_name"] == "workspace.exemplo_models.propensao"
    assert gateway.updates[0]["alias"] == "champion"
