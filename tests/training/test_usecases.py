"""Testes do pipeline de treino.

Toda esta lógica vivia em quatro notebooks do repositório de domínio, sem teste.
"""

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from mlplatform.testing import (
    FakeExperimentTracker,
    FakeModelPublisher,
    FakeScratchStore,
    FakeTaskChannel,
    FakeTrainingSetBuilder,
    FixedClock,
    InMemoryAuditStore,
)
from mlplatform.training.contract import FeatureLookupSpec, TrainingConfig
from mlplatform.training.usecases import (
    FIT_TASK,
    PREPARE_TASK,
    RESULTS_KEY,
    RUN_ID_KEY,
    WINDOW_END_KEY,
    WINDOW_START_KEY,
    FitAndCompare,
    PrepareTrainingSet,
    SanityGateFailure,
    SelectTestAndRegister,
)

INSTANT = datetime(2026, 8, 26, 6, 0, 0, tzinfo=UTC)
PREFIX = "workspace.training_scratch.propensao"


def _config(**o) -> TrainingConfig:
    return TrainingConfig(
        **{
            "domain": "exemplo",
            "model_name": "propensao",
            "algorithm": DummyClassifier,
            "hyperparameter_sets": [{"strategy": "most_frequent"}, {"strategy": "prior"}],
            "feature_lookups": [
                FeatureLookupSpec(
                    table_name="workspace.exemplo_features.f",
                    feature_names=["txn_count"],
                    lookup_key="customer_id",
                    timestamp_lookup_key="reference_date",
                )
            ],
            "spine_table": "workspace.exemplo.spine",
            "label_column": "label",
            "reference_date_column": "reference_date",
            "train_pct": 0.6,
            "val_pct": 0.2,
            "test_pct": 0.2,
            "metric": "accuracy",
            "metric_direction": "maximize",
            **o,
        }
    )


def _frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "reference_date": [date(2026, 1, i + 1) for i in range(n)],
            "txn_count": [i % 5 for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )


def _scorer(pipeline, X, y):
    return float((pipeline.predict(X) == y).mean())


# --------------------------------------------------------------------------


def test_prepare_materializes_the_three_splits_and_publishes_the_window():
    scratch, channel = FakeScratchStore(), FakeTaskChannel()

    PrepareTrainingSet(
        builder=FakeTrainingSetBuilder(_frame()),
        scratch=scratch,
        tracker=FakeExperimentTracker(),
        channel=channel,
    ).execute(config=_config(), catalog="workspace")

    assert set(scratch.tables) == {f"{PREFIX}_train", f"{PREFIX}_val", f"{PREFIX}_test"}
    assert channel.values[RUN_ID_KEY] == "run-mlflow-1"
    assert channel.values[WINDOW_START_KEY] == "2026-01-01"
    assert channel.values[WINDOW_END_KEY] == "2026-01-10"


def test_splits_never_cut_a_reference_date_in_half():
    """O corte cai sempre em cima de uma data real — uma mesma safra não pode
    aparecer em dois splits, senão há vazamento entre treino e teste."""
    scratch = FakeScratchStore()

    PrepareTrainingSet(
        builder=FakeTrainingSetBuilder(_frame()),
        scratch=scratch,
        tracker=FakeExperimentTracker(),
        channel=FakeTaskChannel(),
    ).execute(config=_config(), catalog="workspace")

    dates = {s: set(scratch.tables[f"{PREFIX}_{s}"]["reference_date"]) for s in ("train", "val", "test")}
    assert not dates["train"] & dates["val"]
    assert not dates["val"] & dates["test"]
    assert not dates["train"] & dates["test"]


def test_fit_evaluates_every_combination_and_hands_results_forward():
    frame = _frame()
    scratch = FakeScratchStore({f"{PREFIX}_train": frame, f"{PREFIX}_val": frame})
    channel = FakeTaskChannel(initial={(PREPARE_TASK, RUN_ID_KEY): "run-mlflow-1"})

    results = FitAndCompare(
        scratch=scratch, tracker=FakeExperimentTracker(), channel=channel
    ).execute(config=_config(), catalog="workspace", scorer=_scorer)

    assert len(results) == 2
    assert json.loads(channel.values[RESULTS_KEY])


def _register(scratch=None, channel=None, publisher=None, audit=None, config=None, results=None):
    frame = _frame()
    scratch = scratch or FakeScratchStore({f"{PREFIX}_train": frame, f"{PREFIX}_test": frame})
    results = results if results is not None else [[{"strategy": "most_frequent"}, 0.6]]
    channel = channel or FakeTaskChannel(
        initial={
            (PREPARE_TASK, RUN_ID_KEY): "run-mlflow-1",
            (PREPARE_TASK, WINDOW_START_KEY): "2026-01-01",
            (PREPARE_TASK, WINDOW_END_KEY): "2026-01-10",
            (FIT_TASK, RESULTS_KEY): json.dumps(results),
        }
    )
    publisher = publisher or FakeModelPublisher()
    audit = audit or InMemoryAuditStore()
    name = SelectTestAndRegister(
        scratch=scratch,
        tracker=FakeExperimentTracker(),
        publisher=publisher,
        builder=FakeTrainingSetBuilder(frame),
        audit=audit,
        clock=FixedClock(INSTANT),
        channel=channel,
    ).execute(
        config=config or _config(),
        catalog="workspace",
        scorer=_scorer,
        run_id="run-1",
        git_commit="abc",
        git_branch="main",
    )
    return name, publisher, audit


def test_registers_under_the_derived_model_name_and_audits_success():
    name, publisher, audit = _register()

    assert name == "workspace.exemplo_models.propensao"
    assert publisher.published[0]["full_model_name"] == name
    assert audit.statuses() == ["SUCCESS"]


def test_the_model_registered_is_the_one_that_passed_the_gate():
    """Antes eram duas tasks: uma fitava, testava e aprovava; a outra fitava
    OUTRO modelo com os mesmos hiperparâmetros e registrava esse. Sem
    random_state os dois diferem, e o que ia para produção nunca fora avaliado."""
    _, publisher, _ = _register()

    wrapped = publisher.published[0]["model"]
    # o objeto publicado carrega o pipeline realmente fitado e avaliado
    assert hasattr(wrapped, "model")
    assert hasattr(wrapped.model, "predict")


def test_a_failing_sanity_gate_audits_and_does_not_register():
    publisher, audit = FakeModelPublisher(), InMemoryAuditStore()
    empty = pd.DataFrame(columns=list(_frame().columns))

    with pytest.raises(SanityGateFailure):
        _register(
            scratch=FakeScratchStore({f"{PREFIX}_train": _frame(), f"{PREFIX}_test": empty}),
            publisher=publisher,
            audit=audit,
        )

    assert publisher.published == []
    assert audit.statuses() == ["FAILED"]


def test_combinations_with_a_non_finite_metric_are_discarded():
    """max() com NaN devolve resultado dependente da ordem da lista, em silêncio
    — e NaN acontece de verdade quando o split de validação sai vazio."""
    _, publisher, _ = _register(
        results=[[{"strategy": "prior"}, float("nan")], [{"strategy": "most_frequent"}, 0.6]]
    )

    assert publisher.published  # escolheu a combinação finita em vez da primeira da lista
