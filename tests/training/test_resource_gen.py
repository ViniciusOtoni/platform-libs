"""O gerador do job de treino — que até aqui não tinha teste nenhum.

Era o único componente sem gerador, e virou o único gerador sem cobertura. Os
dois defeitos abaixo vieram daí e só apareceram executando o job no workspace.
"""

import pytest
from sklearn.dummy import DummyClassifier

from mlplatform.training.contract import (
    FeatureLookupSpec,
    TrainingConfig,
    clear_registry,
    register_training_config,
)
from mlplatform.training.resource_gen import generate_job_resource


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _config(model_name: str = "propensao") -> TrainingConfig:
    return TrainingConfig(
        domain="exemplo",
        model_name=model_name,
        algorithm=DummyClassifier,
        hyperparameter_sets=[{"strategy": "prior"}],
        feature_lookups=[
            FeatureLookupSpec(
                table_name="workspace.exemplo_features.f",
                feature_names=["x"],
                lookup_key="customer_id",
                timestamp_lookup_key="reference_date",
            )
        ],
        spine_table="workspace.exemplo.spine",
        label_column="label",
        reference_date_column="reference_date",
        train_pct=0.6,
        val_pct=0.2,
        test_pct=0.2,
        metric="roc_auc",
        metric_direction="maximize",
    )


def _jobs(**kwargs) -> dict:
    return generate_job_resource(environment_dependencies=[], **kwargs)["resources"]["jobs"]


def test_no_parameter_is_left_for_the_user_to_fill_in():
    """`model_name` era um job parameter com default vazio, e os entrypoints o
    exigem. Qualquer execução agendada — ou um clique em Run now — morria com
    `KeyError: ''` antes de tocar em dado nenhum."""
    register_training_config(_config())

    job = _jobs()["training_pipeline"]

    assert all(p["default"] for p in job["parameters"])
    assert "model_name" not in {p["name"] for p in job["parameters"]}


def test_every_task_carries_the_model_name():
    register_training_config(_config())

    job = _jobs()["training_pipeline"]

    for task in job["tasks"]:
        assert task["python_wheel_task"]["named_parameters"]["model_name"] == "propensao"


def test_a_second_model_gets_its_own_job():
    """Com um job só e o modelo vindo por parâmetro, o segundo modelo do
    domínio simplesmente não tinha job."""
    register_training_config(_config("propensao"))
    register_training_config(_config("churn"))

    jobs = _jobs()

    assert set(jobs) == {"training_pipeline_propensao", "training_pipeline_churn"}
    assert jobs["training_pipeline_churn"]["tasks"][0]["python_wheel_task"]["named_parameters"][
        "model_name"
    ] == "churn"


def test_the_three_tasks_run_in_order():
    register_training_config(_config())

    tasks = _jobs()["training_pipeline"]["tasks"]

    assert [t["task_key"] for t in tasks] == [
        "prepare_training_set",
        "fit_and_compare",
        "select_test_and_register",
    ]
    assert tasks[1]["depends_on"] == [{"task_key": "prepare_training_set"}]
    assert tasks[2]["depends_on"] == [{"task_key": "fit_and_compare"}]


def test_an_empty_registry_is_an_error_not_an_empty_bundle():
    with pytest.raises(ValueError, match="nenhuma TrainingConfig"):
        _jobs()
