import pytest

from mlplatform.monitoring.contract import MonitoringConfig, clear_registry, register_monitoring_config
from mlplatform.monitoring.resource_gen import generate_resources


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _config(**overrides):
    defaults = dict(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="feature_table",
        target_table="workspace.exemplo_features.customer_transaction_features",
        columns=["txn_count"],
        threshold=0.2,
        schedule_cron="0 0 7 * * ?",
    )
    defaults.update(overrides)
    return MonitoringConfig(**defaults)


def test_generates_one_job_per_config_with_its_own_schedule():
    register_monitoring_config(_config(target_type="feature_table"))
    register_monitoring_config(
        _config(
            target_type="predictions",
            target_table="workspace.exemplo_predictions.propensao_exemplo",
            schedule_cron="0 0 8 * * ?",
        )
    )

    resources = generate_resources()
    jobs = resources["resources"]["jobs"]

    assert len(jobs) == 2
    feature_job = jobs["drift_check_exemplo_propensao_exemplo_feature_table"]
    predictions_job = jobs["drift_check_exemplo_propensao_exemplo_predictions"]
    assert feature_job["schedule"]["quartz_cron_expression"] == "0 0 7 * * ?"
    assert predictions_job["schedule"]["quartz_cron_expression"] == "0 0 8 * * ?"


def test_each_job_runs_the_entry_point_not_a_notebook():
    """Era o último componente em `notebook_task`, o que obrigava o repositório
    de domínio a versionar `notebooks/evaluate_drift.py`."""
    register_monitoring_config(_config())

    job = list(generate_resources()["resources"]["jobs"].values())[0]

    assert [t["task_key"] for t in job["tasks"]] == ["evaluate_drift"]
    assert "notebook_task" not in job["tasks"][0]
    assert job["tasks"][0]["python_wheel_task"]["entry_point"] == "mlp-evaluate-drift"


def test_the_config_is_identified_statically_not_by_a_parameter():
    """`model_name` e `target_type` iam como job parameters, e o entrypoint os
    exige. Como este job SÓ roda por agendamento, e cron não preenche
    parâmetro, ele morreria antes de tocar em dado nenhum."""
    register_monitoring_config(_config())

    job = list(generate_resources(domain_entry_point="exemplo_monitoring")["resources"]["jobs"].values())[0]
    named = job["tasks"][0]["python_wheel_task"]["named_parameters"]

    assert named == {
        "model_name": "propensao_exemplo",
        "target_type": "feature_table",
        "domain": "exemplo_monitoring",
    }
    assert all(p["default"] for p in job["parameters"])


def test_an_empty_registry_is_an_error_not_an_empty_bundle():
    with pytest.raises(ValueError, match="nenhuma MonitoringConfig"):
        generate_resources()


def test_environment_dependencies_declared_when_given():
    register_monitoring_config(_config())
    deps = ["./dist/*.whl", "https://example.com/mlplatform-1.0.0.whl"]

    resources = generate_resources(environment_dependencies=deps)
    job = list(resources["resources"]["jobs"].values())[0]

    assert job["environments"] == [{"environment_key": "default", "spec": {"client": "3", "dependencies": deps}}]
    assert job["tasks"][0]["environment_key"] == "default"


def test_without_environment_dependencies_falls_back_to_the_defaults():
    """Sem environment o job não instala nada e o entry point não existe: o
    default do framework é o wheel do domínio mais o do próprio framework."""
    register_monitoring_config(_config())

    job = list(generate_resources()["resources"]["jobs"].values())[0]

    assert job["environments"][0]["spec"]["dependencies"]
    assert job["tasks"][0]["environment_key"] == "default"
