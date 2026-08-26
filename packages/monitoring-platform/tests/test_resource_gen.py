import pytest

from monitoring_platform.contract import MonitoringConfig, clear_registry, register_monitoring_config
from monitoring_platform.resource_gen import generate_resources


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


def test_each_job_has_a_single_evaluate_drift_task():
    register_monitoring_config(_config())

    resources = generate_resources()
    job = list(resources["resources"]["jobs"].values())[0]

    assert [t["task_key"] for t in job["tasks"]] == ["evaluate_drift"]
    assert job["tasks"][0]["notebook_task"]["notebook_path"] == "../notebooks/evaluate_drift.py"


def test_job_parameters_carry_domain_model_and_target_type():
    register_monitoring_config(_config())

    resources = generate_resources()
    job = list(resources["resources"]["jobs"].values())[0]
    param_names = {p["name"] for p in job["parameters"]}

    assert param_names == {"domain", "model_name", "target_type", "catalog", "git_commit", "git_branch"}
