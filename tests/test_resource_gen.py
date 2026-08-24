import pytest

from serving_platform.contract import ServingConfig, register_serving_config, clear_registry
from serving_platform.resource_gen import generate_resources


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_batch_config_generates_a_scheduled_job_with_one_task():
    register_serving_config(
        ServingConfig(
            domain="exemplo",
            model_name="modelo_batch",
            mode="batch",
            spine_inference_table="workspace.exemplo.spine_inference",
            schedule_cron="0 0 6 * * ?",
        )
    )

    resources = generate_resources()
    jobs = resources["resources"]["jobs"]

    job = jobs["score_batch_modelo_batch"]
    assert job["schedule"]["quartz_cron_expression"] == "0 0 6 * * ?"
    assert [t["task_key"] for t in job["tasks"]] == ["score_batch"]
    assert job["tasks"][0]["notebook_task"]["notebook_path"] == "../notebooks/score_batch.py"


def test_online_config_generates_a_model_serving_endpoint():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_online", mode="online"))

    resources = generate_resources()
    endpoints = resources["resources"]["model_serving_endpoints"]

    endpoint = endpoints["exemplo-modelo_online-serving"]
    served_entity = endpoint["config"]["served_entities"][0]
    assert served_entity["entity_name"] == "${var.catalog}.exemplo_models.modelo_online@champion"


def test_generate_resources_always_includes_refresh_endpoint_job():
    resources = generate_resources()
    assert "refresh_endpoint" in resources["resources"]["jobs"]


def test_generate_resources_omits_empty_resource_kinds():
    register_serving_config(
        ServingConfig(
            domain="exemplo",
            model_name="modelo_batch",
            mode="batch",
            spine_inference_table="workspace.exemplo.spine_inference",
            schedule_cron="0 0 6 * * ?",
        )
    )

    resources = generate_resources()
    assert "model_serving_endpoints" not in resources["resources"]
