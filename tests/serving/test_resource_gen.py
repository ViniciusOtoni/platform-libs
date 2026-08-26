import pytest

from mlplatform.serving.contract import (
    BatchServingConfig,
    OnlineServingConfig,
    clear_registry,
    register_serving_config,
)
from mlplatform.serving.resource_gen import generate_resources


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _batch(**o) -> BatchServingConfig:
    return BatchServingConfig(
        **{
            "domain": "exemplo",
            "model_name": "modelo_batch",
            "spine_inference_table": "workspace.exemplo.spine",
            "schedule_cron": "0 0 6 * * ?",
            **o,
        }
    )


def _gen(**o):
    return generate_resources(**{"catalog": "workspace", **o})


def test_batch_config_generates_a_scheduled_job_calling_the_entry_point():
    register_serving_config(_batch())

    job = _gen()["resources"]["jobs"]["score_batch_modelo_batch"]

    assert job["schedule"]["quartz_cron_expression"] == "0 0 6 * * ?"
    wheel = job["tasks"][0]["python_wheel_task"]
    assert (wheel["package_name"], wheel["entry_point"]) == ("mlplatform", "mlp-score-batch")
    assert wheel["named_parameters"]["model_name"] == "modelo_batch"


def test_online_config_generates_an_endpoint_with_a_numeric_version():
    """DABs recusa `models:/nome@alias` — só aceita entity_version numérico."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    endpoints = _gen(resolve_alias_version=lambda full, alias: 3)["resources"]["model_serving_endpoints"]

    served = endpoints["exemplo-modelo_online-serving"]["config"]["served_entities"][0]
    assert served["entity_name"] == "${var.catalog}.exemplo_models.modelo_online"
    assert served["entity_version"] == "3"


def test_the_resolver_receives_the_real_catalog_not_a_placeholder():
    """O domínio carregava um CATALOG hardcodado, com um comentário pedindo que
    batesse com o default do databricks.yml — e nada verificava se batia."""
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))
    seen = []

    _gen(catalog="producao", resolve_alias_version=lambda full, alias: seen.append(full) or 1)

    assert seen == ["producao.exemplo_models.modelo_online"]


def test_online_without_a_resolver_fails_with_the_reason():
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    with pytest.raises(ValueError, match="entity_version"):
        _gen()


def test_refresh_endpoint_job_always_exists():
    assert "refresh_endpoint" in _gen()["resources"]["jobs"]


def test_endpoints_are_omitted_when_there_is_no_online_config():
    register_serving_config(_batch())

    assert "model_serving_endpoints" not in _gen()["resources"]


def test_every_job_carries_the_serverless_environment():
    register_serving_config(_batch())

    for job in _gen()["resources"]["jobs"].values():
        assert job["environments"][0]["spec"]["client"] == "3"
        assert all(t["environment_key"] == "default" for t in job["tasks"])
