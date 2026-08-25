import pytest

from serving_platform.contract import (
    ServingConfig,
    register_serving_config,
    get_serving_config,
    get_registry,
    clear_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_online_config_does_not_require_batch_fields():
    config = ServingConfig(domain="exemplo", model_name="modelo_a", mode="online")
    assert config.alias == "champion"
    assert config.spine_inference_table is None


def test_batch_config_requires_spine_and_schedule():
    with pytest.raises(ValueError, match="requires spine_inference_table and schedule_cron"):
        ServingConfig(domain="exemplo", model_name="modelo_b", mode="batch")


def test_batch_config_accepts_required_fields():
    config = ServingConfig(
        domain="exemplo",
        model_name="modelo_b",
        mode="batch",
        spine_inference_table="workspace.exemplo.spine_inference",
        schedule_cron="0 0 6 * * ?",
    )
    assert config.schedule_cron == "0 0 6 * * ?"


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        ServingConfig(domain="exemplo", model_name="modelo_c", mode="streaming")


def test_config_accepts_custom_alias():
    config = ServingConfig(domain="exemplo", model_name="modelo_d", mode="online", alias="challenger")
    assert config.alias == "challenger"


def test_register_and_get_serving_config():
    config = ServingConfig(domain="exemplo", model_name="modelo_e", mode="online")
    register_serving_config(config)

    assert get_serving_config("modelo_e") is config
    assert get_registry() == {"modelo_e": config}


def test_register_serving_config_rejects_duplicate():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_f", mode="online"))

    with pytest.raises(ValueError, match="already registered"):
        register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_f", mode="online"))
