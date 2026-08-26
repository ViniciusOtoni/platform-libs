import pytest

from mlplatform.serving.contract import (
    BatchServingConfig,
    OnlineServingConfig,
    batch_configs,
    clear_registry,
    get_registry,
    get_serving_config,
    online_configs,
    register_serving_config,
)


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


def test_online_config_needs_no_batch_fields():
    config = OnlineServingConfig(domain="exemplo", model_name="modelo_a")

    assert config.alias == "champion"


def test_batch_fields_are_required_by_the_type_itself():
    """Antes eram opcionais num tipo único e a obrigatoriedade era checada em
    runtime pelo __post_init__. Com dois tipos, faltar um campo é TypeError na
    construção — o erro sai antes de qualquer execução."""
    with pytest.raises(TypeError):
        BatchServingConfig(domain="exemplo", model_name="modelo_batch")  # type: ignore[call-arg]


def test_register_and_get():
    config = _batch()
    register_serving_config(config)

    assert get_serving_config("modelo_batch") is config
    assert get_registry() == {"modelo_batch": config}


def test_register_rejects_duplicates():
    register_serving_config(_batch())

    with pytest.raises(ValueError, match="already registered"):
        register_serving_config(_batch())


def test_configs_are_partitioned_by_type_not_by_a_mode_field():
    """É isso que substitui o `if config.mode` que existia no gerador."""
    register_serving_config(_batch())
    register_serving_config(OnlineServingConfig(domain="exemplo", model_name="modelo_online"))

    assert list(batch_configs()) == ["modelo_batch"]
    assert list(online_configs()) == ["modelo_online"]
