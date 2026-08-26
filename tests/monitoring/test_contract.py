import pytest

from mlplatform.monitoring.contract import (
    MonitoringConfig,
    clear_registry,
    get_monitoring_config,
    get_registry,
    register_monitoring_config,
)


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
        columns=["txn_count", "avg_ticket"],
        threshold=0.2,
        schedule_cron="0 0 7 * * ?",
    )
    defaults.update(overrides)
    return MonitoringConfig(**defaults)


def test_register_and_get_monitoring_config():
    config = _config()
    register_monitoring_config(config)

    assert get_monitoring_config("exemplo", "propensao_exemplo", "feature_table") is config


def test_same_model_can_have_feature_table_and_predictions_configs():
    register_monitoring_config(_config(target_type="feature_table"))
    register_monitoring_config(
        _config(target_type="predictions", target_table="workspace.exemplo_predictions.propensao_exemplo")
    )

    assert len(get_registry()) == 2


def test_register_monitoring_config_rejects_duplicate_key():
    register_monitoring_config(_config())

    with pytest.raises(ValueError, match="already registered"):
        register_monitoring_config(_config())
