import pytest

from training_platform.contract import (
    FeatureLookupSpec,
    TrainingConfig,
    clear_registry,
    get_registry,
    get_training_config,
    register_training_config,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _valid_config(**overrides):
    defaults = dict(
        domain="exemplo",
        model_name="modelo_exemplo",
        algorithm=object,
        hyperparameter_sets=[{}],
        feature_lookups=[
            FeatureLookupSpec(
                table_name="workspace.exemplo_features.customer_transaction_features",
                feature_names=["txn_count", "avg_ticket"],
                lookup_key="customer_id",
                timestamp_lookup_key="reference_date",
            )
        ],
        spine_table="workspace.exemplo.spine_train",
        label_column="label_default",
        reference_date_column="reference_date",
        train_pct=0.6,
        val_pct=0.2,
        test_pct=0.2,
        metric="roc_auc",
        metric_direction="maximize",
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_training_config_accepts_valid_split_percentages():
    config = _valid_config()
    assert config.train_pct == 0.6


def test_training_config_rejects_split_not_summing_to_one():
    with pytest.raises(ValueError, match="must equal 1.0"):
        _valid_config(train_pct=0.5, val_pct=0.3, test_pct=0.3)


def test_training_config_defaults_custom_transforms_and_pyfunc_class():
    config = _valid_config()
    assert config.custom_transforms == []
    assert config.pyfunc_model_class is None


def test_register_and_get_training_config():
    config = _valid_config()
    register_training_config(config)

    assert get_training_config("modelo_exemplo") is config
    assert get_registry() == {"modelo_exemplo": config}


def test_register_training_config_rejects_duplicate():
    register_training_config(_valid_config())

    with pytest.raises(ValueError, match="already registered"):
        register_training_config(_valid_config())
