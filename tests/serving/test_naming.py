import pytest

from mlplatform.serving.naming import (
    derive_endpoint_name,
    derive_predictions_table_name,
    validate_endpoint_name,
)


def test_derive_predictions_table_name_follows_convention():
    name = derive_predictions_table_name(catalog="workspace", domain="credito", model_name="propensao_default")
    assert name == "workspace.credito_predictions.propensao_default"


def test_derive_endpoint_name_follows_convention():
    name = derive_endpoint_name(domain="credito", model_name="propensao_default")
    assert name == "credito-propensao_default-serving"


def test_validate_endpoint_name_accepts_convention():
    validate_endpoint_name("credito-propensao_default-serving")


def test_validate_endpoint_name_rejects_dots():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_endpoint_name("credito.propensao_default.serving")
