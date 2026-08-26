import pytest

from training_platform.naming import derive_model_name, validate_model_name


def test_derive_model_name_follows_convention():
    name = derive_model_name(catalog="workspace", domain="credito", model_name="propensao_default")
    assert name == "workspace.credito_models.propensao_default"


def test_validate_model_name_accepts_convention():
    validate_model_name("workspace.credito_models.propensao_default")


def test_validate_model_name_rejects_uppercase():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_model_name("Workspace.Credito.Modelo")
