import pytest

from platform_core.naming import validate_qualified_name


def test_accepts_convention():
    validate_qualified_name("workspace.credito_features.score_features")  # não deve levantar


def test_rejects_uppercase():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_qualified_name("Workspace.Credito.Score")


def test_rejects_wrong_number_of_parts():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_qualified_name("workspace.score_features")


def test_error_message_includes_kind():
    with pytest.raises(ValueError, match="table_name 'Invalid' does not match"):
        validate_qualified_name("Invalid", kind="table_name")
