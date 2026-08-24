import pytest

from feature_platform.naming import derive_table_name, validate_table_name, resolve_table_name


def test_derive_table_name_follows_convention():
    name = derive_table_name(catalog="workspace", domain="credito", function_name="score_features")
    assert name == "workspace.credito_features.score_features"


def test_validate_table_name_accepts_convention():
    validate_table_name("workspace.credito_features.score_features")  # não deve levantar


def test_validate_table_name_rejects_uppercase():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_table_name("Workspace.Credito.Score")


def test_validate_table_name_rejects_wrong_number_of_parts():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_table_name("workspace.score_features")


def test_resolve_table_name_derives_when_none():
    name = resolve_table_name("workspace", "credito", "score_features", None)
    assert name == "workspace.credito_features.score_features"


def test_resolve_table_name_validates_explicit_override():
    name = resolve_table_name("workspace", "credito", "score_features", "workspace.legado.score_v1")
    assert name == "workspace.legado.score_v1"


def test_resolve_table_name_rejects_invalid_explicit_override():
    with pytest.raises(ValueError, match="does not match convention"):
        resolve_table_name("workspace", "credito", "score_features", "Invalid Name")
