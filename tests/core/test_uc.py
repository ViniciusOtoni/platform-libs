"""Criação de schema e concessão de acesso no Unity Catalog."""

import pytest

from mlplatform.core.uc import ensure_schema, grant_statements


class _FakeSpark:
    """Registra o SQL emitido, sem executar nada."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def sql(self, statement: str) -> None:
        self.statements.append(statement)


def test_without_a_group_it_only_creates_the_schema():
    """É o caso do workspace pessoal, onde não existe grupo para conceder."""
    spark = _FakeSpark()

    ensure_schema(spark, "workspace.exemplo_features")

    assert spark.statements == ["CREATE SCHEMA IF NOT EXISTS workspace.exemplo_features"]


def test_the_catalog_and_schema_are_made_usable_before_the_read_grants():
    """USE CATALOG e USE SCHEMA são pré-requisitos: sem eles o SELECT existe mas
    o objeto continua invisível — o modo de falha mais confuso do UC."""
    spark = _FakeSpark()

    ensure_schema(spark, "workspace.exemplo_features", "time-exemplo")

    assert spark.statements[0].startswith("CREATE SCHEMA")
    assert spark.statements[1] == "GRANT USE CATALOG ON CATALOG workspace TO `time-exemplo`"
    assert spark.statements[2] == "GRANT USE SCHEMA ON SCHEMA workspace.exemplo_features TO `time-exemplo`"


def test_both_tables_and_models_are_granted():
    """No mesmo schema convivem tabelas (SELECT) e modelos registrados
    (EXECUTE). Conceder só um deixa metade dos objetos invisível."""
    spark = _FakeSpark()

    ensure_schema(spark, "workspace.exemplo_models", "time-exemplo")

    granted = {s.split()[1] for s in spark.statements if s.startswith("GRANT")}
    assert {"SELECT", "EXECUTE"} <= granted


def test_the_grant_is_on_the_schema_not_on_each_object():
    """Os objetos de um domínio nascem ao longo do tempo — feature table nova,
    versão nova de modelo, a tabela que o monitor cria sozinho. Conceder por
    objeto exigiria lembrar a cada criação, e o esquecimento é silencioso."""
    statements = grant_statements("workspace.exemplo_predictions", "time-exemplo")

    assert all("ON SCHEMA" in s or "ON CATALOG" in s for s in statements)
    assert not any("ON TABLE" in s for s in statements)


@pytest.mark.parametrize("group", ["", None])
def test_an_empty_group_grants_nothing(group):
    spark = _FakeSpark()

    ensure_schema(spark, "workspace.exemplo_features", group)

    assert not [s for s in spark.statements if s.startswith("GRANT")]
