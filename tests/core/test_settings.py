import pytest

from mlplatform.core.settings import BundleSettings


def _write(tmp_path, body: str):
    p = tmp_path / "variables.yml"
    p.write_text(body, encoding="utf-8")
    return p


def test_reads_the_declared_values(tmp_path):
    path = _write(
        tmp_path,
        "catalog: producao\ndomain_package: exemplo_features\ndatabase_instance_name: lakebase-1\n",
    )

    settings = BundleSettings.load(path)

    assert settings.catalog == "producao"
    assert settings.domain_package == "exemplo_features"
    assert settings.database_instance_name == "lakebase-1"


def test_an_empty_file_falls_back_to_defaults(tmp_path):
    settings = BundleSettings.load(_write(tmp_path, ""))

    assert settings.catalog == "workspace"
    assert settings.database_instance_name == ""
    assert settings.targets == {"dev": {"mode": "development", "default": True}}


def test_an_unknown_key_fails_loudly(tmp_path):
    """O modo de falha mais irritante de config em YAML é a chave com typo
    ignorada em silêncio: o default entra no lugar e nada avisa. Aqui isso
    para o processo."""
    path = _write(tmp_path, "catalogo: producao\n")  # typo: deveria ser 'catalog'

    with pytest.raises(ValueError, match="chaves desconhecidas"):
        BundleSettings.load(path)


def test_a_non_mapping_document_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="esperado um mapeamento"):
        BundleSettings.load(_write(tmp_path, "- isto\n- e uma lista\n"))
