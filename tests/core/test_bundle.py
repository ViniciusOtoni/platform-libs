from mlplatform.core.bundle import bundle_name, generate_bundle
from mlplatform.core.settings import BundleSettings


def _bundle(**overrides):
    settings = BundleSettings(**overrides)
    return generate_bundle(settings, domain="exemplo", component="features", wheel_name="exemplo_features")


def test_bundle_name_joins_domain_and_component():
    assert bundle_name("exemplo", "features") == "exemplo-features"


def test_includes_the_generated_resources_directory():
    assert _bundle()["include"] == ["resources/*.yml"]


def test_declares_the_domain_wheel_as_an_artifact():
    """A CLI builda e sobe o wheel do domínio como parte do deploy — sem step
    manual, sem venv empacotada."""
    artifacts = _bundle()["artifacts"]

    assert artifacts["exemplo_features"]["type"] == "whl"
    assert artifacts["exemplo_features"]["build"] == "python -m build --wheel"


def test_catalog_default_comes_from_the_declared_settings():
    assert _bundle(catalog="producao")["variables"]["catalog"]["default"] == "producao"


def test_git_provenance_variables_default_to_local():
    """A esteira sobrescreve com --var; o default cobre execução local."""
    variables = _bundle()["variables"]

    assert variables["git_commit"]["default"] == "local"
    assert variables["git_branch"]["default"] == "local"


def test_lakebase_variable_only_appears_when_configured():
    """Declarar sempre criaria uma variável vazia em todo bundle, inclusive nos
    que não têm nenhuma feature table online."""
    assert "database_instance_name" not in _bundle()["variables"]
    assert (
        _bundle(database_instance_name="lakebase-1")["variables"]["database_instance_name"]["default"]
        == "lakebase-1"
    )


def test_every_variable_carries_a_description():
    """As descrições ficam com o framework: são explicação de um mecanismo da
    plataforma, e repetidas à mão em cada bundle acabavam divergindo."""
    variables = _bundle(database_instance_name="lakebase-1")["variables"]

    assert all(v["description"] for v in variables.values())
