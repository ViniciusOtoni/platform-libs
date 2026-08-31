from mlplatform.core.bundle import bundle_name, generate_bundle
from mlplatform.core.settings import BundleSettings


def _bundle(**overrides):
    settings = BundleSettings(**overrides)
    return generate_bundle(settings, domain="exemplo", component="features", wheel_name="exemplo_features")


def test_bundle_name_falls_back_to_domain_and_component():
    assert bundle_name("exemplo", "features") == "exemplo-features"


def test_bundle_name_prefers_the_domain_package_because_it_is_unique():
    """domínio+componente não é único: serving tem batch e online, e os dois
    virariam "exemplo-serving" — colidindo no mesmo caminho do workspace, um
    sobrescrevendo o outro no deploy."""
    assert bundle_name("exemplo", "serving", "exemplo_serving_batch") == "exemplo-serving-batch"
    assert bundle_name("exemplo", "serving", "exemplo_serving_online") == "exemplo-serving-online"


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


def test_the_domain_group_can_run_but_not_edit_the_resources():
    """A definição do job vem do git. Poder editar pela UI criaria divergência
    silenciosa com o versionado, e o próximo deploy sobrescreveria a edição.

    O nível também precisa ser um que o DABs aceite: `CAN_MANAGE_RUN` existe na
    API de jobs mas não no bundle, e o `bundle validate` reprova."""
    from mlplatform.core.bundle import generate_bundle
    from mlplatform.core.settings import BundleSettings

    bundle = generate_bundle(
        BundleSettings(catalog="workspace", domain_package="exemplo_x", reader_group="time-exemplo"),
        domain="exemplo",
        component="training",
        wheel_name="exemplo_x",
    )

    assert bundle["permissions"] == [
        {"user_name": "${workspace.current_user.userName}", "level": "CAN_MANAGE"},
        {"group_name": "time-exemplo", "level": "CAN_RUN"},
    ]


def test_the_deploying_identity_is_named_explicitly():
    """O `bundle validate` avisa quando a identidade do deploy não está no
    bloco: CAN_MANAGE só se aplica se o deploy sair dela. Em CI ele sai de um
    service principal, não de quem escreveu o bundle."""
    from mlplatform.core.bundle import generate_bundle
    from mlplatform.core.settings import BundleSettings

    bundle = generate_bundle(
        BundleSettings(catalog="workspace", domain_package="exemplo_x", reader_group="time-exemplo"),
        domain="exemplo",
        component="training",
        wheel_name="exemplo_x",
    )

    manage = [p for p in bundle["permissions"] if p["level"] == "CAN_MANAGE"]
    assert manage == [{"user_name": "${workspace.current_user.userName}", "level": "CAN_MANAGE"}]


def test_without_a_group_no_permission_block_is_emitted():
    """Workspace pessoal não tem grupo, e um bloco vazio faria o validate
    reclamar."""
    from mlplatform.core.bundle import generate_bundle
    from mlplatform.core.settings import BundleSettings

    bundle = generate_bundle(
        BundleSettings(catalog="workspace", domain_package="exemplo_x"),
        domain="exemplo",
        component="training",
        wheel_name="exemplo_x",
    )

    assert "permissions" not in bundle


def test_the_platform_variables_are_always_declared():
    """Os jobs referenciam `${var.reader_group}` e `${var.retrain_repository}`
    nos parâmetros. Referência a variável não declarada faz o `bundle validate`
    falhar, mesmo quando o domínio não usa o recurso."""
    from mlplatform.core.bundle import generate_bundle
    from mlplatform.core.settings import BundleSettings

    bundle = generate_bundle(
        BundleSettings(catalog="workspace", domain_package="exemplo_x"),
        domain="exemplo",
        component="training",
        wheel_name="exemplo_x",
    )

    assert bundle["variables"]["reader_group"]["default"] == ""
    assert bundle["variables"]["retrain_repository"]["default"] == ""
