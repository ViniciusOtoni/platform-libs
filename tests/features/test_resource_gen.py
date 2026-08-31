import pytest

from mlplatform.features.contract import clear_registry, feature_table
from mlplatform.features.resource_gen import FeatureResourceGenerator, generate_job_resource


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_generate_job_resource_creates_one_task_per_feature_table():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.b"])
    def feature_b(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    tasks = resource["resources"]["jobs"]["feature_pipeline"]["tasks"]
    task_keys = {t["task_key"] for t in tasks}

    assert task_keys == {"feature_a", "feature_b"}


def test_generate_job_resource_declares_job_parameters():
    resource = generate_job_resource(job_name="feature_pipeline")
    job = resource["resources"]["jobs"]["feature_pipeline"]
    param_names = {p["name"] for p in job["parameters"]}

    assert param_names == {
        "mode",
        "start_date",
        "end_date",
        "git_commit",
        "git_branch",
        "catalog",
        "database_instance_name",
        "reader_group",
    }


def test_generate_job_resource_adds_depends_on_edge():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def base_feature(sources, window):
        return None

    @feature_table(
        domain="exemplo",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.b"],
        depends_on=["base_feature"],
    )
    def derived_feature(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    tasks = {t["task_key"]: t for t in resource["resources"]["jobs"]["feature_pipeline"]["tasks"]}

    assert "depends_on" not in tasks["base_feature"]
    assert tasks["derived_feature"]["depends_on"] == [{"task_key": "base_feature"}]


def test_task_invokes_the_framework_entry_point_not_a_notebook():
    """O domínio não carrega mais notebook nenhum: a task chama o console script
    do wheel do framework."""

    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    task = generate_job_resource(job_name="feature_pipeline")["resources"]["jobs"]["feature_pipeline"]["tasks"][0]

    assert "notebook_task" not in task
    wheel = task["python_wheel_task"]
    assert (wheel["package_name"], wheel["entry_point"]) == ("mlplatform", "mlp-run-feature-table")
    assert wheel["named_parameters"]["feature_table"] == "feature_a"


def test_named_parameters_carry_only_what_is_static_per_task():
    """O Databricks injeta os job parameters no python_wheel_task sozinho.
    Declará-los também aqui os passaria duas vezes, e o argparse morre com
    'unrecognized arguments' — foi assim que o primeiro run real falhou."""

    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    named = generate_job_resource(domain_entry_point="exemplo_features")["resources"]["jobs"]["feature_pipeline"][
        "tasks"
    ][0]["python_wheel_task"]["named_parameters"]

    assert set(named) == {"domain", "feature_table"}


def test_task_carries_the_entry_point_name_not_the_spec_domain():
    """Regressão de um bug que só aparecia em runtime, dentro do job: o gerador
    emitia `--domain` com o campo `domain` da spec ("credito"), mas
    `load_domains(only=...)` casa contra o nome do entry point declarado no
    pyproject ("credito_features"). O YAML era válido e o job falhava com
    DomainLoadError na primeira execução."""

    @feature_table(domain="credito", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    task = generate_job_resource(domain_entry_point="credito_features")["resources"]["jobs"]["feature_pipeline"][
        "tasks"
    ][0]

    assert task["python_wheel_task"]["named_parameters"]["domain"] == "credito_features"


def test_task_omits_domain_when_none_was_declared():
    """Sem entry point declarado, o job carrega todos os domínios instalados —
    mas não inventa um nome que não existe."""

    @feature_table(domain="credito", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    task = generate_job_resource()["resources"]["jobs"]["feature_pipeline"]["tasks"][0]

    assert "domain" not in task["python_wheel_task"]["named_parameters"]


def test_feature_resource_generator_writes_the_same_resource_to_disk(tmp_path):
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    out = tmp_path / "generated_feature_pipeline.job.yml"
    FeatureResourceGenerator(job_name="feature_pipeline").write(str(out))

    content = out.read_text(encoding="utf-8")
    assert "feature_pipeline" in content
    assert "feature_a" in content


def test_dependencies_default_to_the_domain_wheel_plus_the_installed_framework():
    """Antes cada script de domínio hardcodava a URL do wheel, e ela
    dessincronizava da versão realmente instalada. Agora o default é derivado de
    importlib.metadata, que é a verdade sobre o que vai rodar."""
    from mlplatform.core.wheels import DOMAIN_WHEEL_GLOB, framework_version

    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    job = generate_job_resource(job_name="feature_pipeline")["resources"]["jobs"]["feature_pipeline"]
    deps = job["environments"][0]["spec"]["dependencies"]

    assert deps[0] == DOMAIN_WHEEL_GLOB
    assert framework_version() in deps[1]
    assert job["tasks"][0]["environment_key"] == "default"


def test_domain_wheel_path_is_relative_to_the_generated_file_not_the_bundle_root():
    """Regressão do deploy quebrado: o YAML gerado mora em resources/, então um
    './dist/*.whl' resolveria para resources/dist/ e o deploy falha com
    'no files match pattern'."""
    from mlplatform.core.wheels import DOMAIN_WHEEL_GLOB

    assert DOMAIN_WHEEL_GLOB.startswith("../")


def test_generate_job_resource_with_environment_dependencies_declares_native_environment():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    deps = ["../dist/exemplo_features-0.1.0-py3-none-any.whl", "https://example.com/mlplatform-1.0.0.whl"]
    job = generate_job_resource(job_name="feature_pipeline", environment_dependencies=deps)["resources"]["jobs"][
        "feature_pipeline"
    ]

    assert job["environments"] == [{"environment_key": "default", "spec": {"client": "3", "dependencies": deps}}]
    assert job["tasks"][0]["environment_key"] == "default"
