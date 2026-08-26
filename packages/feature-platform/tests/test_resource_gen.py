import pytest

from feature_platform.contract import clear_registry, feature_table
from feature_platform.resource_gen import FeatureResourceGenerator, generate_job_resource


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


def test_generate_job_resource_points_notebook_task_to_relative_path():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    task = resource["resources"]["jobs"]["feature_pipeline"]["tasks"][0]

    assert task["notebook_task"]["notebook_path"] == "../notebooks/run_feature_table.py"
    assert task["notebook_task"]["base_parameters"]["feature_table"] == "feature_a"


def test_feature_resource_generator_writes_the_same_resource_to_disk(tmp_path):
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    out = tmp_path / "generated_feature_pipeline.job.yml"
    FeatureResourceGenerator(job_name="feature_pipeline").write(str(out))

    content = out.read_text(encoding="utf-8")
    assert "feature_pipeline" in content
    assert "feature_a" in content
