from mlplatform.core.resource_gen import dump_yaml, with_environment

from .contract import get_registry

NOTEBOOK_PATH = "../notebooks/run_feature_table.py"

_JOB_PARAMETERS = [
    {"name": "mode", "default": "incremental"},
    {"name": "start_date", "default": ""},
    {"name": "end_date", "default": ""},
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
    {"name": "database_instance_name", "default": "${var.database_instance_name}"},
]




def generate_job_resource(
    job_name: str = "feature_pipeline",
    environment_dependencies: list[str] | None = None,
) -> dict:
    """environment_dependencies: se dado, declara um Environment nativo do
    serverless (client "3") com essas dependências (ex.: URL do wheel do
    domínio + URL do wheel de feature-platform publicado como asset de
    Release) e referencia esse environment em cada task — substitui
    `%pip install`/sys.path hack dentro do notebook por resolução declarativa
    do próprio job, sem cluster/venv manual."""
    registry = get_registry()
    tasks = []
    for name, spec in registry.items():
        task = {
            "task_key": name,
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "base_parameters": {
                    "feature_table": name,
                    "mode": "{{job.parameters.mode}}",
                    "start_date": "{{job.parameters.start_date}}",
                    "end_date": "{{job.parameters.end_date}}",
                    "catalog": "{{job.parameters.catalog}}",
                    "git_commit": "{{job.parameters.git_commit}}",
                    "git_branch": "{{job.parameters.git_branch}}",
                    "database_instance_name": "{{job.parameters.database_instance_name}}",
                },
            },
        }
        if spec.depends_on:
            task["depends_on"] = [{"task_key": dep} for dep in spec.depends_on]
        tasks.append(task)

    job: dict = {
        "name": job_name,
        "parameters": _JOB_PARAMETERS,
        "tasks": tasks,
    }
    return {"resources": {"jobs": {job_name: with_environment(job, environment_dependencies)}}}


def write_job_resource(
    path: str,
    job_name: str = "feature_pipeline",
    environment_dependencies: list[str] | None = None,
) -> None:
    dump_yaml(generate_job_resource(job_name, environment_dependencies), path)


class FeatureResourceGenerator:
    """Implementa mlplatform.core.resource_gen.ResourceGenerator."""

    def __init__(self, job_name: str = "feature_pipeline", environment_dependencies: list[str] | None = None):
        self.job_name = job_name
        self.environment_dependencies = environment_dependencies

    def write(self, path: str) -> None:
        write_job_resource(path, self.job_name, self.environment_dependencies)
