from platform_core.resource_gen import dump_yaml

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


def generate_job_resource(job_name: str = "feature_pipeline") -> dict:
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

    return {
        "resources": {
            "jobs": {
                job_name: {
                    "name": job_name,
                    "parameters": _JOB_PARAMETERS,
                    "tasks": tasks,
                }
            }
        }
    }


def write_job_resource(path: str, job_name: str = "feature_pipeline") -> None:
    dump_yaml(generate_job_resource(job_name), path)


class FeatureResourceGenerator:
    """Implementa platform_core.resource_gen.ResourceGenerator."""

    def __init__(self, job_name: str = "feature_pipeline"):
        self.job_name = job_name

    def write(self, path: str) -> None:
        write_job_resource(path, self.job_name)
