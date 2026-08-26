from platform_core.resource_gen import dump_yaml

from .contract import get_registry
from .naming import derive_monitor_key

NOTEBOOK_PATH = "../notebooks/evaluate_drift.py"


def _monitoring_job(key: str, config) -> dict:
    return {
        "name": f"drift_check_{key}",
        "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
        "parameters": [
            {"name": "domain", "default": config.domain},
            {"name": "model_name", "default": config.model_name},
            {"name": "target_type", "default": config.target_type},
            {"name": "catalog", "default": "${var.catalog}"},
            {"name": "git_commit", "default": "${var.git_commit}"},
            {"name": "git_branch", "default": "${var.git_branch}"},
        ],
        "tasks": [
            {
                "task_key": "evaluate_drift",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    "base_parameters": {
                        "domain": "{{job.parameters.domain}}",
                        "model_name": "{{job.parameters.model_name}}",
                        "target_type": "{{job.parameters.target_type}}",
                        "catalog": "{{job.parameters.catalog}}",
                        "git_commit": "{{job.parameters.git_commit}}",
                        "git_branch": "{{job.parameters.git_branch}}",
                    },
                },
            }
        ],
    }


def generate_resources() -> dict:
    registry = get_registry()
    jobs = {}
    for config in registry.values():
        key = derive_monitor_key(config.domain, config.model_name, config.target_type)
        jobs[f"drift_check_{key}"] = _monitoring_job(key, config)
    return {"resources": {"jobs": jobs}}


def write_resources(path: str) -> None:
    dump_yaml(generate_resources(), path)


class MonitoringResourceGenerator:
    """Implementa platform_core.resource_gen.ResourceGenerator."""

    def write(self, path: str) -> None:
        write_resources(path)
