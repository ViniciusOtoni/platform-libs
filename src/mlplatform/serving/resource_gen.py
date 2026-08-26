from mlplatform.core.resource_gen import dump_yaml, with_environment

from .contract import get_registry
from .naming import derive_endpoint_name, validate_endpoint_name

BATCH_NOTEBOOK_PATH = "../notebooks/score_batch.py"
REFRESH_NOTEBOOK_PATH = "../notebooks/refresh_endpoint.py"


def _batch_job(model_name: str, config, environment_dependencies: list[str] | None = None) -> dict:
    return with_environment(
        {
            "name": f"score_batch_{model_name}",
            "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
            "parameters": [
                {"name": "model_name", "default": model_name},
                {"name": "catalog", "default": "${var.catalog}"},
                {"name": "git_commit", "default": "${var.git_commit}"},
                {"name": "git_branch", "default": "${var.git_branch}"},
            ],
            "tasks": [
                {
                    "task_key": "score_batch",
                    "notebook_task": {
                        "notebook_path": BATCH_NOTEBOOK_PATH,
                        "base_parameters": {
                            "model_name": "{{job.parameters.model_name}}",
                            "catalog": "{{job.parameters.catalog}}",
                            "git_commit": "{{job.parameters.git_commit}}",
                            "git_branch": "{{job.parameters.git_branch}}",
                        },
                    },
                }
            ],
        },
        environment_dependencies,
    )


def _online_endpoint(model_name: str, config, entity_version: int) -> dict:
    # model_serving_endpoints em DABs não aceita a sintaxe models:/nome@alias em
    # entity_name — só um entity_name puro + entity_version numérico fixo (confirmado
    # ao vivo: 404 RESOURCE_DOES_NOT_EXIST tentando "...@champion"). O alias é
    # resolvido para a versão vigente no momento da geração (ver
    # resolve_alias_version); mover o alias depois exige rodar refresh_endpoint
    # (Task 7) ou gerar os recursos de novo.
    endpoint_name = derive_endpoint_name(config.domain, model_name)
    validate_endpoint_name(endpoint_name)
    return {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "name": model_name,
                    "entity_name": f"${{var.catalog}}.{config.domain}_models.{model_name}",
                    "entity_version": str(entity_version),
                    "scale_to_zero_enabled": True,
                    "workload_size": "Small",
                }
            ]
        },
    }


def _refresh_endpoint_job(environment_dependencies: list[str] | None = None) -> dict:
    return with_environment(
        {
            "name": "refresh_endpoint",
            "parameters": [
                {"name": "model_name", "default": ""},
                {"name": "catalog", "default": "${var.catalog}"},
            ],
            "tasks": [
                {
                    "task_key": "refresh_endpoint",
                    "notebook_task": {
                        "notebook_path": REFRESH_NOTEBOOK_PATH,
                        "base_parameters": {
                            "model_name": "{{job.parameters.model_name}}",
                            "catalog": "{{job.parameters.catalog}}",
                        },
                    },
                }
            ],
        },
        environment_dependencies,
    )


def generate_resources(resolve_alias_version=None, environment_dependencies: list[str] | None = None) -> dict:
    """resolve_alias_version: callable (model_name: str, config: ServingConfig) -> int.
    Obrigatório quando há algum ServingConfig com mode="online" — model_serving_endpoints
    em DABs só aceita entity_version (um número), não um alias, então o alias precisa
    ser resolvido para a versão vigente no momento da geração dos recursos.
    environment_dependencies: se dado, declara um Environment nativo do serverless
    nos jobs (batch/refresh) -- não se aplica a model_serving_endpoints."""
    registry = get_registry()
    jobs = {"refresh_endpoint": _refresh_endpoint_job(environment_dependencies)}
    endpoints = {}

    for model_name, config in registry.items():
        if config.mode == "batch":
            jobs[f"score_batch_{model_name}"] = _batch_job(model_name, config, environment_dependencies)
        else:
            if resolve_alias_version is None:
                raise ValueError(
                    f"ServingConfig '{model_name}' has mode='online' but no "
                    "resolve_alias_version resolver was provided to generate_resources()"
                )
            entity_version = resolve_alias_version(model_name, config)
            endpoints[derive_endpoint_name(config.domain, model_name)] = _online_endpoint(
                model_name, config, entity_version
            )

    resources = {"resources": {"jobs": jobs}}
    if endpoints:
        resources["resources"]["model_serving_endpoints"] = endpoints
    return resources


def write_resources(path: str, resolve_alias_version=None, environment_dependencies: list[str] | None = None) -> None:
    dump_yaml(generate_resources(resolve_alias_version, environment_dependencies), path)


class ServingResourceGenerator:
    """Implementa mlplatform.core.resource_gen.ResourceGenerator."""

    def __init__(self, resolve_alias_version=None, environment_dependencies: list[str] | None = None):
        self.resolve_alias_version = resolve_alias_version
        self.environment_dependencies = environment_dependencies

    def write(self, path: str) -> None:
        write_resources(path, self.resolve_alias_version, self.environment_dependencies)
