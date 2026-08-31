"""Geração dos jobs de verificação de drift.

Último componente a sair de `notebook_task`. O YAML apontava para
`../notebooks/evaluate_drift.py`, o que obrigava o repositório de domínio a
versionar o notebook — e era o que sobrava do formato antigo depois de features,
training e serving já terem migrado.
"""

from mlplatform.core.resource_gen import dump_yaml, with_environment
from mlplatform.core.wheels import default_dependencies

from .contract import get_registry
from .naming import derive_monitor_key

PACKAGE_NAME = "mlplatform"
ENTRY_POINT = "mlp-evaluate-drift"

# `model_name` e `target_type` NÃO entram aqui: iam como job parameters e o
# entrypoint os exige, então qualquer execução agendada — que é o único jeito
# como este job roda de verdade — morreria antes de tocar em dado nenhum. Vão
# estáticos em named_parameters, como nos outros geradores.
_JOB_PARAMETERS = [
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
]


def _job(key: str, config, domain_entry_point: str | None) -> dict:
    named = {"model_name": config.model_name, "target_type": config.target_type}
    if domain_entry_point:
        named["domain"] = domain_entry_point

    return {
        "name": f"drift_check_{key}",
        "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
        "parameters": _JOB_PARAMETERS,
        "tasks": [
            {
                "task_key": "evaluate_drift",
                "python_wheel_task": {
                    "package_name": PACKAGE_NAME,
                    "entry_point": ENTRY_POINT,
                    "named_parameters": named,
                },
            }
        ],
    }


def generate_resources(
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> dict:
    registry = get_registry()
    if not registry:
        raise ValueError("nenhuma MonitoringConfig registrada")

    deps = environment_dependencies if environment_dependencies is not None else default_dependencies()

    jobs = {}
    for config in registry.values():
        key = derive_monitor_key(config.domain, config.model_name, config.target_type)
        jobs[f"drift_check_{key}"] = with_environment(_job(key, config, domain_entry_point), deps)
    return {"resources": {"jobs": jobs}}


def write_resources(
    path: str,
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> None:
    dump_yaml(generate_resources(environment_dependencies, domain_entry_point), path)
