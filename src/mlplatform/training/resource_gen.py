"""Geração do job de treino.

Este é o único componente que nunca teve gerador: o YAML das quatro tasks era
escrito à mão no repositório de domínio e copiado por domínio. É também o motivo
de o pipeline de treino ter divergido dos demais sem ninguém notar.
"""

from mlplatform.core.resource_gen import dump_yaml, with_environment
from mlplatform.core.wheels import default_dependencies

from .contract import get_registry

PACKAGE_NAME = "mlplatform"

# Três tasks, não quatro. `select_test_and_register` funde o que eram duas: a
# antiga separação fazia uma task fitar-testar-aprovar um modelo e a seguinte
# fitar OUTRO com os mesmos hiperparâmetros para registrar. Sem random_state, os
# dois objetos diferem — o modelo aprovado era descartado e outro ia para
# produção. Juntá-las torna isso impossível por construção.
_TASKS = [
    ("prepare_training_set", "mlp-prepare-training-set", []),
    ("fit_and_compare", "mlp-fit-compare", ["prepare_training_set"]),
    ("select_test_and_register", "mlp-select-test-register", ["fit_and_compare"]),
]

_JOB_PARAMETERS = [
    {"name": "model_name", "default": ""},
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
]


def generate_job_resource(
    job_name: str = "training_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> dict:
    registry = get_registry()
    if not registry:
        raise ValueError("nenhuma TrainingConfig registrada")

    named_base: dict[str, str] = {}
    if domain_entry_point:
        named_base["domain"] = domain_entry_point

    tasks = []
    for task_key, entry_point, depends_on in _TASKS:
        task: dict = {
            "task_key": task_key,
            "python_wheel_task": {
                "package_name": PACKAGE_NAME,
                "entry_point": entry_point,
                "named_parameters": dict(named_base),
            },
        }
        if depends_on:
            task["depends_on"] = [{"task_key": d} for d in depends_on]
        tasks.append(task)

    job = {"name": job_name, "parameters": _JOB_PARAMETERS, "tasks": tasks}
    deps = environment_dependencies if environment_dependencies is not None else default_dependencies()
    return {"resources": {"jobs": {job_name: with_environment(job, deps)}}}


def write_job_resource(
    path: str,
    job_name: str = "training_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> None:
    dump_yaml(generate_job_resource(job_name, environment_dependencies, domain_entry_point), path)
