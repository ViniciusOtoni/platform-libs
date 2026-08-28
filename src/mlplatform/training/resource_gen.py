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

# `model_name` NÃO entra aqui. Ele era um job parameter com default vazio, e o
# entrypoint o exige — então qualquer execução agendada, ou qualquer clique em
# "Run now", morria com KeyError: ''. O nome do modelo é conhecido na geração;
# vai estático em named_parameters, como o gerador de batch sempre fez.
_JOB_PARAMETERS = [
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
]


def _job(model_name: str, job_name: str, domain_entry_point: str | None) -> dict:
    named_base = {"model_name": model_name}
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

    return {"name": job_name, "parameters": _JOB_PARAMETERS, "tasks": tasks}


def generate_job_resource(
    job_name: str = "training_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> dict:
    """Um job por TrainingConfig registrada.

    Antes era um job só, com `model_name` vindo de um job parameter vazio que
    alguém teria de preencher a cada execução. Com dois modelos no mesmo
    domínio, um deles simplesmente não tinha job.
    """
    registry = get_registry()
    if not registry:
        raise ValueError("nenhuma TrainingConfig registrada")

    deps = environment_dependencies if environment_dependencies is not None else default_dependencies()
    single = len(registry) == 1

    jobs = {}
    for model_name in registry:
        # Com um modelo só o nome do job segue o que era antes; com vários,
        # precisa desambiguar ou os dois colidem no mesmo caminho do workspace.
        key = job_name if single else f"{job_name}_{model_name}"
        jobs[key] = with_environment(_job(model_name, key, domain_entry_point), deps)

    return {"resources": {"jobs": jobs}}


def write_job_resource(
    path: str,
    job_name: str = "training_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> None:
    dump_yaml(generate_job_resource(job_name, environment_dependencies, domain_entry_point), path)
