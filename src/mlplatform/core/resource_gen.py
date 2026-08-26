from typing import Protocol

import yaml


class ResourceGenerator(Protocol):
    """Contrato que cada `<componente>.resource_gen` implementa, formalizando o
    hook hoje descoberto por convenção de nome de arquivo
    (`scripts/generate_resources.py`) sem nenhuma garantia de forma."""

    def write(self, path: str) -> None: ...


def dump_yaml(resource: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resource, f, sort_keys=False)


ENVIRONMENT_KEY = "default"


def with_environment(job: dict, dependencies: list[str] | None) -> dict:
    """Declara um Environment nativo do serverless no job e referencia-o em
    TODAS as tasks.

    Antes existiam três implementações disso — uma por componente — e elas
    divergiam: a de monitoring aplicava o `environment_key` só em
    `job["tasks"][0]`. Isso nunca quebrou porque os jobs de monitoring têm
    exatamente uma task, mas um job com duas deixaria a segunda sem ambiente e
    ela falharia em runtime por dependência ausente. Iterar é o correto.

    Só se aplica a `jobs`; `model_serving_endpoints` resolvem dependências pelo
    modelo MLflow registrado, não por Environment de job.
    """
    if not dependencies:
        return job
    for task in job["tasks"]:
        task["environment_key"] = ENVIRONMENT_KEY
    job["environments"] = [
        {
            "environment_key": ENVIRONMENT_KEY,
            "spec": {"client": "3", "dependencies": list(dependencies)},
        }
    ]
    return job
