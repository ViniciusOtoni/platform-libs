from typing import Protocol

import yaml


class ResourceGenerator(Protocol):
    """Contrato que cada `<componente>.resource_gen` implementa, formalizando o
    hook hoje descoberto por convenção de nome de arquivo
    (`scripts/generate_resources.py`) sem nenhuma garantia de forma."""

    def write(self, path: str) -> None: ...


class _NoAliasDumper(yaml.SafeDumper):
    """Repete o valor em vez de emitir ancora YAML.

    Os geradores compartilham listas por referencia — a de job parameters e a
    mesma para todos os jobs do componente —, e o dumper padrao transforma a
    segunda ocorrencia em `*id001`. O DABs aceita, mas o YAML gerado e a
    principal superficie de debug depois que os notebooks sairam: quem abre o
    arquivo para entender o que a esteira montou encontra uma referencia em vez
    do conteudo, e precisa resolve-la de cabeca.
    """

    def ignore_aliases(self, data) -> bool:
        return True


def dump_yaml(resource: dict, path: str) -> None:
    # allow_unicode: sem isso o safe_dump escapa acentos ("\xE1"), e o YAML
    # gerado é justamente o que alguém abre para entender o que a esteira montou.
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(resource, f, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True)


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
