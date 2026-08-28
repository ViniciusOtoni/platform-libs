from mlplatform.core.naming import derive_model_name
from mlplatform.core.resource_gen import dump_yaml, with_environment
from mlplatform.core.wheels import default_dependencies

from .contract import BatchServingConfig, OnlineServingConfig, batch_configs, online_configs
from .naming import derive_endpoint_name, validate_endpoint_name

PACKAGE_NAME = "mlplatform"
SCORE_ENTRY_POINT = "mlp-score-batch"
REFRESH_ENTRY_POINT = "mlp-refresh-endpoint"

_JOB_PARAMETERS = [
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
]


def _wheel_task(entry_point: str, named: dict[str, str]) -> dict:
    # Só o estático: o Databricks injeta os job parameters sozinho, e declará-los
    # aqui também os passaria duas vezes.
    return {"package_name": PACKAGE_NAME, "entry_point": entry_point, "named_parameters": named}


def _batch_job(config: BatchServingConfig, domain_entry_point: str | None) -> dict:
    named = {"model_name": config.model_name}
    if domain_entry_point:
        named["domain"] = domain_entry_point
    return {
        "name": f"score_batch_{config.model_name}",
        "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
        "parameters": _JOB_PARAMETERS,
        "tasks": [{"task_key": "score_batch", "python_wheel_task": _wheel_task(SCORE_ENTRY_POINT, named)}],
    }


def _refresh_job(config: OnlineServingConfig, endpoint_key: str, domain_entry_point: str | None) -> dict:
    """Um job de refresh por endpoint, com o nome REAL do endpoint embutido.

    O nome não pode ser recalculado em runtime: o DABs prefixa os recursos por
    target — em `dev` o endpoint vira `dev_<usuario>_<nome>` — e o código dentro
    do job não conhece esse prefixo. Derivá-lo lá dentro dava
    `ResourceDoesNotExist` apontando um endpoint que existe, só que com outro
    nome. A referência abaixo é resolvida pelo próprio DABs no deploy, então
    vale para qualquer target.

    `model_name` também vai estático, pelo mesmo motivo do gerador de training:
    era um job parameter vazio que derrubava qualquer execução não parametrizada
    à mão.
    """
    named = {
        "model_name": config.model_name,
        "endpoint_name": f"${{resources.model_serving_endpoints.{endpoint_key}.name}}",
    }
    if domain_entry_point:
        named["domain"] = domain_entry_point
    return {
        "name": f"refresh_endpoint_{config.model_name}",
        "parameters": _JOB_PARAMETERS,
        "tasks": [
            {"task_key": "refresh_endpoint", "python_wheel_task": _wheel_task(REFRESH_ENTRY_POINT, named)}
        ],
    }


def _endpoint(config: OnlineServingConfig, entity_version: int) -> dict:
    # model_serving_endpoints em DABs não aceita `models:/nome@alias` em
    # entity_name — só entity_name puro + entity_version numérico (confirmado ao
    # vivo: 404 RESOURCE_DOES_NOT_EXIST com o alias). O alias é resolvido para a
    # versão vigente no momento da geração; movê-lo depois exige rodar
    # refresh_endpoint ou gerar os recursos de novo.
    endpoint_name = derive_endpoint_name(config.domain, config.model_name)
    validate_endpoint_name(endpoint_name)
    return {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "name": config.model_name,
                    "entity_name": f"${{var.catalog}}.{config.domain}_models.{config.model_name}",
                    "entity_version": str(entity_version),
                    "scale_to_zero_enabled": True,
                    "workload_size": "Small",
                }
            ]
        },
    }


def generate_resources(
    catalog: str,
    resolve_alias_version=None,
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> dict:
    """resolve_alias_version: callable (full_model_name, alias) -> int, exigido
    quando há alguma config online.

    Sem `if config.mode`: cada tipo de config já diz o que produz. Batch vira
    entrada em `jobs`, online em `model_serving_endpoints` — buckets diferentes,
    que é por que os dois nunca foram polimórficos de verdade.
    """
    deps = environment_dependencies if environment_dependencies is not None else default_dependencies()
    jobs: dict = {}

    # refresh_endpoint só faz sentido onde existe endpoint. Emiti-lo sempre —
    # como o código original fazia — deployava um job inútil no bundle de batch,
    # e com dois bundles de serving no mesmo domínio isso vira dois jobs de mesmo
    # nome no workspace, indistinguíveis na UI.
    for config in online_configs().values():
        endpoint_key = derive_endpoint_name(config.domain, config.model_name)
        jobs[f"refresh_endpoint_{config.model_name}"] = with_environment(
            _refresh_job(config, endpoint_key, domain_entry_point), deps
        )

    for name, config in batch_configs().items():
        jobs[f"score_batch_{name}"] = with_environment(_batch_job(config, domain_entry_point), deps)

    endpoints = {}
    for config in online_configs().values():
        if resolve_alias_version is None:
            raise ValueError(
                f"'{config.model_name}' é online mas nenhum resolve_alias_version foi fornecido — "
                "DABs exige entity_version numérico, não alias"
            )
        # Catálogo real, não o placeholder ${var.catalog}: o resolvedor consulta
        # o Model Registry de verdade para descobrir a versão do alias. Antes o
        # repositório de domínio carregava um `CATALOG = "workspace"` hardcodado,
        # com um comentário pedindo que batesse com o default do databricks.yml —
        # e nada verificava se batia.
        full_name = derive_model_name(catalog, config.domain, config.model_name)
        endpoints[derive_endpoint_name(config.domain, config.model_name)] = _endpoint(
            config, resolve_alias_version(full_name, config.alias)
        )

    resources: dict = {"resources": {"jobs": jobs}}
    if endpoints:
        resources["resources"]["model_serving_endpoints"] = endpoints
    return resources


def write_resources(
    path: str,
    catalog: str,
    resolve_alias_version=None,
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> None:
    dump_yaml(
        generate_resources(catalog, resolve_alias_version, environment_dependencies, domain_entry_point),
        path,
    )
