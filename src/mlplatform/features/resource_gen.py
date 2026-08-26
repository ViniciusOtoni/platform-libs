from mlplatform.core.resource_gen import dump_yaml, with_environment
from mlplatform.core.wheels import default_dependencies

from .contract import get_registry

ENTRY_POINT = "mlp-run-feature-table"
PACKAGE_NAME = "mlplatform"

_JOB_PARAMETERS = [
    {"name": "mode", "default": "incremental"},
    {"name": "start_date", "default": ""},
    {"name": "end_date", "default": ""},
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
    {"name": "database_instance_name", "default": "${var.database_instance_name}"},
]


def _task(name: str, domain_entry_point: str | None) -> dict:
    """Uma task por feature table, apontando para o console script do framework.

    Era `notebook_task` apontando para um arquivo no repositório do domínio.
    Virou `python_wheel_task`: o domínio não precisa mais carregar notebook
    nenhum, e o entry point vem do wheel do próprio framework. Confirmado ao
    vivo que serverless suporta python_wheel_task com environment_key.

    `domain_entry_point` é o nome declarado no grupo `mlplatform.domains` do
    pyproject do domínio — NÃO o campo `domain` da spec. São coisas diferentes:
    `load_domains(only=...)` casa contra o nome do entry point. Emitir o campo da
    spec aqui gerava um YAML que só falhava em runtime, dentro do job, com
    DomainLoadError.
    """
    # Só o que é estático por task. O Databricks injeta os job parameters como
    # `--<nome>=<valor>` no python_wheel_task automaticamente — declará-los aqui
    # também os passaria DUAS vezes, e o argparse morre com "unrecognized
    # arguments". Foi assim que o primeiro run real falhou.
    named: dict[str, str] = {}
    if domain_entry_point:
        named["domain"] = domain_entry_point
    named["feature_table"] = name
    return {
        "task_key": name,
        "python_wheel_task": {
            "package_name": PACKAGE_NAME,
            "entry_point": ENTRY_POINT,
            "named_parameters": named,
        },
    }


def generate_job_resource(
    job_name: str = "feature_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> dict:
    """environment_dependencies: quando omitido, deriva o par padrão — wheel do
    domínio buildado pelo `artifacts:` do bundle + wheel do framework publicado
    como asset do Release, na versão que está instalada aqui.

    domain_entry_point: nome declarado no grupo `mlplatform.domains`, propagado
    para as tasks. Sem ele, o job carrega todos os domínios instalados."""
    tasks = []
    for name, spec in get_registry().items():
        task = _task(name, domain_entry_point)
        if spec.depends_on:
            task["depends_on"] = [{"task_key": dep} for dep in spec.depends_on]
        tasks.append(task)

    job: dict = {
        "name": job_name,
        "parameters": _JOB_PARAMETERS,
        "tasks": tasks,
    }
    deps = environment_dependencies if environment_dependencies is not None else default_dependencies()
    return {"resources": {"jobs": {job_name: with_environment(job, deps)}}}


def write_job_resource(
    path: str,
    job_name: str = "feature_pipeline",
    environment_dependencies: list[str] | None = None,
    domain_entry_point: str | None = None,
) -> None:
    dump_yaml(generate_job_resource(job_name, environment_dependencies, domain_entry_point), path)


class FeatureResourceGenerator:
    """Implementa mlplatform.core.resource_gen.ResourceGenerator."""

    def __init__(
        self,
        job_name: str = "feature_pipeline",
        environment_dependencies: list[str] | None = None,
        domain_entry_point: str | None = None,
    ):
        self.job_name = job_name
        self.environment_dependencies = environment_dependencies
        self.domain_entry_point = domain_entry_point

    def write(self, path: str) -> None:
        write_job_resource(path, self.job_name, self.environment_dependencies, self.domain_entry_point)
