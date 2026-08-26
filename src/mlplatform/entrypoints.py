"""Console scripts do framework — o composition root.

Este é o único módulo que monta adapters concretos e o único autorizado a
importar mais de um bounded context (o `ruff.toml` compartilhado abre exceção de
TID251 aqui, e só aqui). Nenhum caso de uso recebe um objeto de contexto: cada um
declara os ports de que precisa, e a fiação acontece aqui.

Substituem os notebooks. Com `python_wheel_task` o repositório de domínio não tem
mais arquivo nenhum de orquestração — mas isso também remove a superfície de
debug que o notebook dava (abrir no workspace, editar uma célula, rerodar). A
compensação é o log da configuração resolvida na entrada de cada comando: antes o
notebook *era* o registro do que rodou.
"""

import argparse
from datetime import date

from .core.adapters import DeltaAuditStore
from .core.discovery import load_config_module, load_domains
from .core.ports import SystemClock


def _spark():
    """A SparkSession do runtime do Databricks. Import tardio: fora de um job
    isso não existe, e o módulo precisa ser importável para `--help` funcionar."""
    from databricks.sdk.runtime import spark

    return spark


def _run_id(default: str = "local") -> str:
    """currentRunId() levanta Py4JSecurityException em compute serverless/shared
    access mode — cai para um id gerado localmente quando o contexto do job não
    expõe o run id."""
    try:
        from databricks.sdk.runtime import dbutils

        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
    except Exception:  # noqa: BLE001
        import uuid

        return f"{default}-{uuid.uuid4()}"


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", help="nome do entry point do domínio a carregar")
    parser.add_argument("--config-module", help="escape hatch: importa o módulo de configs direto")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--git-commit", default="local")
    parser.add_argument("--git-branch", default="local")


def _load(args: argparse.Namespace) -> None:
    if args.config_module:
        load_config_module(args.config_module)
    else:
        load_domains(only=args.domain)


def run_feature_table(argv: list[str] | None = None) -> int:
    from .features.adapters import DeltaFeatureWriter, LakebaseOnlineStore, SparkSourceReader
    from .features.contract import get_registry
    from .features.modes import WriteMode
    from .features.usecases import RunFeatureTable

    parser = argparse.ArgumentParser(prog="mlp-run-feature-table")
    _common_args(parser)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--mode", default=WriteMode.INCREMENTAL, type=WriteMode)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--database-instance-name", default="")
    args = parser.parse_args(argv)

    _load(args)
    spec = get_registry()[args.feature_table]
    spark = _spark()

    # O notebook era o log do que rodou; sem ele, o comando precisa dizer.
    print(
        f"[mlplatform] feature_table={args.feature_table} domain={spec.domain} "
        f"mode={args.mode} catalog={args.catalog} online={spec.online}",
        flush=True,
    )

    RunFeatureTable(
        reader=SparkSourceReader(spark),
        writer=DeltaFeatureWriter(spark),
        audit=DeltaAuditStore(spark),
        clock=SystemClock(),
        online=LakebaseOnlineStore() if spec.online else None,
    ).execute(
        spec=spec,
        catalog=args.catalog,
        mode=args.mode,
        today=date.today(),
        run_id=_run_id(),
        git_commit=args.git_commit,
        git_branch=args.git_branch,
        backfill_start=args.start_date or None,
        backfill_end=args.end_date or None,
        database_instance_name=args.database_instance_name,
    )
    return 0


def generate_resources(argv: list[str] | None = None) -> int:
    """Gera o YAML de recursos DAB a partir do registro do domínio."""
    from .features.resource_gen import write_job_resource

    parser = argparse.ArgumentParser(prog="mlp-generate-resources")
    _common_args(parser)
    parser.add_argument("--component", required=True, choices=["features"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--job-name", default="feature_pipeline")
    args = parser.parse_args(argv)

    _load(args)
    # O mesmo --domain que carregou as configs aqui é propagado para as tasks:
    # é o nome do entry point, que é o que load_domains() casa em runtime.
    write_job_resource(args.out, job_name=args.job_name, domain_entry_point=args.domain)
    print(f"[mlplatform] recursos de '{args.component}' escritos em {args.out}", flush=True)
    return 0
