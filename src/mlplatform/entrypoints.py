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
    # Flags com underscore, não hífen: o Databricks injeta os job parameters
    # como `--<nome>=<valor>` no python_wheel_task, e os nomes deles usam
    # underscore. Com hífen, o argparse rejeita tudo o que o job manda.
    parser.add_argument("--domain", help="nome do entry point do domínio a carregar")
    parser.add_argument("--config_module", help="escape hatch: importa o módulo de configs direto")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--git_commit", default="local")
    parser.add_argument("--git_branch", default="local")


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
    parser.add_argument("--feature_table", required=True)
    parser.add_argument("--mode", default=WriteMode.INCREMENTAL, type=WriteMode)
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--database_instance_name", default="")
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
    parser.add_argument("--job_name", default="feature_pipeline")
    args = parser.parse_args(argv)

    _load(args)
    # O mesmo --domain que carregou as configs aqui é propagado para as tasks:
    # é o nome do entry point, que é o que load_domains() casa em runtime.
    write_job_resource(args.out, job_name=args.job_name, domain_entry_point=args.domain)
    print(f"[mlplatform] recursos de '{args.component}' escritos em {args.out}", flush=True)
    return 0


def _domain_of(component: str) -> str:
    """O campo `domain` das configs registradas.

    Derivado, e não declarado no YAML: já está nas specs, e duas fontes para o
    mesmo fato acabam discordando.
    """
    if component == "features":
        from .features.contract import get_registry
    else:  # pragma: no cover - só features está migrado
        raise ValueError(f"componente não suportado: {component}")

    registry = get_registry()
    if not registry:
        raise ValueError(
            "nenhuma config registrada — o wheel do domínio está instalado e "
            "declara o entry point em 'mlplatform.domains'?"
        )
    domains = {spec.domain for spec in registry.values()}
    if len(domains) > 1:
        raise ValueError(f"um bundle serve um domínio só, mas foram registrados: {sorted(domains)}")
    return domains.pop()


def generate_bundle(argv: list[str] | None = None) -> int:
    """Materializa o bundle DAB inteiro a partir do `conf/variables.yml`.

    O repositório de domínio não versiona `databricks.yml` nem `resources/`:
    ambos são escritos aqui, em tempo de CI, e são gitignorados.
    """
    from pathlib import Path

    from .core.bundle import generate_bundle as build_bundle
    from .core.resource_gen import dump_yaml
    from .core.settings import DEFAULT_PATH, BundleSettings
    from .features.resource_gen import write_job_resource

    parser = argparse.ArgumentParser(prog="mlp-generate-bundle")
    parser.add_argument(
        "--component",
        choices=["features"],
        help="sobrescreve o componente declarado em conf/variables.yml",
    )
    parser.add_argument("--config", default=DEFAULT_PATH)
    parser.add_argument("--root", default=".", help="raiz do bundle onde escrever")
    args = parser.parse_args(argv)

    settings = BundleSettings.load(args.config)
    # A esteira é genérica: roda o mesmo comando para todo bundle, então quem
    # sabe o que este bundle é, é o próprio arquivo do domínio.
    component = args.component or settings.component
    if not component:
        raise ValueError(f"declare 'component' em {args.config} (ou passe --component)")
    args.component = component
    load_domains(only=settings.domain_package)

    domain = _domain_of(component)
    root = Path(args.root)
    (root / "resources").mkdir(parents=True, exist_ok=True)

    # O nome do artifact tem que bater com o pacote Python do domínio para a CLI
    # saber o que buildar; o entry point declarado é justamente esse nome.
    wheel_name = settings.domain_package or f"{domain}_{args.component}"

    dump_yaml(
        build_bundle(settings, domain=domain, component=args.component, wheel_name=wheel_name),
        str(root / "databricks.yml"),
    )
    write_job_resource(
        str(root / "resources" / f"generated_{args.component}.job.yml"),
        job_name=settings.job_name or "feature_pipeline",
        domain_entry_point=settings.domain_package,
    )

    print(
        f"[mlplatform] bundle '{domain}-{args.component}' gerado em {root} "
        f"(catalog={settings.catalog}, domain_package={settings.domain_package})",
        flush=True,
    )
    return 0
