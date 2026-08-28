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
    elif component == "serving":
        from .serving.contract import get_registry
    elif component == "training":
        from .training.contract import get_registry
    else:  # pragma: no cover
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
        choices=["features", "serving", "training"],
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

    bundle = build_bundle(settings, domain=domain, component=component, wheel_name=wheel_name)
    dump_yaml(bundle, str(root / "databricks.yml"))
    # `.yml`, não `.job.yml`: por convenção do DABs um arquivo .job.yml declara UM
    # job, e o gerado pode conter vários — e, no serving, também
    # model_serving_endpoints. Com o sufixo errado o `bundle validate` emite uma
    # recomendação em toda execução, e aviso permanente ensina a ignorar a saída.
    out = str(root / "resources" / f"generated_{component}.yml")
    if component == "features":
        write_job_resource(
            out,
            job_name=settings.job_name or "feature_pipeline",
            domain_entry_point=settings.domain_package,
        )
    elif component == "training":
        from .training.resource_gen import write_job_resource as write_training

        write_training(
            out,
            job_name=settings.job_name or "training_pipeline",
            domain_entry_point=settings.domain_package,
        )
    else:
        from .serving.adapters import SdkModelRegistry
        from .serving.contract import online_configs
        from .serving.resource_gen import write_resources

        # O resolvedor só é construído quando há config online: montá-lo sempre
        # exigiria credenciais de workspace até num bundle puramente batch.
        resolver = None
        if online_configs():
            resolver = SdkModelRegistry().version_for_alias
        write_resources(
            out,
            catalog=settings.catalog,
            resolve_alias_version=resolver,
            domain_entry_point=settings.domain_package,
        )

    # O nome real do bundle, não uma string remontada: com os notebooks removidos,
    # este log é a principal superfície de debug, e um nome que não corresponde ao
    # que foi gerado manda quem está investigando para o lugar errado.
    print(
        f"[mlplatform] bundle '{bundle['bundle']['name']}' gerado em {root} "
        f"(component={component}, catalog={settings.catalog}, "
        f"domain_package={settings.domain_package})",
        flush=True,
    )
    return 0


def score_batch(argv: list[str] | None = None) -> int:
    from .core.adapters import DeltaAuditStore
    from .serving.adapters import DeltaPredictionWriter, FeatureEngineeringScorer
    from .serving.contract import get_serving_config
    from .serving.usecases import ScoreBatch

    parser = argparse.ArgumentParser(prog="mlp-score-batch")
    _common_args(parser)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args(argv)

    _load(args)
    config = get_serving_config(args.model_name)
    spark = _spark()

    print(
        f"[mlplatform] score_batch model={args.model_name} domain={config.domain} "
        f"catalog={args.catalog} spine={config.spine_inference_table}",
        flush=True,
    )

    ScoreBatch(
        scorer=FeatureEngineeringScorer(spark),
        writer=DeltaPredictionWriter(spark),
        audit=DeltaAuditStore(spark),
        clock=SystemClock(),
    ).execute(
        config=config,
        catalog=args.catalog,
        run_id=_run_id(),
        git_commit=args.git_commit,
        git_branch=args.git_branch,
    )
    return 0


def refresh_endpoint(argv: list[str] | None = None) -> int:
    from .serving.adapters import SdkEndpointGateway
    from .serving.contract import get_serving_config
    from .serving.usecases import RefreshEndpoint

    parser = argparse.ArgumentParser(prog="mlp-refresh-endpoint")
    _common_args(parser)
    parser.add_argument("--model_name", required=True)
    # Injetado pelo DABs a partir do recurso do endpoint, já com o prefixo do
    # target. Recalcular o nome aqui erra em qualquer target que não seja prod.
    parser.add_argument("--endpoint_name", required=True)
    args = parser.parse_args(argv)

    _load(args)
    config = get_serving_config(args.model_name)

    name = RefreshEndpoint(gateway=SdkEndpointGateway()).execute(
        config=config, catalog=args.catalog, endpoint_name=args.endpoint_name
    )
    print(f"[mlplatform] endpoint '{name}' reaponta para o alias '{config.alias}'", flush=True)
    return 0


def _training_scorer(config):
    """O scorer sai da config: uma string vira scorer do sklearn, um callable é
    usado como veio."""
    from sklearn.metrics import get_scorer

    return get_scorer(config.metric) if isinstance(config.metric, str) else config.metric


def _training_context(args):
    from .training.adapters import (
        DbutilsTaskChannel,
        DeltaScratchStore,
        FeatureEngineeringTrainingSet,
        MlflowTracker,
    )
    from .training.contract import get_training_config

    config = get_training_config(args.model_name)
    spark = _spark()
    print(
        f"[mlplatform] training model={config.model_name} domain={config.domain} "
        f"catalog={args.catalog} metric={config.metric}",
        flush=True,
    )
    return config, spark, {
        "builder": FeatureEngineeringTrainingSet(spark),
        "scratch": DeltaScratchStore(spark),
        "tracker": MlflowTracker(),
        "channel": DbutilsTaskChannel(),
    }


def _training_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    _common_args(parser)
    parser.add_argument("--model_name", required=True)
    return parser


def prepare_training_set(argv: list[str] | None = None) -> int:
    from .training.usecases import PrepareTrainingSet

    args = _training_parser("mlp-prepare-training-set").parse_args(argv)
    _load(args)
    config, _spark_session, deps = _training_context(args)

    PrepareTrainingSet(
        builder=deps["builder"], scratch=deps["scratch"], tracker=deps["tracker"], channel=deps["channel"]
    ).execute(config=config, catalog=args.catalog)
    return 0


def fit_and_compare(argv: list[str] | None = None) -> int:
    from .training.usecases import FitAndCompare

    args = _training_parser("mlp-fit-compare").parse_args(argv)
    _load(args)
    config, _spark_session, deps = _training_context(args)

    results = FitAndCompare(
        scratch=deps["scratch"], tracker=deps["tracker"], channel=deps["channel"]
    ).execute(config=config, catalog=args.catalog, scorer=_training_scorer(config))
    print(f"[mlplatform] {len(results)} combinações avaliadas", flush=True)
    return 0


def select_test_register(argv: list[str] | None = None) -> int:
    from .core.adapters import DeltaAuditStore
    from .training.adapters import FeatureEngineeringPublisher
    from .training.usecases import SelectTestAndRegister

    args = _training_parser("mlp-select-test-register").parse_args(argv)
    _load(args)
    config, spark, deps = _training_context(args)

    name = SelectTestAndRegister(
        scratch=deps["scratch"],
        tracker=deps["tracker"],
        publisher=FeatureEngineeringPublisher(spark),
        builder=deps["builder"],
        audit=DeltaAuditStore(spark),
        clock=SystemClock(),
        channel=deps["channel"],
    ).execute(
        config=config,
        catalog=args.catalog,
        scorer=_training_scorer(config),
        run_id=_run_id(),
        git_commit=args.git_commit,
        git_branch=args.git_branch,
    )
    print(f"[mlplatform] modelo registrado: {name}", flush=True)
    return 0
