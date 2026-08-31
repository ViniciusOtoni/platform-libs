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
import os
from dataclasses import replace
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
    # Vazio = não concede. Vem do conf/variables.yml do domínio via job parameter.
    parser.add_argument("--reader_group", default="")


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
        writer=DeltaFeatureWriter(spark, args.reader_group),
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
    elif component == "monitoring":
        from .monitoring.contract import get_registry
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
        choices=["features", "serving", "training", "monitoring"],
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
    elif component == "monitoring":
        from .monitoring.resource_gen import write_resources as write_monitoring

        write_monitoring(out, domain_entry_point=settings.domain_package)
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
    from .serving.adapters import DeltaPredictionWriter, FeatureEngineeringScorer, SdkModelRegistry
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
        registry=SdkModelRegistry(),
        scorer=FeatureEngineeringScorer(spark),
        writer=DeltaPredictionWriter(spark, args.reader_group),
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
    from .serving.adapters import SdkEndpointGateway, SdkModelRegistry
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

    name = RefreshEndpoint(gateway=SdkEndpointGateway(), registry=SdkModelRegistry()).execute(
        config=config, catalog=args.catalog, endpoint_name=args.endpoint_name
    )
    print(f"[mlplatform] endpoint '{name}' reaponta para o alias '{config.alias}'", flush=True)
    return 0


def evaluate_drift(argv: list[str] | None = None) -> int:
    from .core.adapters import DeltaAuditStore
    from .monitoring.adapters import (
        AuditTrainingRunReader,
        DatabricksQualityMonitor,
        DeltaDriftMetricsWriter,
        DeltaTableReader,
        GitHubRepositoryDispatch,
    )
    from .monitoring.contract import get_monitoring_config
    from .monitoring.usecases import EvaluateDrift

    parser = argparse.ArgumentParser(prog="mlp-evaluate-drift")
    _common_args(parser)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--target_type", required=True, choices=["feature_table", "predictions"])
    # Repositório da esteira do domínio, no formato owner/repo. Vazio desliga o
    # retreino automático: o componente só mede e registra.
    parser.add_argument("--retrain_repository", default="")
    args = parser.parse_args(argv)

    _load(args)
    # O domínio da config vem do registro, e não da flag: `--domain` carrega o
    # nome do ENTRY POINT do pacote, que é outra coisa. Duas fontes para o mesmo
    # fato acabariam discordando.
    config = get_monitoring_config(_domain_of("monitoring"), args.model_name, args.target_type)
    spark = _spark()

    print(
        f"[mlplatform] evaluate_drift model={args.model_name} domain={config.domain} "
        f"target={config.target_table} type={config.target_type} "
        f"columns={config.columns} threshold={config.threshold}",
        flush=True,
    )

    # O token vem do ambiente, não de parâmetro de job: parâmetro de job fica
    # visível na UI e no histórico de execuções de quem tiver acesso ao job.
    # No Databricks isso se resolve com um secret scope montado como variável de
    # ambiente na task.
    github_token = os.environ.get("GITHUB_TOKEN", "")
    retrain = None
    if args.retrain_repository and github_token:
        retrain = GitHubRepositoryDispatch(args.retrain_repository, github_token)
    elif args.retrain_repository:
        print(
            "[mlplatform] AVISO: retrain_repository declarado mas GITHUB_TOKEN ausente — "
            "drift será medido e registrado, mas nenhum retreino será pedido",
            flush=True,
        )

    results = EvaluateDrift(
        retrain=retrain,
        runs=AuditTrainingRunReader(spark),
        monitor=DatabricksQualityMonitor(spark, args.reader_group),
        reader=DeltaTableReader(spark),
        writer=DeltaDriftMetricsWriter(spark),
        audit=DeltaAuditStore(spark),
        clock=SystemClock(),
    ).execute(
        config=config,
        catalog=args.catalog,
        run_id=_run_id(),
        git_commit=args.git_commit,
        git_branch=args.git_branch,
    )

    drifted = [r.column_name for r in results if r.status == "DRIFT_DETECTED"]
    print(
        f"[mlplatform] {len(results)} colunas avaliadas"
        + (f", drift em {drifted}" if drifted else ", sem drift"),
        flush=True,
    )
    return 0


def model_version(argv: list[str] | None = None) -> int:
    """Imprime a versão mais recente do modelo, uma linha, sem mais nada.

    Existe para a esteira poder FIXAR a versão que vai promover. Sem isso, o
    passo de promoção resolveria "a mais recente" no momento da aprovação — que
    pode ser outra, se um treino agendado rodou nesse meio-tempo. Quem aprovou
    inspecionou uma versão específica; é essa que tem que ir.
    """
    from databricks.sdk import WorkspaceClient

    from .core.naming import derive_model_name

    parser = argparse.ArgumentParser(prog="mlp-model-version")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--domain", required=True, help="domínio da config, não o entry point")
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args(argv)

    full_name = derive_model_name(args.catalog, args.domain, args.model_name)
    versions = WorkspaceClient().model_versions.list(full_name)
    newest = max((v.version for v in versions), default=None)
    if newest is None:
        raise SystemExit(f"nenhuma versão registrada em {full_name}")
    print(newest)
    return 0


def promote_model(argv: list[str] | None = None) -> int:
    """Aponta o alias para uma versão específica.

    Separado do treino de propósito: é o passo que fica atrás da aprovação
    manual do GitHub Environment. Treinar produz um candidato; promover é que
    coloca em produção, e as duas coisas precisam poder acontecer em momentos
    diferentes.
    """
    from databricks.sdk import WorkspaceClient

    from .core.naming import derive_model_name, validate_qualified_name

    parser = argparse.ArgumentParser(prog="mlp-promote-model")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--domain", required=True, help="domínio da config, não o entry point")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args(argv)

    full_name = derive_model_name(args.catalog, args.domain, args.model_name)
    validate_qualified_name(full_name, kind="model name")

    WorkspaceClient().registered_models.set_alias(full_name, args.alias, int(args.version))
    print(f"[mlplatform] {full_name}@{args.alias} -> v{args.version}", flush=True)
    return 0


NO_PROMOTION = "none"


def _with_promotion_override(config, override: str):
    if not override:
        return config
    return replace(config, promotion_alias=None if override == NO_PROMOTION else override)


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

    parser = _training_parser("mlp-select-test-register")
    # Sobrescreve o alias declarado na config:
    #   ""      -> usa o que o domínio declarou (execução agendada normal)
    #   "none"  -> registra e NÃO promove
    #   outro   -> promove para esse alias
    # O modo "none" é o do retreino disparado por drift, em que a promoção fica
    # atrás de uma aprovação humana no GitHub. Sentinela explícita em vez de
    # vazio-significa-não-promover: o Databricks injeta job parameters não
    # preenchidos como string vazia, e isso faria toda execução agendada parar
    # de promover sem ninguém pedir.
    parser.add_argument("--promotion_alias", default="")
    args = parser.parse_args(argv)
    _load(args)
    config, spark, deps = _training_context(args)

    name = SelectTestAndRegister(
        scratch=deps["scratch"],
        tracker=deps["tracker"],
        publisher=FeatureEngineeringPublisher(spark, args.reader_group),
        builder=deps["builder"],
        audit=DeltaAuditStore(spark),
        clock=SystemClock(),
        channel=deps["channel"],
    ).execute(
        # `--promotion_alias` sobrescreve a config quando presente; ausente
        # mantém o que o domínio declarou.
        config=_with_promotion_override(config, args.promotion_alias),
        catalog=args.catalog,
        scorer=_training_scorer(config),
        run_id=_run_id(),
        git_commit=args.git_commit,
        git_branch=args.git_branch,
    )
    print(f"[mlplatform] modelo registrado: {name}", flush=True)
    return 0
