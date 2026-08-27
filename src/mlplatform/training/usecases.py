import json
from datetime import date

from mlplatform.core.audit import RunRecord
from mlplatform.core.naming import derive_model_name, validate_qualified_name
from mlplatform.core.ports import AuditStore, Clock
from mlplatform.core.quality import Finding, gate_passed

from .contract import TrainingConfig
from .features import feature_columns
from .pipeline import build_pipeline
from .ports import ExperimentTracker, ModelPublisher, ScratchStore, TaskChannel, TrainingSetBuilder
from .quality import run_sanity_gate
from .selection import select_best
from .split import assign_split, compute_split_dates

COMPONENT = "training"

# Chaves trocadas entre as tasks. Ficam nomeadas aqui, e não como strings soltas
# em cada notebook, porque um typo de um lado só aparecia em runtime na task
# seguinte, com um KeyError sem relação aparente.
RUN_ID_KEY = "mlflow_run_id"
WINDOW_START_KEY = "window_start"
WINDOW_END_KEY = "window_end"
RESULTS_KEY = "hyperparameter_results"

PREPARE_TASK = "prepare_training_set"
FIT_TASK = "fit_and_compare"


class SanityGateFailure(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__(f"sanity gate failed: {[f.check for f in findings if f.status == 'FAIL']}")


def _scratch_prefix(catalog: str, model_name: str) -> str:
    return f"{catalog}.training_scratch.{model_name}"


class PrepareTrainingSet:
    """Monta o training set, divide cronologicamente e materializa os splits."""

    def __init__(
        self,
        builder: TrainingSetBuilder,
        scratch: ScratchStore,
        tracker: ExperimentTracker,
        channel: TaskChannel,
    ):
        self._builder = builder
        self._scratch = scratch
        self._tracker = tracker
        self._channel = channel

    def execute(self, config: TrainingConfig, catalog: str) -> None:
        training_set = self._builder.build(config.spine_table, config)
        df = self._builder.to_pandas(training_set)

        dates = sorted({d for d in df[config.reference_date_column]})
        train_end, val_end = compute_split_dates(
            dates, config.train_pct, config.val_pct, config.test_pct
        )
        df["_split"] = [assign_split(d, train_end, val_end) for d in df[config.reference_date_column]]

        prefix = _scratch_prefix(catalog, config.model_name)
        for split in ("train", "val", "test"):
            self._scratch.write(df[df["_split"] == split].drop(columns=["_split"]), f"{prefix}_{split}")

        run_id = self._tracker.start_run(f"/Shared/mlplatform/{config.domain}/{config.model_name}")
        self._channel.set(RUN_ID_KEY, run_id)
        self._channel.set(WINDOW_START_KEY, str(min(dates)))
        self._channel.set(WINDOW_END_KEY, str(max(dates)))


class FitAndCompare:
    """Fita cada combinação de hiperparâmetros e mede na validação."""

    def __init__(self, scratch: ScratchStore, tracker: ExperimentTracker, channel: TaskChannel):
        self._scratch = scratch
        self._tracker = tracker
        self._channel = channel

    def execute(self, config: TrainingConfig, catalog: str, scorer) -> list[tuple[dict, float]]:
        prefix = _scratch_prefix(catalog, config.model_name)
        train, val = self._scratch.read(f"{prefix}_train"), self._scratch.read(f"{prefix}_val")
        cols = feature_columns(list(train.columns), config)

        run_id = self._channel.get(PREPARE_TASK, RUN_ID_KEY)
        metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"

        results: list[tuple[dict, float]] = []
        for hyperparams in config.hyperparameter_sets:
            pipeline = build_pipeline(config.custom_transforms, config.algorithm(**hyperparams))
            pipeline.fit(train[cols], train[config.label_column])
            value = (
                float(scorer(pipeline, val[cols], val[config.label_column]))
                if len(val) > 0
                else float("nan")
            )
            self._tracker.log_params(run_id, hyperparams)
            self._tracker.log_metric(run_id, metric_name, value)
            results.append((hyperparams, value))

        self._channel.set(RESULTS_KEY, json.dumps(results))
        return results


class SelectTestAndRegister:
    """Escolhe o melhor, avalia no teste, aplica o gate e registra — o MESMO objeto.

    Antes eram duas tasks: uma fitava um pipeline, media no teste e aprovava o
    gate; a outra fitava OUTRO pipeline com os mesmos hiperparâmetros e
    registrava esse. Com um estimador sem `random_state` — o caso do domínio de
    exemplo — os dois objetos são diferentes, e o modelo que passou no gate era
    descartado. O que ia para produção nunca tinha sido avaliado.

    Juntar as duas etapas resolve isso por construção, e de quebra elimina um
    dos três fits que o pipeline fazia do mesmo modelo.
    """

    def __init__(
        self,
        scratch: ScratchStore,
        tracker: ExperimentTracker,
        publisher: ModelPublisher,
        builder: TrainingSetBuilder,
        audit: AuditStore,
        clock: Clock,
        channel: TaskChannel,
    ):
        self._scratch = scratch
        self._tracker = tracker
        self._publisher = publisher
        self._builder = builder
        self._audit = audit
        self._clock = clock
        self._channel = channel

    def execute(
        self, config: TrainingConfig, catalog: str, scorer, run_id: str, git_commit: str, git_branch: str
    ) -> str:
        mlflow_run_id = self._channel.get(PREPARE_TASK, RUN_ID_KEY)
        window_start = date.fromisoformat(self._channel.get(PREPARE_TASK, WINDOW_START_KEY))
        window_end = date.fromisoformat(self._channel.get(PREPARE_TASK, WINDOW_END_KEY))
        results = [(h, m) for h, m in json.loads(self._channel.get(FIT_TASK, RESULTS_KEY))]

        best = select_best(results, config.metric_direction)

        prefix = _scratch_prefix(catalog, config.model_name)
        train, test = self._scratch.read(f"{prefix}_train"), self._scratch.read(f"{prefix}_test")
        cols = feature_columns(list(train.columns), config)

        pipeline = build_pipeline(config.custom_transforms, config.algorithm(**best))
        pipeline.fit(train[cols], train[config.label_column])

        test_metric = (
            float(scorer(pipeline, test[cols], test[config.label_column]))
            if len(test) > 0
            else float("nan")
        )
        findings = run_sanity_gate(test_metric, num_predictions=len(test))

        full_model_name = derive_model_name(catalog, config.domain, config.model_name)
        validate_qualified_name(full_model_name, kind="model name")

        if not gate_passed(findings):
            self._record(full_model_name, "FAILED", window_start, window_end, run_id, git_commit, git_branch)
            raise SanityGateFailure(findings)

        metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"
        self._tracker.log_params(mlflow_run_id, best, prefix="best__")
        self._tracker.log_metric(mlflow_run_id, f"test_{metric_name}", test_metric)

        wrapped = (config.pyfunc_model_class or _default_model_class())(pipeline)
        self._publisher.publish(
            model=wrapped,
            training_set=self._builder.build(config.spine_table, config),
            full_model_name=full_model_name,
            run_id=mlflow_run_id,
            git_commit=git_commit,
            git_branch=git_branch,
        )
        self._record(full_model_name, "SUCCESS", window_start, window_end, run_id, git_commit, git_branch)
        return full_model_name

    def _record(
        self,
        entity: str,
        status: str,
        window_start: date,
        window_end: date,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> None:
        self._audit.append(
            RunRecord(
                component=COMPONENT,
                entity_name=entity,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode="train",
                status=status,
                window_start=window_start,
                window_end=window_end,
                run_ts=self._clock.now(),
            )
        )


def _default_model_class():
    # Import tardio: pyfunc_model importa mlflow, e o caso de uso não deve puxar
    # mlflow só por existir.
    from .pyfunc_model import FeaturePlatformModel

    return FeaturePlatformModel
