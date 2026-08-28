from datetime import date

from mlplatform.core.audit import RunRecord
from mlplatform.core.naming import derive_model_name, derive_predictions_table_name
from mlplatform.core.ports import AuditStore, Clock
from mlplatform.core.quality import Finding, gate_passed

from .contract import BatchServingConfig, OnlineServingConfig
from .naming import validate_endpoint_name
from .ports import BatchScorer, EndpointGateway, ModelRegistry, PredictionWriter
from .quality import run_predictions_gate, run_structure_gate

COMPONENT = "serving"
PREDICTION_COLUMN = "prediction"


class PredictionsGateFailure(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__(f"predictions quality gate failed: {[f.check for f in findings if f.status == 'FAIL']}")


class ScoreBatch:
    """Pontua a spine de inferência e grava as predições, com gate antes da escrita.

    Era ~60 linhas de notebook no repositório de domínio, copiadas por domínio.
    """

    def __init__(
        self,
        scorer: BatchScorer,
        writer: PredictionWriter,
        registry: ModelRegistry,
        audit: AuditStore,
        clock: Clock,
    ):
        self._scorer = scorer
        self._writer = writer
        self._registry = registry
        self._audit = audit
        self._clock = clock

    def execute(
        self,
        config: BatchServingConfig,
        catalog: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
        today: date | None = None,
    ) -> None:
        today = today or self._clock.now().date()
        full_model_name = derive_model_name(catalog, config.domain, config.model_name)
        predictions_table = derive_predictions_table_name(catalog, config.domain, config.model_name)

        # A versão é resolvida aqui, e não deixada implícita no alias: ela vai
        # gravada em cada linha da saída. Sem isso o monitoramento não consegue
        # atribuir uma mudança de score a troca de modelo em vez de drift.
        model_version = self._registry.version_for_alias(full_model_name, config.alias)

        spine = self._scorer.read_table(config.spine_inference_table)
        input_rows = self._scorer.count(spine)
        predictions = self._scorer.score(
            f"models:/{full_model_name}@{config.alias}", spine, model_version
        )

        scored = self._scorer.to_pandas(predictions)
        # Estrutura antes de conteúdo: uma coluna declarada e ausente invalida
        # os checks seguintes, que passariam a medir outra coisa.
        findings = [
            *run_structure_gate(scored, config.output),
            *run_predictions_gate(scored, PREDICTION_COLUMN, input_rows),
        ]

        if not gate_passed(findings):
            self._record(predictions_table, "FAILED", today, run_id, git_commit, git_branch)
            raise PredictionsGateFailure(findings)

        self._writer.append(predictions, predictions_table)
        self._record(predictions_table, "SUCCESS", today, run_id, git_commit, git_branch)

    def _record(
        self, entity: str, status: str, day: date, run_id: str, git_commit: str, git_branch: str
    ) -> None:
        self._audit.append(
            RunRecord(
                component=COMPONENT,
                entity_name=entity,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode="batch",
                status=status,
                window_start=day,
                window_end=day,
                run_ts=self._clock.now(),
            )
        )


class RefreshEndpoint:
    """Reaponta o endpoint para a resolução corrente do alias.

    O `entity_version` do recurso congela no momento em que os recursos são
    gerados. Mover o alias `champion` para outra versão depois disso não muda o
    endpoint sozinho — este use case é o que fecha essa lacuna.
    """

    def __init__(self, gateway: EndpointGateway, registry: ModelRegistry):
        self._gateway = gateway
        self._registry = registry

    def execute(self, config: OnlineServingConfig, catalog: str, endpoint_name: str) -> str:
        # O nome vem de fora, resolvido pelo DABs no deploy. Derivá-lo aqui
        # ignorava o prefixo de target (`dev_<usuario>_`) e procurava um
        # endpoint que não existe com aquele nome.
        validate_endpoint_name(endpoint_name)
        full_model_name = derive_model_name(catalog, config.domain, config.model_name)

        # O alias é resolvido aqui: a API de serving não o aceita em
        # `entity_name` — trata `<modelo>@champion` como nome de modelo e
        # responde que ele não existe.
        version = self._registry.version_for_alias(full_model_name, config.alias)

        self._gateway.update_to_version(
            endpoint_name=endpoint_name,
            model_name=config.model_name,
            full_model_name=full_model_name,
            version=version,
        )
        return endpoint_name
