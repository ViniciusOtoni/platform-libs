from datetime import date

from mlplatform.core.audit import RunRecord
from mlplatform.core.ports import AuditStore, Clock
from mlplatform.core.quality import Finding, gate_passed

from .contract import FeatureTableSpec
from .modes import WriteMode
from .naming import resolve_table_name
from .ports import FeatureWriter, OnlineStore, SourceReader
from .quality import run_quality_gate
from .types import DateRange
from .window import parse_backfill_window, resolve_incremental_window

COMPONENT = "feature_generation"


class QualityGateFailure(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__(f"quality gate failed: {[f.check for f in findings if f.status == 'FAIL']}")


class RunFeatureTable:
    """Executa uma feature table de ponta a ponta: resolve a janela, computa,
    aplica o gate de qualidade, escreve, audita e sincroniza online.

    As dependências são declaradas na assinatura, uma a uma, e não empacotadas
    num objeto de contexto. Quem lê a classe vê o conjunto completo do que ela
    toca; um teste a constrói com fakes e nada mais. Passar um `PlatformContext`
    aqui transformaria injeção de dependência em service locator — a montagem
    dos adapters é trabalho do composition root, em `entrypoints.py`.

    `online` é opcional: só feature tables com `online=True` precisam dele, e um
    caso de uso não deve exigir uma dependência que talvez nunca use.
    """

    def __init__(
        self,
        reader: SourceReader,
        writer: FeatureWriter,
        audit: AuditStore,
        clock: Clock,
        online: OnlineStore | None = None,
    ):
        self._reader = reader
        self._writer = writer
        self._audit = audit
        self._clock = clock
        self._online = online

    def execute(
        self,
        spec: FeatureTableSpec,
        catalog: str,
        mode: WriteMode,
        today: date,
        run_id: str,
        git_commit: str,
        git_branch: str,
        backfill_start: str | None = None,
        backfill_end: str | None = None,
        database_instance_name: str = "",
    ) -> None:
        window = self._resolve_window(spec, mode, today, backfill_start, backfill_end)
        sources = {name: self._reader.read(name) for name in spec.sources}
        result = spec.compute_fn(sources, window)

        findings = run_quality_gate(
            self._reader.to_pandas(result), spec.entity_keys, spec.timestamp_key, window.end
        )
        table_name = resolve_table_name(catalog, spec.domain, spec.name, spec.table_name)

        if not gate_passed(findings):
            self._record(spec, mode, window, "FAILED", run_id, git_commit, git_branch)
            raise QualityGateFailure(findings)

        self._writer.write(
            df=result,
            table_name=table_name,
            entity_keys=spec.entity_keys,
            timestamp_key=spec.timestamp_key,
            mode=mode,
            partition_cols=spec.partition_cols(),
            enable_cdf=spec.online,
        )
        self._writer.tag_provenance(table_name, git_commit, git_branch)
        self._record(spec, mode, window, "SUCCESS", run_id, git_commit, git_branch)

        if spec.online:
            if self._online is None:
                raise ValueError(
                    f"feature table '{spec.name}' is declared online=True but no OnlineStore was provided"
                )
            self._online.sync(
                table_name, spec.entity_keys, database_instance_name, spec.timestamp_key
            )

    def _resolve_window(
        self,
        spec: FeatureTableSpec,
        mode: WriteMode,
        today: date,
        backfill_start: str | None,
        backfill_end: str | None,
    ) -> DateRange:
        if mode == WriteMode.BACKFILL:
            if not backfill_start or not backfill_end:
                raise ValueError("backfill mode requires start_date and end_date")
            return parse_backfill_window(backfill_start, backfill_end)
        return resolve_incremental_window(
            self._audit.last_success_checkpoint(COMPONENT, spec.name), today
        )

    def _record(
        self,
        spec: FeatureTableSpec,
        mode: WriteMode,
        window: DateRange,
        status: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> None:
        # Um único instante por execução: antes, cada ponto de escrita chamava
        # utcnow() por conta própria e o caminho de falha carimbava um horário
        # diferente do de sucesso.
        self._audit.append(
            RunRecord(
                component=COMPONENT,
                entity_name=spec.name,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode=mode.value,
                status=status,
                window_start=window.start,
                window_end=window.end,
                run_ts=self._clock.now(),
            )
        )
