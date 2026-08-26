from dataclasses import dataclass
from datetime import date, datetime

# ATENÇÃO: esta dataclass é o schema de uma tabela Delta serializada. Mudar,
# remover ou reordenar um campo é uma migração de schema em `platform_audit.
# pipeline_runs` — silenciosa, porque nada aqui falha em tempo de import — e
# quebra o `toPandas()` que o monitoring faz sobre essa tabela para resolver a
# janela de baseline. Há um teste travando os campos justamente para que a
# mudança seja deliberada.
AUDIT_TABLE = "platform_audit.pipeline_runs"


@dataclass(frozen=True)
class RunRecord:
    """Evento de domínio: uma execução de pipeline que terminou."""

    component: str
    entity_name: str
    git_commit: str
    git_branch: str
    run_id: str
    mode: str
    status: str
    window_start: date
    window_end: date
    run_ts: datetime


def to_row(record: RunRecord) -> dict:
    return {
        "component": record.component,
        "entity_name": record.entity_name,
        "git_commit": record.git_commit,
        "git_branch": record.git_branch,
        "run_id": record.run_id,
        "mode": record.mode,
        "status": record.status,
        "window_start": record.window_start,
        "window_end": record.window_end,
        "run_ts": record.run_ts,
    }
