"""Ports compartilhados entre os bounded contexts.

Só entra aqui o que é consumido por mais de um contexto. Um port usado por um
contexto só mora no contexto — senão o shared kernel vira o vocabulário de todo
mundo, que é exatamente o acoplamento que a fronteira existe para impedir.
"""

from datetime import UTC, date, datetime
from typing import Protocol

from .audit import RunRecord


class AuditStore(Protocol):
    """Registro central de execuções. Escrito pelos quatro contextos e lido pelo
    monitoring, que resolve a janela de baseline a partir do último treino."""

    def append(self, record: RunRecord) -> None: ...

    def last_success_checkpoint(self, component: str, entity_name: str) -> date | None:
        """window_end do último run SUCCESS, ou None se não houver nenhum."""
        ...


class Clock(Protocol):
    """O tempo como dependência.

    Antes cada ponto de escrita chamava `datetime.utcnow()` por conta própria, o
    que tinha duas consequências: o caminho de falha e o de sucesso carimbavam
    instantes diferentes dentro da mesma execução, e não havia como um teste
    fixar o relógio. `utcnow()` também está deprecado desde o 3.12 — e o
    serverless do Databricks roda 3.12.
    """

    def now(self) -> datetime: ...


class SystemClock:
    """Implementação padrão. `datetime.now(UTC)` no lugar do `utcnow()` deprecado."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Relógio fixo para testes: a mesma execução carimba sempre o mesmo instante."""

    def __init__(self, instant: datetime):
        self._instant = instant

    def now(self) -> datetime:
        return self._instant
