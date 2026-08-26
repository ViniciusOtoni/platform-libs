"""Test doubles dos ports, publicados como API do framework.

Não são só para os testes do próprio framework: um repositório de domínio pode
importá-los para testar suas feature tables sem subir Spark. Se os fakes ficassem
em `tests/`, não iriam no wheel e cada domínio escreveria os seus — divergindo,
como já aconteceu com o `Finding`.

Estes fakes são deliberadamente burros: guardam o que receberam e devolvem o que
mandaram guardar. Fake que reimplementa a lógica do adapter passa a poder discordar
dele, e aí o teste verde deixa de significar alguma coisa.
"""

from datetime import date, datetime
from typing import Any

import pandas as pd

from .core.audit import RunRecord
from .features.modes import WriteMode


class InMemoryAuditStore:
    """AuditStore em memória. Substitui a tabela Delta que antes tornava
    `write_run` e `get_last_success_checkpoint` intestáveis."""

    def __init__(self, checkpoints: dict[tuple[str, str], date] | None = None):
        self.records: list[RunRecord] = []
        self._checkpoints = checkpoints or {}

    def append(self, record: RunRecord) -> None:
        self.records.append(record)

    def last_success_checkpoint(self, component: str, entity_name: str) -> date | None:
        return self._checkpoints.get((component, entity_name))

    def statuses(self) -> list[str]:
        return [r.status for r in self.records]


class FakeSourceReader:
    """Devolve dataframes pandas pré-carregados por nome de tabela."""

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None):
        self._tables = tables or {}
        self.read_tables: list[str] = []

    def read(self, table_name: str) -> Any:
        self.read_tables.append(table_name)
        return self._tables.get(table_name, pd.DataFrame())

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df


class FakeFeatureWriter:
    """Grava a chamada em vez de escrever em Delta."""

    def __init__(self) -> None:
        self.writes: list[dict] = []
        self.provenance: list[tuple[str, str, str]] = []

    def write(
        self,
        df: Any,
        table_name: str,
        entity_keys: list[str],
        timestamp_key: str,
        mode: WriteMode,
        partition_cols: list[str],
        enable_cdf: bool,
    ) -> None:
        self.writes.append(
            {
                "df": df,
                "table_name": table_name,
                "entity_keys": entity_keys,
                "timestamp_key": timestamp_key,
                "mode": mode,
                "partition_cols": partition_cols,
                "enable_cdf": enable_cdf,
            }
        )

    def tag_provenance(self, table_name: str, git_commit: str, git_branch: str) -> None:
        self.provenance.append((table_name, git_commit, git_branch))


class FakeOnlineStore:
    def __init__(self) -> None:
        self.syncs: list[tuple[str, list[str], str]] = []

    def sync(self, table_name: str, primary_keys: list[str], database_instance_name: str) -> None:
        self.syncs.append((table_name, primary_keys, database_instance_name))


class FixedClock:
    """Relógio parado: toda a execução carimba o mesmo instante."""

    def __init__(self, instant: datetime):
        self._instant = instant

    def now(self) -> datetime:
        return self._instant


class FakeBatchScorer:
    """Devolve predições pré-definidas em vez de chamar fe.score_batch."""

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None, predictions: pd.DataFrame | None = None):
        self._tables = tables or {}
        self._predictions = predictions if predictions is not None else pd.DataFrame()
        self.scored: list[tuple[str, Any]] = []

    def read_table(self, table_name: str) -> Any:
        return self._tables.get(table_name, pd.DataFrame())

    def count(self, df: Any) -> int:
        return len(df)

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df

    def score(self, model_uri: str, spine: Any) -> Any:
        self.scored.append((model_uri, spine))
        return self._predictions


class FakePredictionWriter:
    def __init__(self) -> None:
        self.appends: list[tuple[Any, str]] = []

    def append(self, df: Any, table_name: str) -> None:
        self.appends.append((df, table_name))


class FakeEndpointGateway:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update_to_alias(self, endpoint_name: str, model_name: str, full_model_name: str, alias: str) -> None:
        self.updates.append(
            {
                "endpoint_name": endpoint_name,
                "model_name": model_name,
                "full_model_name": full_model_name,
                "alias": alias,
            }
        )
