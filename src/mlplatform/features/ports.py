"""Ports do contexto de features.

Moram aqui, e não no `core/`, porque só features os consome. Um port no shared
kernel obriga todo contexto a conhecer o vocabulário dele.

São três pequenos em vez de um "DatabricksGateway": separados por capacidade,
cada fake de teste implementa só o que aquele teste exercita.
"""

from typing import Any, Protocol

import pandas as pd

from .modes import WriteMode


class SourceReader(Protocol):
    """Entrada e saída do motor de dataframes.

    `to_pandas` mora aqui, junto de `read`, porque as duas são a mesma coisa:
    mover dados através da fronteira do motor de execução. O gate de qualidade
    opera em pandas, então em algum ponto o resultado precisa atravessar — e o
    caso de uso não deve saber que o outro lado é Spark.
    """

    def read(self, table_name: str) -> Any: ...

    def to_pandas(self, df: Any) -> pd.DataFrame: ...


class FeatureWriter(Protocol):
    """Persistência da feature table."""

    def write(
        self,
        df: Any,
        table_name: str,
        entity_keys: list[str],
        timestamp_key: str,
        mode: WriteMode,
        partition_cols: list[str],
        enable_cdf: bool,
    ) -> None: ...

    def tag_provenance(self, table_name: str, git_commit: str, git_branch: str) -> None:
        """Grava commit/branch como propriedades da tabela, para rastrear qual
        versão do código produziu os dados que estão lá."""
        ...


class OnlineStore(Protocol):
    """Sincronização com o Online Feature Store (Lakebase). Só exercitado por
    feature tables declaradas com `online=True`."""

    def sync(
        self,
        table_name: str,
        primary_keys: list[str],
        database_instance_name: str,
        timeseries_key: str | None = None,
    ) -> None: ...
