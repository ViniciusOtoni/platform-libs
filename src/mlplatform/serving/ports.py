"""Ports do contexto de serving.

Separados por capacidade, e não reunidos num "DatabricksGateway": o SDK é usado
para coisas sem relação entre si — resolver a versão de um alias, atualizar um
endpoint, pontuar em lote. Um port gordo obrigaria todo fake a implementar os
três, mesmo num teste que exercita um só.
"""

from typing import Any, Protocol

import pandas as pd


class BatchScorer(Protocol):
    """Pontuação em lote com resolução automática de FeatureLookup."""

    def score(self, model_uri: str, spine: Any) -> Any: ...

    def count(self, df: Any) -> int: ...

    def to_pandas(self, df: Any) -> pd.DataFrame: ...

    def read_table(self, table_name: str) -> Any: ...


class PredictionWriter(Protocol):
    def append(self, df: Any, table_name: str) -> None: ...


class ModelRegistry(Protocol):
    """Resolve um alias para a versão numérica vigente.

    DABs não aceita `models:/nome@alias` em `model_serving_endpoints` — só um
    `entity_version` numérico (confirmado ao vivo: 404 RESOURCE_DOES_NOT_EXIST
    tentando o alias). Por isso a resolução acontece na geração do recurso, e não
    é um detalhe interno do adapter.
    """

    def version_for_alias(self, full_model_name: str, alias: str) -> int: ...


class EndpointGateway(Protocol):
    """Atualiza o endpoint para a resolução corrente do alias.

    Existe porque o `entity_version` do recurso congela no momento da geração:
    mover o alias depois exige este passo, ou gerar os recursos de novo.
    """

    def update_to_alias(self, endpoint_name: str, model_name: str, full_model_name: str, alias: str) -> None: ...
