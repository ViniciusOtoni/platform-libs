"""Implementações reais dos ports de serving.

Imports de infraestrutura ficam dentro dos métodos — ver a nota em
`core/adapters.py` sobre o container do endpoint de serving.
"""

from typing import Any

import pandas as pd

from .structure import MODEL_VERSION_COLUMN, SCORED_AT_COLUMN


class FeatureEngineeringScorer:
    """Pontuação via Feature Engineering em Unity Catalog.

    `fe.score_batch` resolve os FeatureLookup do modelo automaticamente: a spine
    carrega só as chaves de entidade e o timestamp, e as features vêm da feature
    table (ou da Online Feature Store, no caso online).
    """

    def __init__(self, spark):
        self._spark = spark

    def read_table(self, table_name: str) -> Any:
        return self._spark.table(table_name)

    def count(self, df: Any) -> int:
        return df.count()

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df.toPandas()

    def score(self, model_uri: str, spine: Any, model_version: int) -> Any:
        import pyspark.sql.functions as F
        from databricks.feature_engineering import FeatureEngineeringClient

        # As duas colunas do framework nascem aqui, junto com a predição.
        # `model_version` é o que separa "os scores mudaram porque os dados
        # mudaram" de "os scores mudaram porque o modelo trocou" — sem ela, o
        # monitoramento não consegue distinguir os dois.
        return (
            FeatureEngineeringClient()
            .score_batch(model_uri=model_uri, df=spine, result_type="double")
            .withColumn(SCORED_AT_COLUMN, F.current_timestamp())
            .withColumn(MODEL_VERSION_COLUMN, F.lit(int(model_version)))
        )


class DeltaPredictionWriter:
    def __init__(self, spark):
        self._spark = spark

    def append(self, df: Any, table_name: str) -> None:
        # saveAsTable não cria o schema em Unity Catalog.
        schema = table_name.rsplit(".", 1)[0]
        self._spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        # mergeSchema porque o conjunto de colunas CRESCE por motivos rotineiros:
        # o domínio adiciona uma feature ao FeatureLookup, ou o framework passa a
        # gravar uma coluna nova (foi o caso de `model_version`). Sem ele o
        # append falha com DELTA_METADATA_MISMATCH e o job só volta a rodar
        # depois de alguém alterar a tabela à mão.
        #
        # Só ADIÇÃO de coluna: as linhas antigas leem NULL na coluna nova, o
        # histórico continua intacto, e a tabela segue append-only — que é a
        # propriedade de que o data drift depende. mergeSchema não reescreve nem
        # apaga nada.
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table_name)


class SdkModelRegistry:
    def version_for_alias(self, full_model_name: str, alias: str) -> int:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient().model_versions.get_by_alias(full_model_name, alias).version


class SdkEndpointGateway:
    def update_to_version(
        self, endpoint_name: str, model_name: str, full_model_name: str, version: int
    ) -> None:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ServedEntityInput

        WorkspaceClient().serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    name=model_name,
                    entity_name=full_model_name,
                    entity_version=str(version),
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ],
        )
