"""Implementações reais dos ports de features.

Todos os imports de infraestrutura (pyspark, delta, databricks-sdk) ficam dentro
dos métodos. Ver a nota em `core/adapters.py`: o framework é embarcado no
artefato MLflow e importado no container do endpoint de serving, onde essas
bibliotecas não existem.

O SQL daqui é o código que os testes com fake NÃO exercitam. Ele continua
coberto só por execução real no Databricks — mover para trás de um port torna o
caso de uso testável, mas não testa o adapter. Ver a seção de riscos no plano.
"""

from typing import Any

import pandas as pd

from mlplatform.core.uc import ensure_schema

from .modes import WriteMode, WriteSemantics, write_strategy_for


class SparkSourceReader:
    def __init__(self, spark):
        self._spark = spark

    def read(self, table_name: str) -> Any:
        return self._spark.table(table_name)

    def to_pandas(self, df: Any) -> pd.DataFrame:
        return df.toPandas()


class DeltaFeatureWriter:
    def __init__(self, spark, reader_group: str | None = None):
        self._spark = spark
        # Grupo do domínio que recebe leitura nos schemas criados aqui. `None`
        # não concede — é o certo em workspace pessoal, onde não há grupo.
        self._reader_group = reader_group

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
        # saveAsTable não cria o schema em Unity Catalog — sem isso, a primeira
        # escrita de um domínio novo falha com SCHEMA_NOT_FOUND.
        catalog, schema, _table = table_name.split(".")
        ensure_schema(self._spark, f"{catalog}.{schema}", self._reader_group)

        if write_strategy_for(mode) is WriteSemantics.MERGE:
            self._merge(df, table_name, entity_keys, timestamp_key)
        else:
            self._overwrite_by_partition(df, table_name, partition_cols)

        self._ensure_primary_key(table_name, entity_keys, timestamp_key)

        if enable_cdf:
            # A synced table do Lakebase depende de Change Data Feed para
            # atualização incremental — sem isso, create_synced_database_table
            # falha com InvalidParameterValue.
            self._spark.sql(
                f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )

    def tag_provenance(self, table_name: str, git_commit: str, git_branch: str) -> None:
        self._spark.sql(
            f"ALTER TABLE {table_name} SET TBLPROPERTIES "
            f"('git_commit' = '{git_commit}', 'git_branch' = '{git_branch}')"
        )

    def _merge(self, df: Any, table_name: str, entity_keys: list[str], timestamp_key: str) -> None:
        from delta.tables import DeltaTable

        if not self._spark.catalog.tableExists(table_name):
            df.write.format("delta").saveAsTable(table_name)
            return

        target = DeltaTable.forName(self._spark, table_name)
        condition = " AND ".join(f"target.{k} = source.{k}" for k in [*entity_keys, timestamp_key])
        (
            target.alias("target")
            .merge(df.alias("source"), condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def _overwrite_by_partition(self, df: Any, table_name: str, partition_cols: list[str]) -> None:
        # spark.conf.set desta config global não é permitido em serverless /
        # Spark Connect (CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION). A opção no
        # próprio writer já basta para o modo dinâmico por partição.
        writer = df.write.format("delta").mode("overwrite").option("partitionOverwriteMode", "dynamic")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.saveAsTable(table_name)

    def _has_primary_key(self, catalog: str, schema: str, table: str) -> bool:
        rows = self._spark.sql(
            f"""
            SELECT 1 FROM system.information_schema.table_constraints
            WHERE table_catalog = '{catalog}' AND table_schema = '{schema}'
              AND table_name = '{table}' AND constraint_type = 'PRIMARY KEY'
            """
        ).collect()
        return len(rows) > 0

    def _ensure_primary_key(self, table_name: str, entity_keys: list[str], timestamp_key: str) -> None:
        # O Feature Engineering em Unity Catalog só reconhece a tabela como
        # feature table via FeatureLookup se ela tiver PRIMARY KEY — sem isso,
        # create_training_set falha com BAD_REQUEST. A coluna de timestamp
        # precisa ser TIMESERIES para o FE entender timestamp_lookup_key.
        # Idempotente: escritas repetidas não tentam recriar a constraint.
        catalog, schema, table = table_name.split(".")
        if self._has_primary_key(catalog, schema, table):
            return
        for col in [*entity_keys, timestamp_key]:
            self._spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {col} SET NOT NULL")
        key_list = ", ".join(entity_keys) + f", {timestamp_key} TIMESERIES"
        self._spark.sql(
            f"ALTER TABLE {table_name} ADD CONSTRAINT {schema}_{table}_pk PRIMARY KEY ({key_list})"
        )


def build_synced_table_spec(table_name: str, primary_keys: list[str]) -> dict:
    """Separado do adapter para ser testável sem SDK nem rede."""
    return {
        "source_table_full_name": table_name,
        "primary_key_columns": primary_keys,
        "scheduling_policy": "TRIGGERED",
    }


class LakebaseOnlineStore:
    def sync(self, table_name: str, primary_keys: list[str], database_instance_name: str) -> None:
        """Cria ou sincroniza a synced table no Lakebase.

        `logical_database_name` não é parâmetro livre: é sempre o catalog da
        tabela de origem. Model Serving não resolve FeatureLookup automático
        contra uma synced table cuja database Lakebase difere do catalog UC
        ("Reading online tables whose Lakebase database differs from its catalog
        name is not supported" — confirmado ao vivo).
        """
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.database import (
            SyncedDatabaseTable,
            SyncedTableSchedulingPolicy,
            SyncedTableSpec,
        )

        if not database_instance_name:
            raise ValueError("sync requires a non-empty database_instance_name")

        spec_fields = build_synced_table_spec(table_name, primary_keys)
        # build_synced_table_spec devolve scheduling_policy como string pura para
        # ficar testável sem SDK; SyncedTableSpec exige o enum — passar a string
        # quebra com AttributeError dentro do as_dict() do próprio SDK.
        spec_fields["scheduling_policy"] = SyncedTableSchedulingPolicy(spec_fields["scheduling_policy"])

        WorkspaceClient().database.create_synced_database_table(
            synced_table=SyncedDatabaseTable(
                name=f"{table_name}_online",
                database_instance_name=database_instance_name,
                logical_database_name=table_name.split(".")[0],
                spec=SyncedTableSpec(**spec_fields),
            )
        )
