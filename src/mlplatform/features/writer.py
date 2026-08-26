from enum import StrEnum


class WriteMode(StrEnum):
    INCREMENTAL = "incremental"
    BACKFILL = "backfill"


def write_strategy_for(mode: WriteMode) -> str:
    if mode == WriteMode.INCREMENTAL:
        return "merge"
    if mode == WriteMode.BACKFILL:
        return "overwrite_by_partition"
    raise ValueError(f"unknown mode: {mode}")


def write_feature_table(
    spark,
    df,
    table_name: str,
    entity_keys: list[str],
    timestamp_key: str,
    mode: WriteMode,
    partition_cols: list[str],
    enable_cdf: bool = False,
) -> None:
    """Escreve a feature table no Delta. Requer SparkSession com Delta habilitado —
    exercitado via notebook (Task 12), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — sem isso,
    # a primeira escrita de um domínio novo falha com SCHEMA_NOT_FOUND.
    catalog, schema, _table = table_name.split(".")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    strategy = write_strategy_for(mode)
    if strategy == "merge":
        _merge(spark, df, table_name, entity_keys, timestamp_key)
    else:
        _overwrite_by_partition(spark, df, table_name, partition_cols)
    _ensure_primary_key(spark, table_name, entity_keys, timestamp_key)
    if enable_cdf:
        # A synced table do Lakebase (Online Feature Store) depende de Change Data
        # Feed pra suportar atualizações incrementais — sem isso, create_synced_database_table
        # falha com InvalidParameterValue. Só habilitado para feature tables com online=True.
        spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")


def _merge(spark, df, table_name: str, entity_keys: list[str], timestamp_key: str) -> None:
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(table_name):
        df.write.format("delta").saveAsTable(table_name)
        return

    target = DeltaTable.forName(spark, table_name)
    merge_keys = [*entity_keys, timestamp_key]
    condition = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
    (
        target.alias("target")
        .merge(df.alias("source"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def _overwrite_by_partition(spark, df, table_name: str, partition_cols: list[str]) -> None:
    # spark.conf.set(...) para esta config global não é permitido em
    # serverless/Spark Connect (AnalysisException: CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION).
    # O .option(...) abaixo, no próprio writer, já basta para o modo dinâmico por partição.
    writer = df.write.format("delta").mode("overwrite").option("partitionOverwriteMode", "dynamic")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(table_name)


def _has_primary_key(spark, catalog: str, schema: str, table: str) -> bool:
    rows = spark.sql(
        f"""
        SELECT 1 FROM system.information_schema.table_constraints
        WHERE table_catalog = '{catalog}' AND table_schema = '{schema}'
          AND table_name = '{table}' AND constraint_type = 'PRIMARY KEY'
        """
    ).collect()
    return len(rows) > 0


def _ensure_primary_key(spark, table_name: str, entity_keys: list[str], timestamp_key: str) -> None:
    # O Feature Engineering em Unity Catalog só reconhece uma tabela como feature table
    # via FeatureLookup se ela tiver uma PRIMARY KEY — sem isso, create_training_set
    # falha com BAD_REQUEST "no primary key constraint defined". A coluna de timestamp
    # precisa ser marcada TIMESERIES para o FE entender a semântica de
    # timestamp_lookup_key usada no lookup. Idempotente: escritas repetidas na mesma
    # tabela não tentam recriar a constraint.
    catalog, schema, table = table_name.split(".")
    if _has_primary_key(spark, catalog, schema, table):
        return
    for col in [*entity_keys, timestamp_key]:
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {col} SET NOT NULL")
    constraint_name = f"{schema}_{table}_pk"
    key_list = ", ".join(entity_keys) + f", {timestamp_key} TIMESERIES"
    spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} PRIMARY KEY ({key_list})")
