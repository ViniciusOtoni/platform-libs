from enum import Enum


class WriteMode(str, Enum):
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
) -> None:
    """Escreve a feature table no Delta. Requer SparkSession com Delta habilitado —
    exercitado via notebook (Task 12), não via pytest."""
    strategy = write_strategy_for(mode)
    if strategy == "merge":
        _merge(spark, df, table_name, entity_keys, timestamp_key)
    else:
        _overwrite_by_partition(spark, df, table_name, partition_cols)


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
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    writer = df.write.format("delta").mode("overwrite").option("partitionOverwriteMode", "dynamic")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(table_name)
