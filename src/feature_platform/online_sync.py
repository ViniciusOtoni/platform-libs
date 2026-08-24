def build_synced_table_spec(table_name: str, primary_keys: list[str]) -> dict:
    """Monta a especificação de uma synced table para o Online Feature Store
    (Lakebase). Separado de sync_online_table() para ser testável sem SDK/rede."""
    return {
        "source_table_full_name": table_name,
        "primary_key_columns": primary_keys,
        "scheduling_policy": "TRIGGERED",
    }


def sync_online_table(spark, table_name: str, primary_keys: list[str], database_instance_name: str) -> None:
    """Cria ou sincroniza a synced table no Lakebase para a feature table informada.
    Requer databricks-sdk, um workspace real, e um Database Instance do Lakebase já
    provisionado (database_instance_name) — exercitado via notebook (Task 12), não via
    pytest. Sem um Database Instance existente, a chamada falha com um erro claro do
    próprio SDK/API, não silenciosamente."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.database import SyncedDatabaseTable, SyncedTableSpec

    if not database_instance_name:
        raise ValueError("sync_online_table requires a non-empty database_instance_name")

    client = WorkspaceClient()
    spec_fields = build_synced_table_spec(table_name, primary_keys)
    synced_table = SyncedDatabaseTable(
        name=f"{table_name}_online",
        database_instance_name=database_instance_name,
        spec=SyncedTableSpec(**spec_fields),
    )
    client.database.create_synced_database_table(synced_table=synced_table)
