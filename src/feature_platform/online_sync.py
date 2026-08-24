def build_synced_table_spec(table_name: str, primary_keys: list[str]) -> dict:
    """Monta a especificação de uma synced table para o Online Feature Store
    (Lakebase). Separado de sync_online_table() para ser testável sem SDK/rede."""
    return {
        "source_table_full_name": table_name,
        "primary_key_columns": primary_keys,
        "scheduling_policy": "TRIGGERED",
    }


def sync_online_table(spark, table_name: str, primary_keys: list[str]) -> None:
    """Cria ou sincroniza a synced table no Lakebase para a feature table informada.
    Requer databricks-sdk e um workspace real — exercitado via notebook (Task 12), não
    via pytest. A superfície exata da API (`WorkspaceClient().database.*`) deve ser
    conferida contra a versão do databricks-sdk instalada antes do primeiro deploy real,
    porque o Online Feature Store via Lakebase é uma API recente do Databricks."""
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    spec = build_synced_table_spec(table_name, primary_keys)
    client.database.create_synced_database_table(
        name=f"{table_name}_online",
        spec=spec,
    )
