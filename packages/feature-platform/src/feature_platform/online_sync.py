def build_synced_table_spec(table_name: str, primary_keys: list[str]) -> dict:
    """Monta a especificação de uma synced table para o Online Feature Store
    (Lakebase). Separado de sync_online_table() para ser testável sem SDK/rede."""
    return {
        "source_table_full_name": table_name,
        "primary_key_columns": primary_keys,
        "scheduling_policy": "TRIGGERED",
    }


def sync_online_table(
    spark,
    table_name: str,
    primary_keys: list[str],
    database_instance_name: str,
) -> None:
    """Cria ou sincroniza a synced table no Lakebase para a feature table informada.
    Requer databricks-sdk, um workspace real, e um Database Instance do Lakebase já
    provisionado (database_instance_name), com um Database Catalog já criado
    mapeando esse instance para uma database lógica cujo nome é IGUAL ao catalog UC
    da tabela de origem — Model Serving não resolve FeatureLookup automaticamente
    contra uma synced table cuja Lakebase (Postgres) database difere do catalog UC
    ("Reading online tables whose Lakebase database differs from its catalog name
    is not supported", confirmado ao vivo). Por isso `logical_database_name` não é
    um parâmetro livre: é sempre o catalog da própria table_name, derivado aqui —
    exercitado via notebook (Task 12), não via pytest. Sem essa infraestrutura já
    existente, a chamada falha com um erro claro do próprio SDK/API, não
    silenciosamente."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.database import SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy

    if not database_instance_name:
        raise ValueError("sync_online_table requires a non-empty database_instance_name")

    logical_database_name = table_name.split(".")[0]

    client = WorkspaceClient()
    spec_fields = build_synced_table_spec(table_name, primary_keys)
    # build_synced_table_spec() devolve scheduling_policy como string pura (pra ficar
    # testável sem SDK) — SyncedTableSpec exige o enum: passar a string direto quebra
    # com AttributeError: 'str' object has no attribute 'value' dentro do as_dict() do
    # SDK, na hora de serializar o request.
    spec_fields["scheduling_policy"] = SyncedTableSchedulingPolicy(spec_fields["scheduling_policy"])
    synced_table = SyncedDatabaseTable(
        name=f"{table_name}_online",
        database_instance_name=database_instance_name,
        logical_database_name=logical_database_name,
        spec=SyncedTableSpec(**spec_fields),
    )
    client.database.create_synced_database_table(synced_table=synced_table)
