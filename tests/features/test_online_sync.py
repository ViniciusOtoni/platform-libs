from mlplatform.features.online_sync import build_synced_table_spec


def test_build_synced_table_spec_uses_table_name_as_source():
    spec = build_synced_table_spec("workspace.credito_features.score_features", primary_keys=["customer_id"])

    assert spec["source_table_full_name"] == "workspace.credito_features.score_features"
    assert spec["primary_key_columns"] == ["customer_id"]
    assert spec["scheduling_policy"] == "TRIGGERED"
