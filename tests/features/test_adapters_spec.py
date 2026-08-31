"""Forma da spec da synced table, testavel sem SDK nem rede."""

from mlplatform.features.adapters import build_synced_table_spec


def test_without_a_timeseries_key_the_spec_stays_as_before():
    """Feature table sem coluna de tempo: a chave primaria e so a entidade."""
    spec = build_synced_table_spec("workspace.d_features.t", ["customer_id"])

    assert "timeseries_key" not in spec
    assert spec["primary_key_columns"] == ["customer_id"]


def test_the_timeseries_key_is_declared_when_there_is_one():
    """E o que torna a synced table uma feature table temporal do lado online.
    Sem isso, 24 safras viram 24 linhas para a mesma chave primaria."""
    spec = build_synced_table_spec("workspace.d_features.t", ["customer_id"], "feature_ts")

    assert spec["timeseries_key"] == "feature_ts"
    assert spec["primary_key_columns"] == ["customer_id"]
