import pytest

from feature_platform.contract import feature_table, get_registry, clear_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_feature_table_registers_spec_with_defaults():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.transactions"])
    def minha_feature(sources, window):
        return None

    registry = get_registry()
    spec = registry["minha_feature"]
    assert spec.domain == "exemplo"
    assert spec.entity_keys == ["customer_id"]
    assert spec.timestamp_key == "feature_ts"
    assert spec.sources == ["raw.transactions"]
    assert spec.online is False
    assert spec.depends_on == []
    assert spec.table_name is None
    assert spec.compute_fn is minha_feature


def test_feature_table_requires_domain():
    with pytest.raises(TypeError):
        feature_table(entity_keys=["k"], timestamp_key="ts", sources=[])


def test_feature_table_rejects_duplicate_registration():
    @feature_table(domain="exemplo", entity_keys=["k"], timestamp_key="ts", sources=[])
    def duplicada(sources, window):
        return None

    with pytest.raises(ValueError, match="already registered"):
        feature_table(domain="exemplo", entity_keys=["k"], timestamp_key="ts", sources=[])(duplicada)


def test_feature_table_accepts_online_and_depends_on_and_table_name():
    @feature_table(
        domain="exemplo",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.transactions"],
        online=True,
        depends_on=["outra_feature"],
        table_name="workspace.legado.minha_tabela",
    )
    def com_opcoes(sources, window):
        return None

    spec = get_registry()["com_opcoes"]
    assert spec.online is True
    assert spec.depends_on == ["outra_feature"]
    assert spec.table_name == "workspace.legado.minha_tabela"
