from monitoring_platform.naming import derive_monitor_key


def test_derive_monitor_key_follows_convention():
    key = derive_monitor_key(domain="credito", model_name="propensao_default", target_type="feature_table")
    assert key == "credito_propensao_default_feature_table"


def test_derive_monitor_key_differs_by_target_type():
    key_a = derive_monitor_key("credito", "propensao_default", "feature_table")
    key_b = derive_monitor_key("credito", "propensao_default", "predictions")
    assert key_a != key_b
