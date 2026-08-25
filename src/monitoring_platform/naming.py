def derive_monitor_key(domain: str, model_name: str, target_type: str) -> str:
    return f"{domain}_{model_name}_{target_type}"
