from dataclasses import dataclass
from typing import Literal


@dataclass
class MonitoringConfig:
    domain: str
    model_name: str
    target_type: Literal["feature_table", "predictions"]
    target_table: str
    columns: list[str]
    threshold: float
    schedule_cron: str


def _config_key(domain: str, model_name: str, target_type: str) -> str:
    return f"{domain}.{model_name}.{target_type}"


_REGISTRY: dict[str, MonitoringConfig] = {}


def register_monitoring_config(config: MonitoringConfig) -> None:
    key = _config_key(config.domain, config.model_name, config.target_type)
    if key in _REGISTRY:
        raise ValueError(f"monitoring config '{key}' already registered")
    _REGISTRY[key] = config


def get_monitoring_config(domain: str, model_name: str, target_type: str) -> MonitoringConfig:
    return _REGISTRY[_config_key(domain, model_name, target_type)]


def get_registry() -> dict[str, MonitoringConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
