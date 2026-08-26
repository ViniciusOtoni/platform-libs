from dataclasses import dataclass
from typing import Literal

from mlplatform.core.registry import Registry


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


_registry: Registry[MonitoringConfig] = Registry(kind="monitoring config")


def register_monitoring_config(config: MonitoringConfig) -> None:
    key = _config_key(config.domain, config.model_name, config.target_type)
    _registry.register(key, config)


def get_monitoring_config(domain: str, model_name: str, target_type: str) -> MonitoringConfig:
    return _registry.get(_config_key(domain, model_name, target_type))


def get_registry() -> dict[str, MonitoringConfig]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()
