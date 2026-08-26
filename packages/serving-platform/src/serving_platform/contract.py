from dataclasses import dataclass
from typing import Literal, Optional

from platform_core.registry import Registry

_VALID_MODES = ("online", "batch")


@dataclass
class ServingConfig:
    domain: str
    model_name: str
    mode: Literal["online", "batch"]
    alias: str = "champion"
    spine_inference_table: Optional[str] = None
    schedule_cron: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.mode == "batch" and (self.spine_inference_table is None or self.schedule_cron is None):
            raise ValueError("mode='batch' requires spine_inference_table and schedule_cron")


_registry: Registry[ServingConfig] = Registry(kind="serving config")


def register_serving_config(config: ServingConfig) -> None:
    _registry.register(config.model_name, config)


def get_serving_config(model_name: str) -> ServingConfig:
    return _registry.get(model_name)


def get_registry() -> dict[str, ServingConfig]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()
