from dataclasses import dataclass
from typing import Literal, Optional

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


_REGISTRY: dict[str, ServingConfig] = {}


def register_serving_config(config: ServingConfig) -> None:
    if config.model_name in _REGISTRY:
        raise ValueError(f"serving config '{config.model_name}' already registered")
    _REGISTRY[config.model_name] = config


def get_serving_config(model_name: str) -> ServingConfig:
    return _REGISTRY[model_name]


def get_registry() -> dict[str, ServingConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
