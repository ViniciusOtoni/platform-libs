from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class FeatureLookupSpec:
    table_name: str
    feature_names: list[str]
    lookup_key: str
    timestamp_lookup_key: str


@dataclass
class TrainingConfig:
    domain: str
    model_name: str
    algorithm: type
    hyperparameter_sets: list[dict]
    feature_lookups: list[FeatureLookupSpec]
    spine_table: str
    label_column: str
    reference_date_column: str
    train_pct: float
    val_pct: float
    test_pct: float
    metric: str | Callable
    metric_direction: Literal["maximize", "minimize"]
    custom_transforms: list = field(default_factory=list)
    pyfunc_model_class: Optional[type] = None

    def __post_init__(self) -> None:
        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}"
            )


_REGISTRY: dict[str, TrainingConfig] = {}


def register_training_config(config: TrainingConfig) -> None:
    if config.model_name in _REGISTRY:
        raise ValueError(f"training config '{config.model_name}' already registered")
    _REGISTRY[config.model_name] = config


def get_training_config(model_name: str) -> TrainingConfig:
    return _REGISTRY[model_name]


def get_registry() -> dict[str, TrainingConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
