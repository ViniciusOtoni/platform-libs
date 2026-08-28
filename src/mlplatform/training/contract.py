from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from mlplatform.core.registry import Registry


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
    pyfunc_model_class: type | None = None
    # Alias apontado para a versão recém-registrada, depois do gate passar.
    # É o que fecha o ciclo com o serving: os dois componentes consomem
    # `models:/<nome>@champion`, e sem esta promoção o modelo novo ficava
    # registrado sem que nada o servisse — o serving continuava numa versão
    # antiga, ou falhava porque o alias nem existia.
    # `None` desliga a promoção automática, para o domínio que queira um
    # gate humano entre treinar e servir.
    promotion_alias: str | None = "champion"

    def __post_init__(self) -> None:
        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}"
            )


_registry: Registry[TrainingConfig] = Registry(kind="training config")


def register_training_config(config: TrainingConfig) -> None:
    _registry.register(config.model_name, config)


def get_training_config(model_name: str) -> TrainingConfig:
    return _registry.get(model_name)


def get_registry() -> dict[str, TrainingConfig]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()
