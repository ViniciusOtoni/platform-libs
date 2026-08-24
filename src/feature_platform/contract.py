from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class FeatureTableSpec:
    name: str
    entity_keys: list[str]
    timestamp_key: str
    sources: list[str]
    compute_fn: Callable
    domain: str
    online: bool = False
    depends_on: list[str] = field(default_factory=list)
    table_name: Optional[str] = None


_REGISTRY: dict[str, FeatureTableSpec] = {}


def feature_table(
    *,
    domain: str,
    entity_keys: list[str],
    timestamp_key: str,
    sources: list[str],
    online: bool = False,
    depends_on: list[str] | None = None,
    table_name: str | None = None,
):
    def decorator(fn: Callable) -> Callable:
        spec = FeatureTableSpec(
            name=fn.__name__,
            entity_keys=list(entity_keys),
            timestamp_key=timestamp_key,
            sources=list(sources),
            compute_fn=fn,
            domain=domain,
            online=online,
            depends_on=list(depends_on or []),
            table_name=table_name,
        )
        if spec.name in _REGISTRY:
            raise ValueError(f"feature table '{spec.name}' already registered")
        _REGISTRY[spec.name] = spec
        return fn

    return decorator


def get_registry() -> dict[str, FeatureTableSpec]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
