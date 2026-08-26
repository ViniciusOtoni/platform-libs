from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Container de specs registradas por decorator/chamada explícita, substituindo
    o dict global mutável (`_REGISTRY: dict = {}` em nível de módulo) que cada
    contract.py reimplementava. Uma instância por componente, injetada em vez de
    estado de módulo compartilhado."""

    def __init__(self, *, kind: str = "entry"):
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        if key in self._items:
            raise ValueError(f"{self._kind} '{key}' already registered")
        self._items[key] = value

    def get(self, key: str) -> T:
        return self._items[key]

    def all(self) -> dict[str, T]:
        return dict(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._items
