from typing import Protocol

import yaml


class ResourceGenerator(Protocol):
    """Contrato que cada `<componente>.resource_gen` implementa, formalizando o
    hook hoje descoberto por convenção de nome de arquivo
    (`scripts/generate_resources.py`) sem nenhuma garantia de forma."""

    def write(self, path: str) -> None: ...


def dump_yaml(resource: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resource, f, sort_keys=False)
