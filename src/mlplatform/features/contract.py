from collections.abc import Callable
from dataclasses import dataclass, field

from mlplatform.core.registry import Registry


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
    table_name: str | None = None
    partition_by: list[str] | None = None

    def partition_cols(self) -> list[str]:
        """Colunas de particionamento do backfill.

        O default é a COLUNA DE TEMPO, e isso é o que faz o modo `BACKFILL`
        significar o que promete.

        O default anterior era a primeira entity key. Como o backfill grava com
        `partitionOverwriteMode=dynamic`, cada janela substituía a partição
        inteira do cliente — apagando todas as safras anteriores dele. O
        sintoma: rodar backfill de janeiro a agosto deixava UMA safra na
        tabela, a última, com a auditoria registrando as oito execuções como
        SUCCESS. Uma feature store sem histórico, e nada acusando.

        Particionar pela chave de entidade também é ruim por si numa tabela
        temporal: cardinalidade altíssima e arquivos minúsculos.

        Segue declarável na spec para o caso raro em que o domínio precise de
        outra coisa.
        """
        if self.partition_by is not None:
            return list(self.partition_by)
        return [self.timestamp_key]


_registry: Registry[FeatureTableSpec] = Registry(kind="feature table")


def feature_table(
    *,
    domain: str,
    entity_keys: list[str],
    timestamp_key: str,
    sources: list[str],
    online: bool = False,
    depends_on: list[str] | None = None,
    table_name: str | None = None,
    partition_by: list[str] | None = None,
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
            partition_by=list(partition_by) if partition_by is not None else None,
        )
        _registry.register(spec.name, spec)
        return fn

    return decorator


def get_registry() -> dict[str, FeatureTableSpec]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()
