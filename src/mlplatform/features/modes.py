from enum import StrEnum


class WriteMode(StrEnum):
    """Como a execução foi disparada."""

    INCREMENTAL = "incremental"
    BACKFILL = "backfill"


class WriteSemantics(StrEnum):
    """O que o modo implica na hora de escrever."""

    MERGE = "merge"
    OVERWRITE_BY_PARTITION = "overwrite_by_partition"


def write_strategy_for(mode: WriteMode) -> WriteSemantics:
    """Política de domínio: qual semântica de escrita cada modo implica.

    Fica no domínio, e não no adapter, porque a decisão é de negócio — um run
    incremental faz merge para não perder o que já existe; um backfill reescreve
    a partição inteira. *Como* executar um merge em Delta é que é infraestrutura,
    e isso vive no adapter. Separar assim mantém a política testável sem Spark.
    """
    if mode == WriteMode.INCREMENTAL:
        return WriteSemantics.MERGE
    if mode == WriteMode.BACKFILL:
        return WriteSemantics.OVERWRITE_BY_PARTITION
    raise ValueError(f"unknown mode: {mode}")
