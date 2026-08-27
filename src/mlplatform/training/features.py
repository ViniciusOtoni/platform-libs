from .contract import TrainingConfig


def feature_columns(columns: list[str], config: TrainingConfig) -> list[str]:
    """Colunas que entram no modelo.

    Estava duplicada como list comprehension em três notebooks, e excluía o
    label e as `lookup_key` — mas NÃO a `timestamp_lookup_key`. Só não quebrava
    porque `prepare_training_set` fazia um `.drop(reference_date_column)` antes,
    num arquivo diferente: um acoplamento invisível entre dois passos, que se
    romperia no dia em que alguém mexesse no drop.

    Aqui a regra é explícita e vale por si só.
    """
    excluded = {
        config.label_column,
        config.reference_date_column,
        *(fl.lookup_key for fl in config.feature_lookups),
        *(fl.timestamp_lookup_key for fl in config.feature_lookups),
    }
    return [c for c in columns if c not in excluded]
