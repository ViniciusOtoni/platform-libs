from datetime import date, datetime
from typing import Any


def as_date(value: Any) -> date:
    """Normaliza o que vier da coluna de data de referência para `date`.

    `toPandas()` converte uma coluna DATE do Delta em datetime64, então os
    valores chegam como `pandas.Timestamp` — não como `date`. Comparar
    Timestamps entre si funciona, e por isso o split parecia correto; o que
    quebra é serializar. `str(Timestamp)` produz "2026-08-23 00:00:00", e o
    `date.fromisoformat` do outro lado do taskValues rejeita isso.

    Coerção num ponto só, na fronteira onde o pandas entra no domínio, em vez
    de espalhar `.date()` por toda parte.
    """
    if isinstance(value, datetime):  # cobre pandas.Timestamp, que é subclasse
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def compute_split_dates(
    distinct_dates: list[date], train_pct: float, val_pct: float, test_pct: float
) -> tuple[date, date]:
    """Retorna (train_end, val_end): os dois cortes cronológicos.
    train = data <= train_end; val = train_end < data <= val_end; test = data > val_end.
    Nunca quebra uma safra ao meio — os cortes caem sempre em cima de uma data real."""
    if not distinct_dates:
        raise ValueError("distinct_dates must not be empty")

    sorted_dates = sorted(set(distinct_dates))
    n = len(sorted_dates)

    train_end_idx = max(0, min(round(n * train_pct) - 1, n - 1))
    val_end_idx = max(train_end_idx, min(round(n * (train_pct + val_pct)) - 1, n - 1))

    return sorted_dates[train_end_idx], sorted_dates[val_end_idx]


def assign_split(reference_date: date, train_end: date, val_end: date) -> str:
    if reference_date <= train_end:
        return "train"
    if reference_date <= val_end:
        return "val"
    return "test"
