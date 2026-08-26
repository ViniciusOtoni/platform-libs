from datetime import date


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
