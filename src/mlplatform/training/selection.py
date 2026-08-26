def select_best(results: list[tuple[dict, float]], metric_direction: str) -> dict:
    if not results:
        raise ValueError("results must not be empty")

    if metric_direction == "maximize":
        best = max(results, key=lambda r: r[1])
    elif metric_direction == "minimize":
        best = min(results, key=lambda r: r[1])
    else:
        raise ValueError(f"unknown metric_direction: {metric_direction}")

    return best[0]
