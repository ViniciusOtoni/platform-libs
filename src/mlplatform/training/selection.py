import math


def select_best(results: list[tuple[dict, float]], metric_direction: str) -> dict:
    """Escolhe o melhor conjunto de hiperparâmetros.

    Combinações cuja métrica não é finita são descartadas antes da comparação.
    `max()`/`min()` com NaN devolvem resultado dependente da ordem da lista, em
    silêncio — e NaN acontece de verdade aqui: quando o split de validação sai
    vazio, a métrica vira float("nan"). Sem esse filtro, o "melhor" modelo era
    escolhido pela posição na lista.
    """
    if not results:
        raise ValueError("results must not be empty")

    finite = [r for r in results if isinstance(r[1], (int, float)) and math.isfinite(r[1])]
    if not finite:
        raise ValueError(
            f"no hyperparameter combination produced a finite metric "
            f"(got {[r[1] for r in results]}) — an empty validation split is the usual cause"
        )

    if metric_direction == "maximize":
        return max(finite, key=lambda r: r[1])[0]
    if metric_direction == "minimize":
        return min(finite, key=lambda r: r[1])[0]
    raise ValueError(f"unknown metric_direction: {metric_direction}")
