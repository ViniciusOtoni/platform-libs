"""Catálogo das métricas de drift que o Lakehouse Monitoring produz.

Existe porque o nome da métrica é uma escolha do domínio, e escolher errado é
silencioso: uma coluna que não existe, ou que vem nula para o tipo de dado
observado, produz "sem drift" para sempre. Foi exatamente o que aconteceu — o
código lia uma coluna `statistic` que não existe no topo da tabela (ela é um
campo ANINHADO dentro de `ks_test` e `chi_squared_test`) e registrava 0.0 em
toda medição.
"""

from dataclasses import dataclass

# Tipo de comparação que o monitor emite na coluna `drift_type`.
# CONSECUTIVE compara a janela atual com a anterior; BASELINE compara com a
# tabela de baseline, que só existe se `baseline_table_name` for configurado na
# criação do monitor — hoje não é. Misturar os dois numa leitura só daria
# resultado dependente da ordem das linhas.
CONSECUTIVE = "CONSECUTIVE"
BASELINE = "BASELINE"


@dataclass(frozen=True)
class DriftMetric:
    """Uma métrica da tabela de drift do monitor.

    `nested_field` != None significa que a coluna é um struct e o valor está
    dentro dele — o caso de `ks_test` e `chi_squared_test`, que carregam
    `statistic` e `pvalue`.

    `bounded` diz se a métrica vive em [0,1]. Isso decide se um limiar é
    comparável entre domínios: 0.2 em `js_distance` significa a mesma coisa em
    qualquer tabela, enquanto 0.2 em `wasserstein_distance` está na escala da
    variável e não significa nada sem contexto.
    """

    column: str
    bounded: bool
    nested_field: str | None = None
    note: str = ""


DRIFT_METRICS: dict[str, DriftMetric] = {
    # Índice padrão da indústria para drift de distribuição. Não é limitado a
    # [0,1], mas tem convenção estabelecida: <0.1 estável, 0.1–0.25 moderado,
    # >0.25 significativo. É o default por ser o único escalar preenchido tanto
    # para colunas numéricas quanto categóricas.
    "population_stability_index": DriftMetric(
        column="population_stability_index",
        bounded=False,
        note="convenção: >0.25 significativo",
    ),
    # Limitadas a [0,1] — um limiar vale para qualquer tabela. Mas o monitor as
    # calcula para colunas CATEGÓRICAS; em colunas numéricas vêm nulas.
    "js_distance": DriftMetric(column="js_distance", bounded=True, note="apenas categóricas"),
    "tv_distance": DriftMetric(column="tv_distance", bounded=True, note="apenas categóricas"),
    "l_infinity_distance": DriftMetric(
        column="l_infinity_distance", bounded=True, note="apenas categóricas"
    ),
    # Na escala da variável observada: um limiar aqui não é transferível entre
    # colunas, quanto mais entre domínios.
    "wasserstein_distance": DriftMetric(
        column="wasserstein_distance", bounded=False, note="na escala da variável"
    ),
    # Structs: o valor está no campo aninhado.
    "ks_test": DriftMetric(column="ks_test", bounded=True, nested_field="statistic"),
    "chi_squared_test": DriftMetric(
        column="chi_squared_test", bounded=False, nested_field="statistic"
    ),
}

DEFAULT_DRIFT_METRIC = "population_stability_index"


def resolve(name: str) -> DriftMetric:
    if name not in DRIFT_METRICS:
        raise ValueError(
            f"métrica de drift desconhecida: {name!r}. "
            f"Disponíveis: {sorted(DRIFT_METRICS)}"
        )
    return DRIFT_METRICS[name]
