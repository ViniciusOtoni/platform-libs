"""Estrutura mínima da tabela de saída da inferência batch.

O domínio declara o formato; o framework confere antes de gravar. É um
contrato, não um construtor de dataframe: a tabela continua sendo produzida
pelo `score_batch`, e isto diz o que ela precisa conter para o monitoramento
conseguir ler depois.

A motivação é a fase seguinte. Data drift compara a distribuição das features
entre safras; model drift compara score contra desfecho. Nenhum dos dois
funciona se cada domínio gravar a saída num formato próprio, e nenhum dos dois
sobrevive a descobrir isso só depois de meses de histórico acumulado no formato
errado.
"""

from dataclasses import dataclass, field

# Colunas que o framework acrescenta e o domínio não declara.
SCORED_AT_COLUMN = "scored_at"
MODEL_VERSION_COLUMN = "model_version"
FRAMEWORK_COLUMNS = (SCORED_AT_COLUMN, MODEL_VERSION_COLUMN)


@dataclass(frozen=True)
class InferenceBatchStruct:
    """Formato mínimo da tabela de predições.

    - `primary_key`: as chaves de entidade. Identificam a linha junto com
      `ts_date`.
    - `ts_date`: a coluna de safra — a data de referência da inferência, não o
      instante em que o job rodou. Quem marca a execução é `scored_at`, que o
      framework grava sozinho. Separar as duas é o que permite reprocessar uma
      safra antiga sem que ela se confunda com a safra corrente.
    - `feature_cols`: as features que entraram na predição. Ficam na tabela
      porque são a base do data drift: sem elas, comparar safras exigiria
      reconstruir o join do FeatureLookup a posteriori, contra feature tables
      que já mudaram.
    - `predict_cols`: as colunas de score.
    - `label_col`: opcional, e por padrão NÃO é gravado aqui — ver a nota
      abaixo.

    Sobre o label: ele não existe no momento da inferência. Pontua-se hoje um
    desfecho que se materializa em semanas. Gravá-lo no scoring produziria uma
    coluna sempre nula, a ser preenchida depois por MERGE — o que torna esta
    tabela mutável e destrói a propriedade que faz o data drift ser confiável:
    cada linha é um registro imutável do que foi pontuado, com o quê, e quando.
    Declarar `label_col` aqui serve para o monitoramento saber ONDE o desfecho
    vai aparecer; não faz o serving escrevê-lo.
    """

    primary_key: list[str]
    ts_date: str
    predict_cols: list[str]
    feature_cols: list[str] = field(default_factory=list)
    label_col: str | None = None

    def __post_init__(self) -> None:
        if not self.primary_key:
            raise ValueError("primary_key não pode ser vazia")
        if not self.ts_date:
            raise ValueError("ts_date não pode ser vazia")
        if not self.predict_cols:
            raise ValueError("predict_cols não pode ser vazia")

        declared = [*self.primary_key, self.ts_date, *self.predict_cols, *self.feature_cols]
        if self.label_col:
            declared.append(self.label_col)

        duplicated = sorted({c for c in declared if declared.count(c) > 1})
        if duplicated:
            raise ValueError(
                f"colunas declaradas em mais de um papel: {duplicated}. "
                "Uma coluna é chave, safra, feature, score ou label — não duas coisas."
            )

        colliding = sorted(set(declared) & set(FRAMEWORK_COLUMNS))
        if colliding:
            raise ValueError(
                f"{colliding} são gravadas pelo framework e não podem ser declaradas: "
                f"{list(FRAMEWORK_COLUMNS)}"
            )

    @property
    def required_columns(self) -> list[str]:
        """Tudo que a tabela precisa ter, incluindo o que o framework grava.

        `label_col` fica de fora de propósito: ele chega depois, por outro
        caminho, e exigi-lo aqui reprovaria toda inferência.
        """
        return [
            *self.primary_key,
            self.ts_date,
            *self.feature_cols,
            *self.predict_cols,
            *FRAMEWORK_COLUMNS,
        ]
