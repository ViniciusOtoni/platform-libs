"""Ports do contexto de monitoring.

Separados por capacidade, como em serving: ler os runs de treino, operar o
quality monitor do Databricks, ler uma tabela e gravar as métricas são coisas
sem relação entre si. Um port gordo obrigaria todo fake a implementar os quatro.
"""

from typing import Protocol

import pandas as pd

from .baseline import TrainingRun


class TrainingRunReader(Protocol):
    """Os runs de treino registrados na tabela de auditoria.

    Monitoring lê a auditoria, mas não conhece o schema dela: o adapter traduz
    para `TrainingRun`. É um anticorruption layer deliberado — se monitoring
    importasse `RunRecord`, ficaria acoplado ao formato da tabela de outro
    contexto, e mudar uma coluna de auditoria quebraria o drift.
    """

    def training_runs(self) -> list[TrainingRun]: ...


class QualityMonitor(Protocol):
    """O monitor de qualidade do Databricks sobre a tabela observada.

    Uma chamada só, e não `create`/`refresh`/`poll` separados: por trás disso há
    uma máquina de estados com espera de até 20 minutos, que é infraestrutura
    pura. Expor os três passos obrigaria o caso de uso a orquestrar polling —
    e a ficar impossível de testar sem workspace.
    """

    def refreshed_drift_table(self, target_table: str, assets_dir: str, output_schema: str) -> str:
        """Garante o monitor criado e atualizado; devolve a tabela de métricas."""
        ...


class TableReader(Protocol):
    def to_pandas(self, table_name: str) -> pd.DataFrame: ...


class DriftMetricsWriter(Protocol):
    def append(self, rows: list[dict], table_name: str) -> None: ...


class RetrainTrigger(Protocol):
    """Pede um retreino quando o drift passa do limiar.

    O gatilho sai do Databricks e entra no GitHub, e não o contrário: é o
    GitHub que tem o mecanismo de aprovação manual (Environments com required
    reviewers) que separa "modelo novo existe" de "modelo novo está servindo".
    Um retreino disparado inteiramente dentro do Databricks promoveria sozinho,
    que é justamente o que não se quer quando a causa foi drift.
    """

    def request_retrain(self, domain: str, model_name: str, drifted_columns: list[str]) -> str:
        """Dispara o retreino e devolve uma referência rastreável."""
        ...
