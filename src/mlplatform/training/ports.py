"""Ports do contexto de training."""

from typing import Any, Protocol

import pandas as pd


class TrainingSetBuilder(Protocol):
    """Monta o training set resolvendo os FeatureLookup contra as feature tables."""

    def build(self, spine_table: str, config: Any) -> Any: ...

    def to_pandas(self, df: Any) -> pd.DataFrame: ...


class ScratchStore(Protocol):
    """Materializa os splits entre as tasks do pipeline.

    Os splits precisam sobreviver ao fim de uma task e ser lidos pela seguinte —
    taskValues carrega valores pequenos, não dataframes.
    """

    def write(self, df: pd.DataFrame, table_name: str) -> None: ...

    def read(self, table_name: str) -> pd.DataFrame: ...


class ExperimentTracker(Protocol):
    """Registro de métricas e parâmetros do experimento.

    Um experiment por modelo, um run pai por execução do pipeline e um run
    filho aninhado por combinação de hiperparâmetros. O aninhamento não é
    estético: parâmetros do MLflow são imutáveis, então registrar duas
    combinações no mesmo run levanta `Changing param values is not allowed` na
    segunda. Os métodos recebem o run por id — quem chama nunca depende de um
    run "corrente" implícito.
    """

    def start_run(self, experiment: str) -> str: ...

    def start_child_run(self, parent_run_id: str, name: str) -> str: ...

    def log_params(self, run_id: str, params: dict, prefix: str = "") -> None: ...

    def log_metric(self, run_id: str, name: str, value: float) -> None: ...


class ModelPublisher(Protocol):
    """Registro do modelo no Unity Catalog, com a linhagem de features embutida.

    O modelo é registrado via Feature Engineering, e não pelo MLflow puro: é isso
    que grava a linhagem dos FeatureLookup no artefato e permite que
    `score_batch` e o endpoint online resolvam as features sozinhos, a partir só
    das chaves de entidade.
    """

    def publish(
        self,
        model: Any,
        training_set: Any,
        full_model_name: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> None: ...


class TaskChannel(Protocol):
    """Passagem de valores entre tasks do mesmo job.

    Confirmado ao vivo que funciona sob `python_wheel_task` em serverless —
    era o maior desconhecido do plano, porque sem notebook não há `dbutils`
    global e o acesso a `taskValues` fora de contexto de notebook não é garantido.
    """

    def set(self, key: str, value: str) -> None: ...

    def get(self, task_key: str, key: str) -> str: ...
