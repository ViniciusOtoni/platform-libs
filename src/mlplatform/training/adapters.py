"""Implementações reais dos ports de training."""

from typing import Any

import pandas as pd


class FeatureEngineeringTrainingSet:
    def __init__(self, spark):
        self._spark = spark

    def build(self, spine_table: str, config: Any) -> Any:
        from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

        return FeatureEngineeringClient().create_training_set(
            df=self._spark.table(spine_table),
            feature_lookups=[
                FeatureLookup(
                    table_name=fl.table_name,
                    feature_names=fl.feature_names,
                    lookup_key=fl.lookup_key,
                    timestamp_lookup_key=fl.timestamp_lookup_key,
                )
                for fl in config.feature_lookups
            ],
            label=config.label_column,
        )

    def to_pandas(self, training_set: Any) -> pd.DataFrame:
        return training_set.load_df().toPandas()


class DeltaScratchStore:
    """Splits materializados em Delta entre as tasks."""

    def __init__(self, spark):
        self._spark = spark

    def write(self, df: pd.DataFrame, table_name: str) -> None:
        schema = table_name.rsplit(".", 1)[0]
        self._spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self._spark.createDataFrame(df).write.format("delta").mode("overwrite").saveAsTable(table_name)

    def read(self, table_name: str) -> pd.DataFrame:
        return self._spark.table(table_name).toPandas()


class MlflowTracker:
    def start_run(self, experiment: str) -> str:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run() as run:
            return run.info.run_id

    def log_params(self, run_id: str, params: dict, prefix: str = "") -> None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({f"{prefix}{k}": v for k, v in params.items()})

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(name, value)


class FeatureEngineeringPublisher:
    """Registra o modelo no Unity Catalog com a linhagem de features embutida."""

    def __init__(self, spark):
        self._spark = spark

    def publish(
        self,
        model: Any,
        training_set: Any,
        full_model_name: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> None:
        import mlflow
        from databricks.feature_engineering import FeatureEngineeringClient

        from . import pyfunc_model

        mlflow.set_registry_uri("databricks-uc")
        self._spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_model_name.rsplit('.', 1)[0]}")

        # code_paths aponta para o MÓDULO da classe pyfunc, não para o pacote
        # inteiro. O MLflow importa o que está aqui dentro do container do
        # endpoint de serving, onde pyspark, delta e o databricks-sdk não
        # existem. Empacotar o framework todo levaria os adapters junto, e o
        # endpoint quebraria num import — sem nenhum teste reclamar, porque
        # nenhum teste roda dentro daquele container.
        with mlflow.start_run(run_id=run_id):
            FeatureEngineeringClient().log_model(
                model=model,
                artifact_path="model",
                flavor=mlflow.pyfunc,
                training_set=training_set,
                registered_model_name=full_model_name,
                code_paths=[pyfunc_model.__file__],
            )
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("git_branch", git_branch)


class DbutilsTaskChannel:
    """taskValues do Databricks, obtidos via runtime do SDK (não há notebook)."""

    def set(self, key: str, value: str) -> None:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key=key, value=value)

    def get(self, task_key: str, key: str) -> str:
        from databricks.sdk.runtime import dbutils

        return dbutils.jobs.taskValues.get(taskKey=task_key, key=key)
