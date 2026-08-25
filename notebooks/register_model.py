# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.training_configs  # noqa: F401
import json
from datetime import date, datetime
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline
from training_platform.pyfunc_model import FeaturePlatformModel
from training_platform.naming import derive_model_name, validate_model_name
from training_platform.audit import RunRecord, write_run

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
config = get_training_config(model_name)

mlflow_run_id = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="mlflow_run_id")
window_start = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="window_start")
window_end = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="window_end")
best_hyperparameters = json.loads(
    dbutils.jobs.taskValues.get(taskKey="select_best_and_test", key="best_hyperparameters")
)
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]

estimator = config.algorithm(**best_hyperparameters)
pipeline = build_pipeline(config.custom_transforms, estimator)
pipeline.fit(X_train, y_train)

pyfunc_class = config.pyfunc_model_class or FeaturePlatformModel
wrapped_model = pyfunc_class(pipeline)

# COMMAND ----------
# `fe.log_model` não aceita `feature_lookups` diretamente nem `training_set=None` —
# exige um `TrainingSet` de verdade (o mesmo padrão que `prepare_training_set.py` já
# usa), que carrega o FeatureSpec a ser embarcado no artefato do modelo.
fe = FeatureEngineeringClient()
feature_lookups = [
    FeatureLookup(
        table_name=fl.table_name,
        feature_names=fl.feature_names,
        lookup_key=fl.lookup_key,
        timestamp_lookup_key=fl.timestamp_lookup_key,
    )
    for fl in config.feature_lookups
]
spine = spark.table(config.spine_table)
# Sem exclude_columns aqui de propósito: diferente de prepare_training_set.py, este
# TrainingSet nunca passa por load_df() — só alimenta fe.log_model() para embarcar o
# FeatureSpec no artefato. exclude_columns só afeta o DataFrame de load_df(), então
# seria um parâmetro morto e visualmente confundível com o bug já corrigido em
# prepare_training_set.py.
training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups,
    label=config.label_column,
)

full_model_name = derive_model_name(catalog, config.domain, config.model_name)
validate_model_name(full_model_name)
mlflow.set_registry_uri("databricks-uc")

# Mesmo requisito já confirmado para tabelas (audit.py, writer.py do feature-platform):
# saveAsTable/registro UC não cria o schema automaticamente. Confirmado ao vivo na
# Task 13 — o primeiro registro bem-sucedido precisou desta linha para criar
# `exemplo_models` antes de `fe.log_model`.
model_schema = full_model_name.rsplit(".", 1)[0]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {model_schema}")

# `FeaturePlatformModel` (ou o pyfunc_model_class customizado do usuário) mora em
# training_platform/, que nunca é instalado como dependência de pip — só existe via o
# bootstrap de sys.path deste próprio job. Sem code_paths, o cloudpickle do artefato
# não consegue reimportar esse módulo em nenhum ambiente fora de um job do
# training-platform (ex.: um job de scoragem do serving-platform), falhando com
# ModuleNotFoundError: No module named 'training_platform'. code_paths embarca o
# pacote no artefato do modelo e o adiciona ao sys.path de quem carregar o modelo.
with mlflow.start_run(run_id=mlflow_run_id):
    fe.log_model(
        model=wrapped_model,
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=full_model_name,
        code_paths=[os.path.join(_repo_root, "src", "training_platform")],
    )
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("git_branch", git_branch)

# COMMAND ----------
write_run(
    spark,
    RunRecord(
        component="training",
        entity_name=full_model_name,
        git_commit=git_commit,
        git_branch=git_branch,
        run_id=run_id_job,
        mode="train",
        status="SUCCESS",
        window_start=date.fromisoformat(window_start),
        window_end=date.fromisoformat(window_end),
        run_ts=datetime.utcnow(),
    ),
)
