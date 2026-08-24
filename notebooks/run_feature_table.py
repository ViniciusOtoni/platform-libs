# Databricks notebook source
dbutils.widgets.text("feature_table", "")
dbutils.widgets.text("mode", "incremental")
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
import examples.features  # noqa: F401  (import dispara o registro via decorator)
from datetime import date

from feature_platform.contract import get_registry
from feature_platform.writer import WriteMode
from feature_platform.engine import run_feature_table

# COMMAND ----------
feature_table_name = dbutils.widgets.get("feature_table")
mode = WriteMode(dbutils.widgets.get("mode"))
start_date = dbutils.widgets.get("start_date") or None
end_date = dbutils.widgets.get("end_date") or None
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
run_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().toString()

registry = get_registry()
spec = registry[feature_table_name]

# COMMAND ----------
run_feature_table(
    spec=spec,
    spark=spark,
    catalog=catalog,
    mode=mode,
    today=date.today(),
    run_id=run_id,
    git_commit=git_commit,
    git_branch=git_branch,
    backfill_start=start_date,
    backfill_end=end_date,
)
