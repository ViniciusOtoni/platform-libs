# Databricks notebook source
dbutils.widgets.text("domain", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("target_type", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — nem a raiz
# do bundle (onde mora `examples/`) nem `src/` (onde mora `monitoring_platform`) estão
# no sys.path por padrão.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.monitoring_configs  # noqa: F401
from datetime import date, datetime

from databricks.sdk import WorkspaceClient

from monitoring_platform.contract import get_monitoring_config
from monitoring_platform.baseline import TrainingRun, resolve_baseline_window, NoTrainingRunError
from monitoring_platform.evaluation import evaluate_drift
from monitoring_platform.central_table import build_drift_metric_row, write_drift_metrics
from monitoring_platform.audit import RunRecord, write_run, AUDIT_TABLE

# COMMAND ----------
domain = dbutils.widgets.get("domain")
model_name = dbutils.widgets.get("model_name")
target_type = dbutils.widgets.get("target_type")
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
config = get_monitoring_config(domain, model_name, target_type)
# currentRunId() não está na whitelist do Py4J em compute serverless/shared access
# mode — levanta Py4JSecurityException. Cai para um id gerado localmente quando o
# contexto de job não expõe o run id dessa forma.
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())
full_model_name = f"{catalog}.{domain}_models.{model_name}"

# COMMAND ----------
training_runs_pd = spark.table(AUDIT_TABLE).filter("component = 'training'").toPandas()
training_runs = [
    TrainingRun(
        entity_name=row["entity_name"],
        status=row["status"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        run_ts=row["run_ts"],
    )
    for _, row in training_runs_pd.iterrows()
]

today = date.today()

try:
    baseline_start, baseline_end = resolve_baseline_window(training_runs, full_model_name)
except NoTrainingRunError:
    write_run(
        spark,
        RunRecord(
            component="monitoring",
            entity_name=config.target_table,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id_job,
            mode="drift_check",
            status="FAILED",
            window_start=today,
            window_end=today,
            run_ts=datetime.utcnow(),
        ),
    )
    raise

# COMMAND ----------
# RISCO DOCUMENTADO (spec, seção 6): nomes de método e parâmetros a confirmar contra
# a versão instalada do databricks-sdk e a documentação atual do Lakehouse Monitoring
# antes do primeiro deploy real. Confirmado ao vivo (Task 12): método e nomes de
# parâmetro batem, mas create() exige um de snapshot/time_series/inference_log.
import time

from databricks.sdk.service.catalog import MonitorSnapshot, MonitorRefreshInfoState

client = WorkspaceClient()
try:
    monitor_info = client.quality_monitors.get(table_name=config.target_table)
except Exception:
    monitor_info = None

if monitor_info is not None:
    # Monitor já existe — só dispara um novo refresh. Fora do try/except acima de
    # propósito: uma falha aqui (permissão, quota, erro transiente da API) não deve
    # ser reinterpretada como "monitor não existe, criar de novo" — isso produziria um
    # erro confuso de "já existe" em vez do erro real do run_refresh.
    refresh_info = client.quality_monitors.run_refresh(table_name=config.target_table)
else:
    # snapshot=MonitorSnapshot(): mais adequado que time_series para este caso de uso
    # (comparar contra uma baseline fixa, não janelas internas por timestamp). Sem
    # baseline_table_name (item em aberto — materializar uma tabela de baseline
    # exigiria saber a coluna de timestamp de target_table, ainda não parte do
    # MonitoringConfig): monitores snapshot comparam cada refresh contra o anterior
    # por padrão.
    monitoring_schema = f"{catalog}.{domain}_monitoring"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {monitoring_schema}")
    monitor_info = client.quality_monitors.create(
        table_name=config.target_table,
        assets_dir=f"/Shared/monitoring-platform/{domain}/{model_name}/{target_type}",
        output_schema_name=monitoring_schema,
        snapshot=MonitorSnapshot(),
    )
    # create() dispara um refresh automaticamente — pega o mais recente para esperar.
    refreshes = client.quality_monitors.list_refreshes(table_name=config.target_table).refreshes
    refresh_info = max(refreshes, key=lambda r: r.start_time_ms)

# O cálculo de profile/drift metrics do LHM é assíncrono — sem esperar o refresh
# terminar, a tabela de saída pode não existir ou estar desatualizada quando lida
# abaixo. Timeout de 20min (ajustado ao vivo — refreshes observados entre ~5 e ~14min,
# provavelmente por provisionamento de compute serverless na Free Edition).
_deadline = time.time() + 1200
while refresh_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
    if time.time() > _deadline:
        raise TimeoutError(f"monitor refresh for '{config.target_table}' did not finish within 1200s")
    time.sleep(15)
    refresh_info = client.quality_monitors.get_refresh(
        table_name=config.target_table, refresh_id=refresh_info.refresh_id
    )

if refresh_info.state != MonitorRefreshInfoState.SUCCESS:
    raise RuntimeError(
        f"monitor refresh for '{config.target_table}' ended in state {refresh_info.state}: {refresh_info.message}"
    )

# COMMAND ----------
# RISCO DOCUMENTADO (spec, seção 6): nomes de coluna da própria tabela `_drift_metrics`
# (ex.: `drift_type`, `statistic`) a confirmar contra a versão instalada antes do
# primeiro deploy real — mas o NOME da tabela em si já não é mais adivinhado: vem
# direto de monitor_info.drift_metrics_table_name (confirmado ao vivo).
drift_table = monitor_info.drift_metrics_table_name
drift_pd = spark.table(drift_table).toPandas()

rows = []
for column in config.columns:
    column_rows = drift_pd[drift_pd["column_name"] == column]
    if column_rows.empty:
        continue
    latest = column_rows.iloc[-1]
    result = evaluate_drift(
        column_name=column,
        drift_metric_name=str(latest.get("drift_type", "unknown_metric")),
        drift_metric_value=float(latest.get("statistic", 0.0)),
        threshold=config.threshold,
    )
    rows.append(
        build_drift_metric_row(
            domain=domain,
            model_name=model_name,
            entity_name=config.target_table,
            target_type=target_type,
            result=result,
            window_start=today,
            window_end=today,
            run_ts=datetime.utcnow(),
        )
    )

if rows:
    write_drift_metrics(spark, rows)

# COMMAND ----------
write_run(
    spark,
    RunRecord(
        component="monitoring",
        entity_name=config.target_table,
        git_commit=git_commit,
        git_branch=git_branch,
        run_id=run_id_job,
        mode="drift_check",
        status="SUCCESS",
        window_start=today,
        window_end=today,
        run_ts=datetime.utcnow(),
    ),
)
