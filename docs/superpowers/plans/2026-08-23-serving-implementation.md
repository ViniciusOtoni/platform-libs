# Componente de Serving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o framework `serving_platform`: contrato `ServingConfig` cobrindo as trilhas online (Model Serving endpoint) e batch (`fe.score_batch` em task única), geração dinâmica dos recursos DAB, gate de qualidade nas predições, comando manual de atualização de endpoint, e auditoria reaproveitando o schema dos Componentes 1 e 2.

**Architecture:** Lógica pura (contrato, nomenclatura, gate de qualidade, geração de resources) em `src/serving_platform/`, testável com `pytest` local, sem Spark. Dois notebooks Databricks fazem a parte real: `score_batch.py` (a única task da trilha batch — `fe.score_batch`, gate, escrita, auditoria) e `refresh_endpoint.py` (atualização manual do endpoint online via Databricks SDK). Um domínio de exemplo em modo batch prova o fluxo ponta a ponta sem custo de endpoint online sempre ligado.

**Tech Stack:** Python 3.11, `databricks-feature-engineering`, `databricks-sdk`, PySpark + Delta (runtime Databricks serverless), Databricks Asset Bundles, pytest, GitHub Actions.

---

## Scope Check

Este plano cobre só o Componente 3 (Serving), conforme
`docs/superpowers/specs/2026-08-23-serving-design.md`. O Componente 4 (Monitoramento)
tem spec e plan próprios, a escrever depois.

## File Structure

```
serving-platform/
├── databricks.yml
├── pyproject.toml
├── pytest.ini
├── requirements-dev.txt
├── README.md
├── src/
│   └── serving_platform/
│       ├── __init__.py
│       ├── contract.py          # ServingConfig, registro por model_name
│       ├── naming.py            # convenção de nome de tabela de predições e de endpoint
│       ├── quality.py           # Finding + gate de qualidade das predições batch
│       ├── audit.py             # RunRecord/to_row/write_run (duplicado dos outros componentes)
│       └── resource_gen.py      # gera jobs (batch) e model_serving_endpoints (online) a partir do registro
├── dominios/
│   └── exemplo/
│       ├── __init__.py
│       └── serving_configs.py   # ServingConfig de exemplo, modo batch
├── notebooks/
│   ├── score_batch.py           # única task da trilha batch
│   └── refresh_endpoint.py      # atualização manual do endpoint online
├── scripts/
│   └── generate_resources.py
├── resources/
│   └── (generated_serving.yml — gerado, não escrito à mão)
├── tests/
│   ├── __init__.py
│   ├── test_contract.py
│   ├── test_naming.py
│   ├── test_quality.py
│   ├── test_audit.py
│   └── test_resource_gen.py
└── .github/
    └── workflows/
        └── deploy.yml
```

Igual ao `feature-platform`, e diferente do `training-platform`: o número de
`ServingConfig` registrados varia (cada modelo pode ou não estar em serving, online ou
batch), então os recursos DAB são **gerados dinamicamente**, não escritos à mão.

---

## Task 1: Scaffolding do repositório

**Files:**
- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `src/serving_platform/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "serving-platform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Criar `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = src
```

- [ ] **Step 3: Criar `requirements-dev.txt`**

```
pyyaml>=6.0
pytest>=8.0
pandas>=2.0
pyspark>=3.5
delta-spark>=3.2
databricks-sdk>=0.30
databricks-feature-engineering>=0.7
```

- [ ] **Step 4: Criar `src/serving_platform/__init__.py`** (vazio)

```python
```

- [ ] **Step 5: Criar `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.databricks/
resources/generated_*.yml
```

- [ ] **Step 6: Criar venv, instalar dependências, confirmar pytest**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```
Expected: `no tests ran`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml pytest.ini requirements-dev.txt src/serving_platform/__init__.py .gitignore
git commit -m "chore: scaffold serving-platform package"
```

---

## Task 2: Contrato e registro (`contract.py`)

**Files:**
- Create: `src/serving_platform/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_contract.py
import pytest

from serving_platform.contract import (
    ServingConfig,
    register_serving_config,
    get_serving_config,
    get_registry,
    clear_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_online_config_does_not_require_batch_fields():
    config = ServingConfig(domain="exemplo", model_name="modelo_a", mode="online")
    assert config.alias == "champion"
    assert config.spine_inference_table is None


def test_batch_config_requires_spine_and_schedule():
    with pytest.raises(ValueError, match="requires spine_inference_table and schedule_cron"):
        ServingConfig(domain="exemplo", model_name="modelo_b", mode="batch")


def test_batch_config_accepts_required_fields():
    config = ServingConfig(
        domain="exemplo",
        model_name="modelo_b",
        mode="batch",
        spine_inference_table="workspace.exemplo.spine_inference",
        schedule_cron="0 0 6 * * ?",
    )
    assert config.schedule_cron == "0 0 6 * * ?"


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        ServingConfig(domain="exemplo", model_name="modelo_c", mode="streaming")


def test_config_accepts_custom_alias():
    config = ServingConfig(domain="exemplo", model_name="modelo_d", mode="online", alias="challenger")
    assert config.alias == "challenger"


def test_register_and_get_serving_config():
    config = ServingConfig(domain="exemplo", model_name="modelo_e", mode="online")
    register_serving_config(config)

    assert get_serving_config("modelo_e") is config
    assert get_registry() == {"modelo_e": config}


def test_register_serving_config_rejects_duplicate():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_f", mode="online"))

    with pytest.raises(ValueError, match="already registered"):
        register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_f", mode="online"))
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `ModuleNotFoundError: No module named 'serving_platform.contract'`

- [ ] **Step 3: Implementar**

```python
# src/serving_platform/contract.py
from dataclasses import dataclass
from typing import Literal, Optional

_VALID_MODES = ("online", "batch")


@dataclass
class ServingConfig:
    domain: str
    model_name: str
    mode: Literal["online", "batch"]
    alias: str = "champion"
    spine_inference_table: Optional[str] = None
    schedule_cron: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.mode == "batch" and (self.spine_inference_table is None or self.schedule_cron is None):
            raise ValueError("mode='batch' requires spine_inference_table and schedule_cron")


_REGISTRY: dict[str, ServingConfig] = {}


def register_serving_config(config: ServingConfig) -> None:
    if config.model_name in _REGISTRY:
        raise ValueError(f"serving config '{config.model_name}' already registered")
    _REGISTRY[config.model_name] = config


def get_serving_config(model_name: str) -> ServingConfig:
    return _REGISTRY[model_name]


def get_registry() -> dict[str, ServingConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/serving_platform/contract.py tests/test_contract.py
git commit -m "feat: add ServingConfig contract covering online and batch modes"
```

---

## Task 3: Nomenclatura (`naming.py`)

**Files:**
- Create: `src/serving_platform/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_naming.py
import pytest

from serving_platform.naming import (
    derive_predictions_table_name,
    derive_endpoint_name,
    validate_endpoint_name,
)


def test_derive_predictions_table_name_follows_convention():
    name = derive_predictions_table_name(catalog="workspace", domain="credito", model_name="propensao_default")
    assert name == "workspace.credito_predictions.propensao_default"


def test_derive_endpoint_name_follows_convention():
    name = derive_endpoint_name(domain="credito", model_name="propensao_default")
    assert name == "credito-propensao_default-serving"


def test_validate_endpoint_name_accepts_convention():
    validate_endpoint_name("credito-propensao_default-serving")


def test_validate_endpoint_name_rejects_dots():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_endpoint_name("credito.propensao_default.serving")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `ModuleNotFoundError: No module named 'serving_platform.naming'`

- [ ] **Step 3: Implementar**

```python
# src/serving_platform/naming.py
import re

_ENDPOINT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def derive_predictions_table_name(catalog: str, domain: str, model_name: str) -> str:
    schema = f"{domain}_predictions"
    return f"{catalog}.{schema}.{model_name}"


def derive_endpoint_name(domain: str, model_name: str) -> str:
    return f"{domain}-{model_name}-serving"


def validate_endpoint_name(name: str) -> None:
    if not _ENDPOINT_NAME_RE.match(name):
        raise ValueError(
            f"endpoint name '{name}' does not match convention "
            "'<domain>-<model_name>-serving' (lowercase letters, digits, underscore, hyphen)"
        )
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/serving_platform/naming.py tests/test_naming.py
git commit -m "feat: add predictions table and serving endpoint naming conventions"
```

---

## Task 4: Gate de qualidade das predições (`quality.py`)

**Files:**
- Create: `src/serving_platform/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_quality.py
import pandas as pd

from serving_platform.quality import (
    Finding,
    check_no_nulls_in_predictions,
    check_row_count_matches,
    run_predictions_gate,
    gate_passed,
)


def test_check_no_nulls_in_predictions_passes():
    df = pd.DataFrame({"prediction": [0.1, 0.9, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "PASS"


def test_check_no_nulls_in_predictions_fails():
    df = pd.DataFrame({"prediction": [0.1, None, 0.5]})
    assert check_no_nulls_in_predictions(df, "prediction").status == "FAIL"


def test_check_row_count_matches_passes_when_equal():
    assert check_row_count_matches(100, 100).status == "PASS"


def test_check_row_count_matches_fails_when_different():
    finding = check_row_count_matches(100, 87)
    assert finding.status == "FAIL"
    assert "input=100" in finding.detail


def test_run_predictions_gate_returns_both_checks():
    df = pd.DataFrame({"prediction": [0.1, 0.9]})
    findings = run_predictions_gate(df, "prediction", input_row_count=2)
    assert {f.check for f in findings} == {"no_nulls_in_predictions", "row_count_matches"}


def test_gate_passed_true_when_all_pass():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "PASS")]) is True


def test_gate_passed_false_when_any_fails():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "FAIL")]) is False
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `ModuleNotFoundError: No module named 'serving_platform.quality'`

- [ ] **Step 3: Implementar**

```python
# src/serving_platform/quality.py
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    detail: str = ""


def check_no_nulls_in_predictions(df: pd.DataFrame, prediction_column: str) -> Finding:
    nulls = int(df[prediction_column].isnull().sum())
    return Finding(
        check="no_nulls_in_predictions",
        status="PASS" if nulls == 0 else "FAIL",
        detail=f"nulls={nulls}",
    )


def check_row_count_matches(input_row_count: int, output_row_count: int) -> Finding:
    matches = input_row_count == output_row_count
    return Finding(
        check="row_count_matches",
        status="PASS" if matches else "FAIL",
        detail=f"input={input_row_count}, output={output_row_count}",
    )


def run_predictions_gate(df: pd.DataFrame, prediction_column: str, input_row_count: int) -> list[Finding]:
    return [
        check_no_nulls_in_predictions(df, prediction_column),
        check_row_count_matches(input_row_count, len(df)),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/serving_platform/quality.py tests/test_quality.py
git commit -m "feat: add blocking quality gate for batch predictions"
```

---

## Task 5: Auditoria (`audit.py`)

Duplicado deliberadamente dos outros componentes — mesmo schema, mesma implementação.

**Files:**
- Create: `src/serving_platform/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_audit.py
from datetime import date, datetime

from serving_platform.audit import RunRecord, to_row, AUDIT_TABLE


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="serving",
        entity_name="workspace.credito_predictions.propensao_default",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="batch",
        status="SUCCESS",
        window_start=date(2026, 8, 23),
        window_end=date(2026, 8, 23),
        run_ts=datetime(2026, 8, 23, 6, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "serving",
        "entity_name": "workspace.credito_predictions.propensao_default",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "batch",
        "status": "SUCCESS",
        "window_start": date(2026, 8, 23),
        "window_end": date(2026, 8, 23),
        "run_ts": datetime(2026, 8, 23, 6, 0, 0),
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `ModuleNotFoundError: No module named 'serving_platform.audit'`

- [ ] **Step 3: Implementar**

```python
# src/serving_platform/audit.py
from dataclasses import dataclass
from datetime import date, datetime

AUDIT_TABLE = "platform_audit.pipeline_runs"


@dataclass(frozen=True)
class RunRecord:
    component: str
    entity_name: str
    git_commit: str
    git_branch: str
    run_id: str
    mode: str
    status: str
    window_start: date
    window_end: date
    run_ts: datetime


def to_row(record: RunRecord) -> dict:
    return {
        "component": record.component,
        "entity_name": record.entity_name,
        "git_commit": record.git_commit,
        "git_branch": record.git_branch,
        "run_id": record.run_id,
        "mode": record.mode,
        "status": record.status,
        "window_start": record.window_start,
        "window_end": record.window_end,
        "run_ts": record.run_ts,
    }


def write_run(spark, record: RunRecord) -> None:
    """Requer SparkSession — exercitado via notebook (Task 6), não via pytest."""
    df = spark.createDataFrame([to_row(record)])
    if spark.catalog.tableExists(AUDIT_TABLE):
        df.write.format("delta").mode("append").saveAsTable(AUDIT_TABLE)
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(AUDIT_TABLE)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/serving_platform/audit.py tests/test_audit.py
git commit -m "feat: duplicate audit record type and pipeline_runs writer"
```

---

## Task 6: Geração de recursos (`resource_gen.py`)

**Files:**
- Create: `src/serving_platform/resource_gen.py`
- Test: `tests/test_resource_gen.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_resource_gen.py
import pytest

from serving_platform.contract import ServingConfig, register_serving_config, clear_registry
from serving_platform.resource_gen import generate_resources


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_batch_config_generates_a_scheduled_job_with_one_task():
    register_serving_config(
        ServingConfig(
            domain="exemplo",
            model_name="modelo_batch",
            mode="batch",
            spine_inference_table="workspace.exemplo.spine_inference",
            schedule_cron="0 0 6 * * ?",
        )
    )

    resources = generate_resources()
    jobs = resources["resources"]["jobs"]

    job = jobs["score_batch_modelo_batch"]
    assert job["schedule"]["quartz_cron_expression"] == "0 0 6 * * ?"
    assert [t["task_key"] for t in job["tasks"]] == ["score_batch"]
    assert job["tasks"][0]["notebook_task"]["notebook_path"] == "../notebooks/score_batch"


def test_online_config_generates_a_model_serving_endpoint():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_online", mode="online"))

    resources = generate_resources()
    endpoints = resources["resources"]["model_serving_endpoints"]

    endpoint = endpoints["exemplo-modelo_online-serving"]
    served_entity = endpoint["config"]["served_entities"][0]
    assert served_entity["entity_name"] == "${var.catalog}.exemplo_models.modelo_online@champion"


def test_generate_resources_always_includes_refresh_endpoint_job():
    resources = generate_resources()
    assert "refresh_endpoint" in resources["resources"]["jobs"]


def test_generate_resources_omits_empty_resource_kinds():
    register_serving_config(
        ServingConfig(
            domain="exemplo",
            model_name="modelo_batch",
            mode="batch",
            spine_inference_table="workspace.exemplo.spine_inference",
            schedule_cron="0 0 6 * * ?",
        )
    )

    resources = generate_resources()
    assert "model_serving_endpoints" not in resources["resources"]
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `ModuleNotFoundError: No module named 'serving_platform.resource_gen'`

- [ ] **Step 3: Implementar**

```python
# src/serving_platform/resource_gen.py
import yaml

from .contract import get_registry
from .naming import derive_endpoint_name

BATCH_NOTEBOOK_PATH = "../notebooks/score_batch"
REFRESH_NOTEBOOK_PATH = "../notebooks/refresh_endpoint"


def _batch_job(model_name: str, config) -> dict:
    return {
        "name": f"score_batch_{model_name}",
        "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
        "parameters": [
            {"name": "model_name", "default": model_name},
            {"name": "catalog", "default": "${var.catalog}"},
            {"name": "git_commit", "default": "${var.git_commit}"},
            {"name": "git_branch", "default": "${var.git_branch}"},
        ],
        "tasks": [
            {
                "task_key": "score_batch",
                "notebook_task": {
                    "notebook_path": BATCH_NOTEBOOK_PATH,
                    "base_parameters": {
                        "model_name": "{{job.parameters.model_name}}",
                        "catalog": "{{job.parameters.catalog}}",
                        "git_commit": "{{job.parameters.git_commit}}",
                        "git_branch": "{{job.parameters.git_branch}}",
                    },
                },
            }
        ],
    }


def _online_endpoint(model_name: str, config) -> dict:
    return {
        "name": derive_endpoint_name(config.domain, model_name),
        "config": {
            "served_entities": [
                {
                    "name": model_name,
                    "entity_name": f"${{var.catalog}}.{config.domain}_models.{model_name}@{config.alias}",
                    "scale_to_zero_enabled": True,
                    "workload_size": "Small",
                }
            ]
        },
    }


def _refresh_endpoint_job() -> dict:
    return {
        "name": "refresh_endpoint",
        "parameters": [
            {"name": "model_name", "default": ""},
            {"name": "catalog", "default": "${var.catalog}"},
        ],
        "tasks": [
            {
                "task_key": "refresh_endpoint",
                "notebook_task": {
                    "notebook_path": REFRESH_NOTEBOOK_PATH,
                    "base_parameters": {
                        "model_name": "{{job.parameters.model_name}}",
                        "catalog": "{{job.parameters.catalog}}",
                    },
                },
            }
        ],
    }


def generate_resources() -> dict:
    registry = get_registry()
    jobs = {"refresh_endpoint": _refresh_endpoint_job()}
    endpoints = {}

    for model_name, config in registry.items():
        if config.mode == "batch":
            jobs[f"score_batch_{model_name}"] = _batch_job(model_name, config)
        else:
            endpoints[derive_endpoint_name(config.domain, model_name)] = _online_endpoint(model_name, config)

    resources = {"resources": {"jobs": jobs}}
    if endpoints:
        resources["resources"]["model_serving_endpoints"] = endpoints
    return resources


def write_resources(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(generate_resources(), f, sort_keys=False)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/serving_platform/resource_gen.py tests/test_resource_gen.py
git commit -m "feat: generate DAB jobs/endpoints dynamically from the serving registry"
```

---

## Task 7: Notebooks, domínio de exemplo e bundle DAB

Glue code + configuração. **Risco documentado no spec (seção 6), não um placeholder**:
a superfície exata do SDK usada em `refresh_endpoint.py`
(`WorkspaceClient().serving_endpoints.update_config_and_wait`, `ServedEntityInput`) e o
schema exato do recurso `model_serving_endpoints` do DAB devem ser conferidos contra a
versão instalada do `databricks-sdk` e a documentação atual antes do primeiro deploy
real de um `ServingConfig` com `mode="online"` — passo de verificação explícito na
Task 9.

**Files:**
- Create: `dominios/exemplo/__init__.py`
- Create: `dominios/exemplo/serving_configs.py`
- Create: `notebooks/score_batch.py`
- Create: `notebooks/refresh_endpoint.py`
- Create: `databricks.yml`
- Create: `scripts/generate_resources.py`

- [ ] **Step 1: Criar o domínio de exemplo (modo batch — sem custo de endpoint sempre ligado)**

```python
# dominios/exemplo/__init__.py
```

```python
# dominios/exemplo/serving_configs.py
from serving_platform.contract import ServingConfig, register_serving_config

config = ServingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    mode="batch",
    alias="champion",
    spine_inference_table="workspace.exemplo.spine_inference",
    schedule_cron="0 0 6 * * ?",
)

register_serving_config(config)
```

- [ ] **Step 2: Criar `notebooks/score_batch.py`** (a única task da trilha batch)

```python
# Databricks notebook source
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
import dominios.exemplo.serving_configs  # noqa: F401
from datetime import date, datetime

from databricks.feature_engineering import FeatureEngineeringClient

from serving_platform.contract import get_serving_config
from serving_platform.naming import derive_predictions_table_name
from serving_platform.quality import run_predictions_gate, gate_passed
from serving_platform.audit import RunRecord, write_run

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
git_commit = dbutils.widgets.get("git_commit")
git_branch = dbutils.widgets.get("git_branch")
config = get_serving_config(model_name)
run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().toString()

# COMMAND ----------
full_model_name = f"{catalog}.{config.domain}_models.{model_name}"
spine = spark.table(config.spine_inference_table)
input_row_count = spine.count()

fe = FeatureEngineeringClient()
predictions_df = fe.score_batch(
    model_uri=f"models:/{full_model_name}@{config.alias}",
    df=spine,
    result_type="double",
)

# COMMAND ----------
prediction_column = "prediction"
predictions_pd = predictions_df.select(prediction_column).toPandas()
findings = run_predictions_gate(predictions_pd, prediction_column, input_row_count)
passed = gate_passed(findings)
predictions_table = derive_predictions_table_name(catalog, config.domain, model_name)

if not passed:
    write_run(
        spark,
        RunRecord(
            component="serving",
            entity_name=predictions_table,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id_job,
            mode="batch",
            status="FAILED",
            window_start=date.today(),
            window_end=date.today(),
            run_ts=datetime.utcnow(),
        ),
    )
    failed_checks = [f.check for f in findings if f.status == "FAIL"]
    raise ValueError(f"predictions quality gate failed: {failed_checks}")

predictions_df.write.format("delta").mode("overwrite").saveAsTable(predictions_table)

write_run(
    spark,
    RunRecord(
        component="serving",
        entity_name=predictions_table,
        git_commit=git_commit,
        git_branch=git_branch,
        run_id=run_id_job,
        mode="batch",
        status="SUCCESS",
        window_start=date.today(),
        window_end=date.today(),
        run_ts=datetime.utcnow(),
    ),
)
```

- [ ] **Step 3: Criar `notebooks/refresh_endpoint.py`**

```python
# Databricks notebook source
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
import dominios.exemplo.serving_configs  # noqa: F401
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

from serving_platform.contract import get_serving_config
from serving_platform.naming import derive_endpoint_name

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_serving_config(model_name)

full_model_name = f"{catalog}.{config.domain}_models.{model_name}"
endpoint_name = derive_endpoint_name(config.domain, model_name)

# COMMAND ----------
client = WorkspaceClient()
client.serving_endpoints.update_config_and_wait(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            name=model_name,
            entity_name=f"{full_model_name}@{config.alias}",
            scale_to_zero_enabled=True,
            workload_size="Small",
        )
    ],
)
print(f"endpoint '{endpoint_name}' updated to current '{config.alias}' resolution")
```

- [ ] **Step 4: Criar `databricks.yml`**

```yaml
bundle:
  name: serving-platform

include:
  - resources/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for models, predictions, and the audit table.
    default: workspace
  git_commit:
    description: Git commit SHA of the deploy, propagated to job parameters for auditing.
    default: local
  git_branch:
    description: Git branch of the deploy, propagated to job parameters for auditing.
    default: local

targets:
  dev:
    mode: development
    default: true
```

- [ ] **Step 5: Criar `scripts/generate_resources.py`**

```python
# scripts/generate_resources.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dominios.exemplo.serving_configs  # noqa: F401
from serving_platform.resource_gen import write_resources

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_serving.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path))
    print(f"resources written to {output_path}")
```

- [ ] **Step 6: Rodar o gerador localmente e confirmar o YAML**

Run:
```powershell
.\.venv\Scripts\python.exe scripts\generate_resources.py
```
Expected: `resources written to .../resources/generated_serving.yml`, contendo o job
`score_batch_propensao_exemplo` com schedule e uma task só, e o job utilitário
`refresh_endpoint`. Sem `model_serving_endpoints` (o exemplo é batch).

- [ ] **Step 7: Validar o bundle**

Run:
```powershell
databricks bundle validate -t dev
```
Expected: validação sem erros de estrutura.

- [ ] **Step 8: Commit**

```bash
git add dominios/ notebooks/ databricks.yml scripts/generate_resources.py resources/.gitkeep
git commit -m "feat: add example batch domain, notebooks, and DAB bundle root"
```

---

## Task 8: GitHub Actions — deploy com tracking de commit/branch

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Criar o workflow**

```yaml
# .github/workflows/deploy.yml
name: Deploy serving-platform

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dev dependencies
        run: pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest

      - name: Generate resources
        run: python scripts/generate_resources.py

      - name: Install Databricks CLI
        uses: databricks/setup-cli@main

      - name: Deploy bundle
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks bundle deploy -t dev \
            --var="git_commit=${{ github.sha }}" \
            --var="git_branch=${{ github.ref_name }}"
```

- [ ] **Step 2: Confirmar os secrets `DATABRICKS_HOST`/`DATABRICKS_TOKEN` no GitHub
  (mesmos valores dos outros repositórios, se for o mesmo workspace).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: deploy bundle on push to main with git commit/branch tracking"
```

---

## Task 9: README e verificação ponta a ponta

**Files:**
- Create: `README.md`

- [ ] **Step 1: Escrever o README**

```markdown
# serving-platform

Componente de serving do ecossistema de MLOps no Databricks: `ServingConfig` único
cobrindo online (Model Serving endpoint) e batch (`fe.score_batch` em task única,
sem join manual, sem tabela intermediária — o mesmo Cenário B da POC original).

Design completo em
[`docs/superpowers/specs/2026-08-23-serving-design.md`](docs/superpowers/specs/2026-08-23-serving-design.md).

## Como declarar um serving

```python
from serving_platform.contract import ServingConfig, register_serving_config

config = ServingConfig(
    domain="credito",
    model_name="propensao_default",
    mode="batch",  # ou "online"
    alias="champion",
    spine_inference_table="workspace.credito.spine_inference",  # obrigatório em mode="batch"
    schedule_cron="0 0 6 * * ?",                                  # obrigatório em mode="batch"
)
register_serving_config(config)
```

## Local (sem Spark)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

## No Databricks (Free Edition)

```powershell
python scripts\generate_resources.py
databricks bundle validate -t dev
databricks bundle deploy   -t dev

# batch
databricks bundle run score_batch_propensao_exemplo -t dev

# online, depois de promover um novo champion no training-platform
databricks bundle run refresh_endpoint -t dev --params model_name=propensao_exemplo
```

## Riscos conhecidos

- Resolução automática de `FeatureLookup` num Model Serving endpoint padrão contra o
  Online Feature Store (Lakebase) é **assumida, não validada em produção** — confirmar
  ao vivo antes de depender disso.
- Endpoint online sempre ligado tem custo contínuo na Free Edition; scale-to-zero é
  tentado, não garantido. Derrube o endpoint manualmente quando não estiver testando.

## Secrets necessários no GitHub Actions

| Secret | Descrição |
|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks Free Edition |
| `DATABRICKS_TOKEN` | Personal access token com permissão de deploy |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions and known risks"
```

- [ ] **Step 3: Verificação ponta a ponta — trilha batch (prioridade, sem custo de
  endpoint):** confirmar que `workspace.exemplo.spine_inference` existe, rodar
  `databricks bundle run score_batch_propensao_exemplo -t dev`, confirmar que
  `workspace.exemplo_predictions.propensao_exemplo` foi criada e que há uma linha
  `SUCCESS` em `platform_audit.pipeline_runs` com `component="serving"`,
  `mode="batch"`.

- [ ] **Step 4: Forçar falha do gate** (ex.: apontar `spine_inference_table` para uma
  tabela vazia ou com schema incompatível) e confirmar que **nada é escrito** na
  tabela de predições, e que a auditoria grava `status=FAILED`.

- [ ] **Step 5: Verificação ponta a ponta — trilha online (só depois da batch
  funcionar, e ciente do custo):** registrar um `ServingConfig` de teste com
  `mode="online"`, gerar recursos, `databricks bundle deploy`, confirmar que o endpoint
  sobe e responde a uma chamada de teste com as features resolvidas corretamente — esta
  é a verificação do risco documentado no spec (seção 6). Se a resolução automática de
  `FeatureLookup` não funcionar como assumido, documentar o comportamento real
  encontrado e ajustar o spec antes de prosseguir. **Derrubar o endpoint manualmente ao
  final do teste** para não incorrer em custo contínuo.

---

## Self-Review

**1. Cobertura do spec:** contrato único com `mode` (Task 2), trilha batch em task
única via `fe.score_batch` (Task 7, notebook `score_batch.py` — sem `build_master`,
sem join manual, conforme a ênfase pedida), gate de qualidade bloqueante (Task 4),
convenção de nome de predições e endpoint (Task 3), geração dinâmica de recursos (Task
6), atualização manual de endpoint (Task 7, `refresh_endpoint.py`), reuso exato do
schema de auditoria (Task 5). Todas as seções do spec têm uma task ou notebook
correspondente.

**2. Placeholders:** nenhum "TBD"/"TODO". A incerteza sobre a superfície exata do SDK
de Model Serving (`update_config_and_wait`, `ServedEntityInput`) e o schema do recurso
`model_serving_endpoints` está documentada como risco com uma implementação concreta +
um passo de verificação explícito (Task 7 intro, Task 9 Step 5) — não é uma lacuna
deixada em aberto.

**3. Consistência de tipos:** `ServingConfig`, `RunRecord` e as chaves usadas em
`resource_gen.py` (`entity_name` com sufixo `@{alias}`, `derive_endpoint_name`) foram
conferidas contra o uso nos dois notebooks e nos testes — mesmos nomes, mesma
convenção de string em todos os lugares que constroem `full_model_name` ou
`model_uri`.

---

Plano completo e salvo em `docs/superpowers/plans/2026-08-23-serving-implementation.md`.
