# Componente de Monitoramento — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o framework `monitoring_platform`: contrato `MonitoringConfig` sobre feature tables e tabelas de predições, resolução automática da baseline via `platform_audit.pipeline_runs`, uso do Lakehouse Monitoring nativo (risco aceito) com uma camada de avaliação que centraliza resultados em `platform_monitoring.drift_metrics`, e alerta de retreino que nunca dispara nada sozinho.

**Architecture:** Lógica pura (contrato, resolução de baseline, avaliação de threshold, nomenclatura, construção de linhas da tabela central) em `src/monitoring_platform/`, testável com `pytest` local, sem Spark e sem o Lakehouse Monitoring real. Um notebook (`evaluate_drift.py`) faz a parte real: cria/atualiza o monitor do LHM, lê a saída dele, aplica o threshold, escreve na tabela central e na auditoria. Um domínio de exemplo prova o fluxo ponta a ponta — e é onde o risco de viabilidade do LHM na Free Edition (spec, seção 6) é finalmente testado ao vivo.

**Tech Stack:** Python 3.11, `databricks-sdk` (Lakehouse Monitoring / `quality_monitors`), PySpark + Delta (runtime Databricks serverless), Databricks Asset Bundles, pytest, GitHub Actions.

**Emenda (2026-08-23, durante o design da arquitetura de plataforma):** este
repositório passou a ser um framework puro — `dominios/exemplo/` foi renomeada para
`examples/` (mesmo papel de harness de integração, não domínio real). O contrato
`MonitoringConfig` não muda (`domain` já era um campo explícito). O
`.github/workflows/deploy.yml` (Task 10) passou a ser um caller do reusable workflow
centralizado em `mlops-platform`. Ver
`docs/superpowers/specs/2026-08-23-monitoramento-design.md`, seção 1.1.

**Correção preventiva (2026-08-24, por analogia com bugs já confirmados ao vivo nos
três componentes anteriores, aplicada antes de qualquer implementação deste
componente):**
1. **Extensão `.py` em `notebook_path`** — o CLI instalado rejeita referências sem
   extensão. `resource_gen.py` (Task 8) usa `"../notebooks/evaluate_drift.py"`.
2. **Bootstrap de `sys.path`** — o cwd de um notebook deployado via DAB não inclui a
   raiz do bundle nem `src/` por padrão. `evaluate_drift.py` (Task 9) insere os dois
   antes de importar `examples.monitoring_configs`/`monitoring_platform`.
3. **`currentRunId()` sem `.get()` e sem fallback** — levanta `Py4JSecurityException`
   em compute serverless/shared access mode. `evaluate_drift.py` usa
   `.currentRunId().get().toString()` num `try`, com fallback para `uuid.uuid4()`.
4. **`CREATE SCHEMA IF NOT EXISTS` antes do primeiro `saveAsTable`** — Unity Catalog
   não cria o schema sozinho. Aplicado em dois lugares: `audit.py`'s `write_run` (Task
   7, mesmo padrão exato dos três componentes anteriores) e `central_table.py`'s
   `write_drift_metrics` (Task 6) — `platform_monitoring` é um schema novo, nunca
   criado por nenhum componente anterior.

Nenhum desses foi validado ao vivo neste componente ainda — são inferências por
analogia. A Task 12 (verificação ponta a ponta) confirma se bastam, junto com o risco
central do Lakehouse Monitoring (spec, seção 6).

---

## Scope Check

Este plano cobre só o Componente 4 (Monitoramento), conforme
`docs/superpowers/specs/2026-08-23-monitoramento-design.md`. Com este componente, os
quatro specs e planos do ecossistema estão completos — a implementação de cada um é o
próximo passo, feito separadamente.

## File Structure

```
monitoring-platform/
├── databricks.yml
├── pyproject.toml
├── pytest.ini
├── requirements-dev.txt
├── README.md
├── src/
│   └── monitoring_platform/
│       ├── __init__.py
│       ├── contract.py         # MonitoringConfig, registro por (domain, model_name, target_type)
│       ├── baseline.py         # TrainingRun, resolve_baseline_window a partir de pipeline_runs
│       ├── evaluation.py       # DriftResult, evaluate_drift (compara valor contra threshold)
│       ├── naming.py           # derive_monitor_key
│       ├── central_table.py    # build_drift_metric_row + write_drift_metrics (Spark)
│       ├── audit.py            # RunRecord/to_row/write_run (duplicado dos outros componentes)
│       └── resource_gen.py     # gera 1 job (com schedule próprio) por MonitoringConfig
├── examples/
│   ├── __init__.py
│   └── monitoring_configs.py   # não-produtivo
├── notebooks/
│   └── evaluate_drift.py       # cria/atualiza o monitor LHM, avalia, centraliza, audita
├── scripts/
│   └── generate_resources.py
├── resources/
│   └── (generated_monitoring.yml — gerado, não escrito à mão)
├── tests/
│   ├── __init__.py
│   ├── test_contract.py
│   ├── test_baseline.py
│   ├── test_evaluation.py
│   ├── test_naming.py
│   ├── test_central_table.py
│   ├── test_audit.py
│   └── test_resource_gen.py
└── .github/
    └── workflows/
        └── deploy.yml
```

Igual ao `feature-platform` e ao `serving-platform`: os recursos DAB são **gerados
dinamicamente**, um job por `MonitoringConfig` (cada um com seu próprio
`schedule_cron` — Databricks Workflows agenda no nível do job, não da task, então
schedules diferentes por config exigem jobs diferentes, não tasks dentro de um job
compartilhado).

---

## Task 1: Scaffolding do repositório

**Files:**
- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `src/monitoring_platform/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "monitoring-platform"
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
```

- [ ] **Step 4: Criar `src/monitoring_platform/__init__.py`** (vazio)

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
git add pyproject.toml pytest.ini requirements-dev.txt src/monitoring_platform/__init__.py .gitignore
git commit -m "chore: scaffold monitoring-platform package"
```

---

## Task 2: Contrato e registro (`contract.py`)

Registrado por `(domain, model_name, target_type)`, não só por `model_name` — um
mesmo modelo pode ter um `MonitoringConfig` para a feature table e outro para as
predições ao mesmo tempo.

**Files:**
- Create: `src/monitoring_platform/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_contract.py
import pytest

from monitoring_platform.contract import (
    MonitoringConfig,
    register_monitoring_config,
    get_monitoring_config,
    get_registry,
    clear_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _config(**overrides):
    defaults = dict(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="feature_table",
        target_table="workspace.exemplo_features.customer_transaction_features",
        columns=["txn_count", "avg_ticket"],
        threshold=0.2,
        schedule_cron="0 0 7 * * ?",
    )
    defaults.update(overrides)
    return MonitoringConfig(**defaults)


def test_register_and_get_monitoring_config():
    config = _config()
    register_monitoring_config(config)

    assert get_monitoring_config("exemplo", "propensao_exemplo", "feature_table") is config


def test_same_model_can_have_feature_table_and_predictions_configs():
    register_monitoring_config(_config(target_type="feature_table"))
    register_monitoring_config(_config(target_type="predictions", target_table="workspace.exemplo_predictions.propensao_exemplo"))

    assert len(get_registry()) == 2


def test_register_monitoring_config_rejects_duplicate_key():
    register_monitoring_config(_config())

    with pytest.raises(ValueError, match="already registered"):
        register_monitoring_config(_config())
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.contract'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/contract.py
from dataclasses import dataclass
from typing import Literal


@dataclass
class MonitoringConfig:
    domain: str
    model_name: str
    target_type: Literal["feature_table", "predictions"]
    target_table: str
    columns: list[str]
    threshold: float
    schedule_cron: str


def _config_key(domain: str, model_name: str, target_type: str) -> str:
    return f"{domain}.{model_name}.{target_type}"


_REGISTRY: dict[str, MonitoringConfig] = {}


def register_monitoring_config(config: MonitoringConfig) -> None:
    key = _config_key(config.domain, config.model_name, config.target_type)
    if key in _REGISTRY:
        raise ValueError(f"monitoring config '{key}' already registered")
    _REGISTRY[key] = config


def get_monitoring_config(domain: str, model_name: str, target_type: str) -> MonitoringConfig:
    return _REGISTRY[_config_key(domain, model_name, target_type)]


def get_registry() -> dict[str, MonitoringConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/contract.py tests/test_contract.py
git commit -m "feat: add MonitoringConfig contract keyed by domain/model/target_type"
```

---

## Task 3: Resolução da baseline (`baseline.py`)

**Files:**
- Create: `src/monitoring_platform/baseline.py`
- Test: `tests/test_baseline.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_baseline.py
from datetime import date, datetime

import pytest

from monitoring_platform.baseline import TrainingRun, resolve_baseline_window, NoTrainingRunError


def _run(entity_name, status, window_start, window_end, run_ts):
    return TrainingRun(entity_name=entity_name, status=status, window_start=window_start, window_end=window_end, run_ts=run_ts)


def test_resolve_baseline_window_picks_most_recent_success():
    runs = [
        _run("workspace.exemplo_models.propensao_exemplo", "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1)),
        _run("workspace.exemplo_models.propensao_exemplo", "SUCCESS", date(2026, 2, 1), date(2026, 7, 31), datetime(2026, 8, 1)),
    ]
    start, end = resolve_baseline_window(runs, "workspace.exemplo_models.propensao_exemplo")
    assert start == date(2026, 2, 1)
    assert end == date(2026, 7, 31)


def test_resolve_baseline_window_ignores_failed_runs():
    runs = [
        _run("workspace.exemplo_models.propensao_exemplo", "FAILED", date(2026, 3, 1), date(2026, 8, 31), datetime(2026, 9, 1)),
        _run("workspace.exemplo_models.propensao_exemplo", "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1)),
    ]
    start, end = resolve_baseline_window(runs, "workspace.exemplo_models.propensao_exemplo")
    assert start == date(2026, 1, 1)
    assert end == date(2026, 6, 30)


def test_resolve_baseline_window_ignores_other_models():
    runs = [_run("workspace.exemplo_models.outro_modelo", "SUCCESS", date(2026, 1, 1), date(2026, 6, 30), datetime(2026, 7, 1))]

    with pytest.raises(NoTrainingRunError, match="propensao_exemplo"):
        resolve_baseline_window(runs, "workspace.exemplo_models.propensao_exemplo")


def test_resolve_baseline_window_raises_when_no_runs():
    with pytest.raises(NoTrainingRunError):
        resolve_baseline_window([], "workspace.exemplo_models.propensao_exemplo")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_baseline.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.baseline'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/baseline.py
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class TrainingRun:
    entity_name: str
    status: str
    window_start: date
    window_end: date
    run_ts: datetime


class NoTrainingRunError(Exception):
    """Levantado quando não há nenhum run SUCCESS de treino registrado para o modelo."""


def resolve_baseline_window(training_runs: list[TrainingRun], full_model_name: str) -> tuple[date, date]:
    candidates = [r for r in training_runs if r.entity_name == full_model_name and r.status == "SUCCESS"]
    if not candidates:
        raise NoTrainingRunError(f"no successful training run found for model '{full_model_name}'")

    most_recent = max(candidates, key=lambda r: r.run_ts)
    return most_recent.window_start, most_recent.window_end
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_baseline.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/baseline.py tests/test_baseline.py
git commit -m "feat: resolve monitoring baseline window from platform_audit.pipeline_runs"
```

---

## Task 4: Avaliação de drift (`evaluation.py`)

> **Correção (achada durante a implementação — Task 4 Step 1, teste inconsistente com
> o Step 3 original):** `test_evaluate_drift_carries_input_fields` chamava
> `evaluate_drift("avg_ticket", "ks_test_pvalue", 0.01, threshold=0.05)` esperando
> `DRIFT_DETECTED`, mas a implementação original do Step 3 (`value > threshold`) produz
> `PASS` para `0.01 > 0.05` (falso) — o teste nunca teria passado com a implementação
> como escrita. A intenção provável era testar semântica de p-value (valor baixo =
> drift), mas o spec não pede comparação direction-aware por métrica — `evaluate_drift`
> é descrita como uma comparação uniforme contra o threshold, e os nomes reais de
> métrica que o Lakehouse Monitoring produz são um risco não confirmado (spec, seção
> 6; Task 9 é onde isso se resolve). Adicionar uma heurística baseada em substring do
> nome da métrica (`"pvalue" in name`) dentro da lógica pura seria adivinhar semântica
> de uma API ainda não validada. Corrigido mantendo a implementação simples
> (`value > threshold`) e ajustando os dados do teste para serem consistentes com ela —
> o teste continua verificando que todos os campos são carregados corretamente para o
> `DriftResult`, só com um valor/threshold que realmente produz `DRIFT_DETECTED` sob
> comparação direta. Se a Task 9 revelar que alguma métrica do LHM precisa de
> comparação invertida, isso é tratado no notebook (que já converte
> `latest.get("statistic")` antes de chamar `evaluate_drift`), não nesta função pura.

**Files:**
- Create: `src/monitoring_platform/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_evaluation.py
from monitoring_platform.evaluation import DriftResult, evaluate_drift


def test_evaluate_drift_passes_below_threshold():
    result = evaluate_drift("txn_count", "js_distance", 0.05, threshold=0.2)
    assert result.status == "PASS"


def test_evaluate_drift_detects_above_threshold():
    result = evaluate_drift("txn_count", "js_distance", 0.35, threshold=0.2)
    assert result.status == "DRIFT_DETECTED"


def test_evaluate_drift_boundary_value_passes():
    result = evaluate_drift("txn_count", "js_distance", 0.2, threshold=0.2)
    assert result.status == "PASS"


def test_evaluate_drift_carries_input_fields():
    result = evaluate_drift("avg_ticket", "ks_test_statistic", 0.35, threshold=0.05)
    assert result == DriftResult(
        column_name="avg_ticket",
        drift_metric_name="ks_test_statistic",
        drift_metric_value=0.35,
        threshold=0.05,
        status="DRIFT_DETECTED",
    )
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.evaluation'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/evaluation.py
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftResult:
    column_name: str
    drift_metric_name: str
    drift_metric_value: float
    threshold: float
    status: str  # "PASS" ou "DRIFT_DETECTED"


def evaluate_drift(column_name: str, drift_metric_name: str, drift_metric_value: float, threshold: float) -> DriftResult:
    status = "DRIFT_DETECTED" if drift_metric_value > threshold else "PASS"
    return DriftResult(
        column_name=column_name,
        drift_metric_name=drift_metric_name,
        drift_metric_value=drift_metric_value,
        threshold=threshold,
        status=status,
    )
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/evaluation.py tests/test_evaluation.py
git commit -m "feat: add threshold-based drift evaluation"
```

---

## Task 5: Nomenclatura (`naming.py`)

**Files:**
- Create: `src/monitoring_platform/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_naming.py
from monitoring_platform.naming import derive_monitor_key


def test_derive_monitor_key_follows_convention():
    key = derive_monitor_key(domain="credito", model_name="propensao_default", target_type="feature_table")
    assert key == "credito_propensao_default_feature_table"


def test_derive_monitor_key_differs_by_target_type():
    key_a = derive_monitor_key("credito", "propensao_default", "feature_table")
    key_b = derive_monitor_key("credito", "propensao_default", "predictions")
    assert key_a != key_b
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.naming'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/naming.py
def derive_monitor_key(domain: str, model_name: str, target_type: str) -> str:
    return f"{domain}_{model_name}_{target_type}"
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/naming.py tests/test_naming.py
git commit -m "feat: add monitor key naming convention"
```

---

## Task 6: Tabela central de drift (`central_table.py`)

**Files:**
- Create: `src/monitoring_platform/central_table.py`
- Test: `tests/test_central_table.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_central_table.py
from datetime import date, datetime

from monitoring_platform.evaluation import DriftResult
from monitoring_platform.central_table import build_drift_metric_row, DRIFT_METRICS_TABLE


def test_drift_metrics_table_name():
    assert DRIFT_METRICS_TABLE == "platform_monitoring.drift_metrics"


def test_build_drift_metric_row_maps_all_fields():
    result = DriftResult(
        column_name="txn_count",
        drift_metric_name="js_distance",
        drift_metric_value=0.35,
        threshold=0.2,
        status="DRIFT_DETECTED",
    )

    row = build_drift_metric_row(
        domain="exemplo",
        model_name="propensao_exemplo",
        entity_name="workspace.exemplo_features.customer_transaction_features",
        target_type="feature_table",
        result=result,
        window_start=date(2026, 8, 23),
        window_end=date(2026, 8, 23),
        run_ts=datetime(2026, 8, 23, 7, 0, 0),
    )

    assert row == {
        "domain": "exemplo",
        "model_name": "propensao_exemplo",
        "entity_name": "workspace.exemplo_features.customer_transaction_features",
        "target_type": "feature_table",
        "column_name": "txn_count",
        "drift_metric_name": "js_distance",
        "drift_metric_value": 0.35,
        "threshold": 0.2,
        "status": "DRIFT_DETECTED",
        "window_start": date(2026, 8, 23),
        "window_end": date(2026, 8, 23),
        "run_ts": datetime(2026, 8, 23, 7, 0, 0),
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_central_table.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.central_table'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/central_table.py
from datetime import date, datetime

from .evaluation import DriftResult

DRIFT_METRICS_TABLE = "platform_monitoring.drift_metrics"


def build_drift_metric_row(
    domain: str,
    model_name: str,
    entity_name: str,
    target_type: str,
    result: DriftResult,
    window_start: date,
    window_end: date,
    run_ts: datetime,
) -> dict:
    return {
        "domain": domain,
        "model_name": model_name,
        "entity_name": entity_name,
        "target_type": target_type,
        "column_name": result.column_name,
        "drift_metric_name": result.drift_metric_name,
        "drift_metric_value": result.drift_metric_value,
        "threshold": result.threshold,
        "status": result.status,
        "window_start": window_start,
        "window_end": window_end,
        "run_ts": run_ts,
    }


def write_drift_metrics(spark, rows: list[dict]) -> None:
    """Requer SparkSession — exercitado via notebook (Task 9), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — platform_monitoring
    # é um schema novo, nunca criado por nenhum componente anterior (mesmo bug já
    # confirmado nos três componentes anteriores).
    schema = DRIFT_METRICS_TABLE.split(".")[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    df = spark.createDataFrame(rows)
    if spark.catalog.tableExists(DRIFT_METRICS_TABLE):
        df.write.format("delta").mode("append").saveAsTable(DRIFT_METRICS_TABLE)
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(DRIFT_METRICS_TABLE)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_central_table.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/central_table.py tests/test_central_table.py
git commit -m "feat: add central drift_metrics table row construction and writer"
```

---

## Task 7: Auditoria (`audit.py`)

Duplicado deliberadamente dos outros três componentes — mesmo schema, mesma
implementação.

**Files:**
- Create: `src/monitoring_platform/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_audit.py
from datetime import date, datetime

from monitoring_platform.audit import RunRecord, to_row, AUDIT_TABLE


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="monitoring",
        entity_name="workspace.exemplo_features.customer_transaction_features",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="drift_check",
        status="SUCCESS",
        window_start=date(2026, 8, 23),
        window_end=date(2026, 8, 23),
        run_ts=datetime(2026, 8, 23, 7, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "monitoring",
        "entity_name": "workspace.exemplo_features.customer_transaction_features",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "drift_check",
        "status": "SUCCESS",
        "window_start": date(2026, 8, 23),
        "window_end": date(2026, 8, 23),
        "run_ts": datetime(2026, 8, 23, 7, 0, 0),
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.audit'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/audit.py
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
    """Requer SparkSession — exercitado via notebook (Task 9), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — sem isso,
    # a primeira escrita falha com SCHEMA_NOT_FOUND (mesmo bug confirmado nos três
    # componentes anteriores).
    schema = AUDIT_TABLE.split(".")[0]
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
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
git add src/monitoring_platform/audit.py tests/test_audit.py
git commit -m "feat: duplicate audit record type and pipeline_runs writer"
```

---

## Task 8: Geração de recursos (`resource_gen.py`)

Um job por `MonitoringConfig` (não tasks dentro de um job compartilhado): o schedule
do Databricks Workflows é por job, e cada config pode ter um `schedule_cron` diferente.

**Files:**
- Create: `src/monitoring_platform/resource_gen.py`
- Test: `tests/test_resource_gen.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_resource_gen.py
import pytest

from monitoring_platform.contract import MonitoringConfig, register_monitoring_config, clear_registry
from monitoring_platform.resource_gen import generate_resources


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _config(**overrides):
    defaults = dict(
        domain="exemplo",
        model_name="propensao_exemplo",
        target_type="feature_table",
        target_table="workspace.exemplo_features.customer_transaction_features",
        columns=["txn_count"],
        threshold=0.2,
        schedule_cron="0 0 7 * * ?",
    )
    defaults.update(overrides)
    return MonitoringConfig(**defaults)


def test_generates_one_job_per_config_with_its_own_schedule():
    register_monitoring_config(_config(target_type="feature_table"))
    register_monitoring_config(_config(target_type="predictions", target_table="workspace.exemplo_predictions.propensao_exemplo", schedule_cron="0 0 8 * * ?"))

    resources = generate_resources()
    jobs = resources["resources"]["jobs"]

    assert len(jobs) == 2
    feature_job = jobs["drift_check_exemplo_propensao_exemplo_feature_table"]
    predictions_job = jobs["drift_check_exemplo_propensao_exemplo_predictions"]
    assert feature_job["schedule"]["quartz_cron_expression"] == "0 0 7 * * ?"
    assert predictions_job["schedule"]["quartz_cron_expression"] == "0 0 8 * * ?"


def test_each_job_has_a_single_evaluate_drift_task():
    register_monitoring_config(_config())

    resources = generate_resources()
    job = list(resources["resources"]["jobs"].values())[0]

    assert [t["task_key"] for t in job["tasks"]] == ["evaluate_drift"]
    assert job["tasks"][0]["notebook_task"]["notebook_path"] == "../notebooks/evaluate_drift.py"


def test_job_parameters_carry_domain_model_and_target_type():
    register_monitoring_config(_config())

    resources = generate_resources()
    job = list(resources["resources"]["jobs"].values())[0]
    param_names = {p["name"] for p in job["parameters"]}

    assert param_names == {"domain", "model_name", "target_type", "catalog", "git_commit", "git_branch"}
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `ModuleNotFoundError: No module named 'monitoring_platform.resource_gen'`

- [ ] **Step 3: Implementar**

```python
# src/monitoring_platform/resource_gen.py
import yaml

from .contract import get_registry
from .naming import derive_monitor_key

NOTEBOOK_PATH = "../notebooks/evaluate_drift.py"


def _monitoring_job(key: str, config) -> dict:
    return {
        "name": f"drift_check_{key}",
        "schedule": {"quartz_cron_expression": config.schedule_cron, "timezone_id": "UTC"},
        "parameters": [
            {"name": "domain", "default": config.domain},
            {"name": "model_name", "default": config.model_name},
            {"name": "target_type", "default": config.target_type},
            {"name": "catalog", "default": "${var.catalog}"},
            {"name": "git_commit", "default": "${var.git_commit}"},
            {"name": "git_branch", "default": "${var.git_branch}"},
        ],
        "tasks": [
            {
                "task_key": "evaluate_drift",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_PATH,
                    "base_parameters": {
                        "domain": "{{job.parameters.domain}}",
                        "model_name": "{{job.parameters.model_name}}",
                        "target_type": "{{job.parameters.target_type}}",
                        "catalog": "{{job.parameters.catalog}}",
                        "git_commit": "{{job.parameters.git_commit}}",
                        "git_branch": "{{job.parameters.git_branch}}",
                    },
                },
            }
        ],
    }


def generate_resources() -> dict:
    registry = get_registry()
    jobs = {}
    for config in registry.values():
        key = derive_monitor_key(config.domain, config.model_name, config.target_type)
        jobs[f"drift_check_{key}"] = _monitoring_job(key, config)
    return {"resources": {"jobs": jobs}}


def write_resources(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(generate_resources(), f, sort_keys=False)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/monitoring_platform/resource_gen.py tests/test_resource_gen.py
git commit -m "feat: generate one scheduled DAB job per monitoring config"
```

---

## Task 9: Notebook, domínio de exemplo e bundle DAB

> **Correção (achada ao vivo na Task 12 — o próprio risco da seção 6 sendo resolvido):**
> boa notícia confirmada primeiro: `client.quality_monitors.create/get/run_refresh`
> **existem e respondem** na Free Edition — os nomes de método e de parâmetro
> (`table_name`, `output_schema_name`, `assets_dir`, `baseline_table_name`) batiam
> exatamente com o assumido, confirmado via `inspect.signature` na versão instalada do
> SDK antes do primeiro deploy real. O único gap: `create()` exige explicitamente um
> dos três parâmetros `snapshot`, `time_series` ou `inference_log` (nenhum estava
> presente na chamada original) — falhou ao vivo com
> `InvalidParameterValue: Please specify one of 'snapshot', 'time_series', or
> 'inference_log' field`. Escolhido `snapshot=MonitorSnapshot()` (sem campos
> obrigatórios) em vez de `time_series`: o caso de uso deste componente é comparar o
> estado atual contra uma baseline fixa (já resolvida via `resolve_baseline_window` e
> passada como `baseline_table_name`), não análise de múltiplas janelas internas por
> timestamp — `time_series` exigiria adicionar uma nova coluna obrigatória
> (`timestamp_col`) ao `MonitoringConfig`, uma mudança de contrato desnecessária para o
> que o spec pede. Corrigido importando `MonitorSnapshot` de
> `databricks.sdk.service.catalog` e passando `snapshot=MonitorSnapshot()` na chamada
> de `create()`.
>
> **Gap real encontrado ao corrigir isso, ainda não resolvido:**
> `baseline_table_name=config.target_table` estava autorreferente — o parâmetro de
> baseline apontava para a própria tabela monitorada, o que tornaria qualquer
> comparação de drift sempre zero (baseline == atual). `baseline_start`/`baseline_end`,
> resolvidos por `resolve_baseline_window` logo acima no notebook, nunca eram
> efetivamente usados para materializar uma tabela de baseline de verdade — a janela é
> calculada e descartada. A correção real exigiria ou (a) materializar uma tabela de
> baseline filtrada por `[baseline_start, baseline_end]`, o que precisa saber a coluna
> de timestamp de `target_table` (`feature_ts` para feature tables, `scored_at` para
> predições — nenhuma delas está no `MonitoringConfig` hoje), ou (b) usar `time_series`
> em vez de `snapshot` e deixar o LHM comparar janelas internamente. Isso é uma decisão
> de design real, não uma correção mecânica — fora do escopo de uma correção ao vivo
> sem uma rodada de grilling. Corrigido removendo `baseline_table_name` da chamada (era
> ativamente errado, pior que omitir) — sem baseline explícita, monitores do tipo
> `snapshot` comparam cada refresh contra o refresh anterior automaticamente (padrão do
> LHM), o que pelo menos produz uma comparação não-trivial, mesmo que não seja
> exatamente "contra a janela de treino" como o spec pretende. `resolve_baseline_window`
> continua implementada e testada — só não está conectada à criação do monitor ainda.
> **Item em aberto, não fechado nesta sessão:** decidir como materializar/usar a
> baseline de verdade (provavelmente exige adicionar um campo de coluna de timestamp ao
> `MonitoringConfig`).

Glue code + configuração. **Risco documentado no spec (seção 6), não um placeholder**:
os nomes exatos dos métodos do `databricks-sdk` para criar/atualizar um monitor do
Lakehouse Monitoring (`client.quality_monitors.create/get/run_refresh`, os parâmetros
de perfil e baseline) e o schema exato das tabelas `_drift_metrics` geradas pelo LHM
(nomes de coluna como `column_name`, `drift_type`, `statistic`) **precisam ser
conferidos contra a documentação atual e a versão instalada do `databricks-sdk` antes
do primeiro deploy real** — é exatamente aqui que o risco aceito na seção 6 do spec é
testado. Se a API tiver mudado, ajuste esta implementação; o restante do componente
(contrato, tabela central, filosofia de sugestão) não muda.

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/monitoring_configs.py`
- Create: `notebooks/evaluate_drift.py`
- Create: `databricks.yml`
- Create: `scripts/generate_resources.py`

- [ ] **Step 1: Criar o exemplo não-produtivo**

```python
# examples/__init__.py
```

```python
# examples/monitoring_configs.py
from monitoring_platform.contract import MonitoringConfig, register_monitoring_config

feature_drift = MonitoringConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    target_type="feature_table",
    target_table="workspace.exemplo_features.customer_transaction_features",
    columns=["txn_count", "avg_ticket"],
    threshold=0.2,
    schedule_cron="0 0 7 * * ?",
)
register_monitoring_config(feature_drift)

predictions_drift = MonitoringConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    target_type="predictions",
    target_table="workspace.exemplo_predictions.propensao_exemplo",
    columns=["prediction"],
    threshold=0.2,
    schedule_cron="0 0 8 * * ?",
)
register_monitoring_config(predictions_drift)
```

- [ ] **Step 2: Criar `notebooks/evaluate_drift.py`**

```python
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
# antes do primeiro deploy real.
from databricks.sdk.service.catalog import MonitorSnapshot

client = WorkspaceClient()
try:
    client.quality_monitors.get(table_name=config.target_table)
    client.quality_monitors.run_refresh(table_name=config.target_table)
except Exception:
    # snapshot=MonitorSnapshot(): create() exige um de snapshot/time_series/
    # inference_log. Sem baseline_table_name (item em aberto — ver nota de correção
    # acima): monitores snapshot comparam cada refresh contra o anterior por padrão.
    client.quality_monitors.create(
        table_name=config.target_table,
        assets_dir=f"/Shared/monitoring-platform/{domain}/{model_name}/{target_type}",
        output_schema_name=f"{catalog}.{domain}_monitoring",
        snapshot=MonitorSnapshot(),
    )

# COMMAND ----------
# RISCO DOCUMENTADO (spec, seção 6): nomes de coluna da tabela `_drift_metrics` gerada
# pelo LHM a confirmar contra a versão instalada antes do primeiro deploy real.
drift_table = f"{config.target_table}_drift_metrics"
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
```

- [ ] **Step 3: Criar `databricks.yml`**

```yaml
bundle:
  name: monitoring-platform

include:
  - resources/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for monitored tables and the audit table.
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

- [ ] **Step 4: Criar `scripts/generate_resources.py`**

> **Correção preventiva:** o `sys.path.insert` original só incluía `src/`, não a raiz
> do repositório — rodar o script diretamente não coloca a raiz no `sys.path`
> automaticamente, quebrando `import examples.monitoring_configs` com
> `ModuleNotFoundError: No module named 'examples'` (mesmo bug já confirmado três
> vezes: `feature-platform`, `serving-platform`).

```python
# scripts/generate_resources.py
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "src"):
    sys.path.insert(0, str(_p))

import examples.monitoring_configs  # noqa: F401
from monitoring_platform.resource_gen import write_resources

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_monitoring.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path))
    print(f"resources written to {output_path}")
```

- [ ] **Step 5: Rodar o gerador localmente e validar o bundle**

Run:
```powershell
.\.venv\Scripts\python.exe scripts\generate_resources.py
databricks bundle validate -t dev
```
Expected: dois jobs gerados (`drift_check_exemplo_propensao_exemplo_feature_table` e
`drift_check_exemplo_propensao_exemplo_predictions`), cada um com schedule próprio e
uma task `evaluate_drift`; `validate` sem erros de estrutura.

- [ ] **Step 6: Commit**

```bash
git add examples/ notebooks/ databricks.yml scripts/generate_resources.py resources/.gitkeep
git commit -m "feat: add non-productive example, evaluate_drift notebook, and DAB bundle root"
```

---

## Task 10: GitHub Actions — caller do reusable workflow (`mlops-platform`)

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Criar o workflow**

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    uses: ViniciusOtoni/mlops-platform/.github/workflows/deploy-bundle.yml@main
    with:
      working-directory: .
    secrets: inherit
```

- [ ] **Step 2: Confirmar os secrets `DATABRICKS_HOST`/`DATABRICKS_TOKEN` **neste
  repositório** (`secrets: inherit` propaga os secrets do repositório chamador, não os
  do `mlops-platform`).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: call mlops-platform's shared deploy-bundle reusable workflow"
```

---

## Task 11: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Escrever o README**

```markdown
# monitoring-platform

Componente de monitoramento do ecossistema de MLOps no Databricks: `MonitoringConfig`
sobre feature tables e tabelas de predições, baseline resolvida automaticamente a
partir de `platform_audit.pipeline_runs`, Lakehouse Monitoring nativo como motor de
cálculo (risco aceito, não confirmado na Free Edition — ver o spec, seção 6), e
sugestão de retreino que nunca dispara nada sozinho.

Design completo em
[`docs/superpowers/specs/2026-08-23-monitoramento-design.md`](docs/superpowers/specs/2026-08-23-monitoramento-design.md).

## Como declarar um monitoramento

```python
from monitoring_platform.contract import MonitoringConfig, register_monitoring_config

config = MonitoringConfig(
    domain="credito",
    model_name="propensao_default",
    target_type="feature_table",  # ou "predictions"
    target_table="workspace.credito_features.customer_features",
    columns=["income", "credit_score"],
    threshold=0.2,
    schedule_cron="0 0 7 * * ?",
)
register_monitoring_config(config)
```

Este repositório é um framework puro — não é onde domínios reais declaram seus
monitoramentos. Instale este pacote no repositório do seu domínio
(`pip install git+https://github.com/ViniciusOtoni/monitoring-platform@vX.Y.Z`) e
declare o módulo lá. Convenção completa em
[`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform).

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
databricks bundle run drift_check_exemplo_propensao_exemplo_feature_table -t dev
```

## Como ler um alerta

Consulte `platform_monitoring.drift_metrics` filtrando por `status = 'DRIFT_DETECTED'`.
Uma linha ali **é** a sugestão de retreino — nada neste componente dispara o
`training-platform` automaticamente. Decida manualmente.

## Risco conhecido

Suporte do Lakehouse Monitoring na Databricks Free Edition não foi confirmado antes da
implementação — validar ao vivo (Task 9 do plano de implementação). Se não funcionar,
só o notebook `evaluate_drift.py` precisa de redesenho; o contrato e a tabela central
continuam válidos.

## Secrets necessários no GitHub Actions

| Secret | Descrição |
|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks Free Edition |
| `DATABRICKS_TOKEN` | Personal access token com permissão de deploy |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions and known risk"
```

---

## Task 12: Verificação ponta a ponta no workspace

Esta é a verificação que resolve, na prática, o risco central deste componente.

- [ ] **Step 1:** Confirmar que `workspace.exemplo_features.customer_transaction_features`
  (Componente 1) e `workspace.exemplo_predictions.propensao_exemplo` (Componente 3, já
  em `append` com `scored_at`) têm dados suficientes, e que há pelo menos um run
  `SUCCESS` de treino em `platform_audit.pipeline_runs` para `propensao_exemplo`
  (Componente 2).

- [ ] **Step 2:** Rodar `databricks bundle run drift_check_exemplo_propensao_exemplo_feature_table -t dev`.
  **Se o `client.quality_monitors.create(...)` falhar** com um erro indicando que o
  recurso não está disponível na Free Edition: documentar o erro exato no spec (seção
  6, substituindo "não confirmado" por "confirmado indisponível"), e abrir uma decisão
  de redesenho pontual do notebook (não do contrato) — por exemplo, cálculo de drift
  via PySpark/pandas puro, mantendo `MonitoringConfig` e `drift_metrics` inalterados.

- [ ] **Step 3: Se funcionar**, confirmar que `platform_monitoring.drift_metrics` foi
  criada com uma linha por coluna monitorada, e que `platform_audit.pipeline_runs` tem
  uma linha `SUCCESS` com `component="monitoring"`, `mode="drift_check"`.

- [ ] **Step 4:** Forçar um valor de drift acima do threshold (ajustando `threshold`
  para um valor bem baixo no `MonitoringConfig` de teste) e confirmar que a linha
  correspondente em `drift_metrics` aparece com `status="DRIFT_DETECTED"` — e que
  nenhuma execução do `training-platform` é disparada automaticamente.

---

## Self-Review

**1. Cobertura do spec:** contrato com registro por `(domain, model_name,
target_type)` (Task 2), resolução automática de baseline via auditoria (Task 3),
avaliação por threshold (Task 4), Lakehouse Monitoring nativo com risco documentado
(Task 9), tabela central `drift_metrics` (Task 6), filosofia de sugestão sem disparo
automático (documentada no README e na ausência de qualquer chamada ao
`training-platform` em todo o código), reuso exato do schema de auditoria (Task 7),
geração de um job por config com schedule próprio (Task 8). Todas as seções do spec
têm uma task ou notebook correspondente.

**2. Placeholders:** nenhum "TBD"/"TODO". A incerteza sobre a API do Lakehouse
Monitoring e o schema das suas tabelas de saída está documentada como risco com uma
implementação concreta + um passo de verificação explícito com plano de contingência
(Task 12, Step 2) — não é uma lacuna deixada em aberto.

**3. Consistência de tipos:** `MonitoringConfig`, `TrainingRun`, `DriftResult`,
`RunRecord`, e a chave composta `(domain, model_name, target_type)` foram conferidas
em todos os lugares que constroem ou consultam essas estruturas — contrato, baseline,
avaliação, tabela central, notebook e testes usam os mesmos nomes de campo.

---

Plano completo e salvo em
`docs/superpowers/plans/2026-08-23-monitoramento-implementation.md`.

Com este plano, os quatro componentes do ecossistema (`feature-platform`,
`training-platform`, `serving-platform`, `monitoring-platform`) têm spec e plano de
implementação escritos, revisados e commitados. A implementação de cada um é o
próximo passo — feita separadamente, na mesma ordem de dependência usada para o
design.
