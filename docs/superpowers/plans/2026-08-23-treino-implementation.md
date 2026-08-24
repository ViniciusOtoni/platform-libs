# Componente de Treino — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o framework `training_platform`: contrato `TrainingConfig`, split temporal, comparação de hiperparâmetros via holdout de validação, pipeline com transformações custom, pyfunc sobrescrevível, gate de sanidade, registro no Unity Catalog com `FeatureLookup` embarcado, e auditoria reaproveitando o schema do `feature-platform`.

**Architecture:** Lógica pura (contrato, split, seleção de melhor hiperparametrização, gate de sanidade, nomenclatura, construção do `Pipeline`) em `src/training_platform/`, testável localmente com `pytest`, sem Spark. Quatro notebooks Databricks sequenciais (`prepare_training_set` → `fit_and_compare_hyperparams` → `select_best_and_test` → `register_model`) fazem a orquestração real, passando estado entre tasks via `dbutils.jobs.taskValues` (resultados pequenos: run id do MLflow, datas, hiperparâmetros escolhidos) e via tabelas Delta de scratch (dados grandes: splits de treino/validação/teste). Um exemplo não-produtivo prova o fluxo ponta a ponta.

**Tech Stack:** Python 3.11, scikit-learn, MLflow (Feature Engineering client), PySpark + Delta (runtime Databricks serverless), Databricks Asset Bundles, pytest, GitHub Actions.

**Emenda (2026-08-23, durante o design da arquitetura de plataforma):** este
repositório passou a ser um framework puro — `dominios/exemplo/` foi renomeada para
`examples/` (mesmo papel de harness de integração, não domínio real). O contrato
`TrainingConfig` não muda (já tinha `domain` como campo explícito). O
`.github/workflows/deploy.yml` (Task 11) passou a ser um caller do reusable workflow
centralizado em `mlops-platform`. Ver
`docs/superpowers/specs/2026-08-23-treino-design.md`, seção 1.1.

---

## Scope Check

Este plano cobre só o Componente 2 (Treino), conforme
`docs/superpowers/specs/2026-08-23-treino-design.md`. Os Componentes 3 (Serving) e 4
(Monitoramento) têm specs e plans próprios, a escrever depois.

## File Structure

```
training-platform/
├── databricks.yml
├── pyproject.toml
├── pytest.ini
├── requirements-dev.txt
├── README.md
├── src/
│   └── training_platform/
│       ├── __init__.py
│       ├── contract.py          # FeatureLookupSpec, TrainingConfig, registro por model_name
│       ├── split.py             # cortes temporais a partir das safras distintas
│       ├── selection.py         # escolha da melhor combinação de hiperparâmetros
│       ├── quality.py           # Finding + gate de sanidade pré-registro
│       ├── naming.py            # convenção de nome do modelo no Unity Catalog
│       ├── pipeline.py          # monta sklearn.Pipeline (custom_transforms + estimador)
│       ├── pyfunc_model.py      # FeaturePlatformModel (default do pyfunc)
│       └── audit.py             # RunRecord/to_row/write_run (duplicado do feature-platform)
├── examples/
│   ├── __init__.py
│   └── training_configs.py     # TrainingConfig de exemplo (não-produtivo), prova o fluxo ponta a ponta
├── notebooks/
│   ├── prepare_training_set.py
│   ├── fit_and_compare_hyperparams.py
│   ├── select_best_and_test.py
│   └── register_model.py
├── resources/
│   └── training_pipeline.job.yml   # escrito à mão (job fixo, 4 tasks — não gerado dinamicamente)
├── tests/
│   ├── __init__.py
│   ├── test_contract.py
│   ├── test_split.py
│   ├── test_selection.py
│   ├── test_quality.py
│   ├── test_naming.py
│   ├── test_pipeline.py
│   ├── test_pyfunc_model.py
│   └── test_audit.py
└── .github/
    └── workflows/
        └── deploy.yml
```

Diferença chave em relação ao `feature-platform`: lá, o job era gerado dinamicamente
(1 task por feature table registrada) porque o número de feature tables varia. Aqui, o
job tem uma estrutura **fixa** de 4 tasks (etapas do pipeline de treino, não uma por
modelo) — o `model_name` a treinar é um **parâmetro de execução**, resolvido em runtime
via o registro em `contract.py`. Por isso `resources/training_pipeline.job.yml` é
escrito à mão, sem gerador.

---

## Task 1: Scaffolding do repositório

**Files:**
- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `src/training_platform/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "training-platform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "scikit-learn>=1.4",
    "mlflow>=3.15.0",
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
scikit-learn>=1.4
mlflow>=3.15.0
pytest>=8.0
pyspark>=3.5
delta-spark>=3.2
databricks-sdk>=0.30
databricks-feature-engineering>=0.7
```

- [ ] **Step 4: Criar `src/training_platform/__init__.py`** (vazio)

```python
```

- [ ] **Step 5: Criar `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.databricks/
mlruns/
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
git add pyproject.toml pytest.ini requirements-dev.txt src/training_platform/__init__.py .gitignore
git commit -m "chore: scaffold training-platform package"
```

---

## Task 2: Contrato e registro (`contract.py`)

**Files:**
- Create: `src/training_platform/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_contract.py
import pytest

from training_platform.contract import (
    FeatureLookupSpec,
    TrainingConfig,
    register_training_config,
    get_training_config,
    get_registry,
    clear_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _valid_config(**overrides):
    defaults = dict(
        domain="exemplo",
        model_name="modelo_exemplo",
        algorithm=object,
        hyperparameter_sets=[{}],
        feature_lookups=[
            FeatureLookupSpec(
                table_name="workspace.exemplo_features.customer_transaction_features",
                feature_names=["txn_count", "avg_ticket"],
                lookup_key="customer_id",
                timestamp_lookup_key="reference_date",
            )
        ],
        spine_table="workspace.exemplo.spine_train",
        label_column="label_default",
        reference_date_column="reference_date",
        train_pct=0.6,
        val_pct=0.2,
        test_pct=0.2,
        metric="roc_auc",
        metric_direction="maximize",
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_training_config_accepts_valid_split_percentages():
    config = _valid_config()
    assert config.train_pct == 0.6


def test_training_config_rejects_split_not_summing_to_one():
    with pytest.raises(ValueError, match="must equal 1.0"):
        _valid_config(train_pct=0.5, val_pct=0.3, test_pct=0.3)


def test_training_config_defaults_custom_transforms_and_pyfunc_class():
    config = _valid_config()
    assert config.custom_transforms == []
    assert config.pyfunc_model_class is None


def test_register_and_get_training_config():
    config = _valid_config()
    register_training_config(config)

    assert get_training_config("modelo_exemplo") is config
    assert get_registry() == {"modelo_exemplo": config}


def test_register_training_config_rejects_duplicate():
    register_training_config(_valid_config())

    with pytest.raises(ValueError, match="already registered"):
        register_training_config(_valid_config())
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.contract'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/contract.py
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass(frozen=True)
class FeatureLookupSpec:
    table_name: str
    feature_names: list[str]
    lookup_key: str
    timestamp_lookup_key: str


@dataclass
class TrainingConfig:
    domain: str
    model_name: str
    algorithm: type
    hyperparameter_sets: list[dict]
    feature_lookups: list[FeatureLookupSpec]
    spine_table: str
    label_column: str
    reference_date_column: str
    train_pct: float
    val_pct: float
    test_pct: float
    metric: str | Callable
    metric_direction: Literal["maximize", "minimize"]
    custom_transforms: list = field(default_factory=list)
    pyfunc_model_class: Optional[type] = None

    def __post_init__(self) -> None:
        total = self.train_pct + self.val_pct + self.test_pct
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train_pct + val_pct + test_pct must equal 1.0, got {total}"
            )


_REGISTRY: dict[str, TrainingConfig] = {}


def register_training_config(config: TrainingConfig) -> None:
    if config.model_name in _REGISTRY:
        raise ValueError(f"training config '{config.model_name}' already registered")
    _REGISTRY[config.model_name] = config


def get_training_config(model_name: str) -> TrainingConfig:
    return _REGISTRY[model_name]


def get_registry() -> dict[str, TrainingConfig]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/contract.py tests/test_contract.py
git commit -m "feat: add TrainingConfig contract and registry"
```

---

## Task 3: Split temporal (`split.py`)

**Files:**
- Create: `src/training_platform/split.py`
- Test: `tests/test_split.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_split.py
from datetime import date

import pytest

from training_platform.split import compute_split_dates, assign_split


def _dates(n):
    return [date(2026, 1, 1 + i) for i in range(n)]


def test_compute_split_dates_60_20_20_over_ten_dates():
    train_end, val_end = compute_split_dates(_dates(10), train_pct=0.6, val_pct=0.2, test_pct=0.2)
    assert train_end == date(2026, 1, 6)
    assert val_end == date(2026, 1, 8)


def test_compute_split_dates_rejects_empty_list():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_split_dates([], train_pct=0.6, val_pct=0.2, test_pct=0.2)


def test_compute_split_dates_deduplicates_input():
    dates_with_dupes = _dates(10) + [date(2026, 1, 1)]
    train_end, val_end = compute_split_dates(dates_with_dupes, train_pct=0.6, val_pct=0.2, test_pct=0.2)
    assert train_end == date(2026, 1, 6)
    assert val_end == date(2026, 1, 8)


def test_assign_split_train():
    assert assign_split(date(2026, 1, 3), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "train"


def test_assign_split_val():
    assert assign_split(date(2026, 1, 7), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "val"


def test_assign_split_test():
    assert assign_split(date(2026, 1, 9), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "test"


def test_assign_split_boundary_belongs_to_earlier_split():
    assert assign_split(date(2026, 1, 6), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "train"
    assert assign_split(date(2026, 1, 8), train_end=date(2026, 1, 6), val_end=date(2026, 1, 8)) == "val"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_split.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.split'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/split.py
from datetime import date


def compute_split_dates(
    distinct_dates: list[date], train_pct: float, val_pct: float, test_pct: float
) -> tuple[date, date]:
    """Retorna (train_end, val_end): os dois cortes cronológicos.
    train = data <= train_end; val = train_end < data <= val_end; test = data > val_end.
    Nunca quebra uma safra ao meio — os cortes caem sempre em cima de uma data real."""
    if not distinct_dates:
        raise ValueError("distinct_dates must not be empty")

    sorted_dates = sorted(set(distinct_dates))
    n = len(sorted_dates)

    train_end_idx = max(0, min(round(n * train_pct) - 1, n - 1))
    val_end_idx = max(train_end_idx, min(round(n * (train_pct + val_pct)) - 1, n - 1))

    return sorted_dates[train_end_idx], sorted_dates[val_end_idx]


def assign_split(reference_date: date, train_end: date, val_end: date) -> str:
    if reference_date <= train_end:
        return "train"
    if reference_date <= val_end:
        return "val"
    return "test"
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_split.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/split.py tests/test_split.py
git commit -m "feat: add temporal split computed over distinct reference dates"
```

---

## Task 4: Seleção da melhor hiperparametrização (`selection.py`)

**Files:**
- Create: `src/training_platform/selection.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_selection.py
import pytest

from training_platform.selection import select_best


def test_select_best_maximize_picks_highest_metric():
    results = [({"n_estimators": 10}, 0.7), ({"n_estimators": 50}, 0.9), ({"n_estimators": 100}, 0.85)]
    assert select_best(results, "maximize") == {"n_estimators": 50}


def test_select_best_minimize_picks_lowest_metric():
    results = [({"alpha": 0.1}, 12.0), ({"alpha": 1.0}, 8.5), ({"alpha": 10.0}, 15.0)]
    assert select_best(results, "minimize") == {"alpha": 1.0}


def test_select_best_rejects_empty_results():
    with pytest.raises(ValueError, match="must not be empty"):
        select_best([], "maximize")


def test_select_best_rejects_unknown_direction():
    with pytest.raises(ValueError, match="unknown metric_direction"):
        select_best([({"a": 1}, 0.5)], "sideways")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_selection.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.selection'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/selection.py
def select_best(results: list[tuple[dict, float]], metric_direction: str) -> dict:
    if not results:
        raise ValueError("results must not be empty")

    if metric_direction == "maximize":
        best = max(results, key=lambda r: r[1])
    elif metric_direction == "minimize":
        best = min(results, key=lambda r: r[1])
    else:
        raise ValueError(f"unknown metric_direction: {metric_direction}")

    return best[0]
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_selection.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/selection.py tests/test_selection.py
git commit -m "feat: add best-hyperparameter selection by metric direction"
```

---

## Task 5: Gate de sanidade (`quality.py`)

**Files:**
- Create: `src/training_platform/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_quality.py
import math

from training_platform.quality import (
    Finding,
    check_metric_is_finite,
    check_predictions_not_empty,
    run_sanity_gate,
    gate_passed,
)


def test_check_metric_is_finite_passes_on_valid_float():
    finding = check_metric_is_finite(0.87)
    assert finding.status == "PASS"


def test_check_metric_is_finite_fails_on_nan():
    finding = check_metric_is_finite(float("nan"))
    assert finding.status == "FAIL"


def test_check_metric_is_finite_fails_on_infinite():
    finding = check_metric_is_finite(math.inf)
    assert finding.status == "FAIL"


def test_check_predictions_not_empty_passes_when_positive():
    assert check_predictions_not_empty(100).status == "PASS"


def test_check_predictions_not_empty_fails_when_zero():
    assert check_predictions_not_empty(0).status == "FAIL"


def test_run_sanity_gate_returns_both_checks():
    findings = run_sanity_gate(0.87, num_predictions=100)
    assert {f.check for f in findings} == {"metric_is_finite", "predictions_not_empty"}


def test_gate_passed_true_when_all_pass():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "PASS")]) is True


def test_gate_passed_false_when_any_fails():
    assert gate_passed([Finding("a", "PASS"), Finding("b", "FAIL")]) is False
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.quality'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/quality.py
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    detail: str = ""


def check_metric_is_finite(metric_value: float) -> Finding:
    is_finite = isinstance(metric_value, (int, float)) and math.isfinite(metric_value)
    return Finding(
        check="metric_is_finite",
        status="PASS" if is_finite else "FAIL",
        detail=f"metric={metric_value}",
    )


def check_predictions_not_empty(num_predictions: int) -> Finding:
    return Finding(
        check="predictions_not_empty",
        status="PASS" if num_predictions > 0 else "FAIL",
        detail=f"num_predictions={num_predictions}",
    )


def run_sanity_gate(metric_value: float, num_predictions: int) -> list[Finding]:
    return [
        check_metric_is_finite(metric_value),
        check_predictions_not_empty(num_predictions),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/quality.py tests/test_quality.py
git commit -m "feat: add sanity gate before model registration"
```

---

## Task 6: Nomenclatura de modelo (`naming.py`)

**Files:**
- Create: `src/training_platform/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_naming.py
import pytest

from training_platform.naming import derive_model_name, validate_model_name


def test_derive_model_name_follows_convention():
    name = derive_model_name(catalog="workspace", domain="credito", model_name="propensao_default")
    assert name == "workspace.credito_models.propensao_default"


def test_validate_model_name_accepts_convention():
    validate_model_name("workspace.credito_models.propensao_default")


def test_validate_model_name_rejects_uppercase():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_model_name("Workspace.Credito.Modelo")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.naming'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/naming.py
import re

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def derive_model_name(catalog: str, domain: str, model_name: str) -> str:
    schema = f"{domain}_models"
    return f"{catalog}.{schema}.{model_name}"


def validate_model_name(full_name: str) -> None:
    if not _NAME_RE.match(full_name):
        raise ValueError(
            f"model name '{full_name}' does not match convention "
            "'<catalog>.<schema>.<model>' (lowercase letters, digits, underscore)"
        )
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/naming.py tests/test_naming.py
git commit -m "feat: add model naming convention derivation and validation"
```

---

## Task 7: Montagem do pipeline (`pipeline.py`)

**Files:**
- Create: `src/training_platform/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_pipeline.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier

from training_platform.pipeline import build_pipeline


class _AddOneTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + 1


def test_build_pipeline_puts_estimator_last():
    pipeline = build_pipeline([_AddOneTransform()], DummyClassifier(strategy="constant", constant=0))
    step_names = [name for name, _ in pipeline.steps]
    assert step_names[-1] == "model"
    assert step_names[0] == "custom_0"


def test_build_pipeline_with_no_custom_transforms():
    pipeline = build_pipeline([], DummyClassifier(strategy="constant", constant=0))
    assert [name for name, _ in pipeline.steps] == ["model"]


def test_build_pipeline_is_fittable_and_predictable():
    import pandas as pd

    pipeline = build_pipeline([_AddOneTransform()], DummyClassifier(strategy="constant", constant=1))
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 1, 1])
    pipeline.fit(X, y)
    assert list(pipeline.predict(X)) == [1, 1, 1]
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.pipeline'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/pipeline.py
from sklearn.pipeline import Pipeline


def build_pipeline(custom_transforms: list, estimator) -> Pipeline:
    steps = [(f"custom_{i}", t) for i, t in enumerate(custom_transforms)]
    steps.append(("model", estimator))
    return Pipeline(steps)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/pipeline.py tests/test_pipeline.py
git commit -m "feat: build sklearn Pipeline with custom transforms before the estimator"
```

---

## Task 8: Pyfunc default (`pyfunc_model.py`)

**Files:**
- Create: `src/training_platform/pyfunc_model.py`
- Test: `tests/test_pyfunc_model.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_pyfunc_model.py
import numpy as np

from training_platform.pyfunc_model import FeaturePlatformModel


class _FakeSklearnModel:
    def predict_proba(self, X):
        return np.array([[0.9, 0.1], [0.2, 0.8]])


def test_predict_returns_positive_class_probability():
    wrapped = FeaturePlatformModel(_FakeSklearnModel())
    result = wrapped.predict(context=None, model_input=None)
    assert list(result) == [0.1, 0.8]
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pyfunc_model.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.pyfunc_model'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/pyfunc_model.py
import mlflow.pyfunc


class FeaturePlatformModel(mlflow.pyfunc.PythonModel):
    """Default: retorna P(classe positiva) como coluna double, igual ao
    ProbabilityScorer validado na POC. Sobrescreva predict() para outro comportamento."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pyfunc_model.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training_platform/pyfunc_model.py tests/test_pyfunc_model.py
git commit -m "feat: add default overridable pyfunc wrapper"
```

---

## Task 9: Auditoria (`audit.py`)

Duplicado deliberadamente do `feature-platform` (decisão registrada no spec, seção 9) —
mesmo schema, mesma implementação.

**Correção aplicada preventivamente (2026-08-24):** a implementação do `feature-platform`
(mesmo módulo, código idêntico) precisou de uma correção descoberta rodando de verdade
contra o workspace: `saveAsTable` não cria o schema automaticamente no Unity Catalog —
a primeira escrita falha com `SCHEMA_NOT_FOUND`. Como é o mesmo bug, na mesma tabela
(`platform_audit.pipeline_runs`), o `write_run` abaixo já inclui o `CREATE SCHEMA IF
NOT EXISTS` desde o início, em vez de esperar descobrir o mesmo problema de novo.

**Files:**
- Create: `src/training_platform/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_audit.py
from datetime import date, datetime

from training_platform.audit import RunRecord, to_row, AUDIT_TABLE


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="training",
        entity_name="workspace.credito_models.propensao_default",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="train",
        status="SUCCESS",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        run_ts=datetime(2026, 8, 23, 3, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "training",
        "entity_name": "workspace.credito_models.propensao_default",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "train",
        "status": "SUCCESS",
        "window_start": date(2026, 1, 1),
        "window_end": date(2026, 6, 30),
        "run_ts": datetime(2026, 8, 23, 3, 0, 0),
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `ModuleNotFoundError: No module named 'training_platform.audit'`

- [ ] **Step 3: Implementar**

```python
# src/training_platform/audit.py
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
    """Requer SparkSession — exercitado via notebook (Task 10), não via pytest."""
    # saveAsTable não cria o schema automaticamente em Unity Catalog — sem isso,
    # a primeira escrita falha com SCHEMA_NOT_FOUND (achado no feature-platform).
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
git add src/training_platform/audit.py tests/test_audit.py
git commit -m "feat: duplicate audit record type and pipeline_runs writer from feature-platform"
```

---

## Task 10: Notebooks, domínio de exemplo e bundle DAB

Glue code + configuração — sem TDD, com passos de verificação concretos.

**Correções aplicadas preventivamente (2026-08-24), achadas no `feature-platform`
rodando de verdade contra um job deployado via DAB:**
1. **`sys.path`**: o cwd de um notebook rodando dentro de um job deployado
   (`.../files/notebooks`) não inclui a raiz do bundle (onde mora `examples/`) nem
   `src/` — `import examples.training_configs` quebraria com `ModuleNotFoundError` sem
   isso. Os 4 notebooks abaixo já começam com o mesmo bloco de `sys.path.insert` usado
   no `feature-platform`.
2. **`currentRunId()`**: levanta `Py4JSecurityException` em compute
   serverless/shared access mode (não está na whitelist do Py4J nesse modo). Os dois
   notebooks que capturam `run_id_job` (`select_best_and_test.py`,
   `register_model.py`) já usam `.currentRunId().get().toString()` com fallback para
   um `uuid.uuid4()` gerado localmente.

**Correção (2026-08-24, achada na revisão de qualidade da Task 10 — não numa execução
real, mas verificada contra a API do `databricks-feature-engineering`):**
`register_model.py` chamava `fe.log_model(..., training_set=None,
feature_lookups=feature_lookups, ...)` — `feature_lookups` não é um parâmetro
reconhecido de `log_model` (é descartado silenciosamente) e `training_set=None` sem
mais nada levantaria erro em runtime. Corrigido: `register_model.py` agora chama
`fe.create_training_set(...)` de novo (mesmo padrão do `prepare_training_set.py`,
lendo a spine) para obter um `TrainingSet` de verdade, e passa esse objeto como
`training_set=training_set` — é ele que carrega o FeatureSpec embarcado no artefato do
modelo.

**Contrato de `taskValues` entre as 4 tasks** (documentado aqui porque nenhum módulo
Python isolado o representa sozinho — é o "cimento" entre os notebooks):

| De | Chave | Tipo | Para |
|---|---|---|---|
| `prepare_training_set` | `mlflow_run_id` | str | `fit_and_compare_hyperparams`, `select_best_and_test`, `register_model` |
| `prepare_training_set` | `window_start` | str (ISO date, min `reference_date` da spine) | `register_model` |
| `prepare_training_set` | `window_end` | str (ISO date, max `reference_date` da spine) | `register_model` |
| `fit_and_compare_hyperparams` | `hyperparameter_results` | str (JSON: `[{"hyperparameters": dict, "metric": float}]`) | `select_best_and_test` |
| `select_best_and_test` | `best_hyperparameters` | str (JSON: dict) | `register_model` |

Dados grandes (splits de treino/validação/teste) trafegam via tabelas Delta de scratch:
`<catalog>.training_scratch.<model_name>_train` / `_val` / `_test`.

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/training_configs.py`
- Create: `notebooks/prepare_training_set.py`
- Create: `notebooks/fit_and_compare_hyperparams.py`
- Create: `notebooks/select_best_and_test.py`
- Create: `notebooks/register_model.py`
- Create: `databricks.yml`
- Create: `resources/training_pipeline.job.yml`

- [ ] **Step 1: Criar o exemplo não-produtivo**

```python
# examples/__init__.py
```

```python
# examples/training_configs.py
from sklearn.ensemble import RandomForestClassifier

from training_platform.contract import FeatureLookupSpec, TrainingConfig, register_training_config

config = TrainingConfig(
    domain="exemplo",
    model_name="propensao_exemplo",
    algorithm=RandomForestClassifier,
    hyperparameter_sets=[
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 8},
    ],
    feature_lookups=[
        FeatureLookupSpec(
            table_name="workspace.exemplo_features.customer_transaction_features",
            feature_names=["txn_count", "avg_ticket"],
            lookup_key="customer_id",
            timestamp_lookup_key="reference_date",
        )
    ],
    spine_table="workspace.exemplo.spine_train",
    label_column="label_default",
    reference_date_column="reference_date",
    train_pct=0.6,
    val_pct=0.2,
    test_pct=0.2,
    metric="roc_auc",
    metric_direction="maximize",
)

register_training_config(config)
```

- [ ] **Step 2: Criar `notebooks/prepare_training_set.py`**

> **Correção (achado ao vivo, Task 13 — pipeline completo rodando pela primeira vez):**
> `create_training_set(..., exclude_columns=[config.reference_date_column])` remove
> `reference_date` do `master` retornado por `load_df()` — mas o cálculo do split (logo
> abaixo, via `master_with_split = master.withColumn("_split", F.when(F.col(config.reference_date_column) <= ...))`)
> precisa dessa coluna, e falhava com
> `UNRESOLVED_COLUMN.WITH_SUGGESTION: reference_date cannot be resolved`. Corrigido
> removendo `exclude_columns` da chamada (mantendo `reference_date` em `master` até
> depois do split) e adicionando `config.reference_date_column` ao `.drop(...)` que já
> remove `_split` antes de escrever as tabelas de split — preserva a intenção original
> (a coluna não deve vazar como feature nas tabelas `train`/`val`/`test`), só adiando a
> remoção para depois de ela ser usada.

> **Correção (achado ao vivo, Task 13 — mesma rodada):**
> `mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")`
> falhava com `RestException: NOT_FOUND: Parent directory does not exist:
> /Shared/training-platform/exemplo` — diferente de uma tabela UC, um experimento MLflow
> em workspace path não cria os diretórios pai automaticamente. Corrigido adicionando
> `WorkspaceClient().workspace.mkdirs(f"/Shared/training-platform/{config.domain}")`
> (idempotente — `mkdirs` não falha se o diretório já existir) logo antes do
> `mlflow.set_experiment(...)`.

> **Correção (achado ao vivo, Task 13 — mesma rodada):** as três notebooks que leem as
> tabelas de split (`fit_and_compare_hyperparams.py`, `select_best_and_test.py`,
> `register_model.py`) calculavam `feature_cols` como "toda coluna exceto o label" —
> isso inclui as colunas de `lookup_key` (ex.: `customer_id`), que são chaves de junção,
> não features. `fit_and_compare_hyperparams` falhava com
> `ValueError: could not convert string to float: 'c1'` ao tentar treinar com
> `customer_id` (string) como feature numérica. Corrigido excluindo também os
> `lookup_key` de todos os `config.feature_lookups`:
> `feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]`
> nas três notebooks.

> **Correção (achado ao vivo, Task 13 — verificação de nested runs):** os `combo_i`
> criados por `fit_and_compare_hyperparams.py` via `mlflow.start_run(run_name=f"combo_{i}", nested=True)`
> tinham o `mlflow.parentRunId` correto, mas apareciam no experimento default do
> notebook (`.../files/notebooks/fit_and_compare_hyperparams`), não no experimento
> compartilhado `/Shared/training-platform/<domain>/<model_name>` — cada task de um job
> roda num processo Python separado, então o `mlflow.set_experiment(...)` chamado em
> `prepare_training_set.py` não vale para as tasks seguintes; sem chamar de novo,
> `start_run(nested=True)` cria a run nova no experimento default do notebook atual.
> Corrigido chamando `mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")`
> no início de `fit_and_compare_hyperparams.py`, antes do `with mlflow.start_run(run_id=mlflow_run_id):`
> (o diretório pai já existe a essa altura — foi criado por `prepare_training_set`, task
> anterior no DAG). `select_best_and_test.py` e `register_model.py` não precisam do mesmo
> fix porque só reabrem o run existente via `run_id=mlflow_run_id` para logar
> métricas/tags — não criam runs novos, então não dependem do experimento ativo.

> **Correção (achado ao vivo, Task 13 — Step 4, forçando falha do gate):** com uma
> `TrainingConfig` de `test_pct=0.0` (tabela de teste vazia), `select_best_and_test.py`
> quebrava antes mesmo de chegar no gate: `scorer(pipeline, X_test, y_test)` chama
> `pipeline.predict(X_test)` internamente, e o `RandomForestClassifier` do scikit-learn
> levanta `ValueError: Found array with 0 sample(s) ... while a minimum of 1 is
> required` — um crash não tratado, não uma falha graciosa do gate. Isso quebra o
> propósito do gate (ser a rede de segurança que resulta num registro de auditoria
> `FAILED` limpo, não numa exceção não tratada). Corrigido pulando a chamada ao `scorer`
> quando `X_test` está vazio, deixando `test_metric = float("nan")` — o que também faz
> `check_metric_is_finite` falhar, reforçando o `FAIL` do gate por dois motivos
> independentes:
> ```python
> if len(X_test) > 0:
>     test_metric = float(scorer(pipeline, X_test, y_test))
> else:
>     test_metric = float("nan")
> ```
> substitui a linha única `test_metric = float(scorer(pipeline, X_test, y_test))`.

> **Correção (achado na revisão final de branch, não ao vivo — mesma raiz do fix
> anterior):** `fit_and_compare_hyperparams.py` tinha o mesmo risco de crash não tratado
> — `metric_value = float(scorer(pipeline, X_val, y_val))` sem guarda. Como
> `compute_split_dates` usa arredondamento sobre o número de datas distintas, um
> `spine_table` com poucas datas distintas pode produzir um bucket de `val` vazio (não
> só `test`) — ex.: com `n=1` data distinta, `train_end == val_end` e o bucket de `val`
> fica vazio. Diferente do bug em `select_best_and_test.py`, este roda **antes** de
> qualquer escrita de auditoria no pipeline (só `select_best_and_test.py` e
> `register_model.py` chamam `write_run`) — um crash aqui derruba o job sem nenhuma
> linha em `platform_audit.pipeline_runs`, quebrando a garantia de trilha de auditoria
> que é o propósito do gate. Corrigido com a mesma guarda:
> ```python
> if len(X_val) > 0:
>     metric_value = float(scorer(pipeline, X_val, y_val))
> else:
>     metric_value = float("nan")
> ```
> substitui a linha única `metric_value = float(scorer(pipeline, X_val, y_val))`. O
> `NaN` resultante flui como métrica normal até `select_best` (que não quebra em
> comparações com `NaN`, só se comporta de forma não-determinística ao escolher o
> "melhor") e, na pior hipótese, até o gate de sanidade em `select_best_and_test.py`, que
> tem a garantia final de capturar e gravar `FAILED` — preservado o princípio de que o
> pipeline nunca crasha sem deixar rastro de auditoria.

> **Correção (achado na revisão final de branch):** `register_model.py` tinha dois
> gaps de qualidade:
> 1. `validate_model_name` (Task 6, `naming.py`) era definida e testada mas nunca
>    chamada em nenhum lugar do pipeline real — `domain`/`model_name` são texto livre
>    fornecido por quem escreve o `TrainingConfig`, fora do controle deste framework;
>    sem a validação, um valor inválido (maiúscula, hífen) só apareceria tarde, como um
>    erro opaco do Unity Catalog. Corrigido chamando `validate_model_name(full_model_name)`
>    logo após `derive_model_name(...)`.
> 2. O `exclude_columns=[config.reference_date_column]` neste `create_training_set` era
>    um resquício do mesmo padrão corrigido em `prepare_training_set.py` (Correção
>    acima) — mas aqui é inofensivo (e por isso não achado pela Task 13 ao vivo):
>    diferente de `prepare_training_set.py`, este `TrainingSet` nunca passa por
>    `load_df()`, só alimenta `fe.log_model(training_set=...)` para embarcar o
>    FeatureSpec no artefato — `exclude_columns` só afeta o DataFrame de `load_df()`,
>    então era um parâmetro morto, visualmente idêntico ao padrão que causou um bug
>    real em outro lugar. Removido para não confundir leitura futura.
>
> O comentário sobre `CREATE SCHEMA IF NOT EXISTS` também foi atualizado: a Task 13
> confirmou ao vivo que essa linha é necessária (o primeiro registro bem-sucedido
> precisou criar `exemplo_models` do zero), então o texto "inferência ainda não
> validada" ficou desatualizado e foi trocado por uma confirmação.

> **Correção (achado ao vivo, Task 13):** o compute serverless da Free Edition não vem
> com `databricks-feature-engineering` pré-instalado — `requirements-dev.txt` só afeta
> o `.venv` local, não o runtime remoto. O job falhou com
> `ModuleNotFoundError: No module named 'databricks.feature_engineering'`. O schema do
> DAB (`environments:`/`environment_key`) documenta esse mecanismo como relevante para
> tasks Python script/wheel/dbt — não fica claro que se aplique a notebook tasks. O
> caminho padrão e bem documentado para notebook tasks é `%pip install` na primeira
> célula, seguido de `dbutils.library.restartPython()`. Aplicado abaixo (e no
> `register_model.py`, que também importa `databricks.feature_engineering`).

```python
# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — nem a raiz
# do bundle (onde mora `examples/`) nem `src/` (onde mora `training_platform`) estão
# no sys.path por padrão.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.training_configs  # noqa: F401
import pyspark.sql.functions as F
import mlflow

from training_platform.contract import get_training_config
from training_platform.split import compute_split_dates

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_training_config(model_name)

# COMMAND ----------
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

spine = spark.table(config.spine_table)

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
training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups,
    label=config.label_column,
)
master = training_set.load_df()

# COMMAND ----------
distinct_dates = [
    row[config.reference_date_column]
    for row in spine.select(config.reference_date_column).distinct().collect()
]
train_end, val_end = compute_split_dates(distinct_dates, config.train_pct, config.val_pct, config.test_pct)
window_start = min(distinct_dates)
window_end = max(distinct_dates)

master_with_split = master.withColumn(
    "_split",
    F.when(F.col(config.reference_date_column) <= F.lit(train_end), "train")
    .when(F.col(config.reference_date_column) <= F.lit(val_end), "val")
    .otherwise("test"),
)

scratch_schema = f"{catalog}.training_scratch"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {scratch_schema}")
scratch_prefix = f"{scratch_schema}.{model_name}"
for split_name in ["train", "val", "test"]:
    (
        master_with_split.filter(F.col("_split") == split_name)
        .drop("_split", config.reference_date_column)
        .write.format("delta")
        .mode("overwrite")
        .saveAsTable(f"{scratch_prefix}_{split_name}")
    )

# COMMAND ----------
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs(f"/Shared/training-platform/{config.domain}")
mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")
run = mlflow.start_run()
mlflow.log_params({"train_pct": config.train_pct, "val_pct": config.val_pct, "test_pct": config.test_pct})
mlflow.end_run()

dbutils.jobs.taskValues.set("mlflow_run_id", run.info.run_id)
dbutils.jobs.taskValues.set("window_start", window_start.isoformat())
dbutils.jobs.taskValues.set("window_end", window_end.isoformat())
```

- [ ] **Step 3: Criar `notebooks/fit_and_compare_hyperparams.py`**

```python
# Databricks notebook source
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")

# COMMAND ----------
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.training_configs  # noqa: F401
import json
import mlflow
from sklearn.metrics import get_scorer

from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline

# COMMAND ----------
model_name = dbutils.widgets.get("model_name")
catalog = dbutils.widgets.get("catalog")
config = get_training_config(model_name)

mlflow_run_id = dbutils.jobs.taskValues.get(taskKey="prepare_training_set", key="mlflow_run_id")
mlflow.set_experiment(f"/Shared/training-platform/{config.domain}/{config.model_name}")

# COMMAND ----------
scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
val_df = spark.table(f"{scratch_prefix}_val").toPandas()

feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]
X_val, y_val = val_df[feature_cols], val_df[config.label_column]

scorer = get_scorer(config.metric) if isinstance(config.metric, str) else config.metric
metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"

# COMMAND ----------
results = []
with mlflow.start_run(run_id=mlflow_run_id):
    for i, hyperparams in enumerate(config.hyperparameter_sets):
        with mlflow.start_run(run_name=f"combo_{i}", nested=True):
            estimator = config.algorithm(**hyperparams)
            pipeline = build_pipeline(config.custom_transforms, estimator)
            pipeline.fit(X_train, y_train)
            if len(X_val) > 0:
                metric_value = float(scorer(pipeline, X_val, y_val))
            else:
                metric_value = float("nan")
            mlflow.log_params(hyperparams)
            mlflow.log_metric(metric_name, metric_value)
            results.append({"hyperparameters": hyperparams, "metric": metric_value})

dbutils.jobs.taskValues.set("hyperparameter_results", json.dumps(results))
```

- [ ] **Step 4: Criar `notebooks/select_best_and_test.py`**

```python
# Databricks notebook source
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
from sklearn.metrics import get_scorer

from training_platform.contract import get_training_config
from training_platform.pipeline import build_pipeline
from training_platform.selection import select_best
from training_platform.quality import run_sanity_gate, gate_passed
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
results = json.loads(
    dbutils.jobs.taskValues.get(taskKey="fit_and_compare_hyperparams", key="hyperparameter_results")
)
# currentRunId() levanta Py4JSecurityException em compute serverless/shared access
# mode — cai para um id gerado localmente quando o contexto de job não expõe o run id.
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
best_hyperparameters = select_best(
    [(r["hyperparameters"], r["metric"]) for r in results], config.metric_direction
)

scratch_prefix = f"{catalog}.training_scratch.{model_name}"
train_df = spark.table(f"{scratch_prefix}_train").toPandas()
test_df = spark.table(f"{scratch_prefix}_test").toPandas()

feature_cols = [c for c in train_df.columns if c not in {config.label_column, *[fl.lookup_key for fl in config.feature_lookups]}]
X_train, y_train = train_df[feature_cols], train_df[config.label_column]
X_test, y_test = test_df[feature_cols], test_df[config.label_column]

scorer = get_scorer(config.metric) if isinstance(config.metric, str) else config.metric
metric_name = config.metric if isinstance(config.metric, str) else "custom_metric"

estimator = config.algorithm(**best_hyperparameters)
pipeline = build_pipeline(config.custom_transforms, estimator)
pipeline.fit(X_train, y_train)
if len(X_test) > 0:
    test_metric = float(scorer(pipeline, X_test, y_test))
else:
    test_metric = float("nan")

findings = run_sanity_gate(test_metric, num_predictions=len(X_test))
passed = gate_passed(findings)

# COMMAND ----------
with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.log_params({f"best__{k}": v for k, v in best_hyperparameters.items()})
    mlflow.log_metric(f"test_{metric_name}", test_metric)

if not passed:
    write_run(
        spark,
        RunRecord(
            component="training",
            entity_name=model_name,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id_job,
            mode="train",
            status="FAILED",
            window_start=date.fromisoformat(window_start),
            window_end=date.fromisoformat(window_end),
            run_ts=datetime.utcnow(),
        ),
    )
    failed_checks = [f.check for f in findings if f.status == "FAIL"]
    raise ValueError(f"sanity gate failed: {failed_checks}")

dbutils.jobs.taskValues.set("best_hyperparameters", json.dumps(best_hyperparameters))
```

- [ ] **Step 5: Criar `notebooks/register_model.py`**

> **Correção (mesma raiz do Step 2):** `databricks.feature_engineering` também é
> importado aqui — precisa do mesmo bootstrap `%pip install` + restart.

```python
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

with mlflow.start_run(run_id=mlflow_run_id):
    fe.log_model(
        model=wrapped_model,
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=full_model_name,
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
```

- [ ] **Step 6: Criar `databricks.yml`**

```yaml
bundle:
  name: training-platform

include:
  - resources/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for models, scratch tables, and the audit table.
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

- [ ] **Step 7: Criar `resources/training_pipeline.job.yml`** (escrito à mão — estrutura fixa, não gerada)

**Correção aplicada preventivamente (2026-08-24):** o CLI instalado exige extensão em
referência de notebook local (achado no `feature-platform`) — os 4 `notebook_path`
abaixo já incluem `.py`.

```yaml
resources:
  jobs:
    training_pipeline:
      name: training_pipeline
      parameters:
        - name: model_name
          default: ""
        - name: catalog
          default: "${var.catalog}"
        - name: git_commit
          default: "${var.git_commit}"
        - name: git_branch
          default: "${var.git_branch}"
      tasks:
        - task_key: prepare_training_set
          notebook_task:
            notebook_path: ../notebooks/prepare_training_set.py
            base_parameters:
              model_name: "{{job.parameters.model_name}}"
              catalog: "{{job.parameters.catalog}}"
        - task_key: fit_and_compare_hyperparams
          depends_on:
            - task_key: prepare_training_set
          notebook_task:
            notebook_path: ../notebooks/fit_and_compare_hyperparams.py
            base_parameters:
              model_name: "{{job.parameters.model_name}}"
              catalog: "{{job.parameters.catalog}}"
        - task_key: select_best_and_test
          depends_on:
            - task_key: fit_and_compare_hyperparams
          notebook_task:
            notebook_path: ../notebooks/select_best_and_test.py
            base_parameters:
              model_name: "{{job.parameters.model_name}}"
              catalog: "{{job.parameters.catalog}}"
              git_commit: "{{job.parameters.git_commit}}"
              git_branch: "{{job.parameters.git_branch}}"
        - task_key: register_model
          depends_on:
            - task_key: select_best_and_test
          notebook_task:
            notebook_path: ../notebooks/register_model.py
            base_parameters:
              model_name: "{{job.parameters.model_name}}"
              catalog: "{{job.parameters.catalog}}"
              git_commit: "{{job.parameters.git_commit}}"
              git_branch: "{{job.parameters.git_branch}}"
```

- [ ] **Step 8: Validar o bundle**

Run:
```powershell
databricks bundle validate -t dev
```
Expected: validação sem erros de estrutura (o job referencia `model_name` via
parâmetro; a ausência de dados reais não é um erro de `validate`, que não executa
notebooks).

- [ ] **Step 9: Commit**

```bash
git add examples/ notebooks/ databricks.yml resources/
git commit -m "feat: add non-productive example, 4-stage notebooks, and DAB bundle root"
```

---

## Task 11: GitHub Actions — caller do reusable workflow (`mlops-platform`)

**Emenda (arquitetura de plataforma):** este repositório não tem
`scripts/generate_resources.py` (o job de treino é fixo, não gerado dinamicamente —
ver spec, seção 7), então o reusable workflow simplesmente pula a etapa de geração de
resources, sem precisar de configuração extra aqui.

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

- [ ] **Step 2: Confirmar manualmente no GitHub** (`Settings > Secrets and variables >
  Actions`) que `DATABRICKS_HOST` e `DATABRICKS_TOKEN` estão configurados **neste
  repositório** — `secrets: inherit` propaga os secrets do repositório chamador, não
  os do `mlops-platform`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: call mlops-platform's shared deploy-bundle reusable workflow"
```

---

## Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Escrever o README**

```markdown
# training-platform

Framework de treino do ecossistema de MLOps no Databricks: contrato `TrainingConfig`
declarativo, split temporal, comparação de hiperparâmetros via holdout de validação,
transformações de negócio custom, pyfunc sobrescrevível, gate de sanidade e registro no
Unity Catalog com `FeatureLookup` embarcado.

Design completo em
[`docs/superpowers/specs/2026-08-23-treino-design.md`](docs/superpowers/specs/2026-08-23-treino-design.md).

## Como declarar um treino

```python
from training_platform.contract import FeatureLookupSpec, TrainingConfig, register_training_config
from sklearn.ensemble import RandomForestClassifier

config = TrainingConfig(
    domain="credito",
    model_name="propensao_default",
    algorithm=RandomForestClassifier,
    hyperparameter_sets=[{"n_estimators": 100}, {"n_estimators": 200}],
    feature_lookups=[FeatureLookupSpec(...)],
    spine_table="workspace.credito.spine_train",
    label_column="label_default",
    reference_date_column="reference_date",
    train_pct=0.6, val_pct=0.2, test_pct=0.2,
    metric="roc_auc",
    metric_direction="maximize",
)
register_training_config(config)
```

Este repositório é um framework puro — não é onde domínios reais declaram seus
treinos. Instale este pacote no repositório do seu domínio
(`pip install git+https://github.com/ViniciusOtoni/training-platform@vX.Y.Z`) e
declare o módulo lá. O modelo é registrado como `<catalog>.<domain>_models.<model_name>`.
Convenção completa em [`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform).

## Local (sem Spark)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

## No Databricks (Free Edition)

```powershell
databricks bundle validate -t dev
databricks bundle deploy   -t dev
databricks bundle run training_pipeline -t dev --params model_name=propensao_exemplo
```

Promoção de alias (`champion`/`challenger`) é sempre manual — revise a métrica de teste
no MLflow/Unity Catalog antes de mover o alias.

## Secrets necessários no GitHub Actions

| Secret | Descrição |
|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks Free Edition |
| `DATABRICKS_TOKEN` | Personal access token com permissão de deploy |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Task 13: Verificação ponta a ponta no workspace

**Concluída em 2026-08-24, ao vivo contra o workspace Free Edition.** Cinco bugs reais
foram encontrados e corrigidos ao longo da verificação (ver notas "Correção" acima,
todas datadas 2026-08-24): bootstrap `%pip install databricks-feature-engineering` +
`CREATE SCHEMA` para `training_scratch`; `create_training_set` excluindo
`reference_date` cedo demais; `mlflow.set_experiment` não criando o diretório pai;
`lookup_key` vazando como feature não-numérica; e o `scorer` do sklearn crashando (em
vez do gate falhar gracefully) com `X_test` vazio. Cada um foi corrigido no código, no
plano e revalidado com uma nova execução real antes de passar para o próximo.

- [x] **Step 1:** Confirmar que `workspace.exemplo.spine_train` existe (colunas
  `customer_id`, `reference_date`, `label_default`) e que a feature table
  `workspace.exemplo_features.customer_transaction_features` do Componente 1 já foi
  populada via backfill. Se não existir, rodar o backfill do `feature-platform`
  primeiro (dependência real entre os dois componentes).

  Confirmado: ambas já existiam no workspace (criadas durante a verificação do
  Componente 1). A feature table também precisou de uma correção retroativa no
  `feature-platform` (PRIMARY KEY ausente — ver o plano daquele repositório) antes do
  `FeatureLookup` funcionar.

- [x] **Step 2:** Rodar o pipeline de treino:
```powershell
databricks bundle run training_pipeline -t dev --params model_name=propensao_exemplo
```
Expected: as 4 tasks terminam `SUCCESS`; um modelo novo aparece em
`workspace.exemplo_models.propensao_exemplo` no Unity Catalog; uma linha `SUCCESS` em
`workspace.platform_audit.pipeline_runs` com `component="training"`.

  Confirmado após as correções acima: `prepare_training_set` → `fit_and_compare_hyperparams`
  → `select_best_and_test` → `register_model`, todas `SUCCESS`. Duas versões do modelo
  registradas em `workspace.exemplo_models.propensao_exemplo` (v1 e v2, de execuções
  sucessivas). Linhas `SUCCESS` confirmadas em `platform_audit.pipeline_runs` via query
  SQL direta.

- [x] **Step 3:** Confirmar no MLflow que o experimento
  `/Shared/training-platform/exemplo/propensao_exemplo` tem um run pai com um run
  aninhado por combinação de hiperparâmetros, e que o run pai tem a métrica de teste
  logada.

  Confirmado via `databricks experiments search-runs`: run pai `1105a68f...` com
  `test_roc_auc=0.446` e dois filhos aninhados (`combo_0`, `combo_1`, `mlflow.parentRunId`
  correto, métrica `roc_auc` de cada combinação). Exigiu a correção do `mlflow.set_experiment`
  ausente em `fit_and_compare_hyperparams.py` (sem isso, os runs aninhados apareciam no
  experimento default do notebook, não no experimento compartilhado do domínio/modelo).

- [x] **Step 4:** Forçar uma falha do gate de sanidade (ex.: um `metric` inválido que
  produza `NaN`) e confirmar que **nenhuma versão nova é registrada**, e que a
  auditoria grava `status=FAILED`.

  Forçado via uma `TrainingConfig` temporária com `test_pct=0.0` (tabela de teste
  vazia) — revertida depois da verificação, não faz parte do exemplo permanente.
  Confirmado: `select_best_and_test` falha com
  `ValueError: sanity gate failed: ['metric_is_finite', 'predictions_not_empty']`;
  `register_model` nunca roda (`UPSTREAM_FAILED`); `workspace.exemplo_models.propensao_exemplo_gate_fail`
  não existe; linha `FAILED` gravada em `platform_audit.pipeline_runs`. Exigiu a
  correção do Step 4 acima (o `scorer` do sklearn crashava antes do gate rodar).

- [x] **Step 5:** Confirmar que nenhum alias (`champion`/`challenger`) foi movido
  automaticamente — a versão nova aparece registrada, sem alias, esperando promoção
  manual.

  Confirmado via `databricks model-versions list`: as duas versões registradas (v1, v2)
  não têm nenhum alias atribuído.

---

## Self-Review

**1. Cobertura do spec:** contrato (Task 2), split temporal (Task 3), comparação de
hiperparâmetros via holdout (Tasks 4, notebook 2), pipeline com transforms custom (Task
7), pyfunc sobrescrevível (Task 8), gate de sanidade (Task 5, aplicado no notebook 3),
convenção de nome de modelo (Task 6), promoção manual de alias (documentada no README e
na verificação, sem automação — não há código que mova alias, por design), reuso exato
do schema de auditoria (Task 9). Todas as seções do spec têm uma task ou notebook
correspondente.

**2. Placeholders:** nenhum "TBD"/"TODO". O contrato de `taskValues` entre notebooks,
que o spec não detalhava, foi resolvido com uma decisão concreta e documentada no
início da Task 10 — não ficou como suposição silenciosa.

**3. Consistência de tipos:** `TrainingConfig`, `FeatureLookupSpec`, `RunRecord` e as
chaves de `taskValues` (`mlflow_run_id`, `window_start`, `window_end`,
`hyperparameter_results`, `best_hyperparameters`) foram conferidas em todos os
notebooks que as escrevem e leem — mesmos nomes, mesma ordem de produção/consumo entre
`prepare_training_set` → `fit_and_compare_hyperparams` → `select_best_and_test` →
`register_model`.

---

Plano completo e salvo em `docs/superpowers/plans/2026-08-23-treino-implementation.md`.
