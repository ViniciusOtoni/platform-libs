# Componente de Serving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o framework `serving_platform`: contrato `ServingConfig` cobrindo as trilhas online (Model Serving endpoint) e batch (`fe.score_batch` em task única), geração dinâmica dos recursos DAB, gate de qualidade nas predições, comando manual de atualização de endpoint, e auditoria reaproveitando o schema dos Componentes 1 e 2.

**Architecture:** Lógica pura (contrato, nomenclatura, gate de qualidade, geração de resources) em `src/serving_platform/`, testável com `pytest` local, sem Spark. Dois notebooks Databricks fazem a parte real: `score_batch.py` (a única task da trilha batch — `fe.score_batch`, gate, escrita, auditoria) e `refresh_endpoint.py` (atualização manual do endpoint online via Databricks SDK). Um domínio de exemplo em modo batch prova o fluxo ponta a ponta sem custo de endpoint online sempre ligado.

**Tech Stack:** Python 3.11, `databricks-feature-engineering`, `databricks-sdk`, PySpark + Delta (runtime Databricks serverless), Databricks Asset Bundles, pytest, GitHub Actions.

**Emenda (2026-08-23, durante o design do Componente 4):** a tabela de predições da
trilha batch passou de `overwrite` para `append`, com uma nova coluna `scored_at`,
porque o design do Componente 4 (Monitoramento) descobriu que monitorar drift de
predições exige histórico — um snapshot sobrescrito a cada execução não deixa o que
comparar. Reflexo já aplicado na Task 7 abaixo; ver
`docs/superpowers/specs/2026-08-23-serving-design.md`, seção 1.1.

**Emenda (2026-08-23, durante o design da arquitetura de plataforma):** este
repositório passou a ser um framework puro — `dominios/exemplo/` foi renomeada para
`examples/` (mesmo papel de harness de integração, não domínio real). O contrato
`ServingConfig` não muda (`domain` já era um campo explícito). O
`.github/workflows/deploy.yml` (Task 8) passou a ser um caller do reusable workflow
centralizado em `mlops-platform`. Ver spec, seção 1.2.

**Correção preventiva (2026-08-24, por analogia com bugs já confirmados ao vivo no
`feature-platform` e no `training-platform`, aplicada antes de qualquer implementação
deste componente):** cinco classes de bug já encontradas e corrigidas nos dois
componentes anteriores se aplicam aqui pela mesma causa raiz (compute serverless da
Free Edition, notebooks deployados via DAB, Unity Catalog não criando schema
automaticamente):
1. **Extensão `.py` em `notebook_path`** — o CLI instalado rejeita referências sem
   extensão. `resource_gen.py` (Task 6) usa `"../notebooks/score_batch.py"` e
   `"../notebooks/refresh_endpoint.py"`, não os caminhos sem extensão do rascunho
   original do design.
2. **Bootstrap de `sys.path`** — o cwd de um notebook deployado via DAB
   (`.../files/notebooks`) não inclui a raiz do bundle (onde mora `examples/`) nem
   `src/` por padrão. Ambos os notebooks (Task 7) inserem os dois no `sys.path` antes
   de importar `examples.serving_configs`/`serving_platform`.
3. **`currentRunId()` sem `.get()` e sem fallback** — levanta `Py4JSecurityException`
   em compute serverless/shared access mode. `score_batch.py` usa
   `.currentRunId().get().toString()` num `try`, com fallback para `uuid.uuid4()`.
   `refresh_endpoint.py` não precisa disso — não usa `run_id` (não escreve auditoria).
4. **`%pip install databricks-feature-engineering` + `restartPython()`** — a lib não
   vem pré-instalada no compute serverless da Free Edition. `score_batch.py` importa
   `databricks.feature_engineering`, então precisa do bootstrap. `refresh_endpoint.py`
   só usa `databricks.sdk`, que já vem disponível por padrão (confirmado
   empiricamente no `training-platform` — `WorkspaceClient` funcionou sem bootstrap
   nenhum) — não precisa do `%pip install`.
5. **`CREATE SCHEMA IF NOT EXISTS` antes do primeiro `saveAsTable`** — Unity Catalog
   não cria o schema sozinho. Aplicado em dois lugares: `audit.py`'s `write_run` (Task
   5, mesmo padrão exato de `feature-platform`/`training-platform` — `CREATE SCHEMA IF
   NOT EXISTS platform_audit`) e `score_batch.py`, antes de escrever
   `<catalog>.<domain>_predictions.<model_name>` pela primeira vez (schema novo, nunca
   criado por nenhum componente anterior).

Nenhum desses foi validado ao vivo neste componente ainda — são inferências por
analogia, aplicadas preventivamente para não redescobrir os mesmos bugs já
confirmados duas vezes. A Task 9 (verificação ponta a ponta) confirma se bastam.

**Correção (2026-08-24, achada ao vivo na Task 9 — bug novo, não uma repetição dos
anteriores):** `fe.score_batch()` chama internamente `mlflow.pyfunc.spark_udf` para
rodar a inferência distribuída — e essa chamada resolve para o `mlflow` **pré-instalado
no runtime base** (`/databricks/python/lib/.../mlflow`), não para uma versão mais nova
instalada via `%pip install databricks-feature-engineering` (que só força a instalação
da própria lib de FE, sem forçar upgrade de uma dependência transitiva já satisfeita no
ambiente). O `mlflow` pré-instalado nesta imagem do Free Edition não sabe interpretar o
formato de string de versão do runtime serverless (`'18.x-aarch64-photon-scala2'`),
levantando `InvalidVersion` ao tentar comparar com `Version("15.4")`. Bug real do
`mlflow` (não deste componente), corrigido a partir da versão 3.15.0 (PR
[#24336](https://github.com/mlflow/mlflow/pull/24336)). Corrigido fixando a versão
mínima explicitamente no bootstrap de `score_batch.py`:
`%pip install databricks-feature-engineering "mlflow>=3.15.0"` — substitui
`%pip install databricks-feature-engineering` sozinho. Não se aplica a
`prepare_training_set.py`/`register_model.py` do `training-platform`: nenhum dos dois
chama `spark_udf` (só `fe.create_training_set`/`fe.log_model`), então não hits esse
caminho de código.

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
├── examples/
│   ├── __init__.py
│   └── serving_configs.py       # ServingConfig de exemplo (não-produtivo), modo batch
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

> **Correção (2026-08-24, achada ao vivo na Task 9 — verificação ponta a ponta):** o
> gate original só checava nulos na coluna de predição. Isso não pega o caso mais
> importante: uma entidade da spine sem correspondência na feature table
> (`FeatureLookup` sem match) recebe features nulas, e o modelo (ex.:
> `RandomForestClassifier` do scikit-learn, que tolera `NaN` nativamente desde a versão
> 1.4) ainda produz uma predição não-nula, só que sem sentido — o gate original deixaria
> passar silenciosamente. Verificado ao vivo: uma spine com um `customer_id` inexistente
> na feature table gerou `txn_count=NULL, avg_ticket=NULL,
> prediction=0.4015943427487544`. Corrigido adicionando `check_no_nulls_in_joined_columns`,
> que checa nulos em **todas as colunas exceto a de predição** (chave de entidade, chave
> de timestamp, features) — e mudando `run_predictions_gate` para incluir esse terceiro
> check. Isso exige que `score_batch.py` (Task 7) passe o DataFrame completo das
> predições para o gate, não só a coluna de predição isolada — ver correção na Task 7.
> Ver também `docs/superpowers/specs/2026-08-23-serving-design.md`, emenda 1.3.

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
    check_no_nulls_in_joined_columns,
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


def test_check_no_nulls_in_joined_columns_passes():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "txn_count": [3, 5], "prediction": [0.1, 0.9]})
    assert check_no_nulls_in_joined_columns(df, "prediction").status == "PASS"


def test_check_no_nulls_in_joined_columns_fails_on_unmatched_feature_lookup():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "txn_count": [3, None], "prediction": [0.1, 0.9]})
    finding = check_no_nulls_in_joined_columns(df, "prediction")
    assert finding.status == "FAIL"
    assert "txn_count" in finding.detail


def test_check_row_count_matches_passes_when_equal():
    assert check_row_count_matches(100, 100).status == "PASS"


def test_check_row_count_matches_fails_when_different():
    finding = check_row_count_matches(100, 87)
    assert finding.status == "FAIL"
    assert "input=100" in finding.detail


def test_run_predictions_gate_returns_all_checks():
    df = pd.DataFrame({"customer_id": ["c1", "c2"], "prediction": [0.1, 0.9]})
    findings = run_predictions_gate(df, "prediction", input_row_count=2)
    assert {f.check for f in findings} == {
        "no_nulls_in_predictions",
        "no_nulls_in_joined_columns",
        "row_count_matches",
    }


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


def check_no_nulls_in_joined_columns(df: pd.DataFrame, prediction_column: str) -> Finding:
    # Uma entidade sem correspondência no FeatureLookup recebe features nulas, mas o
    # modelo pode ainda assim produzir uma predição não-nula (ex.: RandomForestClassifier
    # tolera NaN nativamente) — checar só a coluna de predição não pega esse caso.
    joined_cols = [c for c in df.columns if c != prediction_column]
    bad_cols = [c for c in joined_cols if int(df[c].isnull().sum()) > 0]
    return Finding(
        check="no_nulls_in_joined_columns",
        status="PASS" if not bad_cols else "FAIL",
        detail=f"columns_with_nulls={bad_cols}" if bad_cols else "",
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
        check_no_nulls_in_joined_columns(df, prediction_column),
        check_row_count_matches(input_row_count, len(df)),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `9 passed`

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
    # saveAsTable não cria o schema automaticamente em Unity Catalog — sem isso,
    # a primeira escrita falha com SCHEMA_NOT_FOUND (mesmo bug confirmado no
    # feature-platform e no training-platform).
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
git add src/serving_platform/audit.py tests/test_audit.py
git commit -m "feat: duplicate audit record type and pipeline_runs writer"
```

---

## Task 6: Geração de recursos (`resource_gen.py`)

> **Correção (2026-08-24, achada ao vivo na Task 9 — deploy real da trilha online):** o
> recurso `model_serving_endpoints` de um DAB **não aceita** a sintaxe
> `models:/nome@alias` em `entity_name` — só um `entity_name` "puro"
> (`catalog.schema.modelo`) mais um `entity_version` **numérico** fixo. Tentar deployar
> com `@champion` embutido em `entity_name` falhou ao vivo com
> `404 RESOURCE_DOES_NOT_EXIST: Registered model '...@champion' does not exist`. Isso é
> exatamente o risco documentado na intro da Task 7 ("a superfície exata do SDK... e o
> schema exato do recurso model_serving_endpoints do DAB devem ser conferidos... antes
> do primeiro deploy real") — confirmado, e com uma superfície diferente da assumida.
> Corrigido resolvendo o alias para a versão vigente **no momento da geração dos
> recursos** (não no deploy) via `WorkspaceClient().model_versions.get_by_alias(...)` —
> daí o `entity_version` fixo no YAML gerado. Isso significa que promover um novo alias
> não atualiza um endpoint já deployado automaticamente nem gerando/deployando de novo
> sem intervenção — é exatamente o papel do `refresh_endpoint` (Task 7), que já existia
> por esse motivo, e passa a ser ainda mais central: **é o único caminho, além de gerar
> os recursos de novo, para mover um endpoint já no ar para uma versão nova**.
>
> Mudanças: `_online_endpoint` ganha um parâmetro `entity_version`; `generate_resources`
> ganha um parâmetro opcional `resolve_alias_version` (uma função
> `(model_name, config) -> int`), **obrigatório apenas quando algum `ServingConfig`
> registrado tem `mode="online"`** — levanta `ValueError` claro se faltar, em vez de
> gerar um recurso quebrado silenciosamente. Isso mantém `resource_gen.py` testável
> localmente (os testes passam uma função fake) — só `scripts/generate_resources.py`
> (Task 7, nunca testado localmente, sempre exercitado ao vivo) faz a chamada real ao
> workspace.

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
    assert job["tasks"][0]["notebook_task"]["notebook_path"] == "../notebooks/score_batch.py"


def test_online_config_generates_a_model_serving_endpoint():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_online", mode="online"))

    resources = generate_resources(resolve_alias_version=lambda model_name, config: 3)
    endpoints = resources["resources"]["model_serving_endpoints"]

    endpoint = endpoints["exemplo-modelo_online-serving"]
    served_entity = endpoint["config"]["served_entities"][0]
    assert served_entity["entity_name"] == "${var.catalog}.exemplo_models.modelo_online"
    assert served_entity["entity_version"] == "3"


def test_online_config_without_resolver_raises_clear_error():
    register_serving_config(ServingConfig(domain="exemplo", model_name="modelo_online", mode="online"))

    with pytest.raises(ValueError, match="resolve_alias_version"):
        generate_resources()


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

BATCH_NOTEBOOK_PATH = "../notebooks/score_batch.py"
REFRESH_NOTEBOOK_PATH = "../notebooks/refresh_endpoint.py"


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


def _online_endpoint(model_name: str, config, entity_version: int) -> dict:
    # model_serving_endpoints em DABs não aceita a sintaxe models:/nome@alias em
    # entity_name — só um entity_name puro + entity_version numérico fixo (confirmado
    # ao vivo: 404 RESOURCE_DOES_NOT_EXIST tentando "...@champion"). O alias é
    # resolvido para a versão vigente no momento da geração (ver
    # resolve_alias_version); mover o alias depois exige rodar refresh_endpoint
    # (Task 7) ou gerar os recursos de novo.
    return {
        "name": derive_endpoint_name(config.domain, model_name),
        "config": {
            "served_entities": [
                {
                    "name": model_name,
                    "entity_name": f"${{var.catalog}}.{config.domain}_models.{model_name}",
                    "entity_version": str(entity_version),
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


def generate_resources(resolve_alias_version=None) -> dict:
    """resolve_alias_version: callable (model_name: str, config: ServingConfig) -> int.
    Obrigatório quando há algum ServingConfig com mode="online" — model_serving_endpoints
    em DABs só aceita entity_version (um número), não um alias, então o alias precisa
    ser resolvido para a versão vigente no momento da geração dos recursos."""
    registry = get_registry()
    jobs = {"refresh_endpoint": _refresh_endpoint_job()}
    endpoints = {}

    for model_name, config in registry.items():
        if config.mode == "batch":
            jobs[f"score_batch_{model_name}"] = _batch_job(model_name, config)
        else:
            if resolve_alias_version is None:
                raise ValueError(
                    f"ServingConfig '{model_name}' has mode='online' but no "
                    "resolve_alias_version resolver was provided to generate_resources()"
                )
            entity_version = resolve_alias_version(model_name, config)
            endpoints[derive_endpoint_name(config.domain, model_name)] = _online_endpoint(
                model_name, config, entity_version
            )

    resources = {"resources": {"jobs": jobs}}
    if endpoints:
        resources["resources"]["model_serving_endpoints"] = endpoints
    return resources


def write_resources(path: str, resolve_alias_version=None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(generate_resources(resolve_alias_version), f, sort_keys=False)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `5 passed`

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
- Create: `examples/__init__.py`
- Create: `examples/serving_configs.py`
- Create: `notebooks/score_batch.py`
- Create: `notebooks/refresh_endpoint.py`
- Create: `databricks.yml`
- Create: `scripts/generate_resources.py`

- [ ] **Step 1: Criar o exemplo não-produtivo (modo batch — sem custo de endpoint sempre ligado)**

```python
# examples/__init__.py
```

```python
# examples/serving_configs.py
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
# MAGIC %pip install databricks-feature-engineering "mlflow>=3.15.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("git_commit", "local")
dbutils.widgets.text("git_branch", "local")

# COMMAND ----------
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — nem a raiz
# do bundle (onde mora `examples/`) nem `src/` (onde mora `serving_platform`) estão
# no sys.path por padrão.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.serving_configs  # noqa: F401
from datetime import date, datetime

import pyspark.sql.functions as F
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
# currentRunId() não está na whitelist do Py4J em compute serverless/shared access
# mode — levanta Py4JSecurityException. Cai para um id gerado localmente quando o
# contexto de job não expõe o run id dessa forma.
try:
    run_id_job = dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get().toString()
except Exception:
    import uuid

    run_id_job = str(uuid.uuid4())

# COMMAND ----------
full_model_name = f"{catalog}.{config.domain}_models.{model_name}"
spine = spark.table(config.spine_inference_table)
input_row_count = spine.count()

fe = FeatureEngineeringClient()
predictions_df = fe.score_batch(
    model_uri=f"models:/{full_model_name}@{config.alias}",
    df=spine,
    result_type="double",
).withColumn("scored_at", F.current_timestamp())

# COMMAND ----------
prediction_column = "prediction"
# DataFrame completo (não só a coluna de predição) — o gate agora também checa nulos
# nas colunas resolvidas pelo join (chave de entidade, chave de timestamp, features),
# não só na coluna de predição.
predictions_pd = predictions_df.toPandas()
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

# saveAsTable não cria o schema automaticamente em Unity Catalog — <domain>_predictions
# é um schema novo, nunca criado por nenhum componente anterior.
predictions_schema = predictions_table.rsplit(".", 1)[0]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {predictions_schema}")
predictions_df.write.format("delta").mode("append").saveAsTable(predictions_table)

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
# Num job deployado via DAB, o cwd do notebook é .../files/notebooks — nem a raiz
# do bundle (onde mora `examples/`) nem `src/` (onde mora `serving_platform`) estão
# no sys.path por padrão.
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
for _p in (_repo_root, os.path.join(_repo_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples.serving_configs  # noqa: F401
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

> **Correção (achada nas Tasks 7 e 9):** o `sys.path.insert` original só incluía
> `src/`, não a raiz do repositório — rodar o script diretamente (não como módulo)
> não coloca a raiz no `sys.path` automaticamente, quebrando
> `import examples.serving_configs` com `ModuleNotFoundError: No module named
> 'examples'` (mesmo bug já confirmado no `feature-platform`). E o gerador agora
> precisa resolver o alias de qualquer `ServingConfig` `mode="online"` para uma versão
> concreta (correção acima, Task 6) — via uma chamada real ao workspace usando
> `databricks-sdk`, só possível aqui (nunca em `resource_gen.py`, que fica puro e
> testável). `CATALOG` é literal (não `${var.catalog}`) porque a resolução do alias
> precisa de um catalog real no momento da geração — deve bater com o default de
> `catalog` em `databricks.yml`; o YAML gerado continua referenciando `${var.catalog}`
> para o deploy em si.

```python
# scripts/generate_resources.py
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
for _p in (_repo_root, _repo_root / "src"):
    sys.path.insert(0, str(_p))

import examples.serving_configs  # noqa: F401
from databricks.sdk import WorkspaceClient
from serving_platform.resource_gen import write_resources

CATALOG = "workspace"  # deve bater com o default de `catalog` em databricks.yml


def _resolve_alias_version(model_name: str, config) -> int:
    full_name = f"{CATALOG}.{config.domain}_models.{model_name}"
    return WorkspaceClient().model_versions.get_by_alias(full_name, config.alias).version


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_serving.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_resources(str(output_path), resolve_alias_version=_resolve_alias_version)
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
git add examples/ notebooks/ databricks.yml scripts/generate_resources.py resources/.gitkeep
git commit -m "feat: add non-productive batch example, notebooks, and DAB bundle root"
```

---

## Task 8: GitHub Actions — caller do reusable workflow (`mlops-platform`)

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

Este repositório é um framework puro — não é onde domínios reais declaram seus
servings. Instale este pacote no repositório do seu domínio
(`pip install git+https://github.com/ViniciusOtoni/serving-platform@vX.Y.Z`) e
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

- [x] **Step 3: Verificação ponta a ponta — trilha batch (prioridade, sem custo de
  endpoint):** confirmar que `workspace.exemplo.spine_inference` existe, rodar
  `databricks bundle run score_batch_propensao_exemplo -t dev`, confirmar que
  `workspace.exemplo_predictions.propensao_exemplo` foi criada e que há uma linha
  `SUCCESS` em `platform_audit.pipeline_runs` com `component="serving"`,
  `mode="batch"`.

  Confirmado, depois de corrigir 5 bugs reais ao vivo (bootstraps de `%pip
  install`/`sys.path`, `mlflow>=3.15.0` pinado por causa de `spark_udf`, `code_paths`
  ausente em `fe.log_model` do `training-platform`, `FeaturePlatformModel` não
  filtrando `model_input` para `feature_names_in_`). Todas as correções documentadas
  nas notas "Correção" acima. 5 predições escritas (uma por cliente), tabela com
  `prediction`/`scored_at`, linha `SUCCESS` confirmada via SQL direto.

- [x] **Step 4: Forçar falha do gate** (ex.: apontar `spine_inference_table` para uma
  tabela vazia ou com schema incompatível) e confirmar que **nada é escrito** na
  tabela de predições, e que a auditoria grava `status=FAILED`.

  Forçado com uma spine contendo um `customer_id` sem correspondência na feature
  table. Isso revelou o bug do gate (correção na Task 4 — nulos nas colunas do join
  não eram checados) e, depois de corrigido, confirmou o comportamento certo:
  `ValueError: predictions quality gate failed: ['no_nulls_in_joined_columns']`,
  nenhuma linha nova na tabela de predições, linha `FAILED` gravada na auditoria.

- [x] **Step 5: Verificação ponta a ponta — trilha online (só depois da batch
  funcionar, e ciente do custo):** registrar um `ServingConfig` de teste com
  `mode="online"`, gerar recursos, `databricks bundle deploy`, confirmar que o endpoint
  sobe e responde a uma chamada de teste com as features resolvidas corretamente — esta
  é a verificação do risco documentado no spec (seção 6). Se a resolução automática de
  `FeatureLookup` não funcionar como assumido, documentar o comportamento real
  encontrado e ajustar o spec antes de prosseguir. **Derrubar o endpoint manualmente ao
  final do teste** para não incorrer em custo contínuo.

  **Parcialmente concluído — ver spec, emenda 1.4.** O deploy do endpoint em si exigiu
  uma correção real (Task 6 — `model_serving_endpoints` não aceita `@alias` em
  `entity_name`, corrigido resolvendo para `entity_version` numérico via
  `get_by_alias`). Depois disso, o deploy falhou com "Online feature store setup
  failed": a feature table de exemplo tem `online=False` (nunca foi sincronizada como
  Online Table), e investigar revelou um bug real em `feature-platform`'s
  `online_sync.py` (assinatura de `create_synced_database_table` incorreta) **mais**
  uma dependência de infraestrutura não provisionada (nenhum Database Instance do
  Lakebase existe neste workspace — provisionar um é uma decisão à parte, recurso
  cobrado e em Public Preview). Corrigida a assinatura em `feature-platform` (ver o
  plano daquele repositório); a validação completa da resolução automática de
  `FeatureLookup` num endpoint online **continua não confirmada**, pendente de uma
  decisão explícita do usuário sobre provisionar o Database Instance. Endpoint de
  teste derrubado (`databricks serving-endpoints delete`), exemplo revertido para
  `mode="batch"` (o exemplo permanente, sem custo de endpoint sempre ligado).

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
