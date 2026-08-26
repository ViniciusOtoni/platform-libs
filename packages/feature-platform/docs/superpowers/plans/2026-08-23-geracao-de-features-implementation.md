# Componente de Geração de Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o framework `feature_platform` que permite declarar feature tables via decorator, executá-las em modo incremental ou backfill com checkpointing automático, aplicar um gate de qualidade bloqueante, gerar o job DAB automaticamente, registrar toda execução numa tabela de auditoria central e sincronizar opcionalmente com o Online Feature Store (Lakebase).

**Architecture:** Lógica pura (contrato/registro, nomenclatura, resolução de janela, checks de qualidade, geração de resource YAML) vive em `src/feature_platform/`, testável localmente com `pytest`, sem Spark — mesmo padrão já validado na POC `databricks-feature-lookup-poc`. A camada que fala com Spark/Delta/Lakebase (merge, overwrite, leitura/escrita da tabela de auditoria, sync online) fica isolada em funções específicas dentro dos mesmos módulos, exercitadas via notebook (`notebooks/run_feature_table.py`) rodando num job Databricks real — não em pytest. Um exemplo não-produtivo (`examples/`) prova o fluxo ponta a ponta.

**Tech Stack:** Python 3.11, PySpark + Delta Lake (runtime Databricks serverless), Databricks SDK, Databricks Asset Bundles, pytest, pandas, PyYAML, GitHub Actions.

**Emenda (2026-08-23, durante o design da arquitetura de plataforma):** este
repositório passou a ser um framework puro — sem pastas de domínio real dentro dele.
`dominios/exemplo/` foi renomeada para `examples/` (mesmo papel: harness de teste de
integração, não domínio de negócio real), e o decorator `@feature_table` ganhou um
parâmetro `domain` **explícito e obrigatório** (a inferência automática pelo caminho
`dominios/<domínio>/` não faz mais sentido sem essa pasta). O
`.github/workflows/deploy.yml` (Task 13) passou a ser um caller do reusable workflow
centralizado em `mlops-platform`. Ver
`docs/superpowers/specs/2026-08-23-geracao-de-features-design.md`, seção 1.1.

**Correção (2026-08-24, descoberta ao validar a Task 12 contra o Databricks real):**
duas correções pequenas, aplicadas nas Tasks 10 e 12 abaixo — `resource_gen.py`
precisa da extensão `.py` no `NOTEBOOK_PATH` (o CLI instalado exige extensão em
referências de notebook local) e `scripts/generate_resources.py` precisa inserir a
raiz do repositório no `sys.path`, não só `src/`, para que `import examples.features`
funcione ao rodar o script diretamente.

**Correção (2026-08-24, descoberta na Task 15, rodando o pipeline de verdade contra
Unity Catalog):** três bugs a mais, só visíveis com workspace real (nenhum pytest local
os pegaria):
1. `write_run` (Task 8, `audit.py`) e `write_feature_table` (Task 7, `writer.py`):
   `saveAsTable` não cria o schema automaticamente no Unity Catalog — a primeira
   escrita num schema novo falhava com `SCHEMA_NOT_FOUND`. Ambos agora rodam
   `CREATE SCHEMA IF NOT EXISTS` antes de escrever.
2. `_overwrite_by_partition` (Task 7, `writer.py`): a linha
   `spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")` é rejeitada
   em compute serverless/Spark Connect (`CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION`) —
   removida; o `.option("partitionOverwriteMode", "dynamic")` já no próprio writer é
   suficiente sozinho.
3. `notebooks/run_feature_table.py` (Task 12): precisa inserir a raiz do bundle e
   `src/` no `sys.path` no início do notebook — o cwd de um job deployado via DAB
   (`.../files/notebooks`) não inclui nenhum dos dois por padrão, quebrando
   `import examples.features`. E `dbutils.notebook.entry_point...currentRunId()`
   levanta `Py4JSecurityException` nesse modo de compute (não está na whitelist do
   Py4J) — trocado por `.currentRunId().get().toString()` com fallback para um
   `uuid.uuid4()` gerado localmente se o contexto de job não expuser o run id dessa
   forma.

Essas três correções estão commitadas juntas no repositório
(`fix: three bugs found running the pipeline for real in Unity Catalog`); os blocos de
código abaixo, nas Tasks 7, 8 e 12, ainda mostram a versão pré-correção — trate o texto
desta nota como a versão vigente.

**Correção (2026-08-24, descoberta indiretamente — rodando o `training-platform`, um
consumidor deste componente, contra a mesma feature table de exemplo):**
`write_feature_table` (Task 7, `writer.py`) nunca declarava uma `PRIMARY KEY` na tabela
gerada. O Feature Engineering em Unity Catalog exige essa constraint para reconhecer uma
tabela como feature table via `FeatureLookup` — sem ela, `fe.create_training_set(...)`
falha com `BAD_REQUEST: Table can't be used as a feature table because it has no primary
key constraint defined`. Como a tabela tem `timestamp_key`, a constraint precisa marcar
essa coluna com `TIMESERIES` (senão o FE não reconhece a semântica de série temporal
usada pelo `timestamp_lookup_key` no lookup). Corrigido adicionando, ao final de
`write_feature_table`, um passo idempotente `_ensure_primary_key` que:
1. Verifica via `system.information_schema.table_constraints` se a tabela já tem uma
   `PRIMARY KEY` — se sim, não faz nada (self-healing sem custo em escritas repetidas).
2. Caso contrário, marca `entity_keys` + `timestamp_key` como `NOT NULL` (pré-requisito
   de Delta para colunas de PK) e roda
   `ALTER TABLE ... ADD CONSTRAINT ... PRIMARY KEY (<entity_keys>, <timestamp_key> TIMESERIES)`.

Por ser chamado no final de `write_feature_table` independente do modo (merge ou
overwrite), isso também autocura tabelas já existentes criadas antes desta correção — a
próxima escrita nelas adiciona a constraint retroativamente. O bloco de código abaixo,
na Task 7, ainda mostra a versão pré-correção.

---

## Scope Check

Este plano cobre só o Componente 1 (Geração de Features), conforme o spec em
`docs/superpowers/specs/2026-08-23-geracao-de-features-design.md`. Os componentes de
Treino, Serving e Monitoramento têm specs e plans próprios, a serem escritos depois que
este componente estiver implementado e validado — eles dependem de feature tables
maduras produzidas aqui.

## File Structure

```
feature-platform/
├── databricks.yml                          # bundle root
├── pyproject.toml                          # empacotamento da lib feature_platform
├── pytest.ini
├── requirements-dev.txt
├── README.md
├── src/
│   └── feature_platform/
│       ├── __init__.py
│       ├── types.py                        # DateRange
│       ├── naming.py                       # derive/validate/resolve table_name
│       ├── window.py                       # resolução de janela incremental/backfill
│       ├── quality.py                      # Finding + checks + gate
│       ├── contract.py                     # @feature_table + FeatureTableSpec + registry
│       ├── writer.py                       # WriteMode + merge/overwrite (Spark)
│       ├── audit.py                        # RunRecord + leitura/escrita platform_audit.pipeline_runs (Spark)
│       ├── engine.py                       # orquestração: resolve_window + run_feature_table
│       ├── resource_gen.py                 # gera resources/generated_feature_pipeline.job.yml
│       └── online_sync.py                  # sync opcional para Lakebase (Spark/SDK)
├── scripts/
│   └── generate_resources.py               # CLI: roda resource_gen antes do bundle deploy
├── examples/
│   ├── __init__.py
│   └── features.py                         # feature table de exemplo (não-produtiva), prova o fluxo ponta a ponta
├── notebooks/
│   └── run_feature_table.py                # entrypoint Databricks: le widgets, chama engine.run_feature_table
├── resources/
│   └── (generated_feature_pipeline.job.yml)  # gerado por scripts/generate_resources.py, não escrito à mão
├── tests/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_naming.py
│   ├── test_window.py
│   ├── test_quality.py
│   ├── test_contract.py
│   ├── test_writer.py
│   ├── test_audit.py
│   ├── test_engine.py
│   └── test_resource_gen.py
└── .github/
    └── workflows/
        └── deploy.yml                      # CI: valida + deploya o bundle, propaga git_commit/git_branch
```

Cada módulo tem uma responsabilidade só. `engine.py` é o único que importa de todos os
outros — é o ponto de orquestração, não de lógica nova.

---

## Task 1: Scaffolding do repositório

**Files:**
- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `src/feature_platform/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Criar `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "feature-platform"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
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
pandas>=2.0
pyyaml>=6.0
pytest>=8.0
pyspark>=3.5
delta-spark>=3.2
databricks-sdk>=0.30
```

- [ ] **Step 4: Criar `src/feature_platform/__init__.py`** (vazio, só marca o pacote)

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

- [ ] **Step 6: Criar venv e instalar dependências, confirmar que pytest roda (sem testes ainda)**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```
Expected: `no tests ran` (sem erro de import ou de ambiente).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml pytest.ini requirements-dev.txt src/feature_platform/__init__.py .gitignore
git commit -m "chore: scaffold feature-platform package"
```

---

## Task 2: `DateRange`

**Files:**
- Create: `src/feature_platform/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_types.py
from datetime import date
import pytest

from feature_platform.types import DateRange


def test_date_range_holds_start_and_end():
    r = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 31))
    assert r.start == date(2026, 1, 1)
    assert r.end == date(2026, 1, 31)


def test_date_range_rejects_start_after_end():
    with pytest.raises(ValueError, match="must not be after"):
        DateRange(start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_date_range_allows_start_equal_end():
    r = DateRange(start=date(2026, 1, 1), end=date(2026, 1, 1))
    assert r.start == r.end
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_types.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.types'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/types.py
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must not be after end ({self.end})")
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_types.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/types.py tests/test_types.py
git commit -m "feat: add DateRange value type"
```

---

## Task 3: Nomenclatura (`naming.py`)

**Files:**
- Create: `src/feature_platform/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_naming.py
import pytest

from feature_platform.naming import derive_table_name, validate_table_name, resolve_table_name


def test_derive_table_name_follows_convention():
    name = derive_table_name(catalog="workspace", domain="credito", function_name="score_features")
    assert name == "workspace.credito_features.score_features"


def test_validate_table_name_accepts_convention():
    validate_table_name("workspace.credito_features.score_features")  # não deve levantar


def test_validate_table_name_rejects_uppercase():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_table_name("Workspace.Credito.Score")


def test_validate_table_name_rejects_wrong_number_of_parts():
    with pytest.raises(ValueError, match="does not match convention"):
        validate_table_name("workspace.score_features")


def test_resolve_table_name_derives_when_none():
    name = resolve_table_name("workspace", "credito", "score_features", None)
    assert name == "workspace.credito_features.score_features"


def test_resolve_table_name_validates_explicit_override():
    name = resolve_table_name("workspace", "credito", "score_features", "workspace.legado.score_v1")
    assert name == "workspace.legado.score_v1"


def test_resolve_table_name_rejects_invalid_explicit_override():
    with pytest.raises(ValueError, match="does not match convention"):
        resolve_table_name("workspace", "credito", "score_features", "Invalid Name")
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.naming'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/naming.py
import re

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def derive_table_name(catalog: str, domain: str, function_name: str) -> str:
    schema = f"{domain}_features"
    return f"{catalog}.{schema}.{function_name}"


def validate_table_name(table_name: str) -> None:
    if not _NAME_RE.match(table_name):
        raise ValueError(
            f"table_name '{table_name}' does not match convention "
            "'<catalog>.<schema>.<table>' (lowercase letters, digits, underscore)"
        )


def resolve_table_name(catalog: str, domain: str, function_name: str, table_name: str | None) -> str:
    if table_name is None:
        return derive_table_name(catalog, domain, function_name)
    validate_table_name(table_name)
    return table_name
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_naming.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/naming.py tests/test_naming.py
git commit -m "feat: add table naming convention derivation and validation"
```

---

## Task 4: Resolução de janela (`window.py`)

**Files:**
- Create: `src/feature_platform/window.py`
- Test: `tests/test_window.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_window.py
from datetime import date
import pytest

from feature_platform.window import (
    resolve_incremental_window,
    parse_backfill_window,
    NoCheckpointError,
)


def test_incremental_window_starts_at_last_checkpoint():
    w = resolve_incremental_window(last_success_end=date(2026, 1, 10), today=date(2026, 1, 15))
    assert w.start == date(2026, 1, 10)
    assert w.end == date(2026, 1, 15)


def test_incremental_window_raises_without_checkpoint():
    with pytest.raises(NoCheckpointError, match="run a backfill first"):
        resolve_incremental_window(last_success_end=None, today=date(2026, 1, 15))


def test_incremental_window_raises_when_nothing_to_process():
    with pytest.raises(ValueError, match="nothing to process"):
        resolve_incremental_window(last_success_end=date(2026, 1, 15), today=date(2026, 1, 15))


def test_parse_backfill_window():
    w = parse_backfill_window("2026-01-01", "2026-01-31")
    assert w.start == date(2026, 1, 1)
    assert w.end == date(2026, 1, 31)
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_window.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.window'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/window.py
from datetime import date

from .types import DateRange


class NoCheckpointError(Exception):
    """Levantado quando o modo incremental roda sem nenhum run SUCCESS anterior registrado."""


def resolve_incremental_window(last_success_end: date | None, today: date) -> DateRange:
    if last_success_end is None:
        raise NoCheckpointError(
            "no successful run found for this feature table; run a backfill first "
            "to establish an initial checkpoint before scheduling incremental runs"
        )
    if last_success_end >= today:
        raise ValueError(
            f"nothing to process: checkpoint ({last_success_end}) is not before today ({today})"
        )
    return DateRange(start=last_success_end, end=today)


def parse_backfill_window(start_date: str, end_date: str) -> DateRange:
    return DateRange(start=date.fromisoformat(start_date), end=date.fromisoformat(end_date))
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_window.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/window.py tests/test_window.py
git commit -m "feat: add incremental checkpoint and backfill window resolution"
```

---

## Task 5: Gate de qualidade (`quality.py`)

**Files:**
- Create: `src/feature_platform/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_quality.py
from datetime import date

import pandas as pd

from feature_platform.quality import (
    Finding,
    check_schema,
    check_unique_keys,
    check_no_nulls,
    check_freshness,
    run_quality_gate,
    gate_passed,
)


def _valid_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "feature_ts": [date(2026, 1, 15)] * 3,
            "score": [0.1, 0.2, 0.3],
        }
    )


def test_check_schema_passes_with_expected_columns():
    finding = check_schema(_valid_df(), ["customer_id", "feature_ts"])
    assert finding.status == "PASS"
    assert finding.violations == 0


def test_check_schema_fails_with_missing_column():
    finding = check_schema(_valid_df(), ["customer_id", "missing_col"])
    assert finding.status == "FAIL"
    assert "missing_col" in finding.detail


def test_check_unique_keys_fails_on_duplicate():
    df = pd.concat([_valid_df(), _valid_df().iloc[[0]]], ignore_index=True)
    finding = check_unique_keys(df, entity_keys=["customer_id"], timestamp_key="feature_ts")
    assert finding.status == "FAIL"
    assert finding.violations == 1


def test_check_no_nulls_fails_on_null_key():
    df = _valid_df()
    df.loc[0, "customer_id"] = None
    finding = check_no_nulls(df, entity_keys=["customer_id"], timestamp_key="feature_ts")
    assert finding.status == "FAIL"
    assert finding.violations == 1


def test_check_freshness_passes_within_lag():
    finding = check_freshness(_valid_df(), "feature_ts", window_end=date(2026, 1, 15), max_lag_days=1)
    assert finding.status == "PASS"


def test_check_freshness_fails_when_stale():
    finding = check_freshness(_valid_df(), "feature_ts", window_end=date(2026, 2, 1), max_lag_days=1)
    assert finding.status == "FAIL"


def test_run_quality_gate_returns_all_checks():
    findings = run_quality_gate(_valid_df(), ["customer_id"], "feature_ts", date(2026, 1, 15))
    assert {f.check for f in findings} == {"schema", "unique_keys", "no_nulls", "freshness"}


def test_gate_passed_true_when_all_pass():
    findings = [Finding("a", "PASS", 0), Finding("b", "PASS", 0)]
    assert gate_passed(findings) is True


def test_gate_passed_false_when_any_fails():
    findings = [Finding("a", "PASS", 0), Finding("b", "FAIL", 3)]
    assert gate_passed(findings) is False
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.quality'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/quality.py
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    violations: int
    detail: str = ""


def check_schema(df: pd.DataFrame, expected_columns: list[str]) -> Finding:
    missing = [c for c in expected_columns if c not in df.columns]
    return Finding(
        check="schema",
        status="PASS" if not missing else "FAIL",
        violations=len(missing),
        detail=f"missing columns: {missing}" if missing else "",
    )


def check_unique_keys(df: pd.DataFrame, entity_keys: list[str], timestamp_key: str) -> Finding:
    key_cols = [*entity_keys, timestamp_key]
    dupes = int(df.duplicated(subset=key_cols).sum())
    return Finding(
        check="unique_keys",
        status="PASS" if dupes == 0 else "FAIL",
        violations=dupes,
        detail=f"duplicate rows on {key_cols}",
    )


def check_no_nulls(df: pd.DataFrame, entity_keys: list[str], timestamp_key: str) -> Finding:
    cols = [*entity_keys, timestamp_key]
    nulls = int(df[cols].isnull().sum().sum())
    return Finding(
        check="no_nulls",
        status="PASS" if nulls == 0 else "FAIL",
        violations=nulls,
        detail=f"null values in {cols}",
    )


def check_freshness(df: pd.DataFrame, timestamp_key: str, window_end: date, max_lag_days: int = 1) -> Finding:
    if df.empty:
        return Finding(check="freshness", status="FAIL", violations=1, detail="empty dataframe")
    max_ts = pd.to_datetime(df[timestamp_key]).max().date()
    lag = (window_end - max_ts).days
    passed = lag <= max_lag_days
    return Finding(
        check="freshness",
        status="PASS" if passed else "FAIL",
        violations=0 if passed else 1,
        detail=f"max({timestamp_key})={max_ts}, window_end={window_end}, lag={lag}d",
    )


def run_quality_gate(
    df: pd.DataFrame, entity_keys: list[str], timestamp_key: str, window_end: date
) -> list[Finding]:
    expected_columns = [*entity_keys, timestamp_key]
    return [
        check_schema(df, expected_columns),
        check_unique_keys(df, entity_keys, timestamp_key),
        check_no_nulls(df, entity_keys, timestamp_key),
        check_freshness(df, timestamp_key, window_end),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_quality.py -v`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/quality.py tests/test_quality.py
git commit -m "feat: add blocking quality gate with Finding-based checks"
```

---

## Task 6: Contrato e registro (`contract.py`)

**Files:**
- Create: `src/feature_platform/contract.py`
- Test: `tests/test_contract.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_contract.py
import pytest

from feature_platform.contract import feature_table, get_registry, clear_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_feature_table_registers_spec_with_defaults():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.transactions"])
    def minha_feature(sources, window):
        return None

    registry = get_registry()
    spec = registry["minha_feature"]
    assert spec.domain == "exemplo"
    assert spec.entity_keys == ["customer_id"]
    assert spec.timestamp_key == "feature_ts"
    assert spec.sources == ["raw.transactions"]
    assert spec.online is False
    assert spec.depends_on == []
    assert spec.table_name is None
    assert spec.compute_fn is minha_feature


def test_feature_table_requires_domain():
    with pytest.raises(TypeError):
        feature_table(entity_keys=["k"], timestamp_key="ts", sources=[])


def test_feature_table_rejects_duplicate_registration():
    @feature_table(domain="exemplo", entity_keys=["k"], timestamp_key="ts", sources=[])
    def duplicada(sources, window):
        return None

    with pytest.raises(ValueError, match="already registered"):
        feature_table(domain="exemplo", entity_keys=["k"], timestamp_key="ts", sources=[])(duplicada)


def test_feature_table_accepts_online_and_depends_on_and_table_name():
    @feature_table(
        domain="exemplo",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.transactions"],
        online=True,
        depends_on=["outra_feature"],
        table_name="workspace.legado.minha_tabela",
    )
    def com_opcoes(sources, window):
        return None

    spec = get_registry()["com_opcoes"]
    assert spec.online is True
    assert spec.depends_on == ["outra_feature"]
    assert spec.table_name == "workspace.legado.minha_tabela"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.contract'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/contract.py
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class FeatureTableSpec:
    name: str
    entity_keys: list[str]
    timestamp_key: str
    sources: list[str]
    compute_fn: Callable
    domain: str
    online: bool = False
    depends_on: list[str] = field(default_factory=list)
    table_name: Optional[str] = None


_REGISTRY: dict[str, FeatureTableSpec] = {}


def feature_table(
    *,
    domain: str,
    entity_keys: list[str],
    timestamp_key: str,
    sources: list[str],
    online: bool = False,
    depends_on: list[str] | None = None,
    table_name: str | None = None,
):
    def decorator(fn: Callable) -> Callable:
        spec = FeatureTableSpec(
            name=fn.__name__,
            entity_keys=list(entity_keys),
            timestamp_key=timestamp_key,
            sources=list(sources),
            compute_fn=fn,
            domain=domain,
            online=online,
            depends_on=list(depends_on or []),
            table_name=table_name,
        )
        if spec.name in _REGISTRY:
            raise ValueError(f"feature table '{spec.name}' already registered")
        _REGISTRY[spec.name] = spec
        return fn

    return decorator


def get_registry() -> dict[str, FeatureTableSpec]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contract.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/contract.py tests/test_contract.py
git commit -m "feat: add @feature_table decorator and in-memory registry"
```

---

## Task 7: Estratégia de escrita (`writer.py`)

A parte testável localmente é só a decisão de estratégia (`WriteMode` → `merge` ou
`overwrite_by_partition`). As funções que tocam Spark/Delta (`_merge`,
`_overwrite_by_partition`) não são cobertas por pytest — são exercitadas via notebook
real no Databricks (Task 12), porque dependem de uma `SparkSession` com Delta
habilitado que não faz sentido simular localmente.

**Files:**
- Create: `src/feature_platform/writer.py`
- Test: `tests/test_writer.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_writer.py
import pytest

from feature_platform.writer import WriteMode, write_strategy_for


def test_incremental_uses_merge_strategy():
    assert write_strategy_for(WriteMode.INCREMENTAL) == "merge"


def test_backfill_uses_overwrite_by_partition_strategy():
    assert write_strategy_for(WriteMode.BACKFILL) == "overwrite_by_partition"


def test_write_mode_accepts_string_values():
    assert WriteMode("incremental") == WriteMode.INCREMENTAL
    assert WriteMode("backfill") == WriteMode.BACKFILL
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_writer.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.writer'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/writer.py
from enum import Enum


class WriteMode(str, Enum):
    INCREMENTAL = "incremental"
    BACKFILL = "backfill"


def write_strategy_for(mode: WriteMode) -> str:
    if mode == WriteMode.INCREMENTAL:
        return "merge"
    if mode == WriteMode.BACKFILL:
        return "overwrite_by_partition"
    raise ValueError(f"unknown mode: {mode}")


def write_feature_table(
    spark,
    df,
    table_name: str,
    entity_keys: list[str],
    timestamp_key: str,
    mode: WriteMode,
    partition_cols: list[str],
) -> None:
    """Escreve a feature table no Delta. Requer SparkSession com Delta habilitado —
    exercitado via notebook (Task 12), não via pytest."""
    strategy = write_strategy_for(mode)
    if strategy == "merge":
        _merge(spark, df, table_name, entity_keys, timestamp_key)
    else:
        _overwrite_by_partition(spark, df, table_name, partition_cols)


def _merge(spark, df, table_name: str, entity_keys: list[str], timestamp_key: str) -> None:
    from delta.tables import DeltaTable

    if not spark.catalog.tableExists(table_name):
        df.write.format("delta").saveAsTable(table_name)
        return

    target = DeltaTable.forName(spark, table_name)
    merge_keys = [*entity_keys, timestamp_key]
    condition = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
    (
        target.alias("target")
        .merge(df.alias("source"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def _overwrite_by_partition(spark, df, table_name: str, partition_cols: list[str]) -> None:
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    writer = df.write.format("delta").mode("overwrite").option("partitionOverwriteMode", "dynamic")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(table_name)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_writer.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/writer.py tests/test_writer.py
git commit -m "feat: add write strategy selection and Delta merge/overwrite implementation"
```

---

## Task 8: Auditoria (`audit.py`)

A parte testável localmente é a construção do registro (`RunRecord` → `to_row`). A
leitura/escrita real na tabela `platform_audit.pipeline_runs` depende de Spark e é
exercitada via notebook (Task 12).

**Files:**
- Create: `src/feature_platform/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_audit.py
from datetime import date, datetime

from feature_platform.audit import RunRecord, to_row, AUDIT_TABLE


def test_audit_table_name():
    assert AUDIT_TABLE == "platform_audit.pipeline_runs"


def test_to_row_maps_all_fields():
    record = RunRecord(
        component="feature_generation",
        entity_name="customer_transaction_features",
        git_commit="abc123",
        git_branch="main",
        run_id="run-1",
        mode="incremental",
        status="SUCCESS",
        window_start=date(2026, 1, 10),
        window_end=date(2026, 1, 15),
        run_ts=datetime(2026, 1, 15, 3, 0, 0),
    )

    row = to_row(record)

    assert row == {
        "component": "feature_generation",
        "entity_name": "customer_transaction_features",
        "git_commit": "abc123",
        "git_branch": "main",
        "run_id": "run-1",
        "mode": "incremental",
        "status": "SUCCESS",
        "window_start": date(2026, 1, 10),
        "window_end": date(2026, 1, 15),
        "run_ts": datetime(2026, 1, 15, 3, 0, 0),
    }
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.audit'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/audit.py
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
    """Escreve um registro na tabela de auditoria central. Requer SparkSession —
    exercitado via notebook (Task 12), não via pytest."""
    df = spark.createDataFrame([to_row(record)])
    if spark.catalog.tableExists(AUDIT_TABLE):
        df.write.format("delta").mode("append").saveAsTable(AUDIT_TABLE)
    else:
        df.write.format("delta").mode("overwrite").saveAsTable(AUDIT_TABLE)


def get_last_success_checkpoint(spark, component: str, entity_name: str):
    """Retorna o window_end do último run SUCCESS, ou None se não houver nenhum.
    Requer SparkSession — exercitado via notebook (Task 12), não via pytest."""
    import pyspark.sql.functions as F

    if not spark.catalog.tableExists(AUDIT_TABLE):
        return None

    result = (
        spark.table(AUDIT_TABLE)
        .filter(
            (F.col("component") == component)
            & (F.col("entity_name") == entity_name)
            & (F.col("status") == "SUCCESS")
        )
        .orderBy(F.col("window_end").desc())
        .limit(1)
        .collect()
    )
    return result[0]["window_end"] if result else None
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_audit.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/audit.py tests/test_audit.py
git commit -m "feat: add audit record type and platform_audit.pipeline_runs I/O"
```

---

## Task 9: Orquestração (`engine.py`)

`resolve_window` é testável localmente via mock de `get_last_success_checkpoint`.
`run_feature_table` orquestra tudo (sources reais, Spark, escrita, auditoria, sync
online) e só é exercitado via notebook (Task 12).

**Files:**
- Create: `src/feature_platform/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_engine.py
from datetime import date

import pytest

from feature_platform.contract import FeatureTableSpec
from feature_platform.writer import WriteMode
from feature_platform.window import NoCheckpointError
from feature_platform import engine


def _spec(**overrides):
    defaults = dict(
        name="minha_feature",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.transactions"],
        compute_fn=lambda sources, window: None,
        domain="exemplo",
    )
    defaults.update(overrides)
    return FeatureTableSpec(**defaults)


def test_resolve_window_backfill_uses_explicit_range():
    window = engine.resolve_window(
        spec=_spec(),
        mode=WriteMode.BACKFILL,
        today=date(2026, 1, 20),
        backfill_start="2026-01-01",
        backfill_end="2026-01-15",
        spark=None,
    )
    assert window.start == date(2026, 1, 1)
    assert window.end == date(2026, 1, 15)


def test_resolve_window_backfill_requires_dates():
    with pytest.raises(ValueError, match="backfill mode requires"):
        engine.resolve_window(
            spec=_spec(),
            mode=WriteMode.BACKFILL,
            today=date(2026, 1, 20),
            backfill_start=None,
            backfill_end=None,
            spark=None,
        )


def test_resolve_window_incremental_uses_checkpoint(monkeypatch):
    monkeypatch.setattr(engine, "get_last_success_checkpoint", lambda spark, component, entity: date(2026, 1, 10))

    window = engine.resolve_window(
        spec=_spec(),
        mode=WriteMode.INCREMENTAL,
        today=date(2026, 1, 15),
        backfill_start=None,
        backfill_end=None,
        spark=None,
    )
    assert window.start == date(2026, 1, 10)
    assert window.end == date(2026, 1, 15)


def test_resolve_window_incremental_without_checkpoint_raises(monkeypatch):
    monkeypatch.setattr(engine, "get_last_success_checkpoint", lambda spark, component, entity: None)

    with pytest.raises(NoCheckpointError):
        engine.resolve_window(
            spec=_spec(),
            mode=WriteMode.INCREMENTAL,
            today=date(2026, 1, 15),
            backfill_start=None,
            backfill_end=None,
            spark=None,
        )
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.engine'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/engine.py
from datetime import date, datetime

from .contract import FeatureTableSpec
from .naming import resolve_table_name
from .quality import run_quality_gate, gate_passed, Finding
from .writer import WriteMode, write_feature_table
from .audit import RunRecord, write_run, get_last_success_checkpoint
from .window import resolve_incremental_window, parse_backfill_window
from .types import DateRange

COMPONENT = "feature_generation"


class QualityGateFailure(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        failed = [f.check for f in findings if f.status == "FAIL"]
        super().__init__(f"quality gate failed: {failed}")


def resolve_window(
    spec: FeatureTableSpec,
    mode: WriteMode,
    today: date,
    backfill_start: str | None,
    backfill_end: str | None,
    spark,
) -> DateRange:
    if mode == WriteMode.BACKFILL:
        if not backfill_start or not backfill_end:
            raise ValueError("backfill mode requires start_date and end_date")
        return parse_backfill_window(backfill_start, backfill_end)

    checkpoint = get_last_success_checkpoint(spark, COMPONENT, spec.name)
    return resolve_incremental_window(checkpoint, today)


def run_feature_table(
    spec: FeatureTableSpec,
    spark,
    catalog: str,
    mode: WriteMode,
    today: date,
    run_id: str,
    git_commit: str,
    git_branch: str,
    backfill_start: str | None = None,
    backfill_end: str | None = None,
    database_instance_name: str = "",
) -> None:
    """Orquestra uma execução completa: resolve janela, computa, aplica o gate de
    qualidade, escreve (se passar), audita e sincroniza online (se configurado).
    Requer SparkSession real — exercitado via notebook (Task 12), não via pytest.
    database_instance_name só é usado (e obrigatório) quando spec.online=True."""
    window = resolve_window(spec, mode, today, backfill_start, backfill_end, spark)
    sources = {name: spark.table(name) for name in spec.sources}
    result_df = spec.compute_fn(sources, window)
    pandas_df = result_df.toPandas()

    findings = run_quality_gate(pandas_df, spec.entity_keys, spec.timestamp_key, window.end)
    table_name = resolve_table_name(catalog, spec.domain, spec.name, spec.table_name)

    if not gate_passed(findings):
        write_run(
            spark,
            RunRecord(
                component=COMPONENT,
                entity_name=spec.name,
                git_commit=git_commit,
                git_branch=git_branch,
                run_id=run_id,
                mode=mode.value,
                status="FAILED",
                window_start=window.start,
                window_end=window.end,
                run_ts=datetime.utcnow(),
            ),
        )
        raise QualityGateFailure(findings)

    write_feature_table(
        spark,
        result_df,
        table_name,
        spec.entity_keys,
        spec.timestamp_key,
        mode,
        partition_cols=spec.entity_keys[:1],
    )
    spark.sql(
        f"ALTER TABLE {table_name} SET TBLPROPERTIES "
        f"('git_commit' = '{git_commit}', 'git_branch' = '{git_branch}')"
    )

    write_run(
        spark,
        RunRecord(
            component=COMPONENT,
            entity_name=spec.name,
            git_commit=git_commit,
            git_branch=git_branch,
            run_id=run_id,
            mode=mode.value,
            status="SUCCESS",
            window_start=window.start,
            window_end=window.end,
            run_ts=datetime.utcnow(),
        ),
    )

    if spec.online:
        from .online_sync import sync_online_table

        sync_online_table(spark, table_name, spec.entity_keys, database_instance_name)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/engine.py tests/test_engine.py
git commit -m "feat: add engine orchestration (window resolution, gate, write, audit)"
```

---

## Task 10: Geração do resource DAB (`resource_gen.py`)

**Correção (2026-08-24, descoberta durante a validação real da Task 12):** o
`NOTEBOOK_PATH` original (`"../notebooks/run_feature_table"`, sem extensão) faz o
Databricks CLI instalado (v1.10.0) rejeitar `databricks bundle validate` com
`notebook "notebooks/run_feature_table" not found. Did you mean
"notebooks/run_feature_table.py"?` — referências a notebook local exigem uma extensão
(`.py`/`.r`/`.scala`/`.sql`/`.ipynb`). Corrigido abaixo para
`"../notebooks/run_feature_table.py"`, com o teste correspondente já ajustado.

**Files:**
- Create: `src/feature_platform/resource_gen.py`
- Test: `tests/test_resource_gen.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_resource_gen.py
import pytest

from feature_platform.contract import feature_table, clear_registry
from feature_platform.resource_gen import generate_job_resource


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_generate_job_resource_creates_one_task_per_feature_table():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.b"])
    def feature_b(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    tasks = resource["resources"]["jobs"]["feature_pipeline"]["tasks"]
    task_keys = {t["task_key"] for t in tasks}

    assert task_keys == {"feature_a", "feature_b"}


def test_generate_job_resource_declares_job_parameters():
    resource = generate_job_resource(job_name="feature_pipeline")
    job = resource["resources"]["jobs"]["feature_pipeline"]
    param_names = {p["name"] for p in job["parameters"]}

    assert param_names == {"mode", "start_date", "end_date", "git_commit", "git_branch", "catalog"}


def test_generate_job_resource_adds_depends_on_edge():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def base_feature(sources, window):
        return None

    @feature_table(
        domain="exemplo",
        entity_keys=["customer_id"],
        timestamp_key="feature_ts",
        sources=["raw.b"],
        depends_on=["base_feature"],
    )
    def derived_feature(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    tasks = {t["task_key"]: t for t in resource["resources"]["jobs"]["feature_pipeline"]["tasks"]}

    assert "depends_on" not in tasks["base_feature"]
    assert tasks["derived_feature"]["depends_on"] == [{"task_key": "base_feature"}]


def test_generate_job_resource_points_notebook_task_to_relative_path():
    @feature_table(domain="exemplo", entity_keys=["customer_id"], timestamp_key="feature_ts", sources=["raw.a"])
    def feature_a(sources, window):
        return None

    resource = generate_job_resource(job_name="feature_pipeline")
    task = resource["resources"]["jobs"]["feature_pipeline"]["tasks"][0]

    assert task["notebook_task"]["notebook_path"] == "../notebooks/run_feature_table.py"
    assert task["notebook_task"]["base_parameters"]["feature_table"] == "feature_a"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.resource_gen'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/resource_gen.py
import yaml

from .contract import get_registry

NOTEBOOK_PATH = "../notebooks/run_feature_table.py"

_JOB_PARAMETERS = [
    {"name": "mode", "default": "incremental"},
    {"name": "start_date", "default": ""},
    {"name": "end_date", "default": ""},
    {"name": "catalog", "default": "${var.catalog}"},
    {"name": "git_commit", "default": "${var.git_commit}"},
    {"name": "git_branch", "default": "${var.git_branch}"},
]


def generate_job_resource(job_name: str = "feature_pipeline") -> dict:
    registry = get_registry()
    tasks = []
    for name, spec in registry.items():
        task = {
            "task_key": name,
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "base_parameters": {
                    "feature_table": name,
                    "mode": "{{job.parameters.mode}}",
                    "start_date": "{{job.parameters.start_date}}",
                    "end_date": "{{job.parameters.end_date}}",
                    "catalog": "{{job.parameters.catalog}}",
                    "git_commit": "{{job.parameters.git_commit}}",
                    "git_branch": "{{job.parameters.git_branch}}",
                },
            },
        }
        if spec.depends_on:
            task["depends_on"] = [{"task_key": dep} for dep in spec.depends_on]
        tasks.append(task)

    return {
        "resources": {
            "jobs": {
                job_name: {
                    "name": job_name,
                    "parameters": _JOB_PARAMETERS,
                    "tasks": tasks,
                }
            }
        }
    }


def write_job_resource(path: str, job_name: str = "feature_pipeline") -> None:
    resource = generate_job_resource(job_name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resource, f, sort_keys=False)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_resource_gen.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/resource_gen.py tests/test_resource_gen.py
git commit -m "feat: generate DAB job resource with one task per registered feature table"
```

---

## Task 11: Sync opcional para Lakebase (`online_sync.py`)

**Risco conhecido, documentado no spec (seção 11):** a API do Databricks SDK para
criar/sincronizar *synced tables* no Lakebase é recente e pode ter mudado desde o
treinamento deste plano. A implementação abaixo é a melhor tentativa com a API
documentada (`databricks.sdk.service.database`); **valide contra a versão instalada do
SDK antes de rodar em produção** (passo de verificação no final desta task).

> **Correção (2026-08-24, achada indiretamente — tentando deployar um endpoint online
> no `serving-platform`, que depende de uma feature table sincronizada):** o risco
> acima se confirmou, com uma assinatura diferente da assumida.
> `client.database.create_synced_database_table(name=..., spec=...)` não bate com a
> API real do `databricks-sdk` instalado (`>=0.30` na Free Edition atual) —
> `create_synced_database_table(self, synced_table: SyncedDatabaseTable)` recebe um
> **único objeto**, não `name=`/`spec=` soltos. Além disso, `SyncedDatabaseTable`
> exige `database_instance_name` — o Database Instance do Lakebase de destino —, que
> **não existia como parâmetro nenhum** na assinatura original de `sync_online_table`.
> Corrigido reescrevendo `sync_online_table` para montar `SyncedDatabaseTable(name=...,
> database_instance_name=..., spec=SyncedTableSpec(**build_synced_table_spec(...)))` e
> chamar `create_synced_database_table(synced_table=...)`. `database_instance_name`
> vira um parâmetro obrigatório de `sync_online_table` (e de `run_feature_table`, com
> default `""`, validado com `ValueError` claro se `spec.online=True` e o parâmetro
> vier vazio — falha rápida e legível em vez de um erro opaco do SDK).
>
> **Isso NÃO foi validado ao vivo** — nenhum Database Instance do Lakebase existe no
> workspace usado nesta sessão (provisionar um é uma decisão de infraestrutura à
> parte: recurso standalone, cobrado, ainda em Public Preview). A correção acima
> resolve o bug de assinatura confirmado via inspeção direta do SDK instalado
> (`inspect.signature`), mas a sincronização de ponta a ponta continua sem
> confirmação — ver `docs/superpowers/specs/2026-08-23-geracao-de-features-design.md`,
> se aplicável, ou o spec do `serving-platform`, emenda 1.4, para o achado completo.
> `notebooks/run_feature_table.py` e `resource_gen.py` **não foram alterados** — o
> exemplo permanece `online=False` (não exercita este caminho), e adicionar um widget
> `database_instance` ao job seria expandir a superfície de um caminho ainda não
> validável nesta sessão.

> **Correção (2026-08-24, branch `fix/online-sync-database-instance` — validação de
> ponta a ponta agora com um Database Instance real provisionado, `exemplo-lakebase`,
> `CU_1`/`node_count=1`):** o caminho de sincronização online foi exercitado ao vivo e
> revelou três bugs adicionais além do já documentado acima, todos corrigidos e
> confirmados com o synced table chegando a `SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE`:
>
> 1. **`scheduling_policy` precisa ser o enum, não a string pura.** `SyncedTableSpec`
>    chama `.value` em `scheduling_policy` dentro do seu `as_dict()` — passar a string
>    `"TRIGGERED"` direto quebra com `AttributeError: 'str' object has no attribute
>    'value'`. `build_synced_table_spec()` continua retornando a string pura (mantém o
>    teste original passando, sem depender do SDK), e `sync_online_table()` converte
>    para `SyncedTableSchedulingPolicy(spec_fields["scheduling_policy"])` só na hora de
>    montar o `SyncedTableSpec`.
> 2. **`logicalDatabaseName` é obrigatório para "standard catalog".** Erro real:
>    `InvalidParameterValue: logicalDatabaseName must be defined when creating synced
>    table in a standard catalog`. Isso exige um **Database Catalog** do Lakebase já
>    criado (mapeando um catalog UC para uma database lógica dentro do Database
>    Instance — `databricks database create-database-catalog <catalog> <instance>
>    <database> --create-database-if-not-exists`), e um novo parâmetro
>    `logical_database_name`, threaded exatamente como `database_instance_name`:
>    `sync_online_table()`, `run_feature_table()`, o widget do notebook, o job
>    parameter em `resource_gen.py`, e a variável em `databricks.yml`.
> 3. **A tabela de origem precisa de Change Data Feed habilitado.** Erro real:
>    `InvalidParameterValue: Change Data Feed must be enabled on the source table ...
>    Change Data Feed is required in order to support incremental updates from the
>    delta table.` Isso já estava previsto informalmente no pedido original da
>    plataforma ("também vamos precisar (...) habilitar o change data feed nas
>    tabelas") mas nunca tinha sido implementado. Corrigido em `writer.py`:
>    `write_feature_table()` ganhou um parâmetro `enable_cdf: bool = False`; quando
>    `True`, roda `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed =
>    true)` ao final da escrita (idempotente, seguro de repetir). `engine.py` passa
>    `enable_cdf=spec.online`.
>
> **Um quarto problema, não é bug de código:** o arquivo `resources/generated_*.job.yml`
> é gerado por `scripts/generate_resources.py` e **não é regenerado automaticamente**
> por `databricks bundle deploy` — é gitignored e precisa rodar o script antes de cada
> deploy que mude parâmetros do job (a CI do `mlops-platform` já faz isso
> condicionalmente; um deploy manual direto, como o desta sessão, esqueceu o passo na
> primeira tentativa e o parâmetro novo simplesmente não apareceu no job deployado,
> sem erro nenhum). Vale como lembrete operacional, não como mudança de código.
>
> **Confirmado ao vivo, ponta a ponta:** `databricks database get-synced-database-table
> workspace.exemplo_features.customer_transaction_features_online` retornou
> `detailed_state: SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE` após o backfill —
> `online=True` está funcional de ponta a ponta nesta plataforma. O exemplo
> (`examples/features.py`) foi atualizado para `online=True` na feature table
> `customer_transaction_features`, e `notebooks/run_feature_table.py` /
> `resource_gen.py` / `databricks.yml` agora expõem `database_instance_name`.
>
> **Correção (2026-08-24, achada validando o `serving-platform` contra este online
> sync — o synced table existir e sincronizar não bastou):** com a synced table
> criada usando `logical_database_name="exemplo_online"` (um Database Catalog dedicado,
> separado do catalog UC `workspace` onde a feature table vive), o deploy de um
> `model_serving_endpoint` com `FeatureLookup` automático falhou com "Online feature
> store setup failed" — genérico via API, mas o assistente "Diagnose error" da própria
> UI do Databricks revelou a causa real: **"Reading online tables whose Lakebase
> database differs from its catalog name is not supported."** Ou seja,
> `logical_database_name` não é um valor livre — **precisa ser sempre igual ao catalog
> UC da tabela de origem** (`workspace`, neste caso), senão o synced table sincroniza
> normalmente (esse bug não impede a criação nem a sincronização) mas fica invisível
> para a resolução automática de `FeatureLookup` do Model Serving. Corrigido removendo
> `logical_database_name` como parâmetro livre inteiramente: `sync_online_table()`
> agora deriva `logical_database_name = table_name.split(".")[0]` a partir da própria
> `table_name` — torna o mismatch estruturalmente impossível em vez de documentá-lo
> como uma responsabilidade do operador. `notebooks/run_feature_table.py` /
> `resource_gen.py` / `databricks.yml` voltaram a expor só `database_instance_name`.
> Um Database Catalog cujo `database_name` seja igual ao catalog UC (aqui,
> `databricks database create-database-catalog <nome> <instance> workspace
> --create-database-if-not-exists`) continua sendo pré-requisito de infraestrutura,
> assim como `database_instance_name`.
>
> **Confirmado ao vivo, ponta a ponta, incluindo o Model Serving endpoint:** depois da
> correção, um `model_serving_endpoint` de teste (`serving-platform`, `mode="online"`)
> foi deployado com sucesso (`DEPLOYMENT_READY`) e consultado com
> `{"customer_id": "c1", "reference_date": "2026-08-24"}` — **sem** `txn_count` nem
> `avg_ticket` no payload — retornando uma predição real, confirmando que o
> `FeatureLookup` foi resolvido automaticamente contra o Online Feature Store (Lakebase)
> em tempo de inferência. O endpoint e o Online Store de teste extra criado durante a
> investigação foram derrubados ao final para não gerar custo contínuo. Ver o plano do
> `serving-platform` (Task 9, Step 5) para o registro completo dessa verificação.

**Files:**
- Create: `src/feature_platform/online_sync.py`
- Test: `tests/test_online_sync.py`

- [ ] **Step 1: Escrever o teste** (usa um dublê de `WorkspaceClient`, não bate na rede)

```python
# tests/test_online_sync.py
from feature_platform.online_sync import build_synced_table_spec


def test_build_synced_table_spec_uses_table_name_as_source():
    spec = build_synced_table_spec("workspace.credito_features.score_features", primary_keys=["customer_id"])

    assert spec["source_table_full_name"] == "workspace.credito_features.score_features"
    assert spec["primary_key_columns"] == ["customer_id"]
    assert spec["scheduling_policy"] == "TRIGGERED"
```

- [ ] **Step 2: Rodar e confirmar falha**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_online_sync.py -v`
Expected: `ModuleNotFoundError: No module named 'feature_platform.online_sync'`

- [ ] **Step 3: Implementar**

```python
# src/feature_platform/online_sync.py
def build_synced_table_spec(table_name: str, primary_keys: list[str]) -> dict:
    """Monta a especificação de uma synced table para o Online Feature Store
    (Lakebase). Separado de sync_online_table() para ser testável sem SDK/rede."""
    return {
        "source_table_full_name": table_name,
        "primary_key_columns": primary_keys,
        "scheduling_policy": "TRIGGERED",
    }


def sync_online_table(spark, table_name: str, primary_keys: list[str], database_instance_name: str) -> None:
    """Cria ou sincroniza a synced table no Lakebase para a feature table informada.
    Requer databricks-sdk, um workspace real, e um Database Instance do Lakebase já
    provisionado (database_instance_name) — exercitado via notebook (Task 12), não via
    pytest. Sem um Database Instance existente, a chamada falha com um erro claro do
    próprio SDK/API, não silenciosamente."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.database import SyncedDatabaseTable, SyncedTableSpec

    if not database_instance_name:
        raise ValueError("sync_online_table requires a non-empty database_instance_name")

    client = WorkspaceClient()
    spec_fields = build_synced_table_spec(table_name, primary_keys)
    synced_table = SyncedDatabaseTable(
        name=f"{table_name}_online",
        database_instance_name=database_instance_name,
        spec=SyncedTableSpec(**spec_fields),
    )
    client.database.create_synced_database_table(synced_table=synced_table)
```

- [ ] **Step 4: Rodar e confirmar sucesso**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_online_sync.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/feature_platform/online_sync.py tests/test_online_sync.py
git commit -m "feat: add Lakebase synced table spec builder for online feature tables"
```

---

## Task 12: Notebook entrypoint, exemplo não-produtivo e bundle DAB

Esta task não tem TDD (é glue code + configuração), mas tem passos de verificação
concretos rodando no workspace Databricks Free Edition.

**Files:**
- Create: `examples/__init__.py`
- Create: `examples/features.py`
- Create: `notebooks/run_feature_table.py`
- Create: `databricks.yml`
- Create: `scripts/generate_resources.py`

- [ ] **Step 1: Criar o exemplo não-produtivo**

```python
# examples/__init__.py
```

```python
# examples/features.py
import pyspark.sql.functions as F

from feature_platform.contract import feature_table


@feature_table(
    domain="exemplo",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.transactions"],
    online=False,
)
def customer_transaction_features(sources, window):
    raw = sources["raw.transactions"]
    return (
        raw.filter((F.col("event_ts") >= F.lit(window.start)) & (F.col("event_ts") < F.lit(window.end)))
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("txn_count"),
            F.avg("amount").alias("avg_ticket"),
        )
        .withColumn("feature_ts", F.lit(window.end))
    )
```

- [ ] **Step 2: Criar o notebook entrypoint**

```python
# notebooks/run_feature_table.py
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
```

- [ ] **Step 3: Criar o script de geração de resources**

**Correção (2026-08-24, descoberta durante a validação real desta task):** a versão
original só inseria `src/` no `sys.path`, então `import examples.features` (pacote na
raiz do repo, não em `src/`) falhava com `ModuleNotFoundError: No module named
'examples'` ao rodar `python scripts\generate_resources.py` — porque o Python define
`sys.path[0]` como o diretório do próprio script (`scripts/`), não a raiz do repo, nem
o cwd. Corrigido abaixo inserindo também a raiz do repo no `sys.path`.

```python
# scripts/generate_resources.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import examples.features  # noqa: F401  (importa o exemplo para popular o registro)
from feature_platform.resource_gen import write_job_resource

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "resources" / "generated_feature_pipeline.job.yml"
    output_path.parent.mkdir(exist_ok=True)
    write_job_resource(str(output_path))
    print(f"resource written to {output_path}")
```

- [ ] **Step 4: Criar `databricks.yml`**

```yaml
bundle:
  name: feature-platform

include:
  - resources/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for feature tables and the audit table.
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

- [ ] **Step 5: Rodar o gerador de resources localmente e confirmar o YAML**

Run:
```powershell
.\.venv\Scripts\python.exe scripts\generate_resources.py
```
Expected: `resource written to .../resources/generated_feature_pipeline.job.yml`, e o
arquivo contém uma task `customer_transaction_features` sem `depends_on`.

- [ ] **Step 6: Validar o bundle contra o workspace Databricks Free Edition**

Run:
```powershell
databricks bundle validate -t dev
```
Expected: validação sem erros. Se falhar por causa do notebook precisar de
`raw.transactions` inexistente, isso é esperado nesta etapa — `validate` não executa
código, só valida a estrutura do bundle.

- [ ] **Step 7: Commit**

```bash
git add examples/ notebooks/ databricks.yml scripts/generate_resources.py resources/.gitkeep
git commit -m "feat: add non-productive example, notebook entrypoint, and DAB bundle root"
```

---

## Task 13: GitHub Actions — caller do reusable workflow (`mlops-platform`)

**Emenda (arquitetura de plataforma):** este repositório não mantém mais um workflow
de deploy inline — ele chama o reusable workflow centralizado em `mlops-platform`,
que roda testes, gera resources (se existir `scripts/generate_resources.py`, que é o
caso aqui) e faz `bundle deploy`, tudo num lugar só reaproveitado pelos quatro
componentes.

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

## Task 14: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Escrever o README**

```markdown
# feature-platform

Framework para produtizar feature tables no Databricks: contrato declarativo via
decorator, execução em modo incremental (batch) ou backfill, gate de qualidade
bloqueante, geração automática do job de orquestração, auditoria central e
sincronização opcional com o Online Feature Store (Lakebase).

Design completo em
[`docs/superpowers/specs/2026-08-23-geracao-de-features-design.md`](docs/superpowers/specs/2026-08-23-geracao-de-features-design.md).

## Como declarar uma feature table

```python
from feature_platform.contract import feature_table

@feature_table(
    domain="credito",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.transactions"],
    online=False,
)
def customer_transaction_features(sources, window):
    ...
```

Este repositório é um framework puro — não é onde domínios reais declaram suas
features. Instale este pacote no repositório do seu domínio
(`pip install git+https://github.com/ViniciusOtoni/feature-platform@vX.Y.Z`) e declare
o módulo lá. O nome da tabela é derivado automaticamente como
`<catalog>.<domain>_features.<nome_da_função>`. Convenção completa em
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

# Primeira execução de uma feature table nova precisa ser um backfill —
# o modo incremental exige um checkpoint prévio.
databricks bundle run feature_pipeline -t dev --params mode=backfill,start_date=2026-01-01,end_date=2026-08-23

# Execuções seguintes
databricks bundle run feature_pipeline -t dev --params mode=incremental
```

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

## Task 15: Verificação ponta a ponta no workspace

Esta task não modifica código — é a validação manual de que tudo funciona junto,
antes de considerar o componente pronto.

- [ ] **Step 1:** Confirmar que a tabela `raw.transactions` (ou equivalente) existe no
  catálogo `workspace` do workspace de destino, com colunas `customer_id`, `event_ts`,
  `amount`. Se não existir, criar uma tabela sintética mínima para o teste (reaproveitar
  o gerador de dados sintéticos da POC, `databricks-feature-lookup-poc/src/synthetic_data.py`,
  é uma opção razoável).

- [ ] **Step 2:** Rodar o backfill inicial e confirmar sucesso:
```powershell
databricks bundle run feature_pipeline -t dev --params mode=backfill,start_date=2026-01-01,end_date=2026-08-23
```
Expected: job `SUCCESS`; tabela `workspace.exemplo_features.customer_transaction_features`
criada; uma linha `SUCCESS` em `workspace.platform_audit.pipeline_runs`.

- [ ] **Step 3:** Rodar o modo incremental e confirmar que o checkpoint funciona:
```powershell
databricks bundle run feature_pipeline -t dev --params mode=incremental
```
Expected: job `SUCCESS`; nova linha em `platform_audit.pipeline_runs` com
`window_start` igual ao `window_end` do backfill anterior.

- [ ] **Step 4:** Forçar uma falha do gate de qualidade (ex.: apontar `entity_keys` para
  uma coluna que o `compute()` de teste deixa nula) e confirmar que **nada é escrito** na
  feature table, e que a auditoria registra `status=FAILED`.

- [ ] **Step 5:** Confirmar `TBLPROPERTIES` na tabela:
```sql
DESCRIBE TABLE EXTENDED workspace.exemplo_features.customer_transaction_features;
```
Expected: `git_commit` e `git_branch` presentes nas propriedades da tabela.

- [ ] **Step 6:** Se alguma feature table tiver sido marcada `online=True`, confirmar no
  Catalog Explorer que a synced table no Lakebase foi criada. Se a chamada do SDK em
  `online_sync.py` falhar por mudança de API (risco documentado na Task 11), registrar o
  erro exato e ajustar a assinatura contra a documentação atual do
  `databricks-sdk` antes de prosseguir — não silenciar o erro.

---

## Self-Review

**1. Cobertura do spec:** contrato via decorator (Task 6), incremental + backfill com
checkpointing (Tasks 4, 9), 1 task por feature table no job DAB (Task 10), gate
bloqueante (Task 5, aplicado em Task 9), versionamento por nova tabela (garantido pela
convenção de nomenclatura da Task 3 + processo documentado no spec — não há automação
para isso, é uma decisão humana de nomear uma tabela nova, então não há código adicional
a escrever), Online Feature Store via Lakebase (Task 11), auditoria central (Task 8),
tracking de git commit/branch (Tasks 9, 12, 13). Todas as seções do spec têm uma task
correspondente.

**2. Placeholders:** nenhum "TBD"/"TODO" nas tasks. O único ponto sinalizado como
incerto (assinatura exata do SDK para Lakebase synced tables, Task 11) tem uma
implementação concreta + um passo de verificação explícito para validar contra a versão
instalada — não é uma lacuna deixada em aberto, é um risco documentado com ação
associada.

**3. Consistência de tipos:** `FeatureTableSpec`, `DateRange`, `WriteMode`, `RunRecord`
e as assinaturas de `resolve_window`, `write_feature_table`, `run_feature_table` e
`get_last_success_checkpoint` foram conferidas para usar os mesmos nomes de campo e
ordem de parâmetros em todos os módulos que os consomem (contract → engine → writer/audit
→ notebook).

---

Plano completo e salvo em `docs/superpowers/plans/2026-08-23-geracao-de-features-implementation.md`. Duas opções de execução:

**1. Subagent-Driven (recomendado)** — eu disparo um subagente novo por task, reviso entre tasks, iteração rápida.

**2. Inline Execution** — executo as tasks nesta sessão usando executing-plans, execução em lote com checkpoints.

Qual abordagem?
