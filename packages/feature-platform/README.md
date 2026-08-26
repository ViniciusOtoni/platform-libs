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
