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
