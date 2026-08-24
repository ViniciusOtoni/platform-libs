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
implementação — validar ao vivo (Task 12 do plano de implementação). Se não funcionar,
só o notebook `evaluate_drift.py` precisa de redesenho; o contrato e a tabela central
continuam válidos.

## Secrets necessários no GitHub Actions

| Secret | Descrição |
|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks Free Edition |
| `DATABRICKS_TOKEN` | Personal access token com permissão de deploy |
