# monitoring-platform

Componente de monitoramento do ecossistema de MLOps no Databricks: `MonitoringConfig`
sobre feature tables e tabelas de predições, baseline resolvida automaticamente a
partir de `platform_audit.pipeline_runs`, Lakehouse Monitoring nativo como motor de
cálculo (confirmado funcionando na Free Edition — ver "Risco conhecido" abaixo), e
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

**Confirmado ao vivo (2026-08-24): o Lakehouse Monitoring funciona na Databricks Free
Edition.** Um monitor `snapshot` foi criado com sucesso e produziu linhas reais
(`PASS` e `DRIFT_DETECTED`) em `platform_monitoring.drift_metrics`. Cinco ajustes
foram necessários para chegar lá — ver as notas "Correção" no plano de implementação,
Task 12.

**Item em aberto, não fechado nesta sessão:** a janela de baseline resolvida a partir
de `platform_audit.pipeline_runs` (`resolve_baseline_window`) ainda não é usada para
materializar uma baseline de verdade no monitor — o monitor `snapshot` hoje compara
cada refresh contra o anterior, não contra a janela de treino do modelo. Corrigir isso
exige uma decisão de design (provavelmente adicionar uma coluna de timestamp ao
`MonitoringConfig`) — ver spec, seção 11.

## Secrets necessários no GitHub Actions

| Secret | Descrição |
|---|---|
| `DATABRICKS_HOST` | URL do workspace Databricks Free Edition |
| `DATABRICKS_TOKEN` | Personal access token com permissão de deploy |
