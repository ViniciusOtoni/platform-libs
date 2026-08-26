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
