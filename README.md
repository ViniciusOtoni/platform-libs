# mlplatform

Framework único do ecossistema de MLOps no Databricks. Um pacote, quatro contextos:

```
src/mlplatform/
├── core/          # shared kernel: registry, audit, quality, naming, resources
├── features/      # geração de feature tables (janela, gate de qualidade, online sync)
├── training/      # treino declarativo (split temporal, hiperparâmetros, registro no UC)
├── serving/       # serving online (Model Serving) e batch (fe.score_batch)
└── monitoring/    # drift de features e de predições via Lakehouse Monitoring
```

Consumo por um repositório de domínio:

```
mlplatform @ git+https://github.com/ViniciusOtoni/platform-libs@mlplatform-v1.0.0
```

```python
from mlplatform.features.contract import feature_table
from mlplatform.training.contract import TrainingConfig
from mlplatform.serving.contract import ServingConfig
from mlplatform.monitoring.contract import MonitoringConfig
```

## Desenvolvimento

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

O lint usa a configuração central do [`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform):

```bash
ruff check --config ../mlops-platform/ruff.toml .
```

## Versionamento

Uma versão para o framework inteiro, em `pyproject.toml`. O CI bloqueia o merge se a
versão não tiver sido incrementada em relação à última tag; o merge na `main` publica
a tag `mlplatform-vX.Y.Z` e anexa o wheel ao GitHub Release.

Domínios pinam a versão **exata** (`@mlplatform-vX.Y.Z`), nunca um range — com um
pacote só, isso é o que preserva a capacidade de cada bundle subir em momento
diferente.

## Histórico

Este repositório consolidou cinco pacotes que antes viviam em repositórios separados
(`platform-core`, `feature-platform`, `training-platform`, `serving-platform`,
`monitoring-platform`). O histórico de commits de todos eles está preservado aqui.
A documentação de design de cada componente está em [`docs/`](docs/).
