# platform-libs

Frameworks do ecossistema de MLOps no Databricks, consolidados num único repositório:

- `packages/platform-core` — audit, naming e registry compartilhados entre os 4 frameworks.
- `packages/feature-platform` — geração de feature tables.
- `packages/training-platform` — treino declarativo.
- `packages/serving-platform` — serving online e batch.
- `packages/monitoring-platform` — monitoramento de drift.

Cada pacote é versionado e lançado de forma independente (tags `<pacote>-vX.Y.Z`), via os
reusable workflows hospedados em [`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform).
Consumo por repositórios de domínio:

```
feature-platform @ git+https://github.com/ViniciusOtoni/platform-libs@feature-platform-v0.2.0#subdirectory=packages/feature-platform
```
