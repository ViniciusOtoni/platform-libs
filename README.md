# mlplatform

Framework do ecossistema de MLOps no Databricks. Um pacote Python, cinco contextos, 280 testes que rodam sem cluster.

O domínio declara o que é específico dele. O framework monta o job, executa o ciclo e gera o bundle.

```
src/mlplatform/
├── core/          registry, auditoria, naming, geração de bundle, grants no UC
├── features/      janela, gate de qualidade, escrita Delta, sync com o Lakebase
├── training/      split temporal, hiperparâmetros, seleção, registro no UC
├── serving/       score_batch e Model Serving, estrutura da tabela de saída
├── monitoring/    drift via Lakehouse Monitoring, baseline, gatilho de retreino
├── entrypoints.py os console scripts (composition root)
└── testing.py     fakes publicados, para os domínios testarem
```

Instalação num repositório de domínio, pinada por release:

```toml
dependencies = [
  "mlplatform @ https://github.com/ViniciusOtoni/platform-libs/releases/download/mlplatform-v3.13.0/mlplatform-3.13.0-py3-none-any.whl",
]
```

## A forma de cada contexto

Os cinco contextos têm o mesmo esqueleto. Quem entende um, entende os outros.

| arquivo | responsabilidade |
|---|---|
| `contract.py` | o que o domínio declara, como dataclass ou decorator |
| `ports.py` | os `Protocol` que a lógica consome |
| `usecases.py` | a lógica, sem nenhum import de infraestrutura |
| `adapters.py` | as implementações reais, contra Spark, MLflow e o SDK |
| `resource_gen.py` | o YAML do job que vai para o bundle |
| `naming.py` | nomes derivados (tabela, experimento, endpoint, job) |

O use case recebe as portas por parâmetro. Nada é construído lá dentro:

```python
def run_feature_table(
    spec: FeatureTableSpec,
    reader: SourceReader,
    writer: FeatureWriter,
    audit: AuditStore,
    clock: Clock,
    ...
) -> None:
```

Quem constrói é o `entrypoints.py`, e só ele. É onde a árvore de dependências é montada uma vez por processo, e é o único lugar do framework que conhece ao mesmo tempo a lógica e a infraestrutura.

O ganho prático é o tempo de teste. A suíte inteira roda em um processo local, sem Spark, sem workspace, sem rede.

## O contrato de import

Duas regras, e as duas valem mais do que parecem.

**Contextos não se importam entre si.** `features` não importa `training`, e assim por diante. O que for comum sobe para `core`. Com quatro repositórios separados isso era impossível por construção; num pacote só, custa um import, e voltaria em semanas se dependesse de code review.

**Infraestrutura só entra pelos adapters, e com o import dentro do método.** Não no topo do arquivo.

A segunda regra existe por um motivo que não é estilo. O `register_model` chama `fe.log_model(..., code_paths=[...])`, o que empacota o fonte do framework dentro do artefato MLflow. O MLflow então importa esse pacote dentro do container do endpoint de serving, onde pyspark, delta e o `databricks-sdk` não estão instalados.

Se `import mlplatform` puxar infraestrutura transitivamente, o endpoint quebra em produção, e nenhum teste comum pega, porque nenhum teste roda dentro daquele container.

As duas regras são verificadas de dois lados. O lint recusa o import:

```toml
[lint.flake8-tidy-imports.banned-api]
"mlplatform.features".msg = "bounded contexts não se importam entre si, compartilhe via mlplatform.core"
"pyspark".msg = "infraestrutura só em adapters.py, e com o import dentro do método"
```

E um teste mede o que de fato foi carregado, num subprocesso limpo:

```python
INFRA_ROOTS = ("pyspark", "delta", "databricks", "mlflow", "sklearn")

_PROBE = """
import sys
import {module}
found = sorted({{m.split('.')[0] for m in sys.modules}} & {roots})
print(','.join(found))
"""
```

O lint pega a intenção, o teste pega o efeito. Um import indireto, três níveis abaixo, escapa do primeiro e não escapa do segundo.

## Descoberta do domínio

O framework não tem caminho de arquivo hardcoded para o domínio. O repositório se declara no próprio `pyproject.toml`:

```toml
[project.entry-points."mlplatform.domains"]
credito_features = "credito_features.configs"
```

Importar esse módulo dispara os decorators e as chamadas de registro. Antes isso era um `import` escrito à mão dentro de cada notebook; com `python_wheel_task` não existe notebook onde escrevê-lo.

Isso foi verificado ao vivo em serverless: entry points de um wheel entregue via `environments[].spec.dependencies` são visíveis a `importlib.metadata`, e o `load()` dispara o efeito colateral de import.

## Os quatro contratos

**Feature table** é uma função marcada. O nome da função vira o nome da tabela.

```python
@feature_table(
    domain="credito",
    entity_keys=["customer_id"],
    timestamp_key="feature_ts",
    sources=["raw.credito_posicoes"],
    online=True,
)
def perfil_credito_cliente(sources, window):
    ...
```

**Treino** é uma dataclass. O algoritmo entra como classe, não como string, e os hiperparâmetros como lista de dicionários: cada combinação vira um run aninhado no MLflow.

```python
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
    promotion_alias: str | None = "champion"
```

O `promotion_alias` aceita `None`. É o que permite que o retreino disparado por drift registre um candidato sem mover o alias do champion.

**Serving batch** carrega a estrutura da tabela de saída, e essa parte é contrato, não construtor de dataframe:

```python
@dataclass(frozen=True)
class InferenceBatchStruct:
    primary_key: list[str]
    ts_date: str
    predict_cols: list[str]
    feature_cols: list[str] = field(default_factory=list)
    label_col: str | None = None
```

A separação entre `ts_date` (a safra de referência) e `scored_at` (o instante da execução, que o framework grava sozinho) é o que permite reprocessar uma safra antiga sem que ela se confunda com a corrente. As `feature_cols` ficam gravadas porque são a base do data drift: sem elas, comparar safras exigiria refazer o join do `FeatureLookup` a posteriori, contra feature tables que já mudaram.

**Monitoramento** declara o alvo, as colunas e o limiar:

```python
@dataclass
class MonitoringConfig:
    domain: str
    model_name: str
    target_type: Literal["feature_table", "predictions"]
    target_table: str
    columns: list[str]
    threshold: float
    schedule_cron: str
    drift_metric: str = DEFAULT_DRIFT_METRIC
```

## Console scripts

Onze, todos apontando para `entrypoints.py`.

| script | o que faz |
|---|---|
| `mlp-run-feature-table` | executa uma feature table, em modo incremental ou backfill |
| `mlp-prepare-training-set` | monta o conjunto de treino com `FeatureLookup` |
| `mlp-fit-compare` | treina cada combinação de hiperparâmetros num run aninhado |
| `mlp-select-test-register` | escolhe o vencedor, testa uma vez e registra no UC |
| `mlp-score-batch` | pontua a carteira e grava a tabela de predições |
| `mlp-refresh-endpoint` | cria ou atualiza o endpoint de Model Serving |
| `mlp-evaluate-drift` | roda o monitor, lê a métrica e decide o veredito |
| `mlp-model-version` | resolve a versão por trás de um alias |
| `mlp-promote-model` | move o alias para a versão aprovada |
| `mlp-generate-resources` | gera o YAML dos jobs |
| `mlp-generate-bundle` | materializa o bundle DAB inteiro |

O nome tem uma restrição que não é óbvia. Ao rodar um `python_wheel_task`, o Databricks monta uma célula assim:

```python
entry = [ep for ep in metadata.distribution(pkg).entry_points if ep.name == "<nome>"]
if entry:
    entry[0].load()()
else:
    module = importlib.import_module(pkg)
    module.<nome>()          # o nome entra cru, com hífens e tudo
```

A última linha é lixo, porque hífen não é identificador. Mas o Python compila a célula inteira antes de executar qualquer coisa, então ela precisa ao menos parsear, mesmo sendo um ramo morto.

E quase sempre parseia por acidente: `module.mlp-score-batch()` vira a expressão `module.mlp - score - batch()`, uma cadeia de subtrações entre nomes. Feio, mas válido.

O acidente para de funcionar quando um dos pedaços entre hífens é uma palavra reservada. Foi o que derrubou `mlp-fit-and-compare`, que virou `mlp - fit - and - compare()` e não compila. Daí o nome atual, e daí um teste que tenta compilar a linha gerada para cada script declarado.

## Testes

280 testes, nenhum precisa de cluster.

Os fakes ficam em `mlplatform.testing` e são publicados de propósito. Sem isso, cada domínio escreveria os seus, e um fake permissivo esconde bug: o `FakeExperimentTracker` levanta `ImmutableParamError` na reescrita de um parâmetro porque o MLflow real também levanta.

```python
from mlplatform.testing import FakeFeatureWriter, FakeSourceReader, InMemoryAuditStore
```

Três testes existem para pegar falha que só apareceria em produção:

| teste | o que guarda |
|---|---|
| `test_import_hygiene` | o container de serving quebrando por import transitivo de infra |
| `test_entrypoint_names` | o nome de console script que o launcher do Databricks não consegue compilar |
| `test_resource_gen` (por contexto) | job parameter declarado num gerador e não aceito pelo parser |

O terceiro merece nota. O Databricks injeta todo job parameter em todas as tasks do job, não só na que o declarou. Um argumento que um parser não conhece aborta a task com `SystemExit(2)`, sem dizer qual argumento foi.

## Release

A CI roda no PR contra a `main`, consumindo a esteira compartilhada do [`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform):

```yaml
jobs:
  ci:
    uses: ViniciusOtoni/mlops-platform/.github/workflows/ci-validate.yml@main
    with:
      working-directory: .
      ruff-config: ruff-framework.toml
    secrets: inherit
```

O `ruff-framework.toml` é o contrato de arquitetura interna descrito acima. Repositórios de domínio ficam no `ruff.toml` puro, porque eles importam pyspark e o SDK legitimamente e as regras de import misfirariam lá.

O merge na `main` cria a tag, publica a release e anexa o wheel. Cada bundle de domínio pina a URL exata dessa release, o que permite subir um componente sem arrastar os outros.

A versão é pinada exata, nunca por range. Com um pacote único servindo cinco bundles, é isso que preserva a possibilidade de migrar um de cada vez.

## Desenvolvimento

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

```bash
ruff check --config ../mlops-platform/ruff-framework.toml .
```

Os specs de design de cada contexto estão em `docs/<contexto>/superpowers/specs/`.
