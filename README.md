# O ciclo de ML escrito uma vez

**Atenção**: este repositório é o outro lado do comparativo que abri no [`exemplo-domain`](https://github.com/ViniciusOtoni/exemplo-domain). Lá eu mostro o que o cientista de dados vê; aqui eu mostro o que sustenta aquilo. Não vamos nos aprofundar em conceitos de Ciência de Dados como estatística etc. Peguei este recorte por ser o componente que eu de fato mantenho, e porque é nele que a diferença entre "ter processo" e "não ter processo" aparece em código.

Ressalto desde já que os cinco contextos estarem num pacote só é uma decisão de **apresentação**, e não uma recomendação de arquitetura. Em escala, cada componente com o seu próprio repositório é o desenho que eu defenderia, e a seção [Como escalar a solução](#como-escalar-a-solução) trata disso.

O mesmo ciclo estava escrito **quatro** vezes. O que mudou não foi o número de repositórios, foi passar a existir **um** contrato.

## Problema de reescrever o ciclo em cada componente

| Indicador | Valor | Onde se verifica |
| --- | --- | --- |
| Contextos no pacote | 5 | `src/mlplatform/` |
| Console scripts publicados | 11 | `[project.scripts]` |
| Testes | 280 | `tests/` |
| Suíte completa | 12,6 s | `pytest -q` |
| Infraestrutura para rodar a suíte | nenhuma | sem Spark, sem workspace, sem rede |

Dados do próprio repositório, medidos na versão `3.13.2` do wheel.

Dessa forma, podemos ter a visibilidade de que o ciclo inteiro cabe em um processo local e que o custo de mexer nele é de segundos, não de um cluster subindo. Não vou entrar no mérito das horas perdidas esperando o job falhar no ambiente, mas é fato que isso é um grande alerta para todo time de plataforma, levando em consideração que um dos braços mais relevantes é a velocidade com que ele corrige e devolve o componente para o domínio.

Dito isso, podemos levantar uma série de questionamentos:

* Será que o problema era o número de repositórios? (Acredito que não: o problema era a mesma regra existir em quatro versões, sem nada que dissesse qual delas valia.)

* Como o padrão se mantém quando cinco bundles de domínio dependem da mesma peça, estando ela em um repositório ou em cinco?

---

Esse problema de engenharia que foi abordado e muitos outros que estão atrelados à jornada de ML podem ser resolvidos com um contrato explícito, testável e versionado. Mas antes de explicar como resolver o problema, vamos entender qual é a situação oposta dessa ideação de framework.

## O problema de não ter um framework consolidado

O time de plataforma até consegue entregar o primeiro caso de uso, definir qual será o padrão de escrita da feature table, como o split será feito e onde o modelo será registrado. Porém, ele já começa a se deparar com alguns problemas:

* Onde mora a regra do split temporal? Ela foi copiada para cada repositório e agora existem versões diferentes da mesma regra.

* Como testo o ciclo sem subir um cluster e sem depender de um workspace?

* Como o cientista promove o modelo dele sem pedir para a plataforma escrever o job?

* Como sei que a feature usada no treino é a mesma que vai ser usada na inferência, seis meses depois?

Logo, tangibilizando esses questionamentos, podemos enxergar alguns problemas comuns em todo esse processo:

1. A lógica não é isolada da infraestrutura! Logo, para testar uma regra de negócio de três linhas é preciso **Spark**, credencial e rede, e o teste simplesmente deixa de ser escrito.

Também é comum a mesma regra existir em repositórios diferentes com pequenas variações. Isso pode causar um mismatch entre o que roda no treino e o que roda na inferência e dificilmente será notado.

Ressalto que essa duplicação não atrapalha apenas a manutenção do time de plataforma, além de ser mais custosa para a instituição: ela corrói justamente a confiança que a Feature Store tentou construir.

2. Falta de clareza no ponto de entrada. Cada componente descobre o domínio de um jeito e cada bundle é montado à mão. Idem para a promoção do pacote: não existe um padrão de release, então cada bundle pina uma coisa diferente e migrar um componente arrasta todos os outros.

Note que todos esses problemas são causados pela falta de fronteiras explícitas entre o que é do domínio, o que é padrão e o que é infraestrutura. Sem isso, quem escreve o próximo componente fica cego e faz da forma que acha ideal. Pensando em ações tomadas de forma individual e em escala, isso definitivamente é um grande problema e difícil de ser resolvido.

## Como a solução foi pensada

O objetivo é simples: o domínio declara o que é específico dele e o framework monta o job, executa o ciclo e gera o bundle. Porém não é trivial, e chegar lá exigiu decidir três coisas antes de escrever a primeira linha.

**Primeiro, o que é declaração e o que é execução.** O cientista escreve contrato: a função marcada, a dataclass de treino, a estrutura da tabela de saída e o alvo do monitor. Tudo o que vem depois — a orquestração, o YAML do job, os grants no Unity Catalog, o registro no MLflow — é do framework. Essa linha é a decisão mais importante do desenho, porque é ela que define o que o domínio consegue quebrar sozinho.

**Segundo, o mesmo esqueleto para os cinco contextos.** Features, training, serving, monitoring e core têm exatamente os mesmos arquivos, com os mesmos papéis. Isso não é simetria por estética: é o que permite que quem entendeu um contexto consiga abrir outro e saber onde procurar, e é também o que torna a separação em repositórios uma operação mecânica no dia em que ela fizer sentido.

**Terceiro, a lógica não conhece a infraestrutura.** O use case recebe portas por parâmetro e nunca constrói nada. Spark, MLflow e o SDK ficam confinados nos adapters. O ganho aqui é direto e mensurável: a suíte inteira roda em um processo local, e uma regra de negócio pode ser corrigida e verificada em segundos, sem cluster.

O ganho das três decisões juntas é deixar a fronteira visível: o que muda por domínio fica no domínio, o que é padrão fica no framework e o que é infraestrutura fica confinado nos adapters. Isso aplica consistência em toda a jornada e, por fim, também economiza custos para a instituição, já que a mesma correção deixa de ser refeita em cada componente.

## Como esse framework foi montado

![Arquitetura do ecossistema](docs/img/arquitetura.png)

Cada bloco desse desenho é um contexto do pacote, e nenhum deles é escrito pelo domínio.

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

Aqui eu quebrei o framework em contextos. O repositório de domínio, como o [`exemplo-domain`](https://github.com/ViniciusOtoni/exemplo-domain), apenas declara o que é dele e instala o pacote pinado por release:

```toml
dependencies = [
  "mlplatform @ https://github.com/ViniciusOtoni/platform-libs/releases/download/mlplatform-v3.13.2/mlplatform-3.13.2-py3-none-any.whl",
]
```

### Explicação sobre os componentes

**1. A forma de cada contexto:** os cinco contextos têm o mesmo esqueleto. Quem entende um, entende os outros.

| arquivo | responsabilidade |
| --- | --- |
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

**2. Como o framework acha o domínio:** não existe caminho de arquivo hardcoded. O repositório de domínio se declara no próprio `pyproject.toml`, e o framework descobre o módulo a partir daí:

```toml
[project.entry-points."mlplatform.domains"]
credito_features = "credito_features.configs"
```

**3. Os quatro contratos:** é tudo o que o cientista escreve.

A **feature table** é uma função marcada. O nome da função vira o nome da tabela.

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

O `online=True` é uma linha do contrato, e é o framework que transforma isso em pipeline de sync.

**Linhagem entre a Feature Table e a Synced Table**

![lineage-feature](docs/img/lineage-feature.png)

**Synced Table (Tabela no Lakebase)**

![lakabase-table](docs/img/lakabase.png)

O **treino** é uma dataclass. O algoritmo entra como classe, não como string, e os hiperparâmetros como lista de dicionários: cada combinação vira um run aninhado no MLflow.

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

Cada dicionário da lista vira um `child run`, e a comparação pela `metric` declarada é que promove o **champion**. Note na coluna `promoted_alias` que quem move o alias é o framework, não o cientista.

![runs-mlflow](docs/img/runs-mlflow.png)

O `promotion_alias` aceita `None`. É o que permite que o retreino disparado por drift registre um candidato sem mover o alias do champion.

Os `feature_lookups` também são gravados dentro do artefato, e é daí que sai a linhagem entre a feature table e a versão do modelo:

![lineage-model](docs/img/lineage-model.png)

O **serving batch** carrega a estrutura da tabela de saída, e essa parte é contrato, não construtor de dataframe:

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

Do lado online, o mesmo artefato responde sem receber feature nenhuma no request, porque o `FeatureLookup` viajou junto com o modelo:

![score-online](docs/img/score-online.png)

O **monitoramento** declara o alvo, as colunas e o limiar:

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

Do limiar e do cron declarados aqui saem o monitor, o baseline e o dashboard, sem o domínio escrever uma linha de SQL:

![dashboard-drift](docs/img/dashboard-drift.png)

**4. Console scripts:** onze, todos apontando para `entrypoints.py`.

| script | o que faz |
| --- | --- |
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

É assim que o job de um domínio aparece do outro lado: uma task só, apontando para o wheel `mlplatform`, sem notebook e sem a task de tabela master.

![score-batch](docs/img/score-batch.png)

### Qual foi o caminho da minha solução

Hoje meu framework está totalmente atrelado ao Databricks. Talvez com um certo viés pelo fato de a empresa em que eu trabalho utilizar a ferramenta, mas é fato que, com ela, conseguimos amarrar o ciclo inteiro em um artefato só: o wheel entra no bundle, o bundle vira job, o job resolve o entry point e o mesmo pacote ainda viaja dentro do modelo até o endpoint.

Ressalto que essa é a solução para o recorte que eu quis mostrar, com um domínio de exemplo em [`exemplo-domain`](https://github.com/ViniciusOtoni/exemplo-domain) e um time só mantendo os cinco contextos. Troque qualquer uma dessas duas condições e o desenho de entrega muda, ainda que o contrato não mude. É exatamente disso que trata a próxima seção.

## Como escalar a solução

Aqui os cinco contextos moram juntos por um motivo específico: a jornada precisava caber em um lugar só para ser apresentada de ponta a ponta. Não é o desenho que eu levaria para escala, e vale dizer por quê.

Cada contexto já é fechado. Ele tem o seu `contract.py`, as suas `ports.py`, os seus `usecases.py`, os seus `adapters.py`, o seu `resource_gen.py` e o seu `naming.py`, e não importa nenhum outro contexto. A fronteira que separaria os repositórios já existe; o que ela ainda não tem é uma fronteira de **entrega**.

O que cada componente ganharia ao virar repositório próprio:

| o que passa a ser dele | por que importa em escala |
| --- | --- |
| esteira de CI/CD própria | o lint e a suíte rodam sobre o que mudou, não sobre os cinco contextos |
| cadência de release própria | corrigir o monitor não obriga treino e serving a atravessar uma versão nova |
| template de bundle próprio | o job de features evolui sem tocar no YAML dos outros componentes |
| versionamento próprio | o domínio pina `features@2.4.0` e `serving@1.9.0`, e migra um por vez |
| dono explícito | o `CODEOWNERS` deixa de ser do pacote inteiro e passa a ser do componente |

E o que fica mais caro, que é a parte que raramente aparece nas apresentações:

* O que hoje é `core` vira um pacote publicado, e todo contexto passa a depender de uma versão dele. Mudança em `core` vira migração coordenada entre N repositórios, e não mais um import.

* Aparece uma matriz de compatibilidade. Alguém precisa responder se `training@3.2` funciona com `core@1.7`, e essa resposta tem que ser testada em algum lugar.

* A esteira compartilhada passa a ser obrigatória, não conveniente. Sem algo como o [`mlops-platform`](https://github.com/ViniciusOtoni/mlops-platform), cada repositório reinventa a sua CI e o padrão se dissolve exatamente onde ele deveria ser mais forte.

O corte, então, não é ideológico. Ele é o ponto em que os componentes passam a evoluir em ritmos diferentes e a ter donos diferentes: enquanto o mesmo time mexe nos cinco na mesma semana, o pacote único paga menos; quando cada componente tem o seu backlog e a sua janela de deploy, o repositório separado paga menos. O que não muda em nenhum dos dois cenários é o contrato, e é por isso que ele é a parte que este repositório trata como intocável.

## Como executar e testar

A suíte roda local, e é assim que se mexe no framework:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

```bash
ruff check --config ../mlops-platform/ruff-framework.toml .
```

Os fakes usados pelos testes ficam em `mlplatform.testing` e são publicados de propósito, para que o domínio também consiga testar o código dele sem subir cluster:

```python
from mlplatform.testing import FakeFeatureWriter, FakeSourceReader, InMemoryAuditStore
```

Os specs de design de cada contexto estão em `docs/<contexto>/superpowers/specs/`.

### Como o pacote chega no domínio

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

O merge na `main` cria a tag, publica a release e anexa o wheel. Cada bundle de domínio pina a URL exata dessa release, o que permite subir um componente sem arrastar os outros.

Note que a versão é pinada exata, nunca por range. Enquanto um pacote único serve cinco bundles, é isso que preserva a possibilidade de migrar um de cada vez. Repare no efeito colateral: uma correção que só toca `monitoring` gera uma versão nova do pacote inteiro, e os cinco bundles passam a estar atrás de uma release que, para quatro deles, não mudou nada.

## O que muda no fim

Fazendo uma alusão aos questionamentos levantados no começo: o problema era mesmo o número de repositórios? Não posso dizer que essa é a única leitura possível, mas tudo indica que era o mesmo ciclo escrito quatro vezes sem contrato, e o que mudou foi passar a existir um. Onde esse contrato mora, se em um repositório ou em cinco, é a decisão seguinte, e ela depende de escala.

### As perguntas do começo, respondidas

No início, levantei alguns questionamentos do time de plataforma. Todos já foram respondidos de forma mais detalhada em suas respectivas sessões [Explicação sobre os componentes](#explicação-sobre-os-componentes):

| A pergunta do começo | O que responde | A prova |
| --- | --- | --- |
| *Onde mora a regra, se ela foi copiada para todo lado?* | um contrato por contexto, todos com o mesmo esqueleto | [arquitetura](docs/img/arquitetura.png) · [linhagem da feature](docs/img/lineage-feature.png) |
| *Como testo o ciclo sem cluster?* | portas recebidas por parâmetro e fakes publicados | `mlplatform.testing` · 280 testes em 12,6 s |
| *Como o cientista promove o modelo sozinho?* | console scripts e bundle gerados a partir do contrato | [task do wheel](docs/img/score-batch.png) · [child runs e champion](docs/img/runs-mlflow.png) |
| *Como sei que treino e inferência usam a mesma feature?* | `FeatureLookup` gravado dentro do artefato | [linhagem do modelo](docs/img/lineage-model.png) · [score online](docs/img/score-online.png) |

Note que aqui reduzimos a duplicação da lógica, aumentamos a eficiência do ciclo de correção e temos uma maior confiança e maturidade nos processos. O trade-off é o que descrevi em [Como escalar a solução](#como-escalar-a-solução): manter os cinco contextos juntos foi uma escolha de apresentação, e ela cobra em release acoplada e em ponto único de mudança. Em escala, esse mesmo contrato caberia em cinco repositórios, cada um com a sua esteira e a sua cadência, mas isso é uma discussão para outro dia hahaha

### O ganho por persona

**Para o cientista:** ele escreve contrato, e não orquestração. Isso vale igual esteja o framework em um repositório ou em cinco, porque o que ele importa é o contrato.

**Para o time de plataforma:** uma fronteira por contexto que já está pronta para virar um repositório quando o componente pedir dono e cadência próprios.

**Para a instituição:** eficiência e controle. O componente novo já nasce padronizado; o ciclo de correção deixa de depender de cluster e passa a durar segundos.

Um framework não deixa o modelo mais inteligente. Ele faz com que o ciclo escrito uma vez seja exatamente o ciclo que roda em todos os domínios, esteja ele empacotado junto ou separado.
