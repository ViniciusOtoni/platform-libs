"""Implementações reais dos ports de training."""

from typing import Any

import pandas as pd

from mlplatform.core.uc import ensure_schema


class FeatureEngineeringTrainingSet:
    def __init__(self, spark):
        self._spark = spark

    def build(self, spine_table: str, config: Any) -> Any:
        from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

        return FeatureEngineeringClient().create_training_set(
            df=self._spark.table(spine_table),
            feature_lookups=[
                FeatureLookup(
                    table_name=fl.table_name,
                    feature_names=fl.feature_names,
                    lookup_key=fl.lookup_key,
                    timestamp_lookup_key=fl.timestamp_lookup_key,
                )
                for fl in config.feature_lookups
            ],
            label=config.label_column,
        )

    def to_pandas(self, training_set: Any) -> pd.DataFrame:
        return training_set.load_df().toPandas()


class DeltaScratchStore:
    """Splits materializados em Delta entre as tasks.

    A área é privada do pipeline e reescrita inteira a cada execução — nada
    aqui sobrevive de propósito de um run para o outro.
    """

    def __init__(self, spark):
        self._spark = spark

    def write(self, df: pd.DataFrame, table_name: str) -> None:
        # `training_scratch` é área da plataforma, compartilhada entre
        # domínios: não recebe grant do grupo de nenhum deles.
        ensure_schema(self._spark, table_name.rsplit(".", 1)[0])
        # overwriteSchema: sem ele o Delta preserva o schema da tabela que já
        # existe e só troca os dados, falhando com DELTA_FAILED_TO_MERGE_FIELDS
        # quando o formato muda. E ele muda por motivos rotineiros: o domínio
        # adiciona uma feature ao FeatureLookup, ou um lookup point-in-time não
        # acha linha e o nulo resultante transforma um bigint em double no
        # roundtrip por pandas. Como a área é descartável, substituir o schema é
        # a semântica correta — preservá-lo é que era o acidente.
        (
            self._spark.createDataFrame(df)
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )

    def read(self, table_name: str) -> pd.DataFrame:
        return self._spark.table(table_name).toPandas()


class MlflowTracker:
    def start_run(self, experiment: str) -> str:
        import mlflow
        from databricks.sdk import WorkspaceClient

        # `set_experiment` cria o experiment, mas NÃO os diretórios do workspace
        # acima dele: com `/Shared/mlplatform/<dominio>` inexistente ele falha
        # com NOT_FOUND em vez de criar o caminho. Todo domínio novo cai nisso
        # na primeira execução, porque o diretório dele nunca existiu antes.
        # `mkdirs` cria a árvore inteira e é idempotente.
        WorkspaceClient().workspace.mkdirs(experiment.rsplit("/", 1)[0])

        mlflow.set_experiment(experiment)
        with mlflow.start_run() as run:
            return run.info.run_id

    def start_child_run(self, parent_run_id: str, name: str) -> str:
        """Cria um run aninhado sob o pai e devolve seu id.

        O `mlflow.parentRunId` é gravado no momento da criação, com o pai
        ativo — por isso o run pai é reaberto aqui. Depois disso o filho é
        endereçado só pelo id, e reabri-lo para logar não desfaz o vínculo.
        """
        import mlflow

        # O experiment vem do run pai, explicitamente. Cada task do job é um
        # PROCESSO próprio, e `set_experiment` só vale dentro do processo que o
        # chamou: aqui, em `fit_and_compare`, não há experiment ativo nenhum —
        # quem o definiu foi `prepare_training_set`, noutra task. Sem passá-lo,
        # a criação do filho vai com experiment_id=None e o MLflow responde
        # RESOURCE_DOES_NOT_EXIST. `nested=True` sozinho não herda isso.
        experiment_id = mlflow.get_run(parent_run_id).info.experiment_id

        # A ordem importa: o pai é entrado primeiro, e é a presença dele no
        # stack que faz `nested=True` gravar o vínculo.
        with (
            mlflow.start_run(run_id=parent_run_id),
            mlflow.start_run(
                run_name=name, nested=True, experiment_id=experiment_id
            ) as child,
        ):
            return child.info.run_id

    def log_params(self, run_id: str, params: dict, prefix: str = "") -> None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({f"{prefix}{k}": v for k, v in params.items()})

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(name, value)


# O que o container de serving precisa instalar para carregar o modelo.
#
# `mlplatform` NÃO está aqui, e essa é a razão de a lista existir. O MLflow
# infere as dependências a partir dos módulos do modelo, enxerga que a classe
# pyfunc vem da distribuição `mlplatform` e grava `mlplatform==<versão>` no
# requirements.txt do artefato. Só que o framework não está no PyPI — existe
# como wheel de Release no GitHub. O pip do container não resolve o nome, o
# build falha com `user_pip_resolution`, e o endpoint fica preso na versão
# anterior. Confirmado ao vivo: a v5 (sem a linha) serve; a v7 (com ela) não.
#
# E a dependência é redundante: `code_paths` já leva o fonte do módulo pyfunc
# para dentro do artefato, que é justamente como o container o carrega sem o
# framework instalado.
_SERVING_PACKAGES = (
    "mlflow",
    "cloudpickle",
    "scikit-learn",
    "numpy",
    "scipy",
    "pandas",
    "psutil",
)

# Resolve os FeatureLookup dentro do container. O cliente de Feature Engineering
# o acrescenta sozinho quando infere os requisitos; declarando a lista à mão,
# passa a ser nossa responsabilidade.
_FEATURE_LOOKUP_REQUIREMENT = "databricks-feature-lookup==1.*"


def pyfunc_code_path(model: Any) -> str:
    """Monta, num diretório temporário, a árvore de pacotes do módulo da classe
    pyfunc — e devolve a raiz dela para o `code_paths`.

    O cloudpickle serializa a classe POR REFERÊNCIA: o artefato guarda
    `mlplatform.training.pyfunc_model.FeaturePlatformModel`, e para carregar o
    modelo o container precisa importar exatamente esse caminho.

    Passar o arquivo solto não resolve — ele chega como `pyfunc_model.py`, sem
    os pacotes acima, e o import falha com "missing Python dependency". Foi o
    que aconteceu ao vivo: o build do container passou e o modelo não carregou.

    Passar o pacote inteiro também não serve: levaria os adapters junto, com
    pyspark e o SDK, para dentro de um container onde nada disso existe.

    Então reconstruímos só o esqueleto: `__init__.py` VAZIOS mais o módulo da
    classe. Os `__init__` são vazios de propósito — copiar os verdadeiros
    traria de volta o que estamos evitando.

    Também funciona para uma classe pyfunc do próprio domínio, que mora noutro
    pacote: a árvore é derivada do módulo da classe, não fixada no framework.
    """
    import os
    import shutil
    import sys
    import tempfile

    module_name = type(model).__module__
    module = sys.modules[module_name]
    parts = module_name.split(".")

    root = tempfile.mkdtemp(prefix="mlp_code_path_")
    package_dir = root
    for part in parts[:-1]:
        package_dir = os.path.join(package_dir, part)
        os.makedirs(package_dir, exist_ok=True)
        with open(os.path.join(package_dir, "__init__.py"), "w"):
            pass

    shutil.copy(module.__file__, os.path.join(package_dir, f"{parts[-1]}.py"))
    return os.path.join(root, parts[0])


def serving_pip_requirements() -> list[str]:
    """Pina as versões realmente instaladas no ambiente de treino.

    Pinar o que treinou, e não um intervalo, é o que garante que o container de
    serving carregue o pickle do sklearn com a mesma versão que o produziu —
    versões diferentes de scikit-learn nem sempre desserializam entre si.
    """
    import importlib.metadata as metadata

    requirements = []
    for package in _SERVING_PACKAGES:
        try:
            requirements.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            # Não instalado no ambiente de treino: o modelo não depende dele.
            continue
    requirements.append(_FEATURE_LOOKUP_REQUIREMENT)
    return requirements


class FeatureEngineeringPublisher:
    """Registra o modelo no Unity Catalog com a linhagem de features embutida."""

    def __init__(self, spark, reader_group: str | None = None):
        self._spark = spark
        # Grupo do domínio que recebe leitura nos schemas criados aqui. `None`
        # não concede — é o certo em workspace pessoal, onde não há grupo.
        self._reader_group = reader_group

    def publish(
        self,
        model: Any,
        training_set: Any,
        full_model_name: str,
        run_id: str,
        git_commit: str,
        git_branch: str,
    ) -> int:
        import mlflow
        from databricks.feature_engineering import FeatureEngineeringClient

        mlflow.set_registry_uri("databricks-uc")
        ensure_schema(self._spark, full_model_name.rsplit(".", 1)[0], self._reader_group)

        # code_paths leva só o esqueleto de pacotes da classe pyfunc — ver
        # `pyfunc_code_path`. Empacotar o framework todo levaria os adapters
        # junto, e o endpoint quebraria num import, sem nenhum teste reclamar,
        # porque nenhum teste roda dentro daquele container.
        with mlflow.start_run(run_id=run_id):
            FeatureEngineeringClient().log_model(
                model=model,
                artifact_path="model",
                flavor=mlflow.pyfunc,
                training_set=training_set,
                registered_model_name=full_model_name,
                code_paths=[pyfunc_code_path(model)],
                pip_requirements=serving_pip_requirements(),
            )
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("git_branch", git_branch)

        # A versão é encontrada pelo run que a produziu, e não por "a maior
        # versão do modelo": duas execuções concorrentes do pipeline
        # registrariam duas versões, e a maior poderia ser a da outra.
        client = mlflow.MlflowClient(registry_uri="databricks-uc")
        for version in client.search_model_versions(f"name='{full_model_name}'"):
            if version.run_id == run_id:
                return int(version.version)
        raise RuntimeError(f"nenhuma versão de {full_model_name} corresponde ao run {run_id}")

    def promote(self, full_model_name: str, version: int, alias: str) -> None:
        import mlflow

        mlflow.MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
            full_model_name, alias, version
        )


class DbutilsTaskChannel:
    """taskValues do Databricks, obtidos via runtime do SDK (não há notebook)."""

    def set(self, key: str, value: str) -> None:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key=key, value=value)

    def get(self, task_key: str, key: str) -> str:
        from databricks.sdk.runtime import dbutils

        return dbutils.jobs.taskValues.get(taskKey=task_key, key=key)
