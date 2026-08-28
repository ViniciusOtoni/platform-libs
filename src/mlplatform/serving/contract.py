from dataclasses import dataclass

from mlplatform.core.registry import Registry

from .structure import InferenceBatchStruct


@dataclass(frozen=True)
class BatchServingConfig:
    """Scoragem em lote agendada."""

    domain: str
    model_name: str
    spine_inference_table: str
    schedule_cron: str
    # Obrigatório de propósito: é a declaração do formato da tabela de saída, e
    # o ponto de tê-la é que nenhum domínio grave predições fora do padrão que
    # o monitoramento vai ler depois.
    output: InferenceBatchStruct
    alias: str = "champion"


@dataclass(frozen=True)
class OnlineServingConfig:
    """Endpoint de Model Serving."""

    domain: str
    model_name: str
    alias: str = "champion"


# Dois tipos em vez de um `ServingConfig` com `mode: Literal["online","batch"]`.
#
# O tipo único obrigava um `if mode` em dois lugares: no __post_init__, validando
# que batch tinha spine_inference_table e schedule_cron, e no gerador, decidindo
# se produzia um job ou um endpoint. Com tipos distintos, os campos obrigatórios
# do batch são obrigatórios pela própria assinatura — o erro passa a ser pego
# antes de rodar, e não numa exceção em runtime.
#
# Não é polimorfismo: batch produz uma entrada em `jobs`, online em
# `model_serving_endpoints`. São saídas de tipos diferentes, em buckets
# diferentes; forçá-las numa interface comum seria polimorfismo falso.
ServingConfig = BatchServingConfig | OnlineServingConfig

_registry: Registry[ServingConfig] = Registry(kind="serving config")


def register_serving_config(config: ServingConfig) -> None:
    _registry.register(config.model_name, config)


def get_serving_config(model_name: str) -> ServingConfig:
    return _registry.get(model_name)


def get_registry() -> dict[str, ServingConfig]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()


def batch_configs() -> dict[str, BatchServingConfig]:
    return {k: v for k, v in _registry.all().items() if isinstance(v, BatchServingConfig)}


def online_configs() -> dict[str, OnlineServingConfig]:
    return {k: v for k, v in _registry.all().items() if isinstance(v, OnlineServingConfig)}
