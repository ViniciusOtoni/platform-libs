from dataclasses import dataclass
from typing import Literal

from mlplatform.core.registry import Registry

from .metrics import DEFAULT_DRIFT_METRIC, resolve


@dataclass
class MonitoringConfig:
    domain: str
    model_name: str
    target_type: Literal["feature_table", "predictions"]
    target_table: str
    columns: list[str]
    threshold: float
    schedule_cron: str
    # Qual das métricas da tabela de drift do monitor decide o veredito. O
    # default é o único escalar preenchido para colunas numéricas E
    # categóricas; as demais vêm nulas fora do tipo delas, e uma métrica nula
    # produz "sem drift" para sempre.
    drift_metric: str = DEFAULT_DRIFT_METRIC
    # Coluna de tempo da tabela observada. Quando declarada, o monitor passa a
    # comparar contra a JANELA DE TREINO em vez de contra a janela anterior: o
    # framework materializa uma fatia da própria tabela nessa janela e a usa
    # como baseline.
    #
    # Opcional porque nem todo alvo tem baseline possível. A tabela de predições
    # não tem: não existem predições do período de treino — o modelo ainda não
    # existia. Para ela, comparar com a janela anterior é o que faz sentido.
    baseline_timestamp_column: str | None = None

    def __post_init__(self) -> None:
        metric = resolve(self.drift_metric)
        if not metric.bounded and self.threshold <= 0:
            # Métrica sem teto exige limiar positivo explícito: com <= 0 todo
            # valor dispara, e o gate vira ruído em vez de sinal.
            raise ValueError(
                f"'{self.drift_metric}' não é limitada a [0,1] ({metric.note}); "
                f"threshold precisa ser positivo, veio {self.threshold}"
            )


def _config_key(domain: str, model_name: str, target_type: str) -> str:
    return f"{domain}.{model_name}.{target_type}"


_registry: Registry[MonitoringConfig] = Registry(kind="monitoring config")


def register_monitoring_config(config: MonitoringConfig) -> None:
    key = _config_key(config.domain, config.model_name, config.target_type)
    _registry.register(key, config)


def get_monitoring_config(domain: str, model_name: str, target_type: str) -> MonitoringConfig:
    return _registry.get(_config_key(domain, model_name, target_type))


def get_registry() -> dict[str, MonitoringConfig]:
    return _registry.all()


def clear_registry() -> None:
    _registry.clear()
