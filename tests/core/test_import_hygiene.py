"""Guarda o risco de produção mais silencioso do framework.

O `register_model` chama `fe.log_model(..., code_paths=[...])`, o que empacota o
fonte do framework dentro do artefato MLflow. O MLflow então **importa esse
pacote dentro do container do endpoint de serving**, onde pyspark, delta e o
databricks-sdk não estão instalados.

Ou seja: se `import mlplatform` puxar infraestrutura transitivamente, o endpoint
de serving quebra em produção — e nenhum teste comum pega, porque nenhum teste
roda dentro daquele container.

Hoje passa porque os adapters importam infra dentro dos métodos, não no topo.
Este teste existe para falhar no dia em que alguém "arrumar" isso movendo os
imports para o topo, ou adicionar uma fachada em `__init__.py` que importe
adapters.
"""

import subprocess
import sys

INFRA_ROOTS = ("pyspark", "delta", "databricks", "mlflow", "sklearn")

_PROBE = """
import sys
import {module}
found = sorted({{m.split('.')[0] for m in sys.modules}} & {roots})
print(','.join(found))
"""


def _infra_pulled_by(module: str) -> list[str]:
    """Roda num subprocesso limpo: nesta sessão o pytest já importou pandas,
    mlflow e afins, então medir `sys.modules` aqui dentro não diria nada."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, roots=set(INFRA_ROOTS))],
        capture_output=True,
        text=True,
        check=True,
    )
    return [m for m in result.stdout.strip().split(",") if m]


def test_importing_the_package_root_pulls_no_infrastructure():
    assert _infra_pulled_by("mlplatform") == []


def test_importing_the_shared_kernel_pulls_no_infrastructure():
    assert _infra_pulled_by("mlplatform.core.audit") == []


def test_importing_the_domain_contract_pulls_no_infrastructure():
    """É o que o domínio importa para declarar suas feature tables, e é o que
    acaba dentro do artefato MLflow junto com o modelo."""
    assert _infra_pulled_by("mlplatform.features.contract") == []
