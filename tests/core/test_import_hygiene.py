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

import pathlib
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


def test_the_module_shipped_in_code_paths_imports_no_infrastructure_itself():
    """`code_paths=[pyfunc_model.__file__]` é importado pelo MLflow DENTRO do
    container do endpoint de serving, onde pyspark, delta e o SDK não existem.

    A checagem é sobre os imports do PRÓPRIO arquivo, não sobre o fecho
    transitivo: o módulo legitimamente importa mlflow (é uma subclasse de
    PythonModel), e o mlflow importa pyspark quando ele está disponível. Medir
    sys.modules aqui confundiria "este código exige" com "o mlflow aproveitou se
    tinha" — e reprovaria por um motivo que não é risco nenhum.

    Antes o code_paths apontava para o pacote inteiro do framework, o que hoje
    levaria os adapters junto — e aí sim o endpoint quebraria."""
    import ast

    from mlplatform.training import pyfunc_model

    tree = ast.parse(pathlib.Path(pyfunc_model.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])

    assert not roots & {"pyspark", "delta", "databricks"}
    # e não pode arrastar o resto do framework junto
    assert "mlplatform" not in roots


# --------------------------------------------------------------------------
# Dependências declaradas vs. dependências realmente importadas
# --------------------------------------------------------------------------

# Módulo de terceiro -> distribuição que o instala. O prefixo mais longo vence,
# porque `databricks` é um namespace partido entre duas distribuições
# diferentes: olhar só o nome de topo não distingue uma da outra.
_DISTRIBUTION_OF = {
    "databricks.feature_engineering": "databricks-feature-engineering",
    "databricks.sdk": "databricks-sdk",
    "mlflow": "mlflow",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

# Fornecidos pelo runtime do Databricks, nunca instalados por nós. Declará-los
# faria o pip tentar resolver Spark dentro do serverless.
_PROVIDED_BY_RUNTIME = ("pyspark", "delta")

# Ancorado no arquivo de teste, e não em `mlplatform.__file__`: sob install
# editable o `__file__` do pacote vem None, e só `__path__` funciona.
_REPO_ROOT = pathlib.Path(__file__).parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "src" / "mlplatform"


def _imported_modules() -> set[str]:
    """Todos os imports de terceiros do framework, inclusive os feitos dentro
    de funções — que são justamente os que escapam de qualquer verificação
    estática de dependência."""
    import ast

    modules: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module)
    return {
        m
        for m in modules
        if m.split(".")[0] not in sys.stdlib_module_names and not m.startswith("mlplatform")
    }


def _declared_distributions() -> set[str]:
    import re
    import tomllib

    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = data["project"]["dependencies"]
    return {re.split(r"[<>=!~\[ @]", spec, maxsplit=1)[0].strip().lower() for spec in specs}


def test_every_third_party_import_is_a_declared_dependency():
    """Segunda vez que esta classe de bug chega em produção.

    Primeiro foi o `databricks-sdk`, importado pelos adapters mas declarado só
    no extra `dev`. Depois o `databricks-feature-engineering`, que nem sequer
    era declarado — o job de treino morria com ModuleNotFoundError no primeiro
    import, e ninguém viu porque o job nunca tinha sido executado.

    Os dois passaram batido porque os adapters importam infra DENTRO dos
    métodos (o que é proposital: mantém `import mlplatform` leve para o
    container de serving). Isso significa que nenhum import de topo denuncia a
    falta, e a suíte inteira passa verde num ambiente de dev que já tem tudo
    instalado. Só a execução real reprova — e é cara.
    """
    declared = _declared_distributions()

    missing = {}
    for module in _imported_modules():
        if module.split(".")[0] in _PROVIDED_BY_RUNTIME:
            continue
        matches = [p for p in _DISTRIBUTION_OF if module == p or module.startswith(p + ".")]
        assert matches, (
            f"import de terceiro não mapeado: {module!r}. Adicione-o a "
            f"_DISTRIBUTION_OF ou a _PROVIDED_BY_RUNTIME — decidir isso é o "
            f"ponto do teste, não um detalhe de manutenção."
        )
        distribution = _DISTRIBUTION_OF[max(matches, key=len)]
        if distribution.lower() not in declared:
            missing[module] = distribution

    assert not missing, f"importado mas não declarado em pyproject: {missing}"
