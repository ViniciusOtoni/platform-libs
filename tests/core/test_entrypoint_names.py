"""Nomes de console script que o launcher de wheel do Databricks consegue gerar.

Ao rodar um `python_wheel_task`, o Databricks monta uma célula assim:

    entry = [ep for ep in metadata.distribution(pkg).entry_points if ep.name == "<nome>"]
    if entry:
        entry[0].load()()
    else:
        module = importlib.import_module(pkg)
        module.<nome>()          # <- o nome entra CRU, com hífens e tudo

A última linha é lixo — hífen não é identificador. Mas o Python compila a
célula inteira antes de executar qualquer coisa, então essa linha precisa ao
menos *parsear*, mesmo sendo um ramo morto que nunca roda.

E quase sempre ela parseia por acidente: `module.mlp-score-batch()` vira a
expressão `module.mlp - score - batch()`, uma cadeia de subtrações entre nomes.
Feio, mas sintaticamente válido.

O acidente para de funcionar quando um dos pedaços entre hífens é uma PALAVRA
RESERVADA. `mlp-fit-and-compare` virava `module.mlp - fit - and - compare()`, e
`and` é operador — SyntaxError na compilação da célula, antes de o ramo correto
ter chance de rodar. O job morria com um erro que não menciona nada do nosso
código.

Custou três execuções reais para achar. O teste é barato; a descoberta não foi.
"""

import ast
import pathlib
import tomllib

import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[2]


def _console_scripts() -> dict[str, str]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["scripts"]


@pytest.mark.parametrize("name", sorted(_console_scripts()))
def test_the_databricks_fallback_snippet_still_parses(name: str):
    """Reproduz literalmente a linha que o launcher gera."""
    try:
        ast.parse(f"module.{name}()")
    except SyntaxError:
        offenders = [p for p in name.split("-") if not p.isidentifier() or p in _KEYWORDS]
        pytest.fail(
            f"o entry point {name!r} quebra a célula que o Databricks gera para "
            f"o python_wheel_task: {offenders} não pode aparecer entre hífens "
            f"porque é palavra reservada do Python. Renomeie o script."
        )


_KEYWORDS = frozenset(__import__("keyword").kwlist)


def test_the_training_job_only_references_declared_scripts():
    """O gerador do job e o pyproject precisam concordar: um nome que só existe
    de um lado só falha em runtime, dentro do Databricks."""
    from mlplatform.training.resource_gen import _TASKS

    declared = set(_console_scripts())
    referenced = {entry_point for _task, entry_point, _deps in _TASKS}

    assert referenced <= declared, f"referenciados mas não declarados: {referenced - declared}"


# --------------------------------------------------------------------------
# Job parameters chegam em TODAS as tasks do job
# --------------------------------------------------------------------------


def test_every_training_entrypoint_accepts_every_job_parameter():
    """O Databricks injeta cada job parameter em TODAS as tasks do job, como
    `--nome=valor`. Uma task cujo parser não declara o argumento aborta com
    `SystemExit(2)` — sem o nome da flag no erro, o que esconde a causa.

    Já mordeu duas vezes: primeiro com `model_name` declarado em dois lugares
    (passado em dobro), agora com `promotion_alias` declarado só na última
    task. O gerador e os parsers precisam concordar, e nada além deste teste
    os obriga a isso.
    """
    from mlplatform.entrypoints import _training_parser
    from mlplatform.training.resource_gen import _JOB_PARAMETERS, _TASKS

    parser = _training_parser("qualquer")
    aceitos = {action.dest for action in parser._actions}
    injetados = {p["name"] for p in _JOB_PARAMETERS}

    faltando = injetados - aceitos
    assert not faltando, (
        f"o job injeta {sorted(faltando)} em todas as tasks, e o parser de "
        f"training não os declara — as {len(_TASKS)} tasks abortariam com SystemExit(2)"
    )


def test_every_serving_and_monitoring_job_parameter_is_accepted():
    """Mesma regra, nos outros componentes."""
    import argparse

    from mlplatform.entrypoints import _common_args
    from mlplatform.monitoring.resource_gen import _JOB_PARAMETERS as MONITORING_PARAMS
    from mlplatform.serving.resource_gen import _JOB_PARAMETERS as SERVING_PARAMS

    # Os parsers desses componentes acrescentam args próprios ao `_common_args`;
    # o que precisa estar coberto aqui é o que vem do job para todas as tasks.
    base = argparse.ArgumentParser()
    _common_args(base)
    comuns = {action.dest for action in base._actions}

    # Declarados pelos entrypoints específicos, e não pelo `_common_args`.
    # Cada um só chega a tasks que o entrypoint correspondente atende.
    especificos = {
        "model_name",  # score_batch, refresh_endpoint, evaluate_drift
        "target_type",  # evaluate_drift
        "endpoint_name",  # refresh_endpoint
        "github_token",  # evaluate_drift
        "retrain_repository",  # evaluate_drift
    }

    for nome, params in (("serving", SERVING_PARAMS), ("monitoring", MONITORING_PARAMS)):
        injetados = {p["name"] for p in params}
        faltando = injetados - comuns - especificos
        assert not faltando, f"{nome} injeta {sorted(faltando)} sem parser que os aceite"
