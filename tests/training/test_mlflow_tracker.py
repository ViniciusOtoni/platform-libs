"""O tracker não pode depender de estado ambiente do processo.

Cada task do job de treino é um processo próprio. `mlflow.set_experiment` vale
só dentro do processo que o chamou, e quem o chama é `prepare_training_set` —
enquanto quem cria os runs filhos é `fit_and_compare`, noutra task.

Na primeira execução real isso apareceu como `RESOURCE_DOES_NOT_EXIST: Could
not find experiment with ID None`: o filho era criado sem experiment porque
`nested=True` não herda o do pai quando não há experiment ativo.

O port promete endereçamento por run_id. Este teste cobra essa promessa do
adapter, com um mlflow de mentira — ele é o único ponto do adapter verificável
sem um workspace.
"""

import sys
import types

import pytest

from mlplatform.training.adapters import MlflowTracker


class _Run:
    def __init__(self, run_id, experiment_id):
        self.info = types.SimpleNamespace(run_id=run_id, experiment_id=experiment_id)


class _RunContext:
    def __init__(self, run):
        self._run = run

    def __enter__(self):
        return self._run

    def __exit__(self, *exc):
        return False


class _FakeMlflow:
    """Registra como `start_run` foi chamado, sem tocar em nada de verdade."""

    def __init__(self):
        self.calls: list[dict] = []
        self.runs = {"pai-1": _Run("pai-1", "exp-42")}
        self._next = 0

    def get_run(self, run_id):
        return self.runs[run_id]

    def start_run(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("nested"):
            self._next += 1
            return _RunContext(_Run(f"filho-{self._next}", kwargs.get("experiment_id")))
        return _RunContext(self.runs[kwargs["run_id"]])


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def test_the_child_run_carries_the_experiment_of_its_parent(fake_mlflow):
    """Explicitamente, e não pelo experiment ativo do processo — que noutra
    task simplesmente não existe."""
    MlflowTracker().start_child_run("pai-1", "combo_0")

    creation = [c for c in fake_mlflow.calls if c.get("nested")]
    assert len(creation) == 1
    assert creation[0]["experiment_id"] == "exp-42"


def test_the_parent_is_active_while_the_child_is_created(fake_mlflow):
    """A ordem é o que faz o `mlflow.parentRunId` ser gravado: sem o pai no
    stack, `nested=True` cria um run solto em vez de um filho."""
    MlflowTracker().start_child_run("pai-1", "combo_0")

    assert fake_mlflow.calls[0].get("run_id") == "pai-1"
    assert fake_mlflow.calls[1].get("nested") is True


def test_the_child_run_is_named_for_the_combination(fake_mlflow):
    MlflowTracker().start_child_run("pai-1", "combo_3")

    assert fake_mlflow.calls[1]["run_name"] == "combo_3"


# --------------------------------------------------------------------------
# Requisitos gravados no artefato do modelo
# --------------------------------------------------------------------------


def test_the_framework_is_never_a_serving_requirement():
    """O MLflow infere `mlplatform==<versão>` porque a classe pyfunc vem dessa
    distribuição. Só que o framework não está no PyPI — é wheel de Release no
    GitHub. O pip do container de serving não resolve o nome, o build morre com
    `user_pip_resolution`, e o endpoint fica preso na versão anterior.

    Confirmado ao vivo: a versão sem essa linha serve; a versão com ela não.
    """
    from mlplatform.training.adapters import serving_pip_requirements

    assert not [r for r in serving_pip_requirements() if r.startswith("mlplatform")]


def test_the_feature_lookup_client_is_required():
    """É o que resolve os FeatureLookup dentro do container. O cliente de
    Feature Engineering o acrescentava sozinho ao inferir os requisitos;
    declarando a lista à mão, passou a ser responsabilidade nossa."""
    from mlplatform.training.adapters import serving_pip_requirements

    assert any(r.startswith("databricks-feature-lookup") for r in serving_pip_requirements())


def test_every_requirement_is_pinned_to_what_actually_trained():
    """Versões diferentes de scikit-learn nem sempre desserializam o pickle uma
    da outra — o container precisa carregar com a versão que produziu."""
    from mlplatform.training.adapters import serving_pip_requirements

    unpinned = [r for r in serving_pip_requirements() if "==" not in r]

    assert unpinned == ["databricks-feature-lookup==1.*"] or not unpinned
