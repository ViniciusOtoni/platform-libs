import pytest

from mlplatform.core import discovery
from mlplatform.core.discovery import DomainLoadError, load_config_module, load_domains


class _FakeEntryPoint:
    def __init__(self, name: str, on_load=None):
        self.name = name
        self._on_load = on_load or (lambda: object())
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._on_load()


def _patch_entry_points(monkeypatch, eps):
    monkeypatch.setattr(discovery, "entry_points", lambda group: list(eps))


def test_loads_every_installed_domain(monkeypatch):
    eps = [_FakeEntryPoint("a"), _FakeEntryPoint("b")]
    _patch_entry_points(monkeypatch, eps)

    assert load_domains() == ["a", "b"]
    assert all(ep.loaded for ep in eps)


def test_only_loads_the_requested_domain(monkeypatch):
    wanted, other = _FakeEntryPoint("wanted"), _FakeEntryPoint("other")
    _patch_entry_points(monkeypatch, [wanted, other])

    assert load_domains(only="wanted") == ["wanted"]
    assert other.loaded is False


def _explodes():
    raise ModuleNotFoundError("No module named 'sklearn'")


def test_a_broken_domain_does_not_take_down_the_others(monkeypatch, capsys):
    """O módulo de configs de training importa sklearn em escopo de módulo. Num
    ambiente de serving, onde sklearn pode não existir, sem isolamento um único
    domínio quebrado derrubaria TODO entry point daquele ambiente — inclusive os
    que não têm nada a ver com training."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("quebrado", _explodes), _FakeEntryPoint("saudavel")])

    assert load_domains() == ["saudavel"]
    assert "quebrado" in capsys.readouterr().out


def test_failure_is_raised_when_the_broken_domain_is_the_one_requested(monkeypatch):
    """Ignorar em silêncio aqui faria o job rodar sem as configs que ele pediu e
    falhar depois, com um KeyError sem relação aparente."""
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("quebrado", _explodes)])

    with pytest.raises(DomainLoadError, match="quebrado"):
        load_domains(only="quebrado")


def test_requesting_an_unknown_domain_says_the_wheel_may_be_missing(monkeypatch):
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("outro")])

    with pytest.raises(DomainLoadError, match="is the domain wheel installed"):
        load_domains(only="inexistente")


def test_config_module_escape_hatch_imports_by_path():
    load_config_module("mlplatform.features.contract")  # não deve levantar


def test_config_module_failure_names_the_module():
    with pytest.raises(DomainLoadError, match="nao.existe"):
        load_config_module("nao.existe")
