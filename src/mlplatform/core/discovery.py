"""Descoberta dos módulos de configuração dos domínios.

Um repositório de domínio se declara no próprio `pyproject.toml`:

    [project.entry-points."mlplatform.domains"]
    exemplo_features = "exemplo_features.configs"

Importar esse módulo é o que dispara os decorators (`@feature_table`) e as
chamadas de registro. Antes isso era um `import` escrito à mão dentro de cada
notebook; com `python_wheel_task` não há notebook onde escrevê-lo.

Verificado ao vivo em serverless (2026-08-26): entry points de um wheel entregue
via `environments[].spec.dependencies` são visíveis a `importlib.metadata`, e o
`load()` dispara o efeito colateral de import.
"""

import importlib
from importlib.metadata import entry_points

DOMAIN_ENTRY_POINT_GROUP = "mlplatform.domains"


class DomainLoadError(Exception):
    """O domínio pedido existe mas não pôde ser importado."""


def load_domains(only: str | None = None) -> list[str]:
    """Importa os módulos de configuração declarados pelos domínios instalados.

    Um domínio que falha ao importar NÃO derruba os demais. Isso importa porque
    o módulo de configs de training importa sklearn em escopo de módulo: num
    ambiente de serving, onde sklearn pode não existir, um único try/except
    ausente faria todo entry point daquele ambiente falhar — inclusive os que
    não têm nada a ver com training.

    A falha só é propagada quando o domínio que quebrou é justamente o que foi
    pedido em `only`. Caso contrário vira um aviso: o trabalho pedido pode
    prosseguir.
    """
    loaded: list[str] = []
    skipped: list[tuple[str, Exception]] = []

    for ep in entry_points(group=DOMAIN_ENTRY_POINT_GROUP):
        if only is not None and ep.name != only:
            continue
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception as exc:  # noqa: BLE001 — um domínio quebrado não derruba os outros
            skipped.append((ep.name, exc))

    if only is not None and not loaded:
        if skipped:
            name, exc = skipped[0]
            raise DomainLoadError(f"domain '{name}' failed to import: {exc}") from exc
        raise DomainLoadError(
            f"no domain registered under entry point '{only}' in group "
            f"'{DOMAIN_ENTRY_POINT_GROUP}' — is the domain wheel installed?"
        )

    for name, exc in skipped:
        print(f"[mlplatform] aviso: domínio '{name}' ignorado ({type(exc).__name__}: {exc})", flush=True)

    return loaded


def load_config_module(dotted_path: str) -> None:
    """Escape hatch explícito: importa um módulo de configs pelo caminho.

    Existe para depuração e para casos em que a descoberta automática não serve.
    Se algum dia virar o caminho principal, isso é sinal de que a declaração por
    entry point não está funcionando — e o problema é esse, não a falta deste.
    """
    try:
        importlib.import_module(dotted_path)
    except Exception as exc:  # noqa: BLE001 — reembrulha com contexto acionável
        raise DomainLoadError(f"could not import config module '{dotted_path}': {exc}") from exc
