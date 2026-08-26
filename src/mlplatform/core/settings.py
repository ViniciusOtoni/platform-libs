"""Leitura do `conf/variables.yml` — o único arquivo de configuração que o
usuário do ecossistema escreve à mão.

O princípio: só entra aqui o que **varia de verdade por job e não dá para
derivar**. Identidade e versão ficam de fora de propósito:

- `component` vem de qual comando foi invocado;
- `domain` vem das próprias configs registradas;
- a versão do framework vem de `importlib.metadata`, que é a verdade sobre o que
  está instalado — uma cópia escrita à mão dessincroniza sem avisar, e foi
  exatamente o que aconteceu com as URLs de wheel hardcodadas em cada bundle.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = "conf/variables.yml"


@dataclass(frozen=True)
class BundleSettings:
    """Configuração declarada pelo domínio para um bundle."""

    catalog: str = "workspace"
    # Nome do entry point declarado no grupo `mlplatform.domains` do pyproject do
    # domínio. Não é derivável: só se descobre carregando, e carregar exige saber
    # qual. É a única identidade que o arquivo precisa carregar.
    domain_package: str | None = None
    database_instance_name: str = ""
    job_name: str | None = None
    targets: dict = field(default_factory=lambda: {"dev": {"mode": "development", "default": True}})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "BundleSettings":
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: esperado um mapeamento no topo, veio {type(data).__name__}")

        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            # Falhar alto aqui evita o modo de falha mais irritante de config em
            # YAML: uma chave com typo é ignorada em silêncio e o valor default
            # entra no lugar, sem nenhum sinal.
            raise ValueError(
                f"{path}: chaves desconhecidas {sorted(unknown)} — "
                f"aceitas: {sorted(cls.__dataclass_fields__)}"
            )
        return cls(**data)
