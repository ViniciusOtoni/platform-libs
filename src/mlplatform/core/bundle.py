"""Geração do `databricks.yml` completo.

O repositório de domínio deixa de versionar o bundle: ele declara
`conf/variables.yml` e a esteira materializa o resto. Antes, cada bundle novo era
uma cópia manual de um `databricks.yml` vizinho — e as cópias divergiam, como
tudo que se copia.

O gerado é efêmero (gitignorado, escrito em tempo de CI), o que também elimina o
risco de alguém editar um arquivo gerado e ter a edição sobrescrita em silêncio.
"""

from .settings import BundleSettings

# Descrições ficam com o framework: são explicação de um mecanismo da plataforma,
# não configuração do domínio. Repetidas à mão em cada bundle, divergiam.
_VARIABLE_DOCS = {
    "catalog": "Catálogo do Unity Catalog onde o componente escreve.",
    "git_commit": "SHA do commit do deploy, propagado aos parâmetros do job para auditoria.",
    "git_branch": "Branch do deploy, propagada aos parâmetros do job para auditoria.",
    "database_instance_name": (
        "Database Instance do Lakebase, usado só por feature tables com online=True."
    ),
}


def bundle_name(domain: str, component: str) -> str:
    return f"{domain}-{component}"


def generate_bundle(settings: BundleSettings, domain: str, component: str, wheel_name: str) -> dict:
    """Monta o databricks.yml inteiro.

    `wheel_name` é a chave do bloco `artifacts:` — precisa bater com o nome do
    pacote Python do domínio para a CLI achar o que buildar.
    """
    variables = {
        "catalog": {"description": _VARIABLE_DOCS["catalog"], "default": settings.catalog},
        "git_commit": {"description": _VARIABLE_DOCS["git_commit"], "default": "local"},
        "git_branch": {"description": _VARIABLE_DOCS["git_branch"], "default": "local"},
    }
    # Só declara a variável do Lakebase quando o domínio de fato configurou uma:
    # declarar sempre criaria uma variável vazia em todo bundle, inclusive nos que
    # não têm nenhuma feature table online.
    if settings.database_instance_name:
        variables["database_instance_name"] = {
            "description": _VARIABLE_DOCS["database_instance_name"],
            "default": settings.database_instance_name,
        }

    return {
        "bundle": {"name": bundle_name(domain, component)},
        "include": ["resources/*.yml"],
        "artifacts": {
            wheel_name: {
                "type": "whl",
                # A CLI builda e sobe o wheel do domínio como parte do deploy —
                # sem step manual e sem venv empacotada.
                "build": "python -m build --wheel",
                "path": ".",
            }
        },
        "variables": variables,
        "targets": settings.targets,
    }
