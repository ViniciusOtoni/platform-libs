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
    "reader_group": "Grupo do workspace que recebe leitura nos schemas do domínio.",
    "retrain_repository": (
        "Repositório owner/repo avisado por repository_dispatch quando há drift."
    ),
}

# Nível concedido ao grupo do domínio sobre os recursos do bundle.
#
# CAN_RUN, e não CAN_MANAGE: o time precisa ver os jobs e disparar uma execução,
# mas a definição vem do git. Poder editar o job pela UI criaria divergência
# silenciosa entre o que está deployado e o que está versionado — e o próximo
# deploy sobrescreveria a edição sem avisar.
#
# O DABs aceita só CAN_MANAGE, CAN_VIEW e CAN_RUN aqui. CAN_MANAGE_RUN existe na
# API de jobs, não no bundle — e o `bundle validate` reprova.
_RESOURCE_PERMISSION_LEVEL = "CAN_RUN"


def bundle_name(domain: str, component: str, domain_package: str | None = None) -> str:
    """Nome do bundle.

    Deriva de `domain_package` quando ele existe, porque domínio+componente NÃO
    é único: um mesmo componente pode ter mais de um bundle — serving tem batch e
    online. Os dois virariam "exemplo-serving", colidiriam no mesmo caminho do
    workspace e um sobrescreveria o outro no deploy.

    O nome do entry point é único por bundle por construção, e a conversão
    underscore→hífen reproduz exatamente os nomes que já existiam
    (exemplo_serving_batch -> exemplo-serving-batch).
    """
    if domain_package:
        return domain_package.replace("_", "-")
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

    # Declarada SEMPRE, mesmo vazia: os jobs referenciam `${var.reader_group}`
    # nos parâmetros, e uma referência a variável inexistente faz o
    # `bundle validate` falhar. Vazia significa "não conceda nada".
    variables["reader_group"] = {
        "description": _VARIABLE_DOCS["reader_group"],
        "default": settings.reader_group,
    }
    variables["retrain_repository"] = {
        "description": _VARIABLE_DOCS["retrain_repository"],
        "default": settings.retrain_repository,
    }

    bundle: dict = {
        "bundle": {"name": bundle_name(domain, component, settings.domain_package)},
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

    # `permissions` no topo vale para todos os recursos do bundle — jobs e
    # endpoints —, então um domínio novo não precisa lembrar de conceder por
    # recurso.
    if settings.reader_group:
        bundle["permissions"] = [
            # A identidade que deploya precisa estar aqui explicitamente. Sem
            # ela o `bundle validate` avisa que CAN_MANAGE só se aplica quando o
            # deploy sai dessa mesma identidade — e em CI ele sai de um service
            # principal, não de quem escreveu o bundle. `current_user` resolve
            # para qualquer uma das duas.
            {"user_name": "${workspace.current_user.userName}", "level": "CAN_MANAGE"},
            {"group_name": settings.reader_group, "level": _RESOURCE_PERMISSION_LEVEL},
        ]
    return bundle
