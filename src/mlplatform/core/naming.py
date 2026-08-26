import re

_QUALIFIED_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def validate_qualified_name(name: str, *, kind: str = "name") -> None:
    """Valida a convenção '<catalog>.<schema>.<table>' (lowercase, digitos,
    underscore) compartilhada por feature tables e nomes de modelo. Endpoints de
    serving e chaves de monitoring seguem convenções próprias, não compartilhadas
    aqui."""
    if not _QUALIFIED_NAME_RE.match(name):
        raise ValueError(
            f"{kind} '{name}' does not match convention "
            "'<catalog>.<schema>.<table>' (lowercase letters, digits, underscore)"
        )


def derive_model_name(catalog: str, domain: str, model_name: str) -> str:
    """Nome totalmente qualificado do modelo no Unity Catalog.

    A convenção de sufixo de schema (`<domain>_models`) é regra da plataforma,
    e estava hardcodada como f-string em seis lugares — incluindo scripts do
    repositório de domínio, que não deveriam conhecê-la. Aqui ela existe uma vez.
    """
    return f"{catalog}.{domain}_models.{model_name}"


def derive_predictions_table_name(catalog: str, domain: str, model_name: str) -> str:
    return f"{catalog}.{domain}_predictions.{model_name}"
