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
