import re

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def derive_model_name(catalog: str, domain: str, model_name: str) -> str:
    schema = f"{domain}_models"
    return f"{catalog}.{schema}.{model_name}"


def validate_model_name(full_name: str) -> None:
    if not _NAME_RE.match(full_name):
        raise ValueError(
            f"model name '{full_name}' does not match convention "
            "'<catalog>.<schema>.<model>' (lowercase letters, digits, underscore)"
        )
