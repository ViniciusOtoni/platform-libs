import re

_ENDPOINT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def derive_predictions_table_name(catalog: str, domain: str, model_name: str) -> str:
    schema = f"{domain}_predictions"
    return f"{catalog}.{schema}.{model_name}"


def derive_endpoint_name(domain: str, model_name: str) -> str:
    return f"{domain}-{model_name}-serving"


def validate_endpoint_name(name: str) -> None:
    if not _ENDPOINT_NAME_RE.match(name):
        raise ValueError(
            f"endpoint name '{name}' does not match convention "
            "'<domain>-<model_name>-serving' (lowercase letters, digits, underscore, hyphen)"
        )
