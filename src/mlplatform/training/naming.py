from mlplatform.core.naming import validate_qualified_name


def derive_model_name(catalog: str, domain: str, model_name: str) -> str:
    schema = f"{domain}_models"
    return f"{catalog}.{schema}.{model_name}"


def validate_model_name(full_name: str) -> None:
    validate_qualified_name(full_name, kind="model name")
