import re

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def derive_table_name(catalog: str, domain: str, function_name: str) -> str:
    schema = f"{domain}_features"
    return f"{catalog}.{schema}.{function_name}"


def validate_table_name(table_name: str) -> None:
    if not _NAME_RE.match(table_name):
        raise ValueError(
            f"table_name '{table_name}' does not match convention "
            "'<catalog>.<schema>.<table>' (lowercase letters, digits, underscore)"
        )


def resolve_table_name(catalog: str, domain: str, function_name: str, table_name: str | None) -> str:
    if table_name is None:
        return derive_table_name(catalog, domain, function_name)
    validate_table_name(table_name)
    return table_name
