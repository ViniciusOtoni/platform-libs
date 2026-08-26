from platform_core.naming import validate_qualified_name


def derive_table_name(catalog: str, domain: str, function_name: str) -> str:
    schema = f"{domain}_features"
    return f"{catalog}.{schema}.{function_name}"


def validate_table_name(table_name: str) -> None:
    validate_qualified_name(table_name, kind="table_name")


def resolve_table_name(catalog: str, domain: str, function_name: str, table_name: str | None) -> str:
    if table_name is None:
        return derive_table_name(catalog, domain, function_name)
    validate_table_name(table_name)
    return table_name
