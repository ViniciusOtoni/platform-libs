"""Criação de schema e concessão de acesso no Unity Catalog.

Existia um `CREATE SCHEMA IF NOT EXISTS` solto em seis adapters — cada um
resolvendo o mesmo problema (`saveAsTable` não cria schema em UC) e nenhum
resolvendo o seguinte: quem consegue LER o que foi criado.

Numa plataforma multi-domínio isso importa. Os objetos nascem pertencendo ao
principal que rodou a esteira — um service principal de CI. O time do domínio
não enxerga a própria feature table, o próprio modelo, a própria tabela de
inferência. Funciona no dia da entrega e falha na primeira investigação.
"""

# Privilégios concedidos ao grupo do domínio, por tipo de schema.
#
# No schema, e não por objeto: os objetos de um domínio nascem ao longo do tempo
# — uma feature table nova, uma versão nova de modelo, a tabela de métricas que
# o monitor cria sozinho. Conceder por objeto exigiria lembrar de conceder a
# cada criação, e o esquecimento é silencioso. No schema, o que nascer depois já
# nasce acessível.
_READ_PRIVILEGES = ("SELECT", "EXECUTE")


def ensure_schema(spark, schema: str, reader_group: str | None = None) -> None:
    """Garante o schema e, se houver grupo, o acesso de leitura a ele.

    `schema` no formato `<catalog>.<schema>`. `reader_group` é o grupo do
    domínio; `None` mantém o comportamento anterior — criar e não conceder — que
    é o certo em workspace de desenvolvimento pessoal, onde não há grupo.
    """
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    if not reader_group:
        return

    catalog = schema.split(".")[0]
    # USE CATALOG e USE SCHEMA são pré-requisitos: sem eles o SELECT existe mas
    # o objeto continua invisível, que é o modo de falha mais confuso do UC.
    for statement in (
        f"GRANT USE CATALOG ON CATALOG {catalog} TO `{reader_group}`",
        f"GRANT USE SCHEMA ON SCHEMA {schema} TO `{reader_group}`",
        *(f"GRANT {p} ON SCHEMA {schema} TO `{reader_group}`" for p in _READ_PRIVILEGES),
    ):
        spark.sql(statement)


def grant_statements(schema: str, reader_group: str) -> list[str]:
    """Os comandos que `ensure_schema` emitiria. Existe para poder testá-los sem
    uma SparkSession — o adapter em si só é exercitado contra um workspace."""
    catalog = schema.split(".")[0]
    return [
        f"GRANT USE CATALOG ON CATALOG {catalog} TO `{reader_group}`",
        f"GRANT USE SCHEMA ON SCHEMA {schema} TO `{reader_group}`",
        *(f"GRANT {p} ON SCHEMA {schema} TO `{reader_group}`" for p in _READ_PRIVILEGES),
    ]
