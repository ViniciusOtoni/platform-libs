"""Implementações reais dos ports do shared kernel.

Este é o único módulo do `core/` que conhece Spark. O import fica dentro dos
métodos, e não no topo: o artefato MLflow embarca o framework via `code_paths` e
o importa dentro do container do endpoint de serving, onde pyspark não existe.
Um import no topo aqui quebraria o endpoint em produção — sem erro em nenhum
teste, porque nenhum teste roda dentro daquele container.
"""

from datetime import date

from .audit import AUDIT_TABLE, RunRecord, to_row
from .uc import ensure_schema


class DeltaAuditStore:
    """AuditStore sobre uma tabela Delta em Unity Catalog."""

    def __init__(self, spark, table: str = AUDIT_TABLE):
        self._spark = spark
        self._table = table

    def append(self, record: RunRecord) -> None:
        # saveAsTable não cria o schema em Unity Catalog — sem isso a primeira
        # escrita falha com SCHEMA_NOT_FOUND.
        # `platform_audit` é da plataforma: recebe escrita de todos os
        # domínios e não é concedida a nenhum.
        ensure_schema(self._spark, self._table.rsplit(".", 1)[0])
        df = self._spark.createDataFrame([to_row(record)])
        mode = "append" if self._spark.catalog.tableExists(self._table) else "overwrite"
        df.write.format("delta").mode(mode).saveAsTable(self._table)

    def last_success_checkpoint(self, component: str, entity_name: str) -> date | None:
        import pyspark.sql.functions as F

        if not self._spark.catalog.tableExists(self._table):
            return None

        rows = (
            self._spark.table(self._table)
            .filter(
                (F.col("component") == component)
                & (F.col("entity_name") == entity_name)
                & (F.col("status") == "SUCCESS")
            )
            .orderBy(F.col("window_end").desc())
            .limit(1)
            .collect()
        )
        return rows[0]["window_end"] if rows else None
