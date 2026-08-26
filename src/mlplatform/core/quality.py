from dataclasses import dataclass

PASS = "PASS"
FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    """Resultado de uma checagem de qualidade, compartilhado pelos contextos.

    `violations` é opcional porque nem toda checagem tem uma contagem natural:
    `check_unique_keys` conta linhas duplicadas, mas `check_metric_is_finite` só
    responde sim/não. Nesses casos o campo fica `None` — deliberadamente, e não
    `0`, que num FAIL leria como "nenhuma violação" e mentiria sobre o resultado.
    """

    check: str
    status: str  # PASS ou FAIL
    violations: int | None = None
    detail: str = ""


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == PASS for f in findings)
