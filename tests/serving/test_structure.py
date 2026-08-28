"""O contrato da tabela de saída da inferência batch.

Existe para a fase de monitoramento: data drift compara a distribuição das
features entre safras, model drift compara score contra desfecho. Nenhum dos
dois funciona se cada domínio gravar a saída num formato próprio — e nenhum dos
dois sobrevive a descobrir isso depois de meses de histórico no formato errado.
"""

import pandas as pd
import pytest

from mlplatform.serving.quality import run_structure_gate
from mlplatform.serving.structure import (
    MODEL_VERSION_COLUMN,
    SCORED_AT_COLUMN,
    InferenceBatchStruct,
)


def _struct(**o) -> InferenceBatchStruct:
    return InferenceBatchStruct(
        **{
            "primary_key": ["customer_id"],
            "ts_date": "reference_date",
            "feature_cols": ["txn_count", "avg_ticket"],
            "predict_cols": ["prediction"],
            **o,
        }
    )


def _frame(**o) -> pd.DataFrame:
    base = {
        "customer_id": [1, 2],
        "reference_date": ["2026-08-25", "2026-08-25"],
        "txn_count": [3, 5],
        "avg_ticket": [10.0, 20.0],
        "prediction": [0.1, 0.9],
        SCORED_AT_COLUMN: ["2026-08-26", "2026-08-26"],
        MODEL_VERSION_COLUMN: [7, 7],
    }
    base.update(o)
    return pd.DataFrame(base)


def _statuses(df, struct) -> dict[str, str]:
    return {f.check: f.status for f in run_structure_gate(df, struct)}


# -- a declaração em si -----------------------------------------------------


def test_the_framework_columns_are_required_without_being_declared():
    """`scored_at` e `model_version` são gravadas pelo framework: o domínio não
    as declara, mas a tabela tem que tê-las."""
    required = _struct().required_columns

    assert SCORED_AT_COLUMN in required
    assert MODEL_VERSION_COLUMN in required


def test_the_label_is_never_required():
    """O label não existe no momento da inferência — ele se materializa semanas
    depois. Exigi-lo aqui reprovaria toda execução."""
    struct = _struct(label_col="label_default")

    assert "label_default" not in struct.required_columns


@pytest.mark.parametrize(
    "kwargs, motivo",
    [
        ({"primary_key": []}, "sem chave não dá para identificar a linha"),
        ({"ts_date": ""}, "sem safra não dá para comparar entre períodos"),
        ({"predict_cols": []}, "uma tabela de inferência sem score não serve"),
    ],
)
def test_an_incomplete_declaration_is_rejected_at_import_time(kwargs, motivo):
    with pytest.raises(ValueError):
        _struct(**kwargs)


def test_a_column_cannot_play_two_roles():
    """Declarar a mesma coluna como feature e como score faria o data drift
    medir o próprio score como se fosse entrada."""
    with pytest.raises(ValueError, match="mais de um papel"):
        _struct(feature_cols=["txn_count", "prediction"])


def test_the_domain_cannot_claim_a_framework_column():
    with pytest.raises(ValueError, match="gravadas pelo framework"):
        _struct(predict_cols=["prediction", SCORED_AT_COLUMN])


# -- a validação da saída ---------------------------------------------------


def test_a_conforming_frame_passes_every_check():
    assert set(_statuses(_frame(), _struct()).values()) == {"PASS"}


def test_a_missing_declared_column_fails():
    """O caso que motiva o contrato: a coluna de safra nunca gravada, descoberto
    só quando o monitoramento tentar agrupar por ela."""
    df = _frame().drop(columns=["reference_date"])

    assert _statuses(df, _struct())["declared_columns_are_present"] == "FAIL"


def test_a_missing_framework_column_fails():
    df = _frame().drop(columns=[MODEL_VERSION_COLUMN])

    assert _statuses(df, _struct())["declared_columns_are_present"] == "FAIL"


def test_a_repeated_entity_in_the_same_batch_fails():
    """Sem isso a entidade repetida pesa dobrado na distribuição da safra, e
    nada acusa."""
    df = pd.concat([_frame(), _frame().head(1)], ignore_index=True)

    assert _statuses(df, _struct())["primary_key_is_unique"] == "FAIL"


def test_the_same_entity_in_another_reference_date_is_not_a_duplicate():
    """A identidade é chave + safra: pontuar o mesmo cliente em duas safras é o
    caso normal, não erro."""
    outra = _frame().assign(reference_date=["2026-09-25", "2026-09-25"])
    df = pd.concat([_frame(), outra], ignore_index=True)

    assert _statuses(df, _struct())["primary_key_is_unique"] == "PASS"


def test_a_null_key_fails():
    df = _frame(customer_id=[1, None])

    assert _statuses(df, _struct())["primary_key_has_no_nulls"] == "FAIL"
