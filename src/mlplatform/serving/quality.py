
import pandas as pd

from mlplatform.core.quality import Finding

from .structure import InferenceBatchStruct


def check_no_nulls_in_predictions(df: pd.DataFrame, prediction_column: str) -> Finding:
    nulls = int(df[prediction_column].isnull().sum())
    return Finding(
        check="no_nulls_in_predictions",
        status="PASS" if nulls == 0 else "FAIL",
        detail=f"nulls={nulls}",
    )


def check_no_nulls_in_joined_columns(df: pd.DataFrame, prediction_column: str) -> Finding:
    # Uma entidade sem correspondência no FeatureLookup recebe features nulas, mas o
    # modelo pode ainda assim produzir uma predição não-nula (ex.: RandomForestClassifier
    # tolera NaN nativamente) — checar só a coluna de predição não pega esse caso.
    joined_cols = [c for c in df.columns if c != prediction_column]
    bad_cols = [c for c in joined_cols if int(df[c].isnull().sum()) > 0]
    return Finding(
        check="no_nulls_in_joined_columns",
        status="PASS" if not bad_cols else "FAIL",
        detail=f"columns_with_nulls={bad_cols}" if bad_cols else "",
    )


def check_row_count_matches(input_row_count: int, output_row_count: int) -> Finding:
    matches = input_row_count == output_row_count
    return Finding(
        check="row_count_matches",
        status="PASS" if matches else "FAIL",
        detail=f"input={input_row_count}, output={output_row_count}",
    )


def run_predictions_gate(df: pd.DataFrame, prediction_column: str, input_row_count: int) -> list[Finding]:
    return [
        check_no_nulls_in_predictions(df, prediction_column),
        check_no_nulls_in_joined_columns(df, prediction_column),
        check_row_count_matches(input_row_count, len(df)),
    ]



def check_declared_columns_are_present(df: pd.DataFrame, struct: InferenceBatchStruct) -> Finding:
    """A saída tem que conter tudo que o domínio declarou.

    Reprovar aqui é barato; descobrir meses depois que a coluna de safra nunca
    foi gravada custa o histórico inteiro."""
    missing = [c for c in struct.required_columns if c not in df.columns]
    return Finding(
        check="declared_columns_are_present",
        status="PASS" if not missing else "FAIL",
        detail=f"missing={missing}" if missing else "",
    )


def check_primary_key_is_unique(df: pd.DataFrame, struct: InferenceBatchStruct) -> Finding:
    """Chave + safra identificam a linha.

    Duplicata aqui não é detalhe: o monitoramento conta distribuição por safra,
    e uma entidade repetida pesa dobrado sem nada acusar."""
    keys = [*struct.primary_key, struct.ts_date]
    if any(k not in df.columns for k in keys):
        return Finding(check="primary_key_is_unique", status="FAIL", detail="key columns missing")
    duplicated = int(df.duplicated(subset=keys).sum())
    return Finding(
        check="primary_key_is_unique",
        status="PASS" if duplicated == 0 else "FAIL",
        detail=f"duplicated_rows={duplicated}",
    )


def check_primary_key_has_no_nulls(df: pd.DataFrame, struct: InferenceBatchStruct) -> Finding:
    cols = [c for c in struct.primary_key if c in df.columns]
    bad = [c for c in cols if int(df[c].isnull().sum()) > 0]
    return Finding(
        check="primary_key_has_no_nulls",
        status="PASS" if not bad else "FAIL",
        detail=f"columns_with_nulls={bad}" if bad else "",
    )


def run_structure_gate(df: pd.DataFrame, struct: InferenceBatchStruct) -> list[Finding]:
    """Conformidade com o formato declarado, antes de a tabela ser gravada."""
    return [
        check_declared_columns_are_present(df, struct),
        check_primary_key_is_unique(df, struct),
        check_primary_key_has_no_nulls(df, struct),
    ]
