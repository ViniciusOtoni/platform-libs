from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    detail: str = ""


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


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
