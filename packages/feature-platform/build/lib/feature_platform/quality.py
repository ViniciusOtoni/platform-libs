from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Finding:
    check: str
    status: str  # "PASS" ou "FAIL"
    violations: int
    detail: str = ""


def check_schema(df: pd.DataFrame, expected_columns: list[str]) -> Finding:
    missing = [c for c in expected_columns if c not in df.columns]
    return Finding(
        check="schema",
        status="PASS" if not missing else "FAIL",
        violations=len(missing),
        detail=f"missing columns: {missing}" if missing else "",
    )


def check_unique_keys(df: pd.DataFrame, entity_keys: list[str], timestamp_key: str) -> Finding:
    key_cols = [*entity_keys, timestamp_key]
    dupes = int(df.duplicated(subset=key_cols).sum())
    return Finding(
        check="unique_keys",
        status="PASS" if dupes == 0 else "FAIL",
        violations=dupes,
        detail=f"duplicate rows on {key_cols}",
    )


def check_no_nulls(df: pd.DataFrame, entity_keys: list[str], timestamp_key: str) -> Finding:
    cols = [*entity_keys, timestamp_key]
    nulls = int(df[cols].isnull().sum().sum())
    return Finding(
        check="no_nulls",
        status="PASS" if nulls == 0 else "FAIL",
        violations=nulls,
        detail=f"null values in {cols}",
    )


def check_freshness(df: pd.DataFrame, timestamp_key: str, window_end: date, max_lag_days: int = 1) -> Finding:
    if df.empty:
        return Finding(check="freshness", status="FAIL", violations=1, detail="empty dataframe")
    max_ts = pd.to_datetime(df[timestamp_key]).max().date()
    lag = (window_end - max_ts).days
    passed = lag <= max_lag_days
    return Finding(
        check="freshness",
        status="PASS" if passed else "FAIL",
        violations=0 if passed else 1,
        detail=f"max({timestamp_key})={max_ts}, window_end={window_end}, lag={lag}d",
    )


def run_quality_gate(
    df: pd.DataFrame, entity_keys: list[str], timestamp_key: str, window_end: date
) -> list[Finding]:
    expected_columns = [*entity_keys, timestamp_key]
    return [
        check_schema(df, expected_columns),
        check_unique_keys(df, entity_keys, timestamp_key),
        check_no_nulls(df, entity_keys, timestamp_key),
        check_freshness(df, timestamp_key, window_end),
    ]


def gate_passed(findings: list[Finding]) -> bool:
    return all(f.status == "PASS" for f in findings)
