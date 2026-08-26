from datetime import date

import pandas as pd

from mlplatform.core.quality import Finding, gate_passed
from mlplatform.features.quality import (
    check_freshness,
    check_no_nulls,
    check_schema,
    check_unique_keys,
    run_quality_gate,
)


def _valid_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "feature_ts": [date(2026, 1, 15)] * 3,
            "score": [0.1, 0.2, 0.3],
        }
    )


def test_check_schema_passes_with_expected_columns():
    finding = check_schema(_valid_df(), ["customer_id", "feature_ts"])
    assert finding.status == "PASS"
    assert finding.violations == 0


def test_check_schema_fails_with_missing_column():
    finding = check_schema(_valid_df(), ["customer_id", "missing_col"])
    assert finding.status == "FAIL"
    assert "missing_col" in finding.detail


def test_check_unique_keys_fails_on_duplicate():
    df = pd.concat([_valid_df(), _valid_df().iloc[[0]]], ignore_index=True)
    finding = check_unique_keys(df, entity_keys=["customer_id"], timestamp_key="feature_ts")
    assert finding.status == "FAIL"
    assert finding.violations == 1


def test_check_no_nulls_fails_on_null_key():
    df = _valid_df()
    df.loc[0, "customer_id"] = None
    finding = check_no_nulls(df, entity_keys=["customer_id"], timestamp_key="feature_ts")
    assert finding.status == "FAIL"
    assert finding.violations == 1


def test_check_freshness_passes_within_lag():
    finding = check_freshness(_valid_df(), "feature_ts", window_end=date(2026, 1, 15), max_lag_days=1)
    assert finding.status == "PASS"


def test_check_freshness_fails_when_stale():
    finding = check_freshness(_valid_df(), "feature_ts", window_end=date(2026, 2, 1), max_lag_days=1)
    assert finding.status == "FAIL"


def test_run_quality_gate_returns_all_checks():
    findings = run_quality_gate(_valid_df(), ["customer_id"], "feature_ts", date(2026, 1, 15))
    assert {f.check for f in findings} == {"schema", "unique_keys", "no_nulls", "freshness"}


def test_gate_passed_true_when_all_pass():
    findings = [Finding("a", "PASS", 0), Finding("b", "PASS", 0)]
    assert gate_passed(findings) is True


def test_gate_passed_false_when_any_fails():
    findings = [Finding("a", "PASS", 0), Finding("b", "FAIL", 3)]
    assert gate_passed(findings) is False
