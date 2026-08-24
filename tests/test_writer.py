import pytest

from feature_platform.writer import WriteMode, write_strategy_for


def test_incremental_uses_merge_strategy():
    assert write_strategy_for(WriteMode.INCREMENTAL) == "merge"


def test_backfill_uses_overwrite_by_partition_strategy():
    assert write_strategy_for(WriteMode.BACKFILL) == "overwrite_by_partition"


def test_write_mode_accepts_string_values():
    assert WriteMode("incremental") == WriteMode.INCREMENTAL
    assert WriteMode("backfill") == WriteMode.BACKFILL
