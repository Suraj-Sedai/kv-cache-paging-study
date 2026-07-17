import csv

import pytest

from src.runner.fragmentation_report import (
    FRAGMENTATION_REPORT_FIELDS,
    build_fragmentation_report_rows,
    write_fragmentation_report_csv,
)
from src.workloads.bimodal import make_bimodal_workload
from src.workloads.heavy_tail import make_heavy_tail_workload
from src.workloads.uniform import make_uniform_workload


def test_fragmentation_report_for_uniform_workload():
    workload = make_uniform_workload(
        batch_size=4,
        prompt_len=8,
        decode_len=0,
    )

    rows = build_fragmentation_report_rows(
        workload_type="uniform",
        workload=workload,
        block_sizes=[4, 8],
    )

    assert len(rows) == 2

    for row in rows:
        assert list(row.keys()) == FRAGMENTATION_REPORT_FIELDS
        assert row["workload_type"] == "uniform"
        assert row["batch_size"] == 4
        assert row["seq_len_min"] == 8
        assert row["seq_len_mean"] == pytest.approx(8.0)
        assert row["seq_len_max"] == 8
        assert row["total_used_tokens"] == 32
        assert row["total_wasted_tokens"] == 0
        assert row["fragmentation_ratio"] == pytest.approx(0.0)


def test_fragmentation_report_for_bimodal_workload():
    workload = make_bimodal_workload(
        batch_size=8,
        short_prompt_range=(5, 9),
        long_prompt_range=(17, 25),
        decode_range=(0, 0),
        short_fraction=0.75,
        seed=123,
    )

    rows = build_fragmentation_report_rows(
        workload_type="bimodal",
        workload=workload,
        block_sizes=[4, 8, 16],
    )

    assert len(rows) == 3

    for row in rows:
        assert row["workload_type"] == "bimodal"
        assert row["batch_size"] == 8
        assert row["seq_len_max"] > row["seq_len_min"]
        assert row["total_allocated_tokens"] >= row["total_used_tokens"]
        assert row["total_wasted_tokens"] >= 0
        assert 0.0 <= row["fragmentation_ratio"] <= 1.0


def test_fragmentation_report_for_heavy_tail_workload():
    workload = make_heavy_tail_workload(
        batch_size=16,
        min_prompt_len=4,
        max_prompt_len=128,
        decode_range=(0, 0),
        alpha=1.4,
        seed=7,
    )

    rows = build_fragmentation_report_rows(
        workload_type="heavy_tail",
        workload=workload,
        block_sizes=[4, 8, 16],
    )

    assert len(rows) == 3

    for row in rows:
        assert row["workload_type"] == "heavy_tail"
        assert row["batch_size"] == 16
        assert row["seq_len_max"] > row["seq_len_min"]
        assert row["total_allocated_tokens"] >= row["total_used_tokens"]
        assert row["max_pages_per_sequence"] >= row["mean_pages_per_sequence"]


def test_fragmentation_report_rejects_invalid_inputs():
    workload = make_uniform_workload(
        batch_size=2,
        prompt_len=8,
        decode_len=0,
    )

    with pytest.raises(ValueError):
        build_fragmentation_report_rows(
            workload_type="",
            workload=workload,
            block_sizes=[4],
        )

    with pytest.raises(ValueError):
        build_fragmentation_report_rows(
            workload_type="uniform",
            workload=workload,
            block_sizes=[],
        )


def test_write_fragmentation_report_csv(tmp_path):
    workload = make_uniform_workload(
        batch_size=4,
        prompt_len=8,
        decode_len=0,
    )

    rows = build_fragmentation_report_rows(
        workload_type="uniform",
        workload=workload,
        block_sizes=[4, 8],
    )

    output_path = tmp_path / "processed" / "fragmentation_report.csv"
    write_fragmentation_report_csv(rows, output_path)

    assert output_path.exists()

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert reader.fieldnames == FRAGMENTATION_REPORT_FIELDS
    assert len(csv_rows) == 2


def test_write_fragmentation_report_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError):
        write_fragmentation_report_csv([], tmp_path / "empty.csv")


def test_write_fragmentation_report_rejects_missing_fields(tmp_path):
    bad_rows = [{"workload_type": "uniform"}]

    with pytest.raises(ValueError):
        write_fragmentation_report_csv(
            bad_rows,
            tmp_path / "bad.csv",
        )
