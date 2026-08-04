from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_tables(base_dir: Path, sectors: list[str]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    Z = pd.DataFrame(
        [[10.0, 1.0, 2.0, 3.0], [4.0, 20.0, 5.0, 6.0], [7.0, 8.0, 30.0, 9.0], [1.0, 2.0, 3.0, 40.0]],
        index=sectors,
        columns=sectors,
    )
    Y = pd.DataFrame({"P3": [1.0, 2.0, 3.0, 4.0]}, index=sectors)
    Z.to_csv(base_dir / "Z_split.csv")
    Y.to_csv(base_dir / "Y_split.csv")
    (Z.sum(axis=1) + Y.sum(axis=1)).rename("x").to_csv(base_dir / "x_split.csv")


def _write_mapping_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_selected_children_json(path: Path, children: list[str]) -> None:
    path.write_text(json.dumps({"CPA_C20": children}), encoding="utf-8")


def _run_aggregate(indir: Path, outdir: Path, mapping_csv: Path, *extra_args: str) -> int:
    from scripts.aggregate_iot_by_mapping import main

    argv = [
        "--indir",
        str(indir),
        "--outdir",
        str(outdir),
        "--mapping-csv",
        str(mapping_csv),
        "--no-xlsx",
        *extra_args,
    ]
    return main(argv)


def test_fail_on_unmapped_split_children_keeps_unsplit_sectors_and_aggregates_split_children(tmp_path: Path) -> None:
    indir = tmp_path / "indir"
    outdir = tmp_path / "out"
    sectors = ["CPA_A", "CPA_B", "p24.g", "p24.h"]
    _write_tables(indir, sectors)

    mapping_csv = tmp_path / "mapping.csv"
    _write_mapping_csv(
        mapping_csv,
        [
            {"source_code": "p24.g", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
            {"source_code": "p24.h", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
        ],
    )
    _write_selected_children_json(indir / "exio_selected_sectors.json", ["p24.g", "p24.h"])

    rc = _run_aggregate(indir, outdir, mapping_csv, "--fail-on-unmapped-split-children")
    assert rc == 0

    Z_agg = pd.read_csv(outdir / "Z_agg.csv", index_col=0)
    assert set(Z_agg.index) == {"CPA_A", "CPA_B", "ENERGY_GREEN"}
    assert set(Z_agg.columns) == {"CPA_A", "CPA_B", "ENERGY_GREEN"}
    assert "CPA_A" in Z_agg.index and "CPA_B" in Z_agg.index
    assert Z_agg.loc["ENERGY_GREEN", "ENERGY_GREEN"] == pytest.approx(82.0)


def test_fail_on_unmapped_split_children_errors_when_selected_child_is_missing(tmp_path: Path) -> None:
    indir = tmp_path / "indir"
    outdir = tmp_path / "out"
    sectors = ["CPA_A", "CPA_B", "p24.g", "p24.h"]
    _write_tables(indir, sectors)

    mapping_csv = tmp_path / "mapping.csv"
    _write_mapping_csv(
        mapping_csv,
        [
            {"source_code": "p24.g", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
        ],
    )
    _write_selected_children_json(indir / "exio_selected_sectors.json", ["p24.g", "p24.h"])

    with pytest.raises(SystemExit, match="p24\.h"):
        _run_aggregate(indir, outdir, mapping_csv, "--fail-on-unmapped-split-children")


def test_fail_on_unmapped_errors_for_unsplit_sectors(tmp_path: Path) -> None:
    indir = tmp_path / "indir"
    outdir = tmp_path / "out"
    sectors = ["CPA_A", "CPA_B", "p24.g", "p24.h"]
    _write_tables(indir, sectors)

    mapping_csv = tmp_path / "mapping.csv"
    _write_mapping_csv(
        mapping_csv,
        [
            {"source_code": "p24.g", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
            {"source_code": "p24.h", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
        ],
    )
    _write_selected_children_json(indir / "exio_selected_sectors.json", ["p24.g", "p24.h"])

    with pytest.raises(SystemExit, match="CPA_A|CPA_B"):
        _run_aggregate(indir, outdir, mapping_csv, "--fail-on-unmapped")


def test_expect_n_sectors_only_checks_final_sector_count(tmp_path: Path) -> None:
    indir = tmp_path / "indir"
    outdir = tmp_path / "out"
    sectors = ["CPA_A", "CPA_B", "p24.g", "p24.h"]
    _write_tables(indir, sectors)

    mapping_csv = tmp_path / "mapping.csv"
    _write_mapping_csv(
        mapping_csv,
        [
            {"source_code": "p24.g", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
            {"source_code": "p24.h", "target_code": "ENERGY_GREEN", "enabled": True, "notes": "ok"},
        ],
    )
    _write_selected_children_json(indir / "exio_selected_sectors.json", ["p24.g", "p24.h"])

    rc = _run_aggregate(indir, outdir, mapping_csv, "--expect-n-sectors", "3")
    assert rc == 0

    Z_agg = pd.read_csv(outdir / "Z_agg.csv", index_col=0)
    assert len(Z_agg.index) == 3
    assert set(Z_agg.index) == {"CPA_A", "CPA_B", "ENERGY_GREEN"}
