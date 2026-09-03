from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _write_tables(base_dir: Path, sectors: list[str]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    sector_index = pd.Index(sectors)

    Z = pd.DataFrame(
        [[10.0, 1.0, 2.0, 3.0], [4.0, 20.0, 5.0, 6.0], [7.0, 8.0, 30.0, 9.0], [1.0, 2.0, 3.0, 40.0]],
        index=sector_index,
        columns=sector_index,
    )
    Y = pd.DataFrame({"P3": [1.0, 2.0, 3.0, 4.0]}, index=sector_index)
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

    with pytest.raises(SystemExit, match=r"p24\.h"):
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



def test_aggregate_cli_reaggregates_generic_environmental_sidecar(tmp_path: Path) -> None:
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
    pd.DataFrame(
        {
            "sector": ["p24.g", "p24.h"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["air_emissions", "air_emissions"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "value": [114.0, 50.0],
            "monetary_output": [57.0, 50.0],
            "intensity": [2.0, 1.0],
        }
    ).to_csv(indir / "environmental_extensions.csv", index=False)
    raw_exio = pd.DataFrame(
        {
            "source_label": ["p24.g", "p24.g", "P3", "P3"],
            "extension": [
                "air_emissions",
                "deforestation",
                "air_emissions",
                "use_phase_only",
            ],
            "kind": ["F", "F", "F_Y", "F_Y"],
            "stressor": [
                "CO2 - combustion - air",
                "Unsupported selected stressor",
                "CO2 - combustion - air",
                "Final-demand-only stressor",
            ],
            "unit": ["kg", "ha", "kg", "kg"],
            "value": [114.0, 0.0, 5.0, 1.0],
            "monetary_denominator": [57.0, 57.0, 10.0, 2.0],
            "intensity": [2.0, 0.0, 0.5, 0.5],
        }
    )
    raw_exio.to_csv(indir / "exio_environmental_extensions.csv", index=False)

    assert _run_aggregate(indir, outdir, mapping_csv) == 0
    satellite = pd.read_csv(outdir / "environmental_extensions.csv")
    assert list(satellite.columns) == [
        "sector_position",
        "sector",
        "boundary",
        "denominator_role",
        "extension",
        "stressor",
        "unit",
        "value",
        "monetary_output",
        "intensity",
    ]
    air = satellite.loc[satellite["extension"].eq("air_emissions")]
    assert air["sector"].tolist() == ["CPA_A", "CPA_B", "ENERGY_GREEN"]
    assert air["sector_position"].tolist() == [1, 2, 3]
    unavailable = air.loc[air["value"].isna()].set_index("sector")
    assert set(unavailable.index) == {"CPA_A", "CPA_B"}
    assert unavailable["value"].isna().all()
    assert unavailable["intensity"].isna().all()
    assert float(unavailable.loc["CPA_A", "monetary_output"]) == 17.0
    assert float(unavailable.loc["CPA_B", "monetary_output"]) == 37.0
    row = air.loc[air["value"].notna()].iloc[0]
    assert row["sector"] == "ENERGY_GREEN"
    assert float(row["value"]) == pytest.approx(164.0)
    assert float(row["monetary_output"]) == pytest.approx(107.0)
    assert float(row["intensity"]) == pytest.approx(164.0 / 107.0)
    unsupported_group = satellite.loc[
        satellite["extension"].eq("deforestation")
    ]
    assert unsupported_group["sector"].tolist() == [
        "CPA_A",
        "CPA_B",
        "ENERGY_GREEN",
    ]
    assert unsupported_group["value"].isna().all()
    assert unsupported_group["intensity"].isna().all()
    assert "use_phase_only" not in set(satellite["extension"])

    raw_written = pd.read_csv(outdir / "exio_environmental_extensions.csv")
    pd.testing.assert_frame_equal(raw_written, raw_exio)
    assert "sector_position" not in raw_written.columns
    assert "available" not in raw_written.columns
    assert "Unnamed: 0" not in raw_written.columns


def test_environmental_grid_distinguishes_undefined_estimates_from_zero_activity() -> None:
    """Evidence presence, physical zero, and target zero remain distinct states."""
    from scripts.aggregate_iot_by_mapping import (
        _complete_environmental_extension_sector_grid,
        _production_environmental_group_domain,
    )

    actual = pd.DataFrame(
        {
            "sector": ["A", "B"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["synthetic", "synthetic"],
            "stressor": ["Synthetic indicator", "Synthetic indicator"],
            "unit": ["synthetic unit", "synthetic unit"],
            "value": [0.0, 0.0],
            "monetary_output": [100.0, 0.0],
            "intensity": [0.0, float("nan")],
        }
    )
    output = pd.Series({"A": 100.0, "B": 0.0, "C": 100.0, "D": 0.0})
    production_groups = pd.DataFrame(
        {
            "extension": ["synthetic", "unsupported", "final_demand_only"],
            "stressor": ["Synthetic indicator", "No supported targets", "Use phase"],
            "unit": ["synthetic unit", "kg", "kg"],
            "kind": ["F", "F", "F_Y"],
        }
    )

    completed = _complete_environmental_extension_sector_grid(
        actual,
        output=output,
        production_groups=_production_environmental_group_domain(production_groups),
    )

    assert completed is not None
    assert len(completed) == 8
    assert "available" not in completed.columns
    synthetic = completed.loc[completed["extension"].eq("synthetic")]
    assert synthetic["sector"].tolist() == ["A", "B", "C", "D"]
    assert synthetic["sector_position"].tolist() == [1, 2, 3, 4]
    by_sector = synthetic.set_index("sector")
    assert by_sector.loc["A", "value"] == 0.0
    assert by_sector.loc["A", "intensity"] == 0.0
    assert by_sector.loc["B", "value"] == 0.0
    assert pd.isna(by_sector.loc["B", "intensity"])
    for sector, expected_output in (("C", 100.0), ("D", 0.0)):
        assert pd.isna(by_sector.loc[sector, "value"])
        assert pd.isna(by_sector.loc[sector, "intensity"])
        assert by_sector.loc[sector, "monetary_output"] == expected_output
    unsupported = completed.loc[completed["extension"].eq("unsupported")]
    assert len(unsupported) == 4
    assert unsupported["value"].isna().all()
    assert unsupported["intensity"].isna().all()
    assert "final_demand_only" not in set(completed["extension"])


def test_environmental_grid_rejects_actual_group_outside_raw_production_domain() -> None:
    """Scientific rows cannot silently expand the selected raw production domain."""
    from scripts.aggregate_iot_by_mapping import (
        _complete_environmental_extension_sector_grid,
    )

    actual = pd.DataFrame(
        {
            "sector": ["A"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["unexpected"],
            "stressor": ["Indicator"],
            "unit": ["kg"],
            "value": [1.0],
            "monetary_output": [10.0],
            "intensity": [0.1],
        }
    )
    groups = pd.DataFrame(
        {"extension": ["selected"], "stressor": ["Indicator"], "unit": ["kg"]}
    )

    with pytest.raises(ValueError, match="outside raw production F"):
        _complete_environmental_extension_sector_grid(
            actual,
            output=pd.Series({"A": 10.0}),
            production_groups=groups,
        )

    duplicate = pd.concat([actual, actual], ignore_index=True)
    duplicate["extension"] = "selected"
    with pytest.raises(ValueError, match="duplicate sector/group"):
        _complete_environmental_extension_sector_grid(
            duplicate,
            output=pd.Series({"A": 10.0}),
            production_groups=groups,
        )


def test_row_contract_writers_do_not_serialize_range_indices(tmp_path: Path) -> None:
    """Explicit position columns must not be preceded by synthetic CSV indices."""
    from scripts.aggregate_iot_by_mapping import (
        _write_sector_contract,
        _write_source_sector_contract,
    )

    sector_path = tmp_path / "sector_contract.csv"
    _write_sector_contract(
        out_path=sector_path,
        product_order=["CPA_A01", "CPA_B"],
        parent_by_sector={"CPA_A01": "CPA_A01", "CPA_B": "CPA_B"},
        verbose=False,
    )
    sector_contract = pd.read_csv(sector_path)
    assert list(sector_contract.columns) == [
        "position",
        "sector_id",
        "parent_sector_id",
        "nace_section",
    ]

    repo_root = Path(__file__).resolve().parents[1]
    parent_order = pd.read_csv(
        repo_root / "scripts/mappings/eurostat_cpa65_products_dictionary.csv"
    )["code"].astype(str).tolist()
    source_path = tmp_path / "source_sector_contract.csv"
    _write_source_sector_contract(
        out_path=source_path,
        parent_order=parent_order,
        verbose=False,
    )
    source_contract = pd.read_csv(source_path)
    assert list(source_contract.columns) == [
        "source_system",
        "source_position",
        "parent_sector_id",
    ]
