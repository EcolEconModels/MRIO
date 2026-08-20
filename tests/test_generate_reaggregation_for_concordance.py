from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_reagg_module():
    path = Path("scripts/pipelines/generate_reaggregation_for_concordance.py")
    spec = importlib.util.spec_from_file_location("generate_reaggregation_for_concordance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.name is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_edge_children_close_to_selected_target_or_compatible_fuel_bucket(tmp_path: Path) -> None:
    module = load_reagg_module()
    split = tmp_path / "fuels_split.csv"
    write_csv(
        split,
        [
            {"source_code": "CPA_C10-12", "source_label": "Food"},
            {"source_code": "CPA_C19", "source_label": "Coke and refined petroleum"},
            {"source_code": "CPA_C20", "source_label": "Chemicals"},
            {"source_code": "CPA_D", "source_label": "Electricity"},
        ],
    )
    base = tmp_path / "fuels_reaggregation.csv"
    write_csv(
        base,
        [
            {"source_code": "p15.j", "source_label": "Beverages", "target_code": "CPA_C10-12_FOOD_AND_BEVERAGES", "target_label": "CPA_C10-12: Food and beverages", "enabled": True, "notes": "base food row"},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_D_Electricity_by_coal", "target_label": "CPA_D: Electricity by coal", "enabled": True, "notes": "base electricity row"},
        ],
    )
    concordance = tmp_path / "exio_to_cpa.csv"
    write_csv(
        concordance,
        [
            {"source_code": "p15.j", "source_label": "Beverages", "target_code": "CPA_C10-12", "target_label": "Food", "enabled": True},
            {"source_code": "p15.j", "source_label": "Beverages", "target_code": "CPA_C20", "target_label": "Chemicals", "enabled": True},
            {"source_code": "p36", "source_label": "Furniture; other goods", "target_code": "CPA_C10-12", "target_label": "Food", "enabled": True},
            {"source_code": "p36", "source_label": "Furniture; other goods", "target_code": "CPA_C20", "target_label": "Chemicals", "enabled": True},
            {"source_code": "p10.f", "source_label": "Lignite", "target_code": "CPA_C19", "target_label": "Coke", "enabled": True},
            {"source_code": "p10.f", "source_label": "Lignite", "target_code": "CPA_B", "target_label": "Mining", "enabled": True},
            {"source_code": "p23.3", "source_label": "Nuclear fuel", "target_code": "CPA_C20", "target_label": "Chemicals", "enabled": True},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_D", "target_label": "Electricity", "enabled": True},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_E37-39", "target_label": "Waste", "enabled": True},
        ],
    )
    rules = tmp_path / "rules.csv"
    write_csv(
        rules,
        [
            {"source_code": "p10.f@CPA_C19", "source_label": "Lignite edge", "target_code": "CPA_C19_COAL", "target_label": "CPA_C19: Coal", "enabled": True, "notes": "coal rule"},
            {"source_code": "p23.3", "source_label": "Nuclear fuel", "target_code": "CPA_C20_NUCLEAR_FUEL", "target_label": "CPA_C20: Nuclear fuel", "enabled": True, "notes": "nuclear rule"},
        ],
    )

    out_rows, review_rows, manifest = module.build_closed_reaggregation(
        split_mapping_csv=split,
        base_reagg_csv=base,
        concordance_csv=concordance,
        rule_csvs=[rules],
    )

    assert review_rows == []
    mapping = {row["source_code"]: row["target_code"] for row in out_rows}
    assert mapping["p15.j@CPA_C10-12"] == "CPA_C10-12_FOOD_AND_BEVERAGES"
    assert mapping["p15.j@CPA_C20"] == "CPA_C20"
    assert mapping["p36@CPA_C10-12"] == "CPA_C10-12"
    assert mapping["p36@CPA_C20"] == "CPA_C20"
    assert mapping["p10.f@CPA_C19"] == "CPA_C19_COAL"
    assert mapping["p23.3"] == "CPA_C20_NUCLEAR_FUEL"
    assert mapping["p40.11.a@CPA_D"] == "CPA_D_Electricity_by_coal"
    assert manifest["review_required_rows"] == 0


def test_raw_many_target_child_requires_review(tmp_path: Path) -> None:
    module = load_reagg_module()
    child = {"child_code": "p36", "source_code": "p36", "source_label": "Furniture", "edge_target_code": "", "edge_target_label": "", "source_n_targets": 2, "is_edge_child": False}
    row, review = module.route_child(
        child,
        base_by_source={},
        rules_by_source={},
        targets_by_source={"p36": {"CPA_C10-12", "CPA_C20"}},
    )
    assert row is None
    assert review is not None
    assert review["source_code"] == "p36"
    assert "multiple active concordance targets" in review["reason"]


def test_write_rows_uses_lf_line_endings(tmp_path: Path) -> None:
    module = load_reagg_module()
    out_csv = tmp_path / "out.csv"
    module.write_rows(
        out_csv,
        [
            {
                "source_code": "p10.f@CPA_C19",
                "source_label": "Lignite",
                "target_code": "CPA_C19_COAL",
                "target_label": "CPA_C19: Coal",
                "enabled": "True",
                "notes": "test",
            }
        ],
        module.REAGG_COLUMNS,
    )

    data = out_csv.read_bytes()
    assert b"\r\n" not in data
    assert data.endswith(b"\n")


def test_official_direct_fuels_recipe_closes_materialized_children() -> None:
    """Preferred topology must close without losing fuels analytical routes."""
    module = load_reagg_module()
    root = Path(__file__).resolve().parents[1]
    concordance = (
        root
        / "scripts/mappings/generated/official_direct/exio_to_eurostat_cpa65_products_from_official_direct.csv"
    )

    fuels_rows, fuels_review, _ = module.build_closed_reaggregation(
        split_mapping_csv=root / "scripts/pipelines/fuels_split.csv",
        base_reagg_csv=root / "scripts/pipelines/fuels_reaggregation.csv",
        concordance_csv=concordance,
        rule_csvs=[
            root / "scripts/pipelines/fuels_reaggregation_fuel_expanded_edge_rules.csv"
        ],
    )
    assert fuels_review == []
    fuels = {row["source_code"]: row["target_code"] for row in fuels_rows}
    assert fuels["p24.g@CPA_C20"] == "CPA_C20_BIOGASOLINE"
    assert fuels["p24.h@CPA_C20"] == "CPA_C20_BIODIESEL"
    assert fuels["p24.i@CPA_C20"] == "CPA_C20_OTHER_LIQUID_BIOFUELS"
