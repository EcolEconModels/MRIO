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
            {"source_code": "CPA_C24", "source_label": "Basic metals"},
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
            {"source_code": "p23.3", "source_label": "Nuclear fuel", "target_code": "CPA_C24", "target_label": "Basic metals", "enabled": True},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_D", "target_label": "Electricity", "enabled": True},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_E37-39", "target_label": "Waste", "enabled": True},
        ],
    )
    rules = tmp_path / "rules.csv"
    write_csv(
        rules,
        [
            {"source_code": "p10.f@CPA_C19", "source_label": "Lignite edge", "target_code": "CPA_C19_COAL", "target_label": "CPA_C19: Coal", "enabled": True, "notes": "coal rule"},
            {"source_code": "p23.3", "source_label": "Nuclear fuel", "target_code": "CPA_C24_NUCLEAR_FUEL", "target_label": "CPA_C24: Nuclear fuel", "enabled": True, "notes": "nuclear rule"},
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
    assert mapping["p23.3"] == "CPA_C24_NUCLEAR_FUEL"
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
        / "scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv"
    )

    fuels_rows, fuels_review, _ = module.build_closed_reaggregation(
        split_mapping_csv=root / "scripts/pipelines/fuels_split.csv",
        base_reagg_csv=root / "scripts/pipelines/fuels_reaggregation.csv",
        concordance_csv=concordance,
        rule_csvs=[],
    )
    assert fuels_review == []
    fuels = {row["source_code"]: row["target_code"] for row in fuels_rows}
    assert fuels["p24.g"] == "CPA_C20_BIOGASOLINE"
    assert fuels["p24.h"] == "CPA_C20_BIODIESEL"
    assert fuels["p24.i"] == "CPA_C20_OTHER_LIQUID_BIOFUELS"
    assert fuels["p24.d@CPA_C20"] == "CPA_C20_others"
    assert fuels["p23.3"] == "CPA_C24_NUCLEAR_FUEL"
    assert fuels["p27.42"] == "CPA_C24"
    assert "CPA_C20_NUCLEAR_FUEL" not in set(fuels.values())


def test_fuels_exception_routes_live_in_canonical_reaggregation_tables() -> None:
    """Retired fuels rule-overlay rows must be owned by each canonical recipe."""
    root = Path(__file__).resolve().parents[1]
    expected = {
        "p10.f@CPA_C19": "CPA_C19_COAL",
        "p10.h@CPA_C19": "CPA_C19_COAL",
        "p23.3": "CPA_C24_NUCLEAR_FUEL",
    }
    for name in ("fuels_reaggregation.csv", "fuels_more_reaggregation.csv"):
        frame = pd.read_csv(root / "scripts/pipelines" / name, dtype=str).fillna("")
        routes = dict(zip(frame["source_code"], frame["target_code"], strict=True))
        for source, target in expected.items():
            assert routes[source] == target, (name, source)


def test_biofuels_and_feedstocks_recipes_close_without_rule_overlay() -> None:
    """New EXIO-native biofuel recipes close on the reviewed official topology."""
    module = load_reagg_module()
    root = Path(__file__).resolve().parents[1]
    concordance = root / "scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv"

    expected_parents = {
        "biofuels": {"CPA_C20", "CPA_D"},
        "biofuels_feedstocks": {
            "CPA_A01",
            "CPA_C10-12",
            "CPA_C16",
            "CPA_C20",
            "CPA_D",
            "CPA_E37-39",
        },
    }
    # Pin the current reviewed topology and require every emitted child to have an
    # explicit row in these new analytical recipes. The generic closure helper may
    # infer a child's statistical CPA home, but that fallback must not hide a
    # missing biofuels/feedstocks analytical decision.
    expected_output_rows = {"biofuels": 31, "biofuels_feedstocks": 84}
    closed: dict[str, dict[str, str]] = {}
    for suffix, parents in expected_parents.items():
        split_path = root / "scripts/pipelines" / f"{suffix}_split.csv"
        split = pd.read_csv(split_path, dtype=str).fillna("")
        assert set(split["source_code"]) == parents
        rows, review, manifest = module.build_closed_reaggregation(
            split_mapping_csv=split_path,
            base_reagg_csv=root / "scripts/pipelines" / f"{suffix}_reaggregation.csv",
            concordance_csv=concordance,
            rule_csvs=[],
        )
        assert review == []
        assert manifest["review_required_rows"] == 0
        assert manifest["rule_rows"] == 0
        assert manifest["output_rows"] == expected_output_rows[suffix]
        assert manifest["closure_rule_counts"] == {
            "existing_exact_reaggregation_row": expected_output_rows[suffix]
        }
        closed[suffix] = {row["source_code"]: row["target_code"] for row in rows}

    biofuels = closed["biofuels"]
    assert biofuels["p24.g"] == "CPA_C20_BIOGASOLINE"
    assert biofuels["p24.h"] == "CPA_C20_BIODIESEL"
    assert biofuels["p24.i"] == "CPA_C20_OTHER_LIQUID_BIOFUELS"
    assert biofuels["p24.d@CPA_C20"] == "CPA_C20_others"
    assert biofuels["p40.2.e"] == "CPA_D_BIOGAS"
    assert biofuels["p40.11.g"] == "CPA_D_others"

    feedstocks = closed["biofuels_feedstocks"]
    assert feedstocks["p01.b"] == "C_WHEA"
    assert feedstocks["p01.c"] == "C_CEREAL_NEC"
    assert feedstocks["p01.e"] == "C_OIL_SEEDS"
    assert feedstocks["p01.f"] == "C_SUGB"
    assert feedstocks["p15.e"] == "C_VOIL"
    assert feedstocks["p15.h"] == "C_SUGAR"
    assert feedstocks["p20.w"] == "C_WOOW"
    assert feedstocks["p90.2.a"] == "C_BIOF"
    assert feedstocks["p90.2.b"] == "C_BIOP"
    assert feedstocks["p90.2.c"] == "C_BIOS"
    assert feedstocks["p90.1.a"] == "C_FOOD_WASTE_INCINERATION"
    assert feedstocks["p90.3.a"] == "C_FOOD_WASTE_COMPOSTING"
    assert feedstocks["p90.5.a"] == "C_FOOD_WASTE_LANDFILL"
    assert feedstocks["p90.1.f"] == "C_WOOD_WASTE_INCINERATION"
    assert feedstocks["p90.5.f"] == "C_WOOD_WASTE_LANDFILL"
    assert feedstocks["p90.1.g"] == "CPA_E37-39_others"  # never proxy UCO with oil/hazardous waste
    assert feedstocks["p01.h"] == "CPA_A01_others"  # never invent crop residues from generic crops


def test_generic_workbook_wrapper_has_no_fuels_specific_rule_default() -> None:
    """Optional route overlays must be opt-in for the generic OUTPUT_SUFFIX runner."""
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(encoding="utf-8")
    assert 'REAGG_RULE_CSV="${REAGG_RULE_CSV:-}"' in wrapper
    assert "fuels_reaggregation_fuel_expanded_edge_rules.csv" not in wrapper
    assert 'REAGG_RULE_ARGS=(--rule-csv "$REAGG_RULE_CSV")' in wrapper
