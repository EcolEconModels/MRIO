from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.generate_fuels_mapping_from_workbook import build_fuels_mappings
from scripts.aggregate_iot_by_mapping import _resolve_product_template_alignment


def _canonical_base(kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "source_dataset": "SRC",
        "source_system": "pxp",
        "source_version": "1",
        "source_level": "product",
        "target_dataset": "TGT",
        "target_system": "pxp",
        "target_version": "1",
        "target_level": "product",
        "source_key": "",
        "source_code": "",
        "source_label": "",
        "target_key": "",
        "target_code": "",
        "target_label": "",
        "subtarget_key": "",
        "weight": "",
        "share": "",
        "enabled": "1",
        "provenance": "methodological",
        "certainty": "derived",
        "transformation_rule": "test",
        "source_name": "sample.csv",
        "official_source": "",
        "notes": "",
        "source_balance_role": "",
        "target_sector_role": "",
        "include_in_direct_energy": "",
    }


def _write_concordance(path: Path) -> None:
    rows: list[dict[str, str]] = []
    for source_key, source_code, target_key, target_label, enabled in [
        ("Motor Gasoline", "p23.20.a", "CPA_C19", "Coke and refined petroleum products", "1"),
        ("Heavy Fuel Oil", "p23.20.g", "CPA_C19", "Coke and refined petroleum products", "1"),
        ("Lubricants", "p23.20.n", "CPA_C19", "Coke and refined petroleum products", "1"),
        ("Electricity by hydro", "p40.11.d", "CPA_D", "Electricity, gas, steam and air conditioning", "1"),
    ]:
        row = _canonical_base("weighted")
        row["source_dataset"] = "EXIOBASE3"
        row["target_dataset"] = "Eurostat_IO"
        row["target_version"] = "CPA_65"
        row["source_key"] = source_key
        row["source_code"] = source_code
        row["source_label"] = source_key
        row["target_key"] = target_key
        row["target_code"] = target_key
        row["target_label"] = target_label
        row["weight"] = "1.0"
        row["enabled"] = enabled
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_fuels_mappings_uses_unique_source_sector_selection(tmp_path: Path) -> None:
    workbook_path = tmp_path / "fuels.xlsx"
    concordance_path = tmp_path / "concordance.csv"
    outdir = tmp_path / "out"

    sheet = pd.DataFrame(
        [
            ["Eurostat sector", "Disaggregated sector", "Reaggregation target"],
            ["CPA_C19", "Motor Gasoline", "Motor Gasoline"],
            ["", "Heavy Fuel Oil", "Coke and refined petroleum products"],
            ["CPA_D", "Electricity by hydro", "Hydro"],
        ]
    )
    with pd.ExcelWriter(workbook_path) as writer:
        sheet.to_excel(writer, sheet_name="Step2_Reaggregation", header=False, index=False)

    _write_concordance(concordance_path)

    outputs = build_fuels_mappings(
        workbook_path=workbook_path,
        sheet_name="Step2_Reaggregation",
        concordance_path=concordance_path,
        outdir=outdir,
    )

    split_mapping = pd.read_csv(outputs["split_mapping"], dtype=str).fillna("")
    identity_mapping = pd.read_csv(outputs["identity_mapping"], dtype=str).fillna("")
    reaggregation_mapping = pd.read_csv(outputs["reaggregation_mapping"], dtype=str).fillna("")
    selected_sectors = pd.read_csv(outputs["selected_sectors"], dtype=str).fillna("")
    uncovered = pd.read_csv(outputs["uncovered_subsectors"], dtype=str).fillna("")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    assert set(selected_sectors["source_sector_code"]) == {"CPA_C19", "CPA_D"}

    # The split mapping is driven by unique column-1 source sectors, not only by the sheet-listed column-2 rows.
    assert set(split_mapping["target_key"]) == {"CPA_C19", "CPA_D"}
    assert set(split_mapping.loc[split_mapping["target_key"] == "CPA_C19", "source_key"]) == {
        "Motor Gasoline",
        "Heavy Fuel Oil",
        "Lubricants",
    }

    # Identity export should cover all enabled split subsectors.
    assert set(identity_mapping["source_key"]) == {
        "CPA_C19::Motor Gasoline",
        "CPA_C19::Heavy Fuel Oil",
        "CPA_C19::Lubricants",
        "CPA_D::Electricity by hydro",
    }
    assert set(identity_mapping["source_key"]) == set(identity_mapping["target_key"])

    # Real reaggregation uses workbook column 3; mapping back to the original aggregate becomes a derived remaining bucket.
    reagg_targets = dict(zip(reaggregation_mapping["source_key"], reaggregation_mapping["target_key"], strict=False))
    assert reagg_targets["CPA_C19::Motor Gasoline"] == "CPA_C19::Motor Gasoline"
    assert reagg_targets["CPA_C19::Heavy Fuel Oil"] == "CPA_C19_REMAINING"
    assert reagg_targets["CPA_D::Electricity by hydro"] == "CPA_D_HYDRO"

    reagg_labels = dict(zip(reaggregation_mapping["source_key"], reaggregation_mapping["target_label"], strict=False))
    assert reagg_labels["CPA_C19::Heavy Fuel Oil"] == "Coke and refined petroleum products - remaining"
    assert reagg_labels["CPA_D::Electricity by hydro"] == "Hydro"

    # Uncovered subsectors are written explicitly so the sheet can be completed before reaggregation runs.
    assert uncovered["from_sector"].tolist() == ["CPA_C19::Lubricants"]
    assert summary["uncovered_split_subsector_count"] == 1
    assert summary["resolved_sheet_row_count"] == 3
    assert summary["unresolved_sheet_row_count"] == 0


def test_resolve_product_template_alignment_falls_back_for_custom_split_sectors(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    template.write_text("placeholder", encoding="utf-8")

    effective_template, replace_map, product_order = _resolve_product_template_alignment(
        template_xlsx=template,
        base_product_codes=["CPA_C19", "CPA_D"],
        output_product_codes=["CPA_C19::Motor Gasoline", "CPA_D::Electricity by wind"],
        verbose=False,
    )

    assert effective_template is None
    assert replace_map == {}
    assert product_order == ["CPA_C19::Motor Gasoline", "CPA_D::Electricity by wind"]


def test_resolve_product_template_alignment_keeps_energy_replacement_template(tmp_path: Path) -> None:
    template = tmp_path / "template.xlsx"
    template.write_text("placeholder", encoding="utf-8")

    effective_template, replace_map, product_order = _resolve_product_template_alignment(
        template_xlsx=template,
        base_product_codes=["CPA_C19", "CPA_D", "CPA_A01"],
        output_product_codes=["ENERGY_BROWN", "ENERGY_GREEN", "CPA_A01"],
        verbose=False,
    )

    assert effective_template == template
    assert replace_map == {"CPA_C19": "ENERGY_BROWN", "CPA_D": "ENERGY_GREEN"}
    assert product_order == ["ENERGY_BROWN", "ENERGY_GREEN", "CPA_A01"]
