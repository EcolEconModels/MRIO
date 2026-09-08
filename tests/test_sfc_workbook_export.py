# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalOperand=false, reportIndexIssue=false
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from ecolecon_mrio.iot_flow_semantics import flow_use_total


def _build_template(
    path: Path,
    *,
    product_labels: tuple[str, str] = (
        "Coke and refined petroleum products",
        "Electricity, gas, steam and air conditioning",
    ),
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet 1"

    ws.cell(row=11, column=1).value = "PRD_USE (Labels)"
    ws.cell(row=11, column=2).value = "Total"
    ws.cell(row=11, column=3).value = product_labels[0]
    ws.cell(row=11, column=4).value = product_labels[1]
    ws.cell(row=11, column=5).value = "Final consumption expediture"
    ws.cell(row=11, column=6).value = "Gross capital formation"
    ws.cell(row=11, column=7).value = "Exports of goods and services"
    ws.cell(row=11, column=8).value = "Total use"
    ws.cell(row=11, column=9).value = "Total final use"

    ws.cell(row=12, column=1).value = "Total"
    ws.cell(row=13, column=1).value = product_labels[0]
    ws.cell(row=14, column=1).value = product_labels[1]

    lower_rows = {
        15: ("Value added, gross", [10.0, 20.0], [":", ":", ":", ":", ":"]),
        16: ("Taxes less subsidies on products", [1.0, 2.0], [9.0, 8.0, 7.0, 24.0, 24.0]),
        17: ("Output", [30.0, 40.0], [":", ":", ":", ":", ":"]),
        18: ("Total intermediate consumption /final use", [5.0, 6.0], [11.0, 12.0, 13.0, 36.0, 36.0]),
        19: ("Imports of goods and services", [7.0, 8.0], [":", ":", ":", ":", ":"]),
        20: ("Total supply at basic prices", [37.0, 48.0], [":", ":", ":", ":", ":"]),
    }
    for row_idx, (label, product_vals, fd_vals) in lower_rows.items():
        ws.cell(row=row_idx, column=1).value = label
        ws.cell(row=row_idx, column=2).value = sum(product_vals)
        for offset, value in enumerate(product_vals, start=3):
            ws.cell(row=row_idx, column=offset).value = value
        for offset, value in enumerate(fd_vals, start=5):
            ws.cell(row=row_idx, column=offset).value = value

    wb.save(path)


def _row_values(ws, row_idx: int, start_col: int, count: int) -> list[object]:
    return [ws.cell(row=row_idx, column=col).value for col in range(start_col, start_col + count)]


def _with_source_rows(parsed, row_codes: list[str]):
    labels = {
        "P1": "Output",
        "P2_ADJ": "Total intermediate consumption /final use",
        "B1G": "Value added, gross",
        "P7": "Imports of goods and services",
        "TS_BP": "Total supply at basic prices",
        "D21X31": "Taxes less subsidies on products",
    }
    assert set(row_codes).issubset(labels)
    meta = dict(parsed.meta)
    dimensions = {key: dict(value) for key, value in meta["dimensions"].items()}
    # The real SDMX metadata contains the complete codelist even when a
    # particular flow has no numeric observations for a supplementary row.
    dimensions["prd_ava"].update(labels)
    meta["dimensions"] = dimensions
    source_codes = list(labels)
    meta["identified_supplementary_rows"] = source_codes
    fd_codes = list(map(str, meta["identified_final_demand_columns"]))
    source_m = pd.DataFrame(
        float("nan"),
        index=source_codes,
        columns=fd_codes,
        dtype="float64",
    )
    for row_code in {"D21X31", "P2_ADJ"}:
        source_m.loc[row_code, :] = 0.0
    return replace(
        parsed,
        meta=meta,
        V=pd.DataFrame(0.0, index=source_codes, columns=parsed.Z.columns),
        M=source_m,
    )


def _matching_transformed_supplementary(parsed, product_codes):
    row_codes = list(map(str, parsed.meta.get("identified_supplementary_rows", [])))
    V = pd.DataFrame(
        0.0,
        index=row_codes,
        columns=pd.Index(product_codes).astype(str),
        dtype="float64",
    )
    M = parsed.M.copy().astype("float64") if parsed.M is not None else None
    return V, M


def _build_lineage_toy_case():
    from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT

    parsed = ParsedEurostatIOT(
        Z=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=["CPA_C10-12", "CPA_C20"],
            columns=["CPA_C10-12", "CPA_C20"],
        ),
        Y=pd.DataFrame(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            index=["CPA_C10-12", "CPA_C20"],
            columns=["P3", "P5", "P6"],
        ),
        meta={
            "dimensions": {
                "prd_ava": {
                    "CPA_C10-12": "Food, beverages and tobacco products",
                    "CPA_C20": "Chemicals and chemical products",
                },
                "prd_use": {
                    "P3": "Final consumption expediture",
                    "P5": "Gross capital formation",
                    "P6": "Exports of goods and services",
                    "TU": "Total use",
                    "TFU": "Total final use",
                },
            },
            "identified_industry_columns": ["CPA_C10-12", "CPA_C20"],
            "identified_final_demand_columns": ["P3", "P5", "P6", "TU", "TFU"],
        },
    )

    z_before = pd.DataFrame(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 2.0, 4.0],
        ],
        index=["p15.j", "p16", "p24.g"],
        columns=["p15.j", "p16", "p24.g"],
    )
    y_before = pd.DataFrame(
        {
            "P3": [10.0, 5.0, 7.0],
            "P5": [4.0, 3.0, 2.0],
            "P6": [1.0, 2.0, 1.0],
        },
        index=z_before.index,
    )
    z_after = pd.DataFrame(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 2.0, 4.0],
        ],
        index=["CPA_C10-12_FOOD_AND_BEVERAGES", "CPA_C10-12_TOBACCO_PRODUCTS", "CPA_C20_BIOGASOLINE"],
        columns=["CPA_C10-12_FOOD_AND_BEVERAGES", "CPA_C10-12_TOBACCO_PRODUCTS", "CPA_C20_BIOGASOLINE"],
    )
    y_after = pd.DataFrame(
        {
            "P3": [10.0, 5.0, 7.0],
            "P5": [4.0, 3.0, 2.0],
            "P6": [1.0, 2.0, 1.0],
        },
        index=z_after.index,
    )
    mapping_df = pd.DataFrame(
        {
            "from_sector": ["p15.j", "p16", "p24.g"],
            "to_sector": [
                "CPA_C10-12_FOOD_AND_BEVERAGES",
                "CPA_C10-12_TOBACCO_PRODUCTS",
                "CPA_C20_BIOGASOLINE",
            ],
            "target_label": [
                "CPA_C10-12: Food and beverages",
                "CPA_C10-12: Tobacco products",
                "CPA_C20: Biogasoline",
            ],
        }
    )
    return _with_source_rows(parsed, ["P1", "P2_ADJ"]), z_before, y_before, z_after, y_after, mapping_df


def test_pretty_sector_label_formats_energy_codes() -> None:
    from scripts.aggregate_iot_by_mapping import _pretty_sector_label

    assert _pretty_sector_label("ENERGY_BROWN") == "Energy Brown"
    assert _pretty_sector_label("ENERGY_GREEN") == "Energy Green"


def test_infer_variant_value_uses_source_dir_and_filters() -> None:
    from scripts.aggregate_iot_by_mapping import _infer_variant_value

    assert _infer_variant_value(
        out_base=Path("/tmp/out"),
        source_dir=Path("/tmp/stk_flow=IMP"),
        variant_dim="stk_flow",
        eurostat_filters={"stk_flow": "DOM"},
    ) == "IMP"

    assert _infer_variant_value(
        out_base=Path("/tmp/out"),
        source_dir=Path("/tmp/input"),
        variant_dim="stk_flow",
        eurostat_filters={"stk_flow": "DOM"},
    ) == "DOM"


def test_write_sfc_workbook_serializes_canonical_supplementary_rows(tmp_path: Path, monkeypatch) -> None:
    """Supplementary controls must serialize their signed industry totals in column B."""
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(template_path)

    parsed = ParsedEurostatIOT(
        Z=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=["CPA_C19", "CPA_D"],
            columns=["CPA_C19", "CPA_D"],
        ),
        Y=pd.DataFrame(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            index=["CPA_C19", "CPA_D"],
            columns=["P3", "P5", "P6"],
        ),
        meta={
            "dimensions": {
                "prd_ava": {
                    "CPA_C19": "Coke and refined petroleum products",
                    "CPA_D": "Electricity, gas, steam and air conditioning",
                    "D21X31": "Taxes less subsidies on products",
                },
                "prd_use": {
                    "P3": "Final consumption expediture",
                    "P5": "Gross capital formation",
                    "P6": "Exports of goods and services",
                    "TU": "Total use",
                    "TFU": "Total final use",
                },
            },
            "identified_industry_columns": ["CPA_C19", "CPA_D"],
            "identified_final_demand_columns": ["P3", "P5", "P6", "TU", "TFU"],
        },
    )
    parsed = _with_source_rows(parsed, ["D21X31", "B1G", "P1", "P2_ADJ", "P7", "TS_BP"])
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    z_before = pd.DataFrame(
        [
            [2.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 2.0, 4.0],
        ],
        index=["CPA_C19::Motor Gasoline", "CPA_C19::Heavy Fuel Oil", "CPA_D::Electricity by hydro"],
        columns=["CPA_C19::Motor Gasoline", "CPA_C19::Heavy Fuel Oil", "CPA_D::Electricity by hydro"],
    )
    y_before = pd.DataFrame(
        {
            "P3": [8.0, 12.0, 6.0],
            "P5": [4.0, 6.0, 3.0],
            "P6": [2.0, 3.0, 1.0],
        },
        index=z_before.index,
    )

    z_after = pd.DataFrame(
        [
            [4.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [5.0, 2.0, 3.0],
        ],
        index=["CPA_C19::Motor Gasoline", "CPA_C19_remaining", "CPA_D::Hydro"],
        columns=["CPA_C19::Motor Gasoline", "CPA_C19_remaining", "CPA_D::Hydro"],
    )
    y_after = pd.DataFrame(
        {
            "P3": [8.0, 12.0, 6.0],
            "P5": [4.0, 6.0, 3.0],
            "P6": [2.0, 3.0, 1.0],
        },
        index=z_after.index,
    )
    v_after = pd.DataFrame(
        [
            [-0.25, 0.75, -2.0],
            [999.0, 999.0, 999.0],
            [888.0, 888.0, 888.0],
            [777.0, 777.0, 777.0],
            [1.0, 2.0, 3.0],
            [666.0, 666.0, 666.0],
        ],
        index=[
            "D21X31",
            "B1G",
            "P1",
            "P2_ADJ",
            "P7",
            "TS_BP",
        ],
        columns=z_after.columns,
    )
    m_after = pd.DataFrame(
        [
            [9.0, 8.0, 7.0, 27.0, 24.0],
            [999.0, 999.0, 999.0, 999.0, 999.0],
            [float("nan")] * 5,
        ],
        index=["D21X31", "P2_ADJ", "P7"],
        columns=["P3", "P5", "P6", "TU", "TFU"],
    )

    mapping_df = pd.DataFrame(
        {
            "from_sector": [
                "CPA_C19::Motor Gasoline",
                "CPA_C19::Heavy Fuel Oil",
                "CPA_D::Electricity by hydro",
            ],
            "to_sector": [
                "CPA_C19::Motor Gasoline",
                "CPA_C19_remaining",
                "CPA_D::Hydro",
            ],
            "target_label": [
                "CPA_C19: Motor Gasoline",
                "CPA_C19: Coke and refined petroleum products_remaining",
                "CPA_D: Hydro",
            ],
        }
    )

    mod._write_sfc_workbook(
        parsed=parsed,
        out_path=out_path,
        template_xlsx=template_path,
        eurostat_dataset="naio_10_cp1700",
        eurostat_filters={"geo": "AT", "time": "2021"},
        eurostat_row_dim="prd_ava",
        eurostat_col_dim="prd_use",
        eurostat_storage_folder=tmp_path,
        variant_dim="stk_flow",
        variant_value="TOTAL",
        Z_before=z_before,
        Y_before=y_before,
        Z_after=z_after,
        Y_after=y_after,
        V_after=v_after,
        M_after=m_after,
        mapping_df=mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        verbose=False,
        output_after=flow_use_total(z_after, y_after),
    )

    out_wb = load_workbook(out_path, data_only=True)
    ws = out_wb["Sheet 1"]
    header_labels = _row_values(ws, 11, 3, 3)
    assert header_labels == [
        "CPA_C19: Motor Gasoline",
        "CPA_C19: Coke and refined petroleum products_remaining",
        "CPA_D: Hydro",
    ]

    row_lookup = {
        str(ws.cell(row=row_idx, column=1).value): row_idx
        for row_idx in range(1, ws.max_row + 1)
        if ws.cell(row=row_idx, column=1).value is not None
    }

    expected_value_added = [999.0, 999.0, 999.0]
    expected_output = [888.0, 888.0, 888.0]
    expected_intermediate = [777.0, 777.0, 777.0]
    expected_taxes = [-0.25, 0.75, -2.0]
    expected_imports = [1.0, 2.0, 3.0]
    expected_supply = [666.0, 666.0, 666.0]

    # Right block above the lower rows is the current aggregated Y matrix.
    assert _row_values(ws, 13, 6, 5) == pytest.approx([8.0, 4.0, 2.0, 21.0, 14.0])
    assert _row_values(ws, 14, 6, 5) == pytest.approx([12.0, 6.0, 3.0, 27.0, 21.0])
    assert _row_values(ws, 15, 6, 5) == pytest.approx([6.0, 3.0, 1.0, 20.0, 10.0])

    assert _row_values(ws, row_lookup["Value added, gross"], 3, 3) == pytest.approx(expected_value_added)
    assert _row_values(ws, row_lookup["Output"], 3, 3) == pytest.approx(expected_output)
    assert _row_values(ws, row_lookup["Total intermediate consumption /final use"], 3, 3) == pytest.approx(expected_intermediate)
    assert _row_values(ws, row_lookup["Taxes less subsidies on products"], 3, 3) == pytest.approx(expected_taxes)
    assert _row_values(ws, row_lookup["Imports of goods and services"], 3, 3) == pytest.approx(expected_imports)
    assert _row_values(ws, row_lookup["Total supply at basic prices"], 3, 3) == pytest.approx(expected_supply)

    # Column B must be regenerated from the canonical V rows after stale
    # template cells are cleared; signed tax-minus-subsidy totals are valid.
    assert ws.cell(row=row_lookup["Taxes less subsidies on products"], column=2).value == pytest.approx(-1.5)
    assert ws.cell(row=row_lookup["Value added, gross"], column=2).value == pytest.approx(2997.0)
    assert ws.cell(row=row_lookup["Output"], column=2).value == pytest.approx(2664.0)
    assert ws.cell(row=row_lookup["Total intermediate consumption /final use"], column=2).value == pytest.approx(2331.0)
    assert ws.cell(row=row_lookup["Imports of goods and services"], column=2).value == pytest.approx(6.0)
    assert ws.cell(row=row_lookup["Total supply at basic prices"], column=2).value == pytest.approx(1998.0)

    assert _row_values(ws, 12, 6, 5) == pytest.approx([26.0, 13.0, 6.0, 68.0, 45.0])
    assert _row_values(ws, row_lookup["Taxes less subsidies on products"], 6, 5) == pytest.approx([9.0, 8.0, 7.0, 27.0, 24.0])
    assert _row_values(ws, row_lookup["Total intermediate consumption /final use"], 6, 5) == pytest.approx([999.0] * 5)
    out_wb.close()

def test_write_sfc_workbook_raises_on_validation_failure(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(template_path)

    parsed = ParsedEurostatIOT(
        Z=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=["CPA_C19", "CPA_D"],
            columns=["CPA_C19", "CPA_D"],
        ),
        Y=pd.DataFrame(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            index=["CPA_C19", "CPA_D"],
            columns=["P3", "P5", "P6"],
        ),
        meta={
            "dimensions": {
                "prd_ava": {
                    "CPA_C19": "Coke and refined petroleum products",
                    "CPA_D": "Electricity, gas, steam and air conditioning",
                },
                "prd_use": {
                    "P3": "Final consumption expediture",
                    "P5": "Gross capital formation",
                    "P6": "Exports of goods and services",
                    "TU": "Total use",
                    "TFU": "Total final use",
                },
            },
            "identified_industry_columns": ["CPA_C19", "CPA_D"],
            "identified_final_demand_columns": ["P3", "P5", "P6", "TU", "TFU"],
        },
    )
    parsed = _with_source_rows(parsed, ["B1G", "P1", "P2_ADJ"])
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_validate_parent_final_demand_reconstruction",
        lambda **_: (_ for _ in ()).throw(ValueError("Parent final-demand reconstruction failed")),
    )

    z_before = pd.DataFrame(
        [[2.0, 1.0], [1.0, 3.0]],
        index=["CPA_C19::Motor Gasoline", "CPA_D::Electricity by hydro"],
        columns=["CPA_C19::Motor Gasoline", "CPA_D::Electricity by hydro"],
    )
    y_before = pd.DataFrame(
        {"P3": [8.0, 6.0], "P5": [4.0, 3.0], "P6": [2.0, 1.0]},
        index=z_before.index,
    )

    mapping_df = pd.DataFrame(
        {
            "from_sector": ["CPA_C19::Motor Gasoline", "CPA_D::Electricity by hydro"],
            "to_sector": ["CPA_C19::Motor Gasoline", "CPA_D::Hydro"],
            "target_label": ["CPA_C19: Motor Gasoline", "CPA_D: Hydro"],
        }
    )

    v_after, m_after = _matching_transformed_supplementary(
        parsed, z_before.columns
    )

    with pytest.raises(ValueError, match="Parent final-demand reconstruction failed"):
        mod._write_sfc_workbook(
            parsed=parsed,
            out_path=out_path,
            template_xlsx=template_path,
            eurostat_dataset="naio_10_cp1700",
            eurostat_filters={"geo": "AT", "time": "2021"},
            eurostat_row_dim="prd_ava",
            eurostat_col_dim="prd_use",
            eurostat_storage_folder=tmp_path,
            variant_dim="stk_flow",
            variant_value="TOTAL",
            Z_before=z_before,
            Y_before=y_before,
            Z_after=z_before,
            Y_after=y_before,
            V_after=v_after,
            M_after=m_after,
            mapping_df=mapping_df,
            mapping_from_col="from_sector",
            mapping_to_col="to_sector",
            verbose=False,
            output_after=flow_use_total(z_before, y_before),
        )


def test_build_source_parent_by_input_uses_target_lineage_for_split_children() -> None:
    from scripts.aggregate_iot_by_mapping import _build_source_parent_by_input

    _, _, _, _, _, mapping_df = _build_lineage_toy_case()
    source_parent_by_input = _build_source_parent_by_input(
        mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        base_product_codes=["CPA_C10-12", "CPA_C20"],
    )

    assert source_parent_by_input["p15.j"] == "CPA_C10-12"
    assert source_parent_by_input["p16"] == "CPA_C10-12"
    assert source_parent_by_input["p24.g"] == "CPA_C20"


def test_parent_validation_uses_detailed_tfu_without_double_counting() -> None:
    from scripts.aggregate_iot_by_mapping import _validate_parent_final_demand_reconstruction

    _, _, y_before, _, y_after, _ = _build_lineage_toy_case()
    y_before = y_before.copy()
    y_after = y_after.copy()
    y_before["TFU"] = [999.0, 999.0, 999.0]
    y_after["TFU"] = [888.0, 888.0, 888.0]

    _validate_parent_final_demand_reconstruction(
        product_codes_order=y_after.index.tolist(),
        source_parent_by_input={
            "p15.j": "CPA_C10-12",
            "p16": "CPA_C10-12",
            "p24.g": "CPA_C20",
        },
        parents_by_output={
            "CPA_C10-12_FOOD_AND_BEVERAGES": ("CPA_C10-12",),
            "CPA_C10-12_TOBACCO_PRODUCTS": ("CPA_C10-12",),
            "CPA_C20_BIOGASOLINE": ("CPA_C20",),
        },
        base_product_codes=["CPA_C10-12", "CPA_C20"],
        Y_before=y_before,
        Y_ord=y_after,
        fd_codes_order_no_total=["P3", "P5", "P6", "TFU"],
    )


def test_write_sfc_workbook_passes_strict_lineage_validation(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(
        template_path,
        product_labels=(
            "Food, beverages and tobacco products",
            "Chemicals and chemical products",
        ),
    )

    parsed, z_before, y_before, z_after, y_after, mapping_df = _build_lineage_toy_case()
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    v_after, m_after = _matching_transformed_supplementary(
        parsed, z_after.columns
    )

    mod._write_sfc_workbook(
        parsed=parsed,
        out_path=out_path,
        template_xlsx=template_path,
        eurostat_dataset="naio_10_cp1700",
        eurostat_filters={"geo": "AT", "time": "2021"},
        eurostat_row_dim="prd_ava",
        eurostat_col_dim="prd_use",
        eurostat_storage_folder=tmp_path,
        variant_dim="stk_flow",
        variant_value="TOTAL",
        Z_before=z_before,
        Y_before=y_before,
        Z_after=z_after,
        Y_after=y_after,
        V_after=v_after,
        M_after=m_after,
        mapping_df=mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        verbose=False,
        output_after=flow_use_total(z_after, y_after),
    )

    assert out_path.is_file()


def test_write_sfc_workbook_fails_when_lineage_parent_total_drifts(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(
        template_path,
        product_labels=(
            "Food, beverages and tobacco products",
            "Chemicals and chemical products",
        ),
    )

    parsed, z_before, y_before, z_after, y_after, mapping_df = _build_lineage_toy_case()
    y_after = y_after.copy()
    y_after.loc["CPA_C10-12_TOBACCO_PRODUCTS", "P3"] = 6.0

    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    v_after, m_after = _matching_transformed_supplementary(
        parsed, z_after.columns
    )

    with pytest.raises(ValueError, match="Parent final-demand reconstruction failed"):
        mod._write_sfc_workbook(
            parsed=parsed,
            out_path=out_path,
            template_xlsx=template_path,
            eurostat_dataset="naio_10_cp1700",
            eurostat_filters={"geo": "AT", "time": "2021"},
            eurostat_row_dim="prd_ava",
            eurostat_col_dim="prd_use",
            eurostat_storage_folder=tmp_path,
            variant_dim="stk_flow",
            variant_value="TOTAL",
            Z_before=z_before,
            Y_before=y_before,
            Z_after=z_after,
            Y_after=y_after,
            V_after=v_after,
            M_after=m_after,
            mapping_df=mapping_df,
            mapping_from_col="from_sector",
            mapping_to_col="to_sector",
            verbose=False,
            output_after=flow_use_total(z_after, y_after),
        )



def test_write_sfc_workbook_template_numeric_cells_are_not_source_applicability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template_path=tmp_path/"template.xlsx"
    _build_template(template_path,product_labels=(
        "Food, beverages and tobacco products", "Chemicals and chemical products",
    ))
    parsed,z_before,y_before,z_after,y_after,mapping_df=_build_lineage_toy_case()
    assert parsed.V is not None and parsed.M is not None
    parsed=replace(parsed,V=parsed.V.drop(index="D21X31"),M=parsed.M.drop(index="D21X31"))
    monkeypatch.setattr(eurostat_iot,"load_eurostat_iot",lambda **_: parsed)
    monkeypatch.setattr(mod,"_write_excel_mapping_sheet",lambda *args,**kwargs: None)
    v_after,m_after=_matching_transformed_supplementary(parsed,z_after.columns)
    out_path=tmp_path/"out.xlsx"
    mod._write_sfc_workbook(
        parsed=parsed,
        out_path=out_path,template_xlsx=template_path,eurostat_dataset="naio_10_cp1700",
        eurostat_filters={"geo":"AT","time":"2021","freq":"A","unit":"MIO_EUR"},
        eurostat_row_dim="prd_ava",eurostat_col_dim="prd_use",eurostat_storage_folder=tmp_path,
        variant_dim="stk_flow",variant_value="TOTAL",Z_before=z_before,Y_before=y_before,
        Z_after=z_after,Y_after=y_after,V_after=v_after,M_after=m_after,mapping_df=mapping_df,
        mapping_from_col="from_sector",mapping_to_col="to_sector",verbose=False,
        output_after=flow_use_total(z_after,y_after),
    )
    assert out_path.is_file()


def test_sfc_layout_template_sentinels_cannot_affect_generated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    labels=("Food, beverages and tobacco products","Chemicals and chemical products")
    normal=tmp_path/"normal.xlsx"; sentinel=tmp_path/"sentinel.xlsx"
    _build_template(normal,product_labels=labels); _build_template(sentinel,product_labels=labels)
    wb=load_workbook(sentinel); ws=wb["Sheet 1"]
    ws["A1"]="STALE AUSTRIA TEMPLATE"; ws["B1"]=2021; ws["A2"]="=1+2"
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value,(int,float)): cell.value=9.87654321e123
    stale=wb.create_sheet("STALE NUMERICAL SOURCE"); stale["A1"]=123456789.0
    wb.save(sentinel); wb.close()

    parsed,z_before,y_before,z_after,y_after,mapping_df=_build_lineage_toy_case()
    monkeypatch.setattr(eurostat_iot,"load_eurostat_iot",lambda **_: parsed)
    monkeypatch.setattr(mod,"_write_excel_mapping_sheet",lambda *args,**kwargs: None)
    v_after,m_after=_matching_transformed_supplementary(parsed,z_after.columns)
    def generate(template,out):
        mod._write_sfc_workbook(
            parsed=parsed,
            out_path=out,template_xlsx=template,eurostat_dataset="naio_10_cp1700",
            eurostat_filters={"geo":"EU27_2020","time":"2023","freq":"A","unit":"MIO_EUR"},
            eurostat_row_dim="prd_ava",eurostat_col_dim="prd_use",eurostat_storage_folder=tmp_path,
            variant_dim="stk_flow",variant_value="TOTAL",Z_before=z_before,Y_before=y_before,
            Z_after=z_after,Y_after=y_after,V_after=v_after,M_after=m_after,mapping_df=mapping_df,
            mapping_from_col="from_sector",mapping_to_col="to_sector",verbose=False,
            output_after=flow_use_total(z_after,y_after),
        )
    normal_out=tmp_path/"normal_out.xlsx"; sentinel_out=tmp_path/"sentinel_out.xlsx"
    generate(normal,normal_out); generate(sentinel,sentinel_out)
    a=load_workbook(normal_out,data_only=False); b=load_workbook(sentinel_out,data_only=False)
    assert a.sheetnames == b.sheetnames
    assert "STALE NUMERICAL SOURCE" not in b.sheetnames
    av=[[c.value for c in row] for row in a["Sheet 1"].iter_rows()]
    bv=[[c.value for c in row] for row in b["Sheet 1"].iter_rows()]
    assert av == bv
    assert b["Sheet 1"]["A1"].value is None and b["Sheet 1"]["B1"].value is None
    assert b["Sheet 1"]["A2"].value is None
    summary=b["Summary"]
    md={str(summary.cell(r,1).value):summary.cell(r,2).value for r in range(2,summary.max_row+1)}
    assert md["Dataset"] == "naio_10_cp1700"
    assert md["Geo"] == "EU27_2020" and md["Time"] == "2023"
    assert md["Frequency"] == "A" and md["Unit"] == "MIO_EUR"
    assert md["Stock or flow"] == "TOTAL"
    assert "no template numerical evidence" in str(md["Template contract"])
    a.close(); b.close()



def test_historical_layout_cannot_define_current_product_schema_or_csv_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workbook toggle and stale Austria product labels cannot alter current CSV outputs."""
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template = tmp_path / "austria_layout.xlsx"
    _build_template(
        template,
        product_labels=(
            "Food, beverages and tobacco products",
            "Imputed rents of owner-occupied dwellings",
        ),
    )
    parsed, z_before, y_before, _, _, _ = _build_lineage_toy_case()
    parsed = replace(parsed, V=None, M=None)
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    indir = tmp_path / "input"
    indir.mkdir()
    z_before.to_csv(indir / "Z_split.csv")
    y_before.to_csv(indir / "Y_split.csv")
    flow_use_total(z_before, y_before).rename("x").to_csv(indir / "x_split.csv")

    mapping_csv = tmp_path / "mapping.csv"
    pd.DataFrame(
        {
            "mapping_set": ["test", "test", "test"],
            "source_code": ["p15.j", "p16", "p24.g"],
            "target_code": [
                "CPA_C10-12_FOOD_AND_BEVERAGES",
                "CPA_C10-12_TOBACCO_PRODUCTS",
                "CPA_C20_BIOGASOLINE",
            ],
            "source_parent_code": ["CPA_C10-12", "CPA_C10-12", "CPA_C20"],
            "enabled": [True, True, True],
            "notes": ["", "", ""],
        }
    ).to_csv(mapping_csv, index=False)

    filters = '{"geo":"EU27_2020","time":"2021","freq":"A","unit":"MIO_EUR","stk_flow":"TOTAL"}'
    common = [
        "--indir", str(indir),
        "--mapping-csv", str(mapping_csv),
        "--no-xlsx",
        "--eurostat-dataset", "naio_10_cp1700",
        "--eurostat-filters", filters,
        "--eurostat-row-dim", "prd_ava",
        "--eurostat-col-dim", "prd_use",
    ]
    without_workbook = tmp_path / "without_workbook"
    with_workbook = tmp_path / "with_workbook"
    assert mod.main([*common, "--outdir", str(without_workbook)]) == 0
    assert mod.main(
        [
            *common,
            "--outdir", str(with_workbook),
            "--write-sfc-workbook",
            "--sfc-template-xlsx", str(template),
            "--sfc-outname", "presentation.xlsx",
        ]
    ) == 0

    off_csv = sorted(path.name for path in without_workbook.glob("*.csv"))
    on_csv = sorted(path.name for path in with_workbook.glob("*.csv"))
    assert off_csv == on_csv
    assert "sector_contract.csv" in off_csv
    assert "source_sector_contract.csv" not in off_csv
    for name in off_csv:
        assert (without_workbook / name).read_bytes() == (with_workbook / name).read_bytes()
    assert (with_workbook / "presentation.xlsx").is_file()



def test_sfc_supplementary_alignment_rejects_template_label_as_code_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A presentation label must not repair a transformed V row-code mismatch."""
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template = tmp_path / "template.xlsx"
    _build_template(
        template,
        product_labels=(
            "Food, beverages and tobacco products",
            "Chemicals and chemical products",
        ),
    )
    parsed, z_before, y_before, z_after, y_after, mapping_df = _build_lineage_toy_case()
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)
    v_after, m_after = _matching_transformed_supplementary(parsed, z_after.columns)
    v_after = v_after.rename(index={"D21X31": "Taxes less subsidies on products"})

    with pytest.raises(
        ValueError, match="Transformed V is missing source supplementary row 'D21X31'"
    ):
        mod._write_sfc_workbook(
            parsed=parsed,
            out_path=tmp_path / "bad.xlsx",
            template_xlsx=template,
            eurostat_dataset="naio_10_cp1700",
            eurostat_filters={"geo": "AT", "time": "2021"},
            eurostat_row_dim="prd_ava",
            eurostat_col_dim="prd_use",
            eurostat_storage_folder=tmp_path,
            variant_dim="stk_flow",
            variant_value="TOTAL",
            Z_before=z_before,
            Y_before=y_before,
            Z_after=z_after,
            Y_after=y_after,
            V_after=v_after,
            M_after=m_after,
            mapping_df=mapping_df,
            mapping_from_col="from_sector",
            mapping_to_col="to_sector",
            verbose=False,
            output_after=flow_use_total(z_after, y_after),
        )


def test_sfc_workbook_rejects_incomplete_authoritative_output_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod
    template_path=tmp_path/"template.xlsx"
    _build_template(template_path,product_labels=(
        "Food, beverages and tobacco products","Chemicals and chemical products",
    ))
    parsed,z_before,y_before,z_after,y_after,mapping_df=_build_lineage_toy_case()
    monkeypatch.setattr(eurostat_iot,"load_eurostat_iot",lambda **_: parsed)
    monkeypatch.setattr(mod,"_write_excel_mapping_sheet",lambda *args,**kwargs: None)
    v_after,m_after=_matching_transformed_supplementary(parsed,z_after.columns)
    incomplete=flow_use_total(z_after,y_after).iloc[:-1]
    with pytest.raises(ValueError,match="must cover the transformed product axis exactly"):
        mod._write_sfc_workbook(
            parsed=parsed,
            out_path=tmp_path/"bad.xlsx",template_xlsx=template_path,eurostat_dataset="naio_10_cp1700",
            eurostat_filters={"geo":"AT","time":"2021"},eurostat_row_dim="prd_ava",
            eurostat_col_dim="prd_use",eurostat_storage_folder=tmp_path,variant_dim="stk_flow",
            variant_value="TOTAL",Z_before=z_before,Y_before=y_before,Z_after=z_after,Y_after=y_after,
            V_after=v_after,M_after=m_after,mapping_df=mapping_df,mapping_from_col="from_sector",
            mapping_to_col="to_sector",verbose=False,output_after=incomplete,
        )


def test_write_excel_mapping_sheet_allows_missing_co2_for_import_flow(capsys) -> None:
    from openpyxl import Workbook
    from scripts.aggregate_iot_by_mapping import _write_excel_mapping_sheet

    wb = Workbook()
    mapping_df = pd.DataFrame(
        {
            "from_sector": ["p10.f@CPA_C19"],
            "to_sector": ["CPA_C19_COAL"],
        }
    )
    disaggregated_y = pd.DataFrame(
        {"P3": [1.0], "P5": [0.0], "P6": [0.0]},
        index=["p10.f@CPA_C19"],
    )
    aggregated_y = pd.DataFrame(
        {"P3": [1.0], "P5": [0.0], "P6": [0.0]},
        index=["CPA_C19_COAL"],
    )
    aggregated_x = pd.Series([1.0], index=["CPA_C19_COAL"], dtype="float64")

    _write_excel_mapping_sheet(
        wb,
        sheet_name="IO Aggregation Mapping",
        mapping_df=mapping_df,
        disaggregated_Y=disaggregated_y,
        aggregated_Y=aggregated_y,
        aggregated_x=aggregated_x,
        aggregated_sector_order=["CPA_C19_COAL"],
        variant_value="IMP",
        allow_missing_sfc_co2=True,
        verbose=True,
    )

    stdout = capsys.readouterr().out
    assert "WARNING: SFC workbook mapping sheet is omitting production-based CO2 intensity columns" in stdout
    assert "WARNING: SFC workbook mapping sheet is omitting use-phase CO2 intensity columns" in stdout
    assert "WARNING: SFC workbook re-aggregated summary is omitting production-based CO2" in stdout
    assert "WARNING: SFC workbook re-aggregated summary is omitting use-phase CO2" in stdout
    assert "for IMP flow" in stdout

    ws = wb["IO Aggregation Mapping"]
    headers = [cell.value for cell in ws[1]]
    assert not any("CO2 intensity" in str(header) for header in headers)

    summary_ws = wb["Re-aggregated Summary"]
    summary_headers = [cell.value for cell in summary_ws[1]]
    assert summary_headers == [
        "To Sector",
        "Monetary output (M€)",
        "Domestic final demand (M€)",
        "Exports (M€)",
        "Stocks & valuables (M€)",
    ]
    assert summary_ws.cell(row=2, column=2).value == 1.0


def test_write_excel_mapping_sheet_requires_co2_for_non_import_flow() -> None:
    from openpyxl import Workbook
    from scripts.aggregate_iot_by_mapping import _write_excel_mapping_sheet

    with pytest.raises(ValueError, match="disaggregated_sector_emission_intensity is required"):
        _write_excel_mapping_sheet(
            Workbook(),
            sheet_name="IO Aggregation Mapping",
            mapping_df=pd.DataFrame({"from_sector": ["p10.f"], "to_sector": ["CPA_C19"]}),
            allow_missing_sfc_co2=False,
        )


def test_generated_workbook_metadata_is_rebuilt() -> None:
    """Guardrail: layout templates cannot seed or mutate canonical data."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_iot_by_mapping.py"
    text = source.read_text(encoding="utf-8")
    assert "for sheet_name in list(wb.sheetnames)" in text
    assert 'if sheet_name != "Sheet 1"' in text
    assert 'ws_sum = wb.create_sheet("Summary")' in text
    assert "numeric primitive cells absent from SDMX" not in text
    assert "template_v_values" not in text and "template_m_values" not in text
    assert "_reorder_aggregation_result" not in text
    assert "normalise_supplementary_row_label" not in text
    assert "template_xlsx=None" in text
    assert "_supplementary_fd_row_from_current_shares" not in text
    assert 'output_after=res.x' in text


def test_identity_mapping_parent_lineage_keeps_template_alignment() -> None:
    from scripts.aggregate_iot_by_mapping import (
        _build_source_parent_by_input,
        _resolve_product_template_alignment,
    )

    template = Path("template.xlsx")
    mapping_df = pd.DataFrame(
        {
            "from_sector": ["p15.a", "p24.d"],
            "to_sector": ["p15.a", "p24.d"],
            "source_parent_code": ["CPA_C10-12", "CPA_C20"],
        }
    )
    lineage = _build_source_parent_by_input(
        mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        base_product_codes=["CPA_C10-12", "CPA_C20"],
    )
    assert lineage == {"p15.a": "CPA_C10-12", "p24.d": "CPA_C20"}

    effective, order, primary, parents = _resolve_product_template_alignment(
        template_xlsx=template,
        base_product_codes=["CPA_C10-12", "CPA_C20"],
        output_product_codes=["p15.a", "p24.d"],
        mapping_df=mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        verbose=False,
    )
    assert effective == template
    assert order == ["p15.a", "p24.d"]
    assert primary == {"p15.a": "CPA_C10-12", "p24.d": "CPA_C20"}
    assert parents == {
        "p15.a": ("CPA_C10-12",),
        "p24.d": ("CPA_C20",),
    }



def test_missing_sfc_co2_reason_distinguishes_imp_and_total() -> None:
    from scripts.aggregate_iot_by_mapping import _missing_sfc_co2_reason

    imp_prod = _missing_sfc_co2_reason("IMP", use_phase=False)
    imp_use = _missing_sfc_co2_reason("IMP", use_phase=True)
    total_prod = _missing_sfc_co2_reason("TOTAL", use_phase=False)
    total_use = _missing_sfc_co2_reason("TOTAL", use_phase=True)

    assert "IMP flow" in imp_prod
    assert "origin-weighted import" in imp_prod
    assert "IMP flow" in imp_use
    assert "TOTAL flow" in total_prod
    assert "domestic-production" in total_prod
    assert "TOTAL flow" in total_use
    assert "for import flow" not in " ".join([imp_prod, imp_use, total_prod, total_use])


def test_template_alignment_failure_is_independent_of_verbose() -> None:
    from scripts.aggregate_iot_by_mapping import _resolve_product_template_alignment

    with pytest.raises(ValueError, match="Could not resolve current Eurostat parents"):
        _resolve_product_template_alignment(
            template_xlsx=Path("template.xlsx"),
            base_product_codes=["CPA_A"],
            output_product_codes=["CUSTOM"],
            mapping_df=pd.DataFrame(
                {"from_sector": ["unmapped"], "to_sector": ["CUSTOM"]}
            ),
            mapping_from_col="from_sector",
            mapping_to_col="to_sector",
            verbose=False,
        )


def test_clear_managed_aggregation_outputs_removes_only_stage_files(tmp_path: Path) -> None:
    from scripts.aggregate_iot_by_mapping import _clear_managed_aggregation_outputs

    stale = [
        "V_agg.csv",
        "M_agg.csv",
        "aggregated_sector_co2_intensity.csv",
        "exio_final_demand_co2_intensity.csv",
        "aggregation_info.json",
    ]
    for name in stale:
        (tmp_path / name).write_text("stale\\n", encoding="utf-8")
    unrelated = tmp_path / "keep_me.txt"
    unrelated.write_text("keep\\n", encoding="utf-8")

    _clear_managed_aggregation_outputs(tmp_path)

    assert all(not (tmp_path / name).exists() for name in stale)
    assert unrelated.read_text(encoding="utf-8") == "keep\\n"



def test_production_based_intensity_preserves_unavailable_emissions() -> None:
    from scripts.aggregate_iot_by_mapping import _compute_production_based_intensity

    frame = pd.DataFrame(
        {
            "disaggregated_sector": ["supported", "unavailable"],
            "emission": [4.0, float("nan")],
            "monetary_output": [2.0, 3.0],
            "unit": ["kg", "kg"],
        }
    )
    output, intensity, unit = _compute_production_based_intensity(frame)
    assert unit == "kg"
    assert float(output.loc["unavailable"]) == 3.0
    assert pd.isna(intensity.loc["unavailable"])
    assert float(intensity.loc["supported"]) == 2.0
