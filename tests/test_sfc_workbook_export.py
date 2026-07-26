# pyright: reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalOperand=false, reportIndexIssue=false
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from ecolecon_mrio.iot_flow_semantics import flow_use_total


def _build_template(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet 1"

    ws.cell(row=11, column=1).value = "PRD_USE (Labels)"
    ws.cell(row=11, column=2).value = "Total"
    ws.cell(row=11, column=3).value = "Coke and refined petroleum products"
    ws.cell(row=11, column=4).value = "Electricity, gas, steam and air conditioning"
    ws.cell(row=11, column=5).value = "Final consumption expediture"
    ws.cell(row=11, column=6).value = "Gross capital formation"
    ws.cell(row=11, column=7).value = "Exports of goods and services"
    ws.cell(row=11, column=8).value = "Total use"
    ws.cell(row=11, column=9).value = "Total final use"

    ws.cell(row=12, column=1).value = "Total"
    ws.cell(row=13, column=1).value = "Coke and refined petroleum products"
    ws.cell(row=14, column=1).value = "Electricity, gas, steam and air conditioning"

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
    meta = dict(parsed.meta)
    dimensions = {key: dict(value) for key, value in meta["dimensions"].items()}
    dimensions["prd_ava"].update({code: labels[code] for code in row_codes})
    meta["dimensions"] = dimensions
    meta["identified_supplementary_rows"] = list(row_codes)
    return replace(
        parsed,
        meta=meta,
        V=pd.DataFrame(0.0, index=row_codes, columns=parsed.Z.columns),
        M=pd.DataFrame(0.0, index=row_codes, columns=parsed.Y.columns),
    )


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


def test_write_sfc_workbook_rebuilds_identity_rows_from_aggregated_iot(tmp_path: Path, monkeypatch) -> None:
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
            [0.25, 0.75, 2.0],
            [999.0, 999.0, 999.0],
            [888.0, 888.0, 888.0],
            [777.0, 777.0, 777.0],
            [1.0, 2.0, 3.0],
            [666.0, 666.0, 666.0],
        ],
        index=[
            "D21X31",
            "Value added, gross",
            "Output",
            "Total intermediate consumption /final use",
            "Imports of goods and services",
            "Total supply at basic prices",
        ],
        columns=z_after.columns,
    )
    m_after = pd.DataFrame(
        [
            [9.0, 8.0, 7.0, 27.0, 24.0],
            [999.0, 999.0, 999.0, 999.0, 999.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        index=["D21X31", "Total intermediate consumption /final use", "Imports of goods and services"],
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

    ws = load_workbook(out_path, data_only=True)["Sheet 1"]
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

    x_after = (z_after.sum(axis=1) + y_after.sum(axis=1)).astype(float)
    col_sums_z = z_after.sum(axis=0).astype(float)
    expected_value_added = [999.0, 999.0, 999.0]
    expected_output = [888.0, 888.0, 888.0]
    expected_intermediate = [777.0, 777.0, 777.0]
    expected_taxes = [0.25, 0.75, 2.0]
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

    assert _row_values(ws, 12, 6, 5) == pytest.approx([26.0, 13.0, 6.0, 68.0, 45.0])
    assert _row_values(ws, row_lookup["Taxes less subsidies on products"], 6, 5) == pytest.approx([9.0, 8.0, 7.0, 27.0, 24.0])
    assert _row_values(ws, row_lookup["Total intermediate consumption /final use"], 6, 5) == pytest.approx([999.0] * 5)


def test_validate_template_reaggregation_raises_on_supplementary_parent_drift(tmp_path: Path) -> None:
    from openpyxl import load_workbook
    from scripts.aggregate_iot_by_mapping import _template_table_layout, _validate_template_reaggregation

    template_path = tmp_path / "template.xlsx"
    _build_template(template_path)
    tpl_ws = load_workbook(template_path, data_only=True)["Sheet 1"]
    tpl_layout = _template_table_layout(tpl_ws)

    Z_ord = pd.DataFrame(
        [[4.0, 1.0, 2.0], [3.0, 2.0, 1.0], [5.0, 2.0, 3.0]],
        index=["CPA_C19::Motor Gasoline", "CPA_C19_remaining", "CPA_D::Hydro"],
        columns=["CPA_C19::Motor Gasoline", "CPA_C19_remaining", "CPA_D::Hydro"],
    )
    Y_ord = pd.DataFrame(
        {"P3": [8.0, 12.0, 6.0], "P5": [4.0, 6.0, 3.0], "P6": [2.0, 3.0, 1.0]},
        index=Z_ord.index,
    )
    lower_row_vectors = {
        "Imports of goods and services": pd.Series(
            [99.0, 1.0, 8.0],
            index=Z_ord.index,
            dtype="float64",
        ),
    }
    lower_fd_vectors: dict[str, pd.Series] = {}

    with pytest.raises(ValueError, match="Template reaggregation validation failed"):
        _validate_template_reaggregation(
            product_codes_order=Z_ord.index.tolist(),
            base_product_codes=["CPA_C19", "CPA_D"],
            parents_by_output={
                "CPA_C19::Motor Gasoline": ("CPA_C19",),
                "CPA_C19_remaining": ("CPA_C19",),
                "CPA_D::Hydro": ("CPA_D",),
            },
            fd_codes_order_no_total=["P3", "P5", "P6", "TU", "TFU"],
            Y_before=Y_ord,
            tpl_ws=tpl_ws,
            tpl_layout=tpl_layout,
            Z_ord=Z_ord,
            Y_ord=Y_ord,
            output_after=flow_use_total(Z_ord, Y_ord),
            lower_row_vectors=lower_row_vectors,
            lower_fd_vectors=lower_fd_vectors,
        )


def test_validate_template_reaggregation_skips_parent_fd_check_for_multi_parent_outputs(tmp_path: Path) -> None:
    from openpyxl import load_workbook
    from scripts.aggregate_iot_by_mapping import _template_table_layout, _validate_template_reaggregation

    template_path = tmp_path / "template.xlsx"
    _build_template(template_path)
    tpl_ws = load_workbook(template_path, data_only=True)["Sheet 1"]
    tpl_layout = _template_table_layout(tpl_ws)

    Z_ord = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["ENERGY_BROWN", "ENERGY_GREEN"],
        columns=["ENERGY_BROWN", "ENERGY_GREEN"],
    )
    Y_ord = pd.DataFrame(
        {"P3": [50.0, 60.0], "P5": [10.0, 20.0], "P6": [5.0, 6.0]},
        index=Z_ord.index,
    )
    Y_before = pd.DataFrame(
        {"P3": [7.0, 8.0], "P5": [2.0, 3.0], "P6": [1.0, 1.0]},
        index=["CPA_C19::fuel", "CPA_D::power"],
    )

    _validate_template_reaggregation(
        product_codes_order=Z_ord.index.tolist(),
        base_product_codes=["CPA_C19", "CPA_D"],
        parents_by_output={
            "ENERGY_BROWN": ("CPA_C19", "CPA_D"),
            "ENERGY_GREEN": ("CPA_C19", "CPA_D"),
        },
        fd_codes_order_no_total=["P3", "P5", "P6", "TU", "TFU"],
        Y_before=Y_before,
        tpl_ws=tpl_ws,
        tpl_layout=tpl_layout,
        Z_ord=Z_ord,
        Y_ord=Y_ord,
        output_after=flow_use_total(Z_ord, Y_ord),
        lower_row_vectors={},
        lower_fd_vectors={},
    )


def test_validate_template_reaggregation_skips_supplementary_parent_check_when_disabled(tmp_path: Path) -> None:
    from openpyxl import load_workbook
    from scripts.aggregate_iot_by_mapping import _template_table_layout, _validate_template_reaggregation

    template_path = tmp_path / "template.xlsx"
    _build_template(template_path)
    tpl_ws = load_workbook(template_path, data_only=True)["Sheet 1"]
    tpl_layout = _template_table_layout(tpl_ws)

    Z_ord = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]],
        index=["CPA_C19::Motor Gasoline", "CPA_D::Hydro"],
        columns=["CPA_C19::Motor Gasoline", "CPA_D::Hydro"],
    )
    Y_ord = pd.DataFrame(
        {"P3": [0.0, 0.0], "P5": [0.0, 0.0], "P6": [0.0, 0.0]},
        index=Z_ord.index,
    )
    Y_before = Y_ord.copy()

    _validate_template_reaggregation(
        product_codes_order=Z_ord.index.tolist(),
        base_product_codes=["CPA_C19", "CPA_D"],
        parents_by_output={
            "CPA_C19::Motor Gasoline": ("CPA_C19",),
            "CPA_D::Hydro": ("CPA_D",),
        },
        fd_codes_order_no_total=["P3", "P5", "P6", "TU", "TFU"],
        Y_before=Y_before,
        tpl_ws=tpl_ws,
        tpl_layout=tpl_layout,
        Z_ord=Z_ord,
        Y_ord=Y_ord,
        output_after=flow_use_total(Z_ord, Y_ord),
        lower_row_vectors={
            "Taxes less subsidies on products": pd.Series(0.0, index=Z_ord.index, dtype="float64"),
        },
        lower_fd_vectors={},
        validate_template_parent_totals=False,
    )


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
        "_validate_template_reaggregation",
        lambda **_: (_ for _ in ()).throw(ValueError("Template reaggregation validation failed for total-row fd=P3")),
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

    with pytest.raises(ValueError, match="Template reaggregation validation failed"):
        mod._write_sfc_workbook(
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


def test_template_parent_total_audit_uses_detailed_tfu_columns_without_double_counting() -> None:
    from scripts.aggregate_iot_by_mapping import _build_template_parent_total_audit

    _, _, y_before, _, y_after, _ = _build_lineage_toy_case()
    y_before = y_before.copy()
    y_after = y_after.copy()
    y_before["TFU"] = [999.0, 999.0, 999.0]
    y_after["TFU"] = [888.0, 888.0, 888.0]

    audit = _build_template_parent_total_audit(
        variant_value="TOTAL",
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
        Y_after=y_after,
        fd_codes_order_no_total=["P3", "P5", "P6", "TFU"],
    )

    c10_tfu = audit[(audit["parent_code"] == "CPA_C10-12") & (audit["fd_code"] == "TFU")].iloc[0]
    assert float(c10_tfu["before_parent_total"]) == pytest.approx(25.0)
    assert float(c10_tfu["after_child_sum"]) == pytest.approx(25.0)
    assert float(c10_tfu["difference"]) == pytest.approx(0.0)
    assert c10_tfu["status"] == "ok"
    assert float(c10_tfu["before_parent_total"]) != pytest.approx(999.0)
    assert float(c10_tfu["after_child_sum"]) != pytest.approx(888.0)


def test_write_sfc_workbook_writes_lineage_audit_and_passes_strict_validation(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(template_path)

    parsed, z_before, y_before, z_after, y_after, mapping_df = _build_lineage_toy_case()
    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    mod._write_sfc_workbook(
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
        mapping_df=mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        verbose=False,
        output_after=flow_use_total(z_after, y_after),
    )

    audit_path = out_path.with_name("template_parent_total_audit.csv")
    audit = pd.read_csv(audit_path)
    c10 = audit[(audit["parent_code"] == "CPA_C10-12") & (audit["fd_code"] == "P3")].iloc[0]
    assert float(c10["before_parent_total"]) == pytest.approx(15.0)
    assert float(c10["after_child_sum"]) == pytest.approx(15.0)
    assert float(c10["difference"]) == pytest.approx(0.0)
    assert c10["status"] == "ok"
    assert int(c10["n_children"]) == 2
    assert c10["child_codes"] == "CPA_C10-12_FOOD_AND_BEVERAGES|CPA_C10-12_TOBACCO_PRODUCTS"


def test_write_sfc_workbook_fails_when_lineage_parent_total_drifts(tmp_path: Path, monkeypatch) -> None:
    import ecolecon_mrio.eurostat_iot as eurostat_iot
    from scripts import aggregate_iot_by_mapping as mod

    template_path = tmp_path / "template.xlsx"
    out_path = tmp_path / "out.xlsx"
    _build_template(template_path)

    parsed, z_before, y_before, z_after, y_after, mapping_df = _build_lineage_toy_case()
    y_after = y_after.copy()
    y_after.loc["CPA_C10-12_TOBACCO_PRODUCTS", "P3"] = 6.0

    monkeypatch.setattr(eurostat_iot, "load_eurostat_iot", lambda **_: parsed)
    monkeypatch.setattr(mod, "_write_excel_mapping_sheet", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Template reaggregation validation failed"):
        mod._write_sfc_workbook(
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
            mapping_df=mapping_df,
            mapping_from_col="from_sector",
            mapping_to_col="to_sector",
            verbose=False,
            output_after=flow_use_total(z_after, y_after),
        )

    audit_path = out_path.with_name("template_parent_total_audit.csv")
    audit = pd.read_csv(audit_path)
    c10 = audit[(audit["parent_code"] == "CPA_C10-12") & (audit["fd_code"] == "P3")].iloc[0]
    assert float(c10["before_parent_total"]) == pytest.approx(15.0)
    assert float(c10["after_child_sum"]) == pytest.approx(16.0)
    assert float(c10["difference"]) == pytest.approx(1.0)
    assert c10["status"] == "drift"


def test_canonical_workbook_wrapper_does_not_pass_parent_total_skip_flag() -> None:
    script_text = Path("scripts/pipelines/at_2021_workbook_reaggregation.sh").read_text(encoding="utf-8")
    assert "--skip-template-parent-totals" not in script_text


def test_validate_template_reaggregation_raises_on_workbook_fd_total_drift(tmp_path: Path, monkeypatch) -> None:
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
        mapping_df=mapping_df,
        mapping_from_col="from_sector",
        mapping_to_col="to_sector",
        verbose=False,
        output_after=flow_use_total(z_after, y_after),
    )

    tpl_ws = load_workbook(template_path, data_only=True)["Sheet 1"]
    workbook = load_workbook(out_path)
    workbook_ws = workbook["Sheet 1"]
    workbook_layout = mod._template_table_layout(workbook_ws)
    workbook_ws.cell(row=int(workbook_layout["total_row"]), column=int(workbook_layout["first_fd_col"])).value = 999.0
    workbook.save(out_path)
    workbook_ws = load_workbook(out_path, data_only=True)["Sheet 1"]

    with pytest.raises(ValueError, match=r"Template reaggregation validation failed for total-row fd=P3"):
        mod._validate_template_reaggregation(
            product_codes_order=z_after.index.tolist(),
            base_product_codes=["CPA_C19", "CPA_D"],
            parents_by_output={
                "CPA_C19::Motor Gasoline": ("CPA_C19",),
                "CPA_C19_remaining": ("CPA_C19",),
                "CPA_D::Hydro": ("CPA_D",),
            },
            fd_codes_order_no_total=["P3", "P5", "P6", "TU", "TFU"],
            Y_before=y_before,
            tpl_ws=tpl_ws,
            tpl_layout=mod._template_table_layout(tpl_ws),
            workbook_ws=workbook_ws,
            workbook_layout=mod._template_table_layout(workbook_ws),
            Z_ord=z_after,
            Y_ord=y_after,
            output_after=flow_use_total(z_after, y_after),
            lower_row_vectors={},
            lower_fd_vectors={},
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
        allow_missing_sfc_co2=True,
        verbose=True,
    )

    stdout = capsys.readouterr().out
    assert "WARNING: SFC workbook mapping sheet is omitting production-based CO2 intensity columns" in stdout
    assert "WARNING: SFC workbook mapping sheet is omitting use-phase CO2 intensity columns" in stdout
    assert "WARNING: SFC workbook re-aggregated summary is omitting production-based CO2" in stdout
    assert "WARNING: SFC workbook re-aggregated summary is omitting use-phase CO2" in stdout
    assert "import flow" in stdout

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


def test_generated_workbook_metadata_does_not_retain_template_values() -> None:
    """Guardrail: canonical writers must generate Summary/Structure, not preserve template data."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_iot_by_mapping.py"
    text = source.read_text(encoding="utf-8")
    assert 'for sheet_name in ("Mapping", "Aggregation", "Summary", "Structure")' in text
    assert 'ws_sum = wb.create_sheet("Summary")' in text
    assert "canonical_V_serializer" in text
    assert "_supplementary_fd_row_from_current_shares" not in text
    assert 'output_after=res.x' in text
