from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook


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


def test_pretty_sector_label_formats_energy_codes() -> None:
    from scripts.aggregate_iot_by_mapping import _pretty_sector_label

    assert _pretty_sector_label("ENERGY_BROWN") == "Energy Brown"
    assert _pretty_sector_label("ENERGY_GREEN") == "Energy Green"


def test_supplementary_fd_row_uses_current_final_demand_shares() -> None:
    from scripts.aggregate_iot_by_mapping import _supplementary_fd_row_from_current_shares

    template_fd = pd.Series(
        {
            "P3": 100.0,
            "P3_S13": float("nan"),
            "P3_S14": float("nan"),
            "P3_S15": float("nan"),
            "P5": 60.0,
            "P51G": float("nan"),
            "P52": float("nan"),
            "P53": float("nan"),
            "P5M": float("nan"),
            "P6": 40.0,
            "P6_B0": float("nan"),
            "P6_D0": float("nan"),
            "P6_U2": float("nan"),
            "P6_U3": float("nan"),
            "TU": float("nan"),
            "TFU": float("nan"),
        },
        dtype="float64",
    )
    current_fd = pd.Series(
        {
            "P3": 100.0,
            "P3_S13": 10.0,
            "P3_S14": 70.0,
            "P3_S15": 20.0,
            "P5": 60.0,
            "P51G": 30.0,
            "P52": 10.0,
            "P53": 20.0,
            "P5M": 30.0,
            "P6": 40.0,
            "P6_B0": 25.0,
            "P6_D0": 10.0,
            "P6_U2": 3.0,
            "P6_U3": 2.0,
            "TU": 0.0,
            "TFU": 0.0,
        },
        dtype="float64",
    )

    out = _supplementary_fd_row_from_current_shares(
        template_fd_values=template_fd,
        current_fd_totals=current_fd,
        product_total=15.0,
    )

    assert out[["P3_S13", "P3_S14", "P3_S15"]].tolist() == pytest.approx([10.0, 70.0, 20.0])
    assert out[["P51G", "P52", "P53"]].tolist() == pytest.approx([30.0, 10.0, 20.0])
    assert out[["P6_B0", "P6_D0", "P6_U2", "P6_U3"]].tolist() == pytest.approx([25.0, 10.0, 3.0, 2.0])
    assert float(out["P3"]) == pytest.approx(100.0)
    assert float(out["P5M"]) == pytest.approx(30.0)
    assert float(out["P5"]) == pytest.approx(60.0)
    assert float(out["P6"]) == pytest.approx(40.0)
    assert float(out["TFU"]) == pytest.approx(200.0)
    assert float(out["TU"]) == pytest.approx(215.0)


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
    x_before = (z_before.sum(axis=1) + y_before.sum(axis=1)).astype(float)
    c19_share_motor = float(x_before.loc["CPA_C19::Motor Gasoline"] / x_before.loc[["CPA_C19::Motor Gasoline", "CPA_C19::Heavy Fuel Oil"]].sum())
    c19_share_remaining = float(x_before.loc["CPA_C19::Heavy Fuel Oil"] / x_before.loc[["CPA_C19::Motor Gasoline", "CPA_C19::Heavy Fuel Oil"]].sum())
    expected_value_added = (x_after - col_sums_z).tolist()
    expected_output = x_after.tolist()
    expected_intermediate = col_sums_z.tolist()
    expected_taxes = [c19_share_motor, c19_share_remaining, 2.0]
    expected_imports = [7.0 * c19_share_motor, 7.0 * c19_share_remaining, 8.0]
    expected_supply = [round(a + b, 10) for a, b in zip(expected_output, expected_imports, strict=True)]

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
    assert _row_values(ws, row_lookup["Total intermediate consumption /final use"], 6, 5) == pytest.approx([26.0, 13.0, 6.0, 68.0, 45.0])


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
        lower_row_vectors={},
        lower_fd_vectors={},
    )