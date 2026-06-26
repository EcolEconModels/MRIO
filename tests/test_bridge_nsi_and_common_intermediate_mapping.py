from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = REPO_ROOT / "scripts" / "mappings" / "build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py"
COMPOSE_SCRIPT = REPO_ROOT / "scripts" / "mappings" / "compose_mapping_via_common_intermediate.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_dictionary(path: Path) -> None:
    rows = [
        {"code": "CPA_C10-12", "label": "Food products, beverages and tobacco products", "description": "test"},
        {"code": "CPA_C20", "label": "Chemicals and chemical products", "description": "test"},
        {"code": "CPA_C21", "label": "Basic pharmaceutical products", "description": "test"},
        {"code": "CPA_D", "label": "Electricity, gas, steam and air-conditioning", "description": "test"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def make_bridge_workbook_bytes() -> bytes:
    frame = pd.DataFrame("", index=range(7), columns=range(5))
    frame.iat[0, 2] = "Food product"
    frame.iat[1, 2] = "C_FOOD"
    frame.iat[2, 2] = "p15.a"
    frame.iat[0, 3] = "Chemical product"
    frame.iat[1, 3] = "C_CHEM"
    frame.iat[2, 3] = "p24.d"
    frame.iat[0, 4] = "Final demand"
    frame.iat[2, 4] = "export"
    frame.iat[4, 0] = "Food products and beverages"
    frame.iat[4, 1] = "p09"
    frame.iat[4, 2] = 1
    frame.iat[5, 0] = "Chemicals, chemical products and man-made fibres"
    frame.iat[5, 1] = "p18"
    frame.iat[5, 3] = 1
    payload = io.BytesIO()
    with pd.ExcelWriter(payload, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="AT_r_sup", header=False, index=False)
    return payload.getvalue()


def test_bridge_nsi_generator_writes_current_schema_many_target_mapping(tmp_path: Path) -> None:
    module = load_module(BRIDGE_SCRIPT, "bridge_nsi_generator")
    dictionary = tmp_path / "dictionary.csv"
    make_dictionary(dictionary)
    result = module.build_mapping(
        workbook_bytes=make_bridge_workbook_bytes(),
        workbook_sha256="synthetic",
        bridge_member="Bridge_NSI_AT.xlsx",
        sheet_name="AT_r_sup",
        target_version="cpa65",
        target_dictionary=dictionary,
    )

    assert set(result.mapping["source_code"]) == {"p15.a", "p24.d"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p24.d", "target_code"]) == {"CPA_C20", "CPA_C21"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p24.d", "allocation_rule"]) == {"split-by-output-share"}
    assert list(result.mapping.columns) == list(module.CANONICAL_COLUMNS)
    assert result.diagnostics.empty


def test_common_intermediate_composer_collapses_edges_to_direct_eb_cpa_mapping(tmp_path: Path) -> None:
    module = load_module(COMPOSE_SCRIPT, "common_intermediate_composer")
    left = pd.DataFrame(
        [
            {
                "source_code": "FC_IND_CPC_E",
                "source_label": "chemical and petrochemical",
                "source_balance_role": "final_use",
                "include_in_direct_energy": "1",
                "exio3_product_code": "p24.d",
                "exio3_product_name": "Chemicals nec",
            },
            {
                "source_code": "FC_IND_CPC_E",
                "source_label": "chemical and petrochemical",
                "source_balance_role": "final_use",
                "include_in_direct_energy": "1",
                "exio3_product_code": "p15.a",
                "exio3_product_name": "Food product",
            },
        ]
    )
    right = pd.DataFrame(
        [
            {"source_code": "p24.d", "source_label": "Chemicals nec", "target_code": "CPA_C20", "target_label": "Chemicals", "enabled": "1"},
            {"source_code": "p24.d", "source_label": "Chemicals nec", "target_code": "CPA_C21", "target_label": "Pharma", "enabled": "1"},
            {"source_code": "p15.a", "source_label": "Food product", "target_code": "CPA_C10-12", "target_label": "Food", "enabled": "1"},
        ]
    )
    left_prepared = module.prepare_left(
        left,
        path=tmp_path / "left.csv",
        source_code_col="source_code",
        source_label_col="source_label",
        intermediate_code_col="exio3_product_code",
        intermediate_label_col="exio3_product_name",
        source_balance_role_col="source_balance_role",
        include_col="include_in_direct_energy",
    )
    right_prepared = module.prepare_right(
        right,
        path=tmp_path / "right.csv",
        intermediate_code_col="source_code",
        intermediate_label_col="source_label",
        target_code_col="target_code",
        target_label_col="target_label",
        enabled_col="enabled",
    )

    result = module.compose_common_intermediate_mapping(
        left_prepared,
        right_prepared,
        target_version="CPA_65",
        source_name="test",
        official_source="test",
    )

    assert set(result.mapping["source_code"]) == {"FC_IND_CPC_E"}
    assert set(result.mapping["target_code"]) == {"CPA_C10-12", "CPA_C20", "CPA_C21"}
    assert set(result.mapping["allocation_rule"]) == {"split-by-energy-product-prd-use"}
    assert set(result.mapping["target_sector_role"]) == {"end_use_sector"}
    assert result.diagnostics.empty


def test_common_intermediate_composer_filters_to_final_use_direct_energy_rows(tmp_path: Path) -> None:
    module = load_module(COMPOSE_SCRIPT, "common_intermediate_filter")
    left = pd.DataFrame(
        [
            {
                "source_code": "FC_IND_CPC_E",
                "source_label": "chemical and petrochemical",
                "source_balance_role": "final_use",
                "include_in_direct_energy": "1",
                "exio3_product_code": "p24.d",
                "exio3_product_name": "Chemicals nec",
            },
            {
                "source_code": "TI_E",
                "source_label": "transformation input",
                "source_balance_role": "transformation_input",
                "include_in_direct_energy": "1",
                "exio3_product_code": "p24.d",
                "exio3_product_name": "Chemicals nec",
            },
            {
                "source_code": "NE_E",
                "source_label": "non energy",
                "source_balance_role": "final_use",
                "include_in_direct_energy": "0",
                "exio3_product_code": "p24.d",
                "exio3_product_name": "Chemicals nec",
            },
        ]
    )
    prepared = module.prepare_left(
        left,
        path=tmp_path / "left.csv",
        source_code_col="source_code",
        source_label_col="source_label",
        intermediate_code_col="exio3_product_code",
        intermediate_label_col="exio3_product_name",
        source_balance_role_col="source_balance_role",
        include_col="include_in_direct_energy",
        source_role_filter={"final_use"},
        include_direct_energy_only=True,
    )
    assert prepared["left_source_code"].tolist() == ["FC_IND_CPC_E"]


def test_common_intermediate_cli_writes_outputs(tmp_path: Path) -> None:
    module = load_module(COMPOSE_SCRIPT, "common_intermediate_cli")
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    out = tmp_path / "out.csv"
    edges = tmp_path / "edges.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame(
        [
            {
                "source_code": "EB_A",
                "source_label": "Energy Balance A",
                "source_balance_role": "final_use",
                "include_in_direct_energy": "1",
                "exio3_product_code": "p01.a",
                "exio3_product_name": "Paddy rice",
            }
        ]
    ).to_csv(left_path, index=False)
    pd.DataFrame(
        [
            {"source_code": "p01.a", "source_label": "Paddy rice", "target_code": "CPA_A01", "target_label": "Agriculture", "enabled": "1"}
        ]
    ).to_csv(right_path, index=False)

    assert module.main([
        "--left-csv", str(left_path),
        "--right-csv", str(right_path),
        "--out-csv", str(out),
        "--edge-out-csv", str(edges),
        "--diagnostics-csv", str(diagnostics),
        "--manifest-json", str(manifest),
        "--source-balance-role-filter", "final_use",
        "--include-direct-energy-only",
    ]) == 0
    assert out.exists()
    assert edges.exists()
    assert diagnostics.exists()
    assert manifest.exists()
    generated = pd.read_csv(out, dtype=str)
    assert set(generated["allocation_rule"]) == {"fixed-share"}


def test_bridge_cli_can_read_bridge_from_zip(tmp_path: Path) -> None:
    module = load_module(BRIDGE_SCRIPT, "bridge_nsi_cli")
    dictionary = tmp_path / "dictionary.csv"
    make_dictionary(dictionary)
    zip_path = tmp_path / "concordances.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Concordances/Archive/National classifications/Bridge_NSI_AT.xlsx", make_bridge_workbook_bytes())
    out_dir = tmp_path / "out"
    assert module.main([
        "--concordances-zip", str(zip_path),
        "--target-dictionary", str(dictionary),
        "--target-version", "cpa65",
        "--out-dir", str(out_dir),
    ]) == 0
    assert (out_dir / "exio_to_eurostat_cpa65_products_from_bridge_nsi_at.csv").exists()
