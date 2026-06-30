from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCRIPT = REPO_ROOT / "scripts" / "mappings" / "build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py"
COMPOSE_SCRIPT = REPO_ROOT / "scripts" / "mappings" / "compose_mapping_via_common_intermediate.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_generator_has_no_hard_coded_nsi_to_cpa_table() -> None:
    text = OFFICIAL_SCRIPT.read_text(encoding="utf-8")
    assert "NSI_PRODUCT_TO_CPA64" not in text
    assert "No hard-coded NSI->CPA table" in text



def test_curated_target_division_maps_do_not_treat_a64_as_division_64() -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_target_division_map_loader")
    cases = [
        ("eurostat_cpa64_to_cpa21_divisions.csv", "eurostat_cpa64_products_dictionary.csv", 1),
        ("eurostat_cpa65_to_cpa21_divisions.csv", "eurostat_cpa65_products_dictionary.csv", 2),
    ]
    for map_name, dictionary_name, expected_div68_targets in cases:
        labels, division_to_targets = module.load_target_division_map(
            REPO_ROOT / "scripts" / "mappings" / map_name,
            target_dictionary=REPO_ROOT / "scripts" / "mappings" / dictionary_name,
        )
        assert labels["CPA_K64"] == "Financial services, except insurance and pension funding"
        assert division_to_targets["64"] == {"CPA_K64"}
        assert "CPA_A01" not in division_to_targets["64"]
        assert len(division_to_targets["68"]) == expected_div68_targets

def test_official_generator_builds_mapping_from_concordance_chain(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_generator")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\tReal estate services (70)\n"
        "Paddy rice\t1\t0\n"
        "Real estate activities\t0\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [
            {"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"},
            {"EXIOBASE 2 Name": "Real estate activities", "EXIOBASE 2 code": "p47.a", "EXIOBASE 3 name": "Real estate services", "EXIOBASE 3 code": "p47.a"},
        ]
    ).to_excel(exio_bridge, index=False)
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame(
        [
            {"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"},
            {"CPA2002_SOURCE_CODE": "70.20", "CPA2008_TARGET_CODE": "68.20"},
        ]
    ).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame(
        [
            {"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"},
            {"CPA2008_SOURCE_CODE": "68.20", "CPA21_TARGET_CODE": "68.20"},
        ]
    ).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"},
            {"code": "CPA_L68A", "label": "Imputed rents", "description": "dictionary text is not parsed"},
            {"code": "CPA_L68B", "label": "Real estate excluding imputed rents", "description": "dictionary text is not parsed"},
        ]
    ).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [
            {"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"},
            {"target_version": "CPA_65", "target_code": "CPA_L68A", "target_label": "Imputed rents", "iot_a64_code": "L68", "cpa21_division": "68"},
            {"target_version": "CPA_65", "target_code": "CPA_L68B", "target_label": "Real estate excluding imputed rents", "iot_a64_code": "L68", "cpa21_division": "68"},
        ]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
    )

    assert result.diagnostics.empty
    assert set(result.mapping.loc[result.mapping["source_code"] == "p01.a", "target_code"]) == {"CPA_A01"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p47.a", "target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p47.a", "allocation_rule"]) == {"split-by-output-share"}
    assert set(result.mapping["target_version"]) == {"CPA_65"}


def test_official_cli_reads_exiobase_members_from_zip(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_zip_cli")
    concordances_zip = tmp_path / "exiobase_concordances.zip"
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [
            {"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"},
            {"EXIOBASE 2 Name": "Real estate activities", "EXIOBASE 2 code": "p47.a", "EXIOBASE 3 name": "Real estate services", "EXIOBASE 3 code": "p47.a"},
        ]
    ).to_excel(exio_bridge, index=False)
    esa_text = (
        "\tProducts of agriculture, hunting and related services (01)\tReal estate services (70)\n"
        "Paddy rice\t1\t0\n"
        "Real estate activities\t0\t1\n"
    )
    with zipfile.ZipFile(concordances_zip, "w") as zf:
        zf.writestr("Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt", esa_text)
        zf.write(
            exio_bridge,
            "Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx",
        )

    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame(
        [
            {"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"},
            {"CPA2002_SOURCE_CODE": "70.20", "CPA2008_TARGET_CODE": "68.20"},
        ]
    ).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame(
        [
            {"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"},
            {"CPA2008_SOURCE_CODE": "68.20", "CPA21_TARGET_CODE": "68.20"},
        ]
    ).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"},
            {"code": "CPA_L68A", "label": "Imputed rents", "description": "dictionary text is not parsed"},
            {"code": "CPA_L68B", "label": "Real estate excluding imputed rents", "description": "dictionary text is not parsed"},
        ]
    ).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [
            {"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"},
            {"target_version": "CPA_65", "target_code": "CPA_L68A", "target_label": "Imputed rents", "iot_a64_code": "L68", "cpa21_division": "68"},
            {"target_version": "CPA_65", "target_code": "CPA_L68B", "target_label": "Real estate excluding imputed rents", "iot_a64_code": "L68", "cpa21_division": "68"},
        ]
    ).to_csv(division_map, index=False)
    out_dir = tmp_path / "out"

    assert module.main([
        "--concordances-zip", str(concordances_zip),
        "--cpa2002-cpa2008", str(cpa02_08),
        "--cpa2008-cpa21", str(cpa08_21),
        "--target-dictionary", str(dictionary),
        "--target-division-map", str(division_map),
        "--target-version", "cpa65",
        "--out-dir", str(out_dir),
    ]) == 0

    generated = pd.read_csv(out_dir / "exio_to_eurostat_cpa65_products_from_official_concordances.csv", dtype=str)
    assert not (out_dir / "exio_to_eurostat_cpa65_products_from_official_concordances_edge_audit.csv").exists()
    assert set(generated.loc[generated["source_code"] == "p01.a", "target_code"]) == {"CPA_A01"}
    assert set(generated.loc[generated["source_code"] == "p47.a", "target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(generated["target_version"]) == {"CPA_65"}


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
    ]) == 0
    assert out.exists()
    assert edges.exists()
    assert diagnostics.exists()
    assert manifest.exists()
    generated = pd.read_csv(out, dtype=str)
    assert set(generated["allocation_rule"]) == {"fixed-share"}
