from __future__ import annotations
import ast
import json

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


def _write_cpa2002_exiobase_workbook(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a tiny direct CPA2002->EXIOBASE2 fixture with real workbook columns."""

    pd.DataFrame(rows, columns=["CPA 2002", "CPA 2002.1", "CPA 2002.2", "EXIOBASE_2.0"]).to_excel(path, index=False)


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


def test_final_aggregation_marks_multi_target_products_for_output_share_split(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_final_aggregation_split")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tReal estate services (70)\n"
        "Real estate services\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [
            {
                "EXIOBASE 2 Name": "Real estate services",
                "EXIOBASE 2 code": "p70",
                "EXIOBASE 3 name": "Real estate services",
                "EXIOBASE 3 code": "p70",
            }
        ]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [
            {
                "CPA 2002": "Real estate services on a fee or contract basis",
                "CPA 2002.1": "70.20.10",
                "CPA 2002.2": "702010",
                "EXIOBASE_2.0": "p70",
            }
        ],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "70.20", "CPA2008_TARGET_CODE": "68.20"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "68.20", "CPA21_TARGET_CODE": "68.20"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "CPA_L68A", "label": "Imputed rents of owner-occupied dwellings", "description": "dictionary text is not parsed"},
            {"code": "CPA_L68B", "label": "Real estate services excluding imputed rents", "description": "dictionary text is not parsed"},
        ]
    ).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [
            {"target_version": "CPA_65", "target_code": "CPA_L68A", "target_label": "Imputed rents of owner-occupied dwellings", "iot_a64_code": "L68A", "cpa21_division": "68"},
            {"target_version": "CPA_65", "target_code": "CPA_L68B", "target_label": "Real estate services excluding imputed rents", "iot_a64_code": "L68B", "cpa21_division": "68"},
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
        cpa2002_exiobase_path=direct,
    )

    assert result.diagnostics.empty
    assert set(result.mapping["target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(result.mapping["allocation_rule"]) == {"split-by-output-share"}
    notes = " ".join(result.mapping["notes"].astype(str))
    assert module.TARGET_DIVISION_SPLIT_WARNING in notes
    assert module.FINAL_AGGREGATION_SPLIT_WARNING in notes
    assert result.coverage["target_division_split_edge_rows"] == 2
    assert result.coverage["final_output_share_split_mapping_rows"] == 2
    assert result.coverage["distinct_exio3_products_requiring_output_share_split"] == 1


def test_default_eurostat_correspondence_paths_are_tracked_renamed_csvs() -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_default_paths")
    assert module.DEFAULT_CPA2002_CPA2008.name == "CPA 2002 - CPA 2008 (CSV format).csv"
    assert module.DEFAULT_CPA2008_CPA21.name == "CPA 2008 - CPA 2.1.csv"
    assert module.DEFAULT_CPA2002_CPA2008.exists()
    assert module.DEFAULT_CPA2008_CPA21.exists()


def test_real_ramon_correspondence_files_parse_as_level_aware_edges() -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_real_level_parser")
    old_to_2008 = module.read_correspondence(
        module.DEFAULT_CPA2002_CPA2008,
        source_tokens=("source", "from", "2002"),
        target_tokens=("target", "to", "2008"),
    )
    current_to_21 = module.read_correspondence(
        module.DEFAULT_CPA2008_CPA21,
        source_tokens=("source", "from", "2008"),
        target_tokens=("target", "to", "21", "2.1"),
    )

    required = {"source_level", "target_level", "source_division", "target_division", "edge_kind"}
    assert required <= set(old_to_2008.columns)
    assert required <= set(current_to_21.columns)
    assert not old_to_2008.loc[(old_to_2008["source_raw"] == "01.11.11") & (old_to_2008["target_raw"] == "01.64.10")].empty
    assert not old_to_2008.loc[(old_to_2008["source_level"] == "section") & (old_to_2008["target_level"] == "subcategory")].empty
    assert not old_to_2008.loc[(old_to_2008["source_raw"] == "C") & (old_to_2008["target_raw"] == "09.90.19")].empty
    assert not current_to_21.loc[(current_to_21["source_raw"] == "A") & (current_to_21["target_raw"] == "A")].empty
    assert not current_to_21.loc[(current_to_21["source_raw"] == "01") & (current_to_21["target_raw"] == "01")].empty

    old_audit = module.audit_correspondence_level_step(old_to_2008, step_name="real CPA2002 -> CPA2008")
    current_audit = module.audit_correspondence_level_step(current_to_21, step_name="real CPA2008 -> CPA2.1")
    assert old_audit.edge_kind_counts["subcategory_to_subcategory"] > 0
    assert old_audit.edge_kind_counts["section_to_subcategory"] > 0
    assert current_audit.edge_kind_counts["section_to_section"] > 0
    assert current_audit.edge_kind_counts["division_to_division"] > 0


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
    # The stricter first-step audit now requires the direct CPA2002 workbook in
    # every success fixture. This is not duplicate test boilerplate: it proves the
    # ESA59 old-product grouping and the direct CPA2002 product concordance agree
    # on the same old two-digit codomain before any CPA2002->CPA2008 bridge is used.
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [
            {"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"},
            {"CPA 2002": "Real estate services", "CPA 2002.1": "70.20.10", "CPA 2002.2": "702010", "EXIOBASE_2.0": "p47.a"},
        ],
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

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.diagnostics.empty
    assert set(result.mapping.loc[result.mapping["source_code"] == "p01.a", "target_code"]) == {"CPA_A01"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p47.a", "target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p47.a", "allocation_rule"]) == {"split-by-output-share"}
    assert set(result.mapping["target_version"]) == {"CPA_65"}




def _write_minimal_official_cli_fixture(tmp_path: Path, *, resolvable: bool) -> dict[str, Path]:
    concordances_zip = tmp_path / "exiobase_concordances.zip"
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    direct_cpa = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    pd.DataFrame(
        [
            {
                "EXIOBASE 2 Name": "Paddy rice",
                "EXIOBASE 2 code": "p01.a",
                "EXIOBASE 3 name": "Paddy rice",
                "EXIOBASE 3 code": "p01.a",
            }
        ]
    ).to_excel(exio_bridge, index=False)
    _write_cpa2002_exiobase_workbook(
        direct_cpa,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    with zipfile.ZipFile(concordances_zip, "w") as zf:
        zf.writestr("Concordances/Different aggregations/EXIOBASE20p_ESA59p.txt", "\tProducts of agriculture, hunting and related services (01)\nPaddy rice\t1\n")
        zf.write(exio_bridge, "Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx")
        zf.write(direct_cpa, "Concordances/Archive/International classifications/CPA2002_6digit_EXIOBASE_FINAL.xlsx")
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    if resolvable:
        pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"}]).to_csv(cpa02_08, index=False)
    else:
        pd.DataFrame([{"CPA2002_SOURCE_CODE": "02.10", "CPA2008_TARGET_CODE": "02.10"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame(
        [
            {"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"},
            {"CPA2008_SOURCE_CODE": "02.10", "CPA21_TARGET_CODE": "02.10"},
        ]
    ).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame([{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]).to_csv(division_map, index=False)
    return {
        "concordances_zip": concordances_zip,
        "cpa02_08": cpa02_08,
        "cpa08_21": cpa08_21,
        "dictionary": dictionary,
        "division_map": division_map,
    }


def test_official_cli_reads_exiobase_members_from_zip(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_zip_cli")
    concordances_zip = tmp_path / "exiobase_concordances.zip"
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    direct_cpa = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    pd.DataFrame(
        [
            {"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"},
            {"EXIOBASE 2 Name": "Real estate activities", "EXIOBASE 2 code": "p47.a", "EXIOBASE 3 name": "Real estate services", "EXIOBASE 3 code": "p47.a"},
        ]
    ).to_excel(exio_bridge, index=False)
    _write_cpa2002_exiobase_workbook(
        direct_cpa,
        [
            {"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"},
            {"CPA 2002": "Real estate services", "CPA 2002.1": "70.20.10", "CPA 2002.2": "702010", "EXIOBASE_2.0": "p47.a"},
        ],
    )
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
        zf.write(
            direct_cpa,
            "Concordances/Archive/International classifications/CPA2002_6digit_EXIOBASE_FINAL.xlsx",
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
    edge_audit = out_dir / "diagnostics" / "exio_to_eurostat_cpa65_products_from_official_concordances_edges.csv"
    assert edge_audit.exists()
    edge_rows = pd.read_csv(edge_audit, dtype=str)
    assert {"exio3_product_code", "target_code", "cpa21_division"} <= set(edge_rows.columns)
    assert set(generated.loc[generated["source_code"] == "p01.a", "target_code"]) == {"CPA_A01"}
    assert set(generated.loc[generated["source_code"] == "p47.a", "target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(generated["target_version"]) == {"CPA_65"}





def test_official_cli_does_not_write_outputs_before_failing_diagnostics(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_cli_diag_gate")
    paths = _write_minimal_official_cli_fixture(tmp_path, resolvable=False)
    out_dir = tmp_path / "out"

    try:
        module.main([
            "--concordances-zip", str(paths["concordances_zip"]),
            "--cpa2002-cpa2008", str(paths["cpa02_08"]),
            "--cpa2008-cpa21", str(paths["cpa08_21"]),
            "--target-dictionary", str(paths["dictionary"]),
            "--target-division-map", str(paths["division_map"]),
            "--target-version", "cpa65",
            "--out-dir", str(out_dir),
        ])
    except SystemExit as exc:
        assert "diagnostics" in str(exc)
    else:
        raise AssertionError("Expected unresolved detailed anchor diagnostics to stop the default CLI run")

    stem = "exio_to_eurostat_cpa65_products_from_official_concordances"
    assert not (out_dir / f"{stem}.csv").exists()
    assert not (out_dir / f"{stem}_diagnostics.csv").exists()
    assert not (out_dir / f"{stem}_manifest.json").exists()
    assert not (out_dir / "diagnostics" / f"{stem}_edges.csv").exists()


def test_official_cli_allow_diagnostics_marks_outputs_exploratory(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_cli_diag_allow")
    paths = _write_minimal_official_cli_fixture(tmp_path, resolvable=False)
    out_dir = tmp_path / "out"

    assert module.main([
        "--concordances-zip", str(paths["concordances_zip"]),
        "--cpa2002-cpa2008", str(paths["cpa02_08"]),
        "--cpa2008-cpa21", str(paths["cpa08_21"]),
        "--target-dictionary", str(paths["dictionary"]),
        "--target-division-map", str(paths["division_map"]),
        "--target-version", "cpa65",
        "--out-dir", str(out_dir),
        "--allow-diagnostics",
    ]) == 0

    stem = "exio_to_eurostat_cpa65_products_from_official_concordances"
    manifest = json.loads((out_dir / f"{stem}_manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostics_allowed"] is True
    assert manifest["output_status"] == "diagnostic_exploratory_unresolved_anchors_present"
    assert manifest["coverage"]["diagnostic_rows"] == 1
    assert manifest["coverage"]["anchor_source_mode"] == "direct-only"
    assert manifest["coverage"]["diagnostics_allowed"] is True
    assert "--allow-diagnostics" in " ".join(manifest["method_notes"])
    assert (out_dir / f"{stem}.csv").exists()
    assert (out_dir / f"{stem}_diagnostics.csv").exists()


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


def test_first_step_closure_fills_missing_direct_cpa_from_esa(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_closure_fill")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\tChemicals, chemical products and man-made fibres (24)\n"
        "Paddy rice\t1\t0\n"
        "Basic chemicals\t0\t1\n"
        "N-fertiliser\t0\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [
            {"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"},
            {"EXIOBASE 2 Name": "Basic chemicals", "EXIOBASE 2 code": "p24.a", "EXIOBASE 3 name": "Basic chemicals", "EXIOBASE 3 code": "p24.a"},
            {"EXIOBASE 2 Name": "N-fertiliser", "EXIOBASE 2 code": "p24.b", "EXIOBASE 3 name": "N-fertiliser", "EXIOBASE 3 code": "p24.b"},
        ]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [
            {"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"},
            {"CPA 2002": "Basic chemicals", "CPA 2002.1": "24.15.10", "CPA 2002.2": "241510", "EXIOBASE_2.0": "p24.a"},
        ],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame(
        [
            {"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"},
            {"CPA2002_SOURCE_CODE": "24.15", "CPA2008_TARGET_CODE": "20.15"},
        ]
    ).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame(
        [
            {"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"},
            {"CPA2008_SOURCE_CODE": "20.15", "CPA21_TARGET_CODE": "20.15"},
        ]
    ).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"},
            {"code": "CPA_C20", "label": "Chemicals", "description": "dictionary text is not parsed"},
        ]
    ).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [
            {"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"},
            {"target_version": "CPA_65", "target_code": "CPA_C20", "target_label": "Chemicals", "iot_a64_code": "C20", "cpa21_division": "20"},
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
        cpa2002_exiobase_path=direct,
        anchor_source_mode="direct-plus-curated-gaps",
    )

    assert result.diagnostics.empty
    # p24.b is not in the direct CPA workbook fixture. It is mapped only in
    # direct-plus-curated-gaps mode because the product-anchor conciliation file
    # gives an explicit detailed CPA2002 anchor; broad ESA division 24 is not fanned out implicitly.
    assert set(result.mapping.loc[result.mapping["source_code"] == "p24.b", "target_code"]) == {"CPA_C20"}
    p24b_edges = result.edges.loc[result.edges["exio20_product_code"] == "p24.b"]
    assert set(p24b_edges["cpa2002_anchor_display_code"]) == {"24.15.10"}
    assert set(p24b_edges["cpa2002_anchor_source"]) == {"curated_CPA2002_product_anchor"}
    assert set(p24b_edges["cpa2002_anchor_basis"]) == {"curated_from_product_label"}
    assert "curated_from_product_label" in " ".join(result.mapping.loc[result.mapping["source_code"] == "p24.b", "notes"].astype(str))

    empty_anchor = tmp_path / "empty_product_anchor_conciliation.csv"
    pd.DataFrame(
        columns=[
            "exio20_product_code",
            "exio20_product_label",
            "esa59_old_cpa2002_division",
            "esa59_old_cpa2002_label",
            "chosen_cpa2002_code",
            "chosen_cpa2002_label",
            "chosen_cpa2002_level",
            "anchor_basis",
            "notes",
        ]
    ).to_csv(empty_anchor, index=False)
    without_anchor = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
        product_anchor_conciliation_path=empty_anchor,
        anchor_source_mode="direct-plus-curated-gaps",
    )
    assert "p24.b" not in set(without_anchor.mapping["source_code"])
    assert set(without_anchor.diagnostics["diagnostic_kind"]) == {"missing_curated_cpa2002_product_anchor"}

    direct_only = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
        anchor_source_mode="direct-only",
    )
    assert "p24.b" not in set(direct_only.mapping["source_code"])
    assert set(direct_only.diagnostics["diagnostic_kind"]) == {"missing_direct_cpa2002_anchor"}
    assert direct_only.coverage["anchor_source_mode"] == "direct-only"

def test_first_step_closure_fails_on_two_digit_inconsistency(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_closure_fail")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Sewage removal", "CPA 2002.1": "90.01.11", "CPA 2002.2": "900111", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]
    ).to_csv(division_map, index=False)

    try:
        module.build_official_mapping(
            esa59_path=esa,
            exio20_exio30_path=exio_bridge,
            cpa2002_cpa2008_path=cpa02_08,
            cpa2008_cpa21_path=cpa08_21,
            target_dictionary=dictionary,
            target_division_map=division_map,
            target_version="cpa65",
            cpa2002_exiobase_path=direct,
        )
    except ValueError as exc:
        assert "First-step EXIOBASE2->old-CPA two-source closure failed" in str(exc)
    else:
        raise AssertionError("Expected coarse first-step inconsistency to fail")


def test_first_step_closure_warns_on_one_source_codomain_gap_when_closed_union_maps(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_right_side_warning")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tPrivate households with employed persons (95)\n"
        "Private households with employed persons\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [
            {
                "EXIOBASE 2 Name": "Private households with employed persons",
                "EXIOBASE 2 code": "p95",
                "EXIOBASE 3 name": "Private households with employed persons",
                "EXIOBASE 3 code": "p95",
            }
        ]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [
            {"CPA 2002": "Domestic personnel services", "CPA 2002.1": "95.00.10", "CPA 2002.2": "950010", "EXIOBASE_2.0": "p95"},
            {"CPA 2002": "Own-use household goods", "CPA 2002.1": "96.00.10", "CPA 2002.2": "960010", "EXIOBASE_2.0": "p95"},
            {"CPA 2002": "Own-use household services", "CPA 2002.1": "97.00.10", "CPA 2002.2": "970010", "EXIOBASE_2.0": "p95"},
        ],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame(
        [
            {"CPA2002_SOURCE_CODE": "95.00", "CPA2008_TARGET_CODE": "97.00"},
            {"CPA2002_SOURCE_CODE": "96.00", "CPA2008_TARGET_CODE": "98.00"},
            {"CPA2002_SOURCE_CODE": "97.00", "CPA2008_TARGET_CODE": "98.00"},
        ]
    ).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame(
        [
            {"CPA2008_SOURCE_CODE": "97.00", "CPA21_TARGET_CODE": "97.00"},
            {"CPA2008_SOURCE_CODE": "98.00", "CPA21_TARGET_CODE": "98.00"},
        ]
    ).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame(
        [{"code": "CPA_T", "label": "Household services", "description": "dictionary text is not parsed"}]
    ).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [
            {"target_version": "CPA_65", "target_code": "CPA_T", "target_label": "Household services", "iot_a64_code": "T", "cpa21_division": "97"},
            {"target_version": "CPA_65", "target_code": "CPA_T", "target_label": "Household services", "iot_a64_code": "T", "cpa21_division": "98"},
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
        cpa2002_exiobase_path=direct,
    )

    assert result.diagnostics.empty
    assert set(result.edges.loc[result.edges["exio20_product_code"] == "p95", "esa59_product_code"]) == {"95", "96", "97"}
    assert set(result.edges.loc[result.edges["exio20_product_code"] == "p95", "cpa2002_anchor_display_code"]) == {"95.00.10", "96.00.10", "97.00.10"}
    assert set(result.mapping.loc[result.mapping["source_code"] == "p95", "target_code"]) == {"CPA_T"}

def test_level_aware_composition_uses_direct_detail_exact_match(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_level_exact")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11.14", "CPA2008_TARGET_CODE": "01.64.10"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "01.64.10", "CPA21_TARGET_CODE": "01.64.10"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.diagnostics.empty
    assert set(result.mapping.loc[result.mapping["source_code"] == "p01.a", "target_code"]) == {"CPA_A01"}
    assert set(result.edges["cpa2002_anchor_display_code"]) == {"01.11.14"}
    assert set(result.edges["cpa2002_match_rule"]) == {"exact"}
    assert set(result.edges["cpa2008_display_code"]) == {"01.64.10"}


def test_level_aware_composition_uses_numeric_ancestor_without_section_fallback(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_level_ancestor")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.diagnostics.empty
    assert set(result.edges["cpa2002_match_rule"]) == {"ancestor:0111"}
    assert set(result.edges["cpa2008_match_rule"]) == {"exact"}


def test_level_aware_composition_reports_unresolved_first_step_path(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_level_first_unresolved")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "C", "CPA2008_TARGET_CODE": "09.90.19"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "09.90.19", "CPA21_TARGET_CODE": "09.90.19"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.mapping.empty
    assert set(result.diagnostics["diagnostic_kind"]) == {"cpa2002_to_cpa2008_unresolved"}


def test_level_aware_composition_reports_unresolved_second_step_path(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_level_second_unresolved")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "02.10", "CPA21_TARGET_CODE": "02.10"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_A01", "label": "Agriculture", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_A01", "target_label": "Agriculture", "iot_a64_code": "A01", "cpa21_division": "01"}]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.mapping.empty
    assert set(result.diagnostics["diagnostic_kind"]) == {"cpa2008_to_cpa21_unresolved"}


def test_level_aware_composition_reports_final_target_gap(tmp_path: Path) -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_level_final_gap")
    esa = tmp_path / "EXIOBASE20p_ESA59p.txt"
    esa.write_text(
        "\tProducts of agriculture, hunting and related services (01)\n"
        "Paddy rice\t1\n",
        encoding="utf-8",
    )
    exio_bridge = tmp_path / "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx"
    pd.DataFrame(
        [{"EXIOBASE 2 Name": "Paddy rice", "EXIOBASE 2 code": "p01.a", "EXIOBASE 3 name": "Paddy rice", "EXIOBASE 3 code": "p01.a"}]
    ).to_excel(exio_bridge, index=False)
    direct = tmp_path / "CPA2002_6digit_EXIOBASE_FINAL.xlsx"
    _write_cpa2002_exiobase_workbook(
        direct,
        [{"CPA 2002": "Rice, not husked", "CPA 2002.1": "01.11.14", "CPA 2002.2": "11114", "EXIOBASE_2.0": "p01.a"}],
    )
    cpa02_08 = tmp_path / "CPA_2002_to_CPA_2008.csv"
    pd.DataFrame([{"CPA2002_SOURCE_CODE": "01.11", "CPA2008_TARGET_CODE": "01.11"}]).to_csv(cpa02_08, index=False)
    cpa08_21 = tmp_path / "CPA_2008_to_CPA_21.csv"
    pd.DataFrame([{"CPA2008_SOURCE_CODE": "01.11", "CPA21_TARGET_CODE": "01.11"}]).to_csv(cpa08_21, index=False)
    dictionary = tmp_path / "cpa65_dictionary.csv"
    pd.DataFrame([{"code": "CPA_B", "label": "Mining", "description": "dictionary text is not parsed"}]).to_csv(dictionary, index=False)
    division_map = tmp_path / "cpa65_to_cpa21_divisions.csv"
    pd.DataFrame(
        [{"target_version": "CPA_65", "target_code": "CPA_B", "target_label": "Mining", "iot_a64_code": "B", "cpa21_division": "05"}]
    ).to_csv(division_map, index=False)

    result = module.build_official_mapping(
        esa59_path=esa,
        exio20_exio30_path=exio_bridge,
        cpa2002_cpa2008_path=cpa02_08,
        cpa2008_cpa21_path=cpa08_21,
        target_dictionary=dictionary,
        target_division_map=division_map,
        target_version="cpa65",
        cpa2002_exiobase_path=direct,
    )

    assert result.mapping.empty
    assert set(result.diagnostics["diagnostic_kind"]) == {"cpa21_division_without_target_aggregation"}


def test_first_step_conciliation_fails_when_recorded_evidence_is_stale() -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_conciliation_evidence_fail")
    esa = pd.DataFrame(
        [
            {
                "exio20_product_code": "p01.a",
                "exio20_product_name": "Paddy rice",
                "esa59_product_code": "01",
                "esa59_product_label": "Agriculture",
            }
        ]
    )
    bridge = pd.DataFrame(
        [
            {
                "exio20_product_name": "Paddy rice",
                "exio20_product_name_norm": "paddyrice",
                "exio20_product_code": "p01.a",
                "exio3_product_name": "Paddy rice",
                "exio3_product_code": "p01.a",
            }
        ]
    )
    direct = pd.DataFrame(
        [
            {
                "exio20_product_code": "p01.a",
                "cpa2002_code": "90.01.11",
                "cpa2002_compact_code": "900111",
                "cpa2002_label": "Sewage removal",
                "cpa2002_division": "90",
            }
        ]
    )
    conciliation = pd.DataFrame(
        [
            {
                "exio20_product_code": "p01.a",
                "exio20_product_label": "Paddy rice",
                "chosen_old_cpa2002_divisions": "90",
                "chosen_old_cpa2002_labels": "Sewage removal",
                "conciliation_basis": "fixture",
                "esa59_old_cpa2002_divisions": "02",
                "direct_old_cpa2002_divisions": "90",
                "notes": "stale evidence should fail",
            }
        ]
    )

    try:
        module.build_closed_first_step_edges(esa, bridge, direct, conciliation)
    except ValueError as exc:
        assert "First-step EXIOBASE2->old-CPA two-source closure failed" in str(exc)
    else:
        raise AssertionError("Expected stale conciliation evidence to fail")


def test_curated_product_anchor_file_documents_esa_only_broad_products() -> None:
    module = load_module(OFFICIAL_SCRIPT, "official_concordance_product_anchor_file")
    anchors = module.read_product_anchor_conciliation_map(module.DEFAULT_PRODUCT_ANCHOR_CONCILIATION)
    assert set(anchors["exio20_product_code"]) == {"p24.b", "p90.5.a", "p90.5.b", "p90.5.c", "p90.5.d", "p90.5.e", "p90.5.f"}
    p905d = anchors.loc[anchors["exio20_product_code"] == "p90.5.d"]
    assert set(p905d["chosen_cpa2002_display_code"]) == {"90.02.13", "90.02.14"}
    assert set(p905d["chosen_cpa2002_division"]) == {"90"}


def test_phase4_removed_legacy_two_digit_architecture_symbols() -> None:
    source_text = OFFICIAL_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    legacy_symbols = {
        "DivisionStepAudit",
        "_active_prefix_edges",
        "_correspondence_division_edges",
        "_division_labels_from_edges",
        "audit_correspondence_division_step",
        "assert_code_universes_equal",
        "_targets_by_source_division",
        "compose_old_divisions_to_cpa21_divisions_checked",
        "compose_old_division_to_cpa21_divisions",
    }
    assert definitions.isdisjoint(legacy_symbols)
