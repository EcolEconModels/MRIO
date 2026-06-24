from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio.mapping_csv import read_weighted_mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mappings" / "build_exiobase_concordance_compat_mappings.py"
DERIVED_COLLAPSED = REPO_ROOT / "data" / "exiobase_concordance_sources" / "derived" / "exiobase_eurostat_iot_a64" / "exiobase3_products_to_eurostat_iot_a64_collapsed.csv"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_exiobase_concordance_compat_mappings", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_collapsed() -> pd.DataFrame:
    rows = []
    for code in ["p01.a", "p24.d", "p24.d", "p70"]:
        if code == "p01.a":
            target = "A01"
            target_label = "Products of crop and animal production, hunting and related services"
            source_label = "Paddy rice"
            multi = False
            n_targets = 1
        elif code == "p24.d":
            source_label = "Chemicals nec"
            if len([row for row in rows if row["exio3_product_code"] == code]) == 0:
                target = "C20"
                target_label = "Chemicals and chemical products"
            else:
                target = "C21"
                target_label = "Basic pharmaceutical products and pharmaceutical preparations"
            multi = True
            n_targets = 2
        else:
            target = "L68"
            target_label = "Real estate services"
            source_label = "Real estate services (70)"
            multi = False
            n_targets = 1
        rows.append(
            {
                "exio3_product_code": code,
                "exio3_product_name": source_label,
                "iot_a64_code": target,
                "iot_a64_label": target_label,
                "iot_a64_divisions": "",
                "n_source_nace2_codes": "1",
                "source_nace2_codes": "1.11",
                "source_nace2_divisions": "01",
                "source_exio_codes": code,
                "bridge_key_kinds": "exio20_code",
                "n_iot_a64_targets_for_exio3_product": str(n_targets),
                "is_multi_target": str(multi),
                "equal_split_weight_if_required": "1.0" if not multi else "0.5",
                "weight_status": "placeholder_equal_split_not_empirical",
            }
        )
    # Pad to the full expected EXIO source count used by the converter guard.
    for idx in range(197):
        rows.append(
            {
                "exio3_product_code": f"px.synthetic.{idx:03d}",
                "exio3_product_name": f"Synthetic product {idx}",
                "iot_a64_code": "A02",
                "iot_a64_label": "Products of forestry and logging",
                "iot_a64_divisions": "02",
                "n_source_nace2_codes": "1",
                "source_nace2_codes": "2.10",
                "source_nace2_divisions": "02",
                "source_exio_codes": f"px.synthetic.{idx:03d}",
                "bridge_key_kinds": "synthetic-test",
                "n_iot_a64_targets_for_exio3_product": "1",
                "is_multi_target": "False",
                "equal_split_weight_if_required": "1.0",
                "weight_status": "placeholder_equal_split_not_empirical",
            }
        )
    return pd.DataFrame(rows)


def test_cpa64_dictionary_has_64_targets_and_unsplit_real_estate() -> None:
    module = _load_module()
    labels = module.read_target_label_map(module.DEFAULT_CPA64_DICTIONARY)
    codes = set(labels)
    assert len(codes) == 64
    assert "CPA_L68" in codes
    assert "CPA_L68A" not in codes
    assert "CPA_L68B" not in codes


def test_converter_writes_current_schema_without_static_weight_column() -> None:
    module = _load_module()
    collapsed = _synthetic_collapsed()
    cpa64 = module.convert_collapsed_to_canonical(collapsed, projection="cpa64")
    cpa65 = module.convert_collapsed_to_canonical(collapsed, projection="cpa65")

    assert list(cpa64.columns) == list(module.CANONICAL_COLUMNS)
    assert list(cpa65.columns) == list(module.CANONICAL_COLUMNS)
    assert "weight" not in cpa64.columns
    assert "weight" not in cpa65.columns

    p24_rules = set(cpa64.loc[cpa64["source_code"] == "p24.d", "allocation_rule"])
    assert p24_rules == {"split-by-output-share"}
    assert set(cpa64.loc[cpa64["source_code"] == "p70", "target_code"]) == {"CPA_L68"}
    assert set(cpa65.loc[cpa65["source_code"] == "p70", "target_code"]) == {"CPA_L68A", "CPA_L68B"}
    assert set(cpa64["target_version"]) == {"CPA_64"}
    assert set(cpa65["target_version"]) == {"CPA_65"}
    cpa64_labels = module.read_target_label_map(module.DEFAULT_CPA64_DICTIONARY)
    cpa65_labels = module.read_target_label_map(module.DEFAULT_CPA65_DICTIONARY)
    assert set(cpa64.loc[cpa64["source_code"] == "p01.a", "target_label"]) == {cpa64_labels["CPA_A01"]}
    assert set(cpa65.loc[cpa65["source_code"] == "p01.a", "target_label"]) == {cpa65_labels["CPA_A01"]}


def test_existing_weighted_mapping_api_resolves_runtime_target_output_shares(tmp_path: Path) -> None:
    module = _load_module()
    mapping = module.convert_collapsed_to_canonical(_synthetic_collapsed(), projection="cpa65")
    path = tmp_path / "mapping.csv"
    mapping.to_csv(path, index=False)

    resolved = read_weighted_mapping(
        path,
        target_output_by_code={
            "CPA_A01": 10.0,
            "CPA_A02": 1.0,
            "CPA_C20": 30.0,
            "CPA_C21": 70.0,
            "CPA_L68A": 25.0,
            "CPA_L68B": 75.0,
        },
    )
    assert resolved["CPA_C20"]["p24.d"] == pytest.approx(0.3)
    assert resolved["CPA_C21"]["p24.d"] == pytest.approx(0.7)
    assert resolved["CPA_L68A"]["p70"] == pytest.approx(0.25)
    assert resolved["CPA_L68B"]["p70"] == pytest.approx(0.75)


def test_real_generated_collapsed_file_if_present(tmp_path: Path) -> None:
    if not DERIVED_COLLAPSED.exists():
        pytest.skip("Generated EXIOBASE concordance collapsed file is not present locally.")
    module = _load_module()
    manifest = module.write_outputs(DERIVED_COLLAPSED, tmp_path)
    assert manifest["coverage"]["cpa64"]["distinct_sources"] == 200
    assert manifest["coverage"]["cpa64"]["distinct_targets"] == 64
    assert manifest["coverage"]["cpa64"]["multi_target_sources"] >= 1
    assert (tmp_path / module.CPA64_OUTPUT_NAME).exists()
    assert (tmp_path / module.CPA65_OUTPUT_NAME).exists()
