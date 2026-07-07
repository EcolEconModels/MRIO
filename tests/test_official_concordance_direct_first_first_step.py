from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_mapping_module():
    path = Path("scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_official_concordances.py")
    spec = importlib.util.spec_from_file_location("official_cpa_mapping", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.name is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_step_uses_direct_rows_before_esa_gap_fill_and_keeps_manure_override() -> None:
    module = load_mapping_module()
    bridge = pd.DataFrame(
        [
            {"exio20_product_code": "pA", "exio20_product_name": "Direct product", "exio20_product_name_norm": "directproduct", "exio3_product_code": "pA", "exio3_product_name": "Direct product"},
            {"exio20_product_code": "pAg", "exio20_product_name": "Agriculture product", "exio20_product_name_norm": "agricultureproduct", "exio3_product_code": "pAg", "exio3_product_name": "Agriculture product"},
            {"exio20_product_code": "p01.w.1", "exio20_product_name": "Manure (conventional treatment)", "exio20_product_name_norm": "manureconventionaltreatment", "exio3_product_code": "p01.w.1", "exio3_product_name": "Manure (conventional treatment)"},
            {"exio20_product_code": "pGap", "exio20_product_name": "Direct gap", "exio20_product_name_norm": "directgap", "exio3_product_code": "pGap", "exio3_product_name": "Direct gap"},
        ]
    )
    esa = pd.DataFrame(
        [
            {"exio20_product_code": "pA", "exio20_product_name": "Direct product", "esa59_product_code": "10", "esa59_product_label": "ESA food"},
            {"exio20_product_code": "pAg", "exio20_product_name": "Agriculture product", "esa59_product_code": "01", "esa59_product_label": "ESA agriculture"},
            {"exio20_product_code": "p01.w.1", "exio20_product_name": "Manure (conventional treatment)", "esa59_product_code": "01", "esa59_product_label": "ESA agriculture"},
            {"exio20_product_code": "pGap", "exio20_product_name": "Direct gap", "esa59_product_code": "24", "esa59_product_label": "ESA chemicals"},
        ]
    )
    direct = pd.DataFrame(
        [
            {"exio20_product_code": "pA", "cpa2002_division": "10", "cpa2002_code": "10.11.11", "cpa2002_label": "Direct food detail"},
            {"exio20_product_code": "pAg", "cpa2002_division": "01", "cpa2002_code": "01.11.11", "cpa2002_label": "Direct agriculture detail"},
            {"exio20_product_code": "p01.w.1", "cpa2002_division": "90", "cpa2002_code": "90.02.11", "cpa2002_label": "Direct waste treatment detail"},
        ]
    )
    conciliation = pd.DataFrame(
        [
            {
                "exio20_product_code": "p01.w.1",
                "exio20_product_label": "Manure (conventional treatment)",
                "chosen_old_cpa2002_divisions": "90",
                "chosen_old_cpa2002_labels": "Sewage and refuse disposal services, sanitation and similar services",
                "conciliation_basis": "direct_cpa2002_more_specific",
                "esa59_old_cpa2002_divisions": "01",
                "direct_old_cpa2002_divisions": "90",
                "notes": "Use direct treatment-service classification.",
            }
        ]
    )

    closed = module.build_closed_first_step_edges(esa, bridge, direct, conciliation)
    by_code = {code: group.reset_index(drop=True) for code, group in closed.groupby("exio20_product_code")}

    assert by_code["pA"].loc[0, "esa59_product_code"] == "10"
    assert by_code["pA"].loc[0, "first_step_source"] == "CPA2002_6digit_EXIOBASE_FINAL_direct"
    assert by_code["pAg"].loc[0, "esa59_product_code"] == "01"
    assert by_code["pAg"].loc[0, "first_step_source"] == "CPA2002_6digit_EXIOBASE_FINAL_direct"
    assert by_code["p01.w.1"].loc[0, "esa59_product_code"] == "90"
    assert by_code["p01.w.1"].loc[0, "first_step_source"] == "curated_first_step_conciliation"
    assert by_code["pGap"].loc[0, "esa59_product_code"] == "24"
    assert by_code["pGap"].loc[0, "first_step_source"] == "ESA59_EXIOBASE20p_ESA59p_direct_gap_fill"
