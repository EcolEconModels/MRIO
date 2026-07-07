from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "mappings" / "compare_exiobase3_to_cpa_mapping_routes.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_route_maps_writes_expected_diagnostics(tmp_path: Path) -> None:
    module = load_module(COMPARE_SCRIPT, "compare_exio3_cpa_routes")
    old_csv = tmp_path / "old.csv"
    new_csv = tmp_path / "new.csv"
    out_dir = tmp_path / "compare"

    pd.DataFrame(
        [
            {"source_code": "p01", "source_label": "Product 1", "target_code": "CPA_A", "target_label": "A", "enabled": "1"},
            {"source_code": "p02", "source_label": "Product 2 old", "target_code": "CPA_B", "target_label": "B", "enabled": "1"},
            {"source_code": "p03", "source_label": "Product 3", "target_code": "CPA_C", "target_label": "C", "enabled": "1"},
            {"source_code": "p03", "source_label": "Product 3", "target_code": "CPA_D", "target_label": "D", "enabled": "1"},
            {"source_code": "p04", "source_label": "Disabled", "target_code": "CPA_Z", "target_label": "Z", "enabled": "0"},
        ]
    ).to_csv(old_csv, index=False)
    pd.DataFrame(
        [
            {"source_code": "p01", "source_label": "Product 1", "target_code": "CPA_A", "target_label": "A", "enabled": "1"},
            {"source_code": "p02", "source_label": "Product 2 new", "target_code": "CPA_C", "target_label": "C", "enabled": "1"},
            {"source_code": "p03", "source_label": "Product 3", "target_code": "CPA_C", "target_label": "C", "enabled": "1"},
            {"source_code": "p03", "source_label": "Product 3", "target_code": "CPA_D", "target_label": "D", "enabled": "1"},
        ]
    ).to_csv(new_csv, index=False)

    assert module.main([
        "--old-csv", str(old_csv),
        "--new-csv", str(new_csv),
        "--out-dir", str(out_dir),
        "--target-version", "cpa65",
    ]) == 0

    summary = json.loads((out_dir / "cpa65_route_comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["old_rows"] == 4
    assert summary["new_rows"] == 4
    assert summary["same_source_target_pairs"] == 3
    assert summary["only_old_pairs"] == 1
    assert summary["only_new_pairs"] == 1
    assert summary["changed_exio_products"] == 1
    assert summary["old_multi_target_products"] == 1
    assert summary["new_multi_target_products"] == 1

    only_old = pd.read_csv(out_dir / "cpa65_source_target_pairs_only_old.csv", dtype=str, keep_default_na=False)
    only_new = pd.read_csv(out_dir / "cpa65_source_target_pairs_only_new.csv", dtype=str, keep_default_na=False)
    per_source = pd.read_csv(out_dir / "cpa65_per_source_target_set_differences.csv", dtype=str, keep_default_na=False)
    old_multi = pd.read_csv(out_dir / "cpa65_multi_target_sources_old.csv", dtype=str, keep_default_na=False)
    new_multi = pd.read_csv(out_dir / "cpa65_multi_target_sources_new.csv", dtype=str, keep_default_na=False)

    assert set(only_old["source_code"]) == {"p02"}
    assert set(only_old["target_code"]) == {"CPA_B"}
    assert set(only_new["source_code"]) == {"p02"}
    assert set(only_new["target_code"]) == {"CPA_C"}
    assert set(per_source["source_code"]) == {"p02"}
    assert set(old_multi["source_code"]) == {"p03"}
    assert set(new_multi["source_code"]) == {"p03"}


def test_default_new_path_reflects_anchor_source_mode() -> None:
    module = load_module(COMPARE_SCRIPT, "compare_exio3_cpa_routes_defaults")
    direct_only = module.default_new_csv("cpa65", anchor_source_mode="direct-only")
    curated = module.default_new_csv("cpa65", anchor_source_mode="direct-plus-curated-gaps")
    assert direct_only.name == "exio_to_eurostat_cpa65_products_from_official_concordances.csv"
    assert curated.name == "exio_to_eurostat_cpa65_products_from_official_concordances_direct_plus_curated_gaps.csv"
