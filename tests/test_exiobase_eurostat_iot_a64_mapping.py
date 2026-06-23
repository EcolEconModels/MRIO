from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mappings" / "build_exiobase3_to_eurostat_iot_a64_mapping.py"


def load_builder_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_exiobase3_to_eurostat_iot_a64_mapping", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a64_codelist_has_64_unique_categories_and_divisions() -> None:
    mod = load_builder_module()
    categories = mod.a64_categories()
    assert len(categories) == 64
    assert len({cat.code for cat in categories}) == 64

    lookup = mod.division_to_a64_lookup()
    assert lookup["01"].code == "A01"
    assert lookup["68"].code == "L68"
    assert lookup["99"].code == "U"
    assert "04" not in lookup


def test_nace2_to_division_parser() -> None:
    mod = load_builder_module()
    assert mod.nace2_to_division("1.11") == "01"
    assert mod.nace2_to_division("10.12") == "10"
    assert mod.nace2_to_division("99") == "99"
    with pytest.raises(ValueError):
        mod.nace2_to_division("not-a-code")


def test_generated_collapsed_mapping_if_present() -> None:
    path = (
        REPO_ROOT
        / "data"
        / "derived"
        / "concordances"
        / "exiobase_eurostat_iot_a64"
        / "exiobase3_products_to_eurostat_iot_a64_collapsed.csv"
    )
    if not path.exists():
        pytest.skip("generated derived mapping is not present in this checkout")

    df = pd.read_csv(path)
    required = {
        "exio3_product_code",
        "exio3_product_name",
        "iot_a64_code",
        "iot_a64_label",
        "equal_split_weight_if_required",
        "weight_status",
    }
    assert required.issubset(df.columns)
    assert df["exio3_product_code"].nunique() == 200
    assert df["iot_a64_code"].nunique() == 64
    assert not df["exio3_product_code"].isna().any()
    assert not df["iot_a64_code"].isna().any()

    sums = df.groupby("exio3_product_code")["equal_split_weight_if_required"].sum().round(12)
    assert (sums == 1.0).all()
    assert set(df["weight_status"]) == {"placeholder_equal_split_not_empirical"}
