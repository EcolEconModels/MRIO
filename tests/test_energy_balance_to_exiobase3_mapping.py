from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "mappings" / "build_energy_balance_to_exiobase3_mapping.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_energy_balance_to_exiobase3_mapping", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_synthetic_zip(path: Path) -> None:
    nace_csv = "\n".join(
        [
            "NACE2,EXIOBASE name,EXIOBASE code",
            "24.10,Basic iron and steel,e20.basic_metals",
            "49.10,Rail transport,e20.rail",
            "49.41,Road freight transport,e20.road",
            "35.11,Electricity generation,e20.electricity",
        ]
    )
    bridge = pd.DataFrame(
        [
            {
                "EXIOBASE 2 Name": "Basic iron and steel",
                "EXIOBASE 2 code": "e20.basic_metals",
                "EXIOBASE 3 name": "Basic iron and steel and of ferro-alloys and first products thereof",
                "EXIOBASE 3 code": "p24.a",
            },
            {
                "EXIOBASE 2 Name": "Rail transport",
                "EXIOBASE 2 code": "e20.rail",
                "EXIOBASE 3 name": "Railway transportation services",
                "EXIOBASE 3 code": "p49.rail",
            },
            {
                "EXIOBASE 2 Name": "Road freight transport",
                "EXIOBASE 2 code": "e20.road",
                "EXIOBASE 3 name": "Other land transportation services",
                "EXIOBASE 3 code": "p49.road",
            },
            {
                "EXIOBASE 2 Name": "Electricity generation",
                "EXIOBASE 2 code": "e20.electricity",
                "EXIOBASE 3 name": "Production of electricity by gas",
                "EXIOBASE 3 code": "p40.gas",
            },
        ]
    )
    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
        bridge.to_excel(writer, index=False)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Concordances/Archive/International classifications/NACE2_EXIOBASE20p_list.csv", nace_csv)
        zf.writestr("Concordances/EXIOBASE/EXIOBASE2 products comparison with EXIOBASE3 products.xlsx", excel_bytes.getvalue())


def make_rules(path: Path) -> None:
    columns = list(load_module().RULE_COLUMNS)
    rows = [
        {
            "order": "1",
            "enabled": "1",
            "source_dataset": "Eurostat_Energy_Balances",
            "source_system": "nrg_bal",
            "source_version": "latest",
            "source_level": "balance-flow",
            "source_code": "FC_IND_IS_E",
            "source_label": "Final consumption - industry sector - iron and steel - energy use",
            "source_balance_role": "final_use",
            "mapping_basis": "nace2_division",
            "selector_codes": "24",
            "selector_label": "NACE 24",
            "relation_type": "official-definition-to-nace-selector",
            "allocation_rule": "unweighted-relational-needs-runtime-weight",
            "certainty": "definition-rule",
            "official_source": "test",
            "include_in_direct_energy": "1",
            "notes": "test rule",
        },
        {
            "order": "2",
            "enabled": "1",
            "source_dataset": "Eurostat_Energy_Balances",
            "source_system": "nrg_bal",
            "source_version": "latest",
            "source_level": "balance-flow",
            "source_code": "FC_TRA_RAIL_E",
            "source_label": "Final consumption - transport sector - rail - energy use",
            "source_balance_role": "final_use",
            "mapping_basis": "nace2_exact_or_prefix",
            "selector_codes": "49.1",
            "selector_label": "NACE 49.1",
            "relation_type": "official-definition-to-nace-selector",
            "allocation_rule": "unweighted-relational-needs-runtime-weight",
            "certainty": "definition-rule",
            "official_source": "test",
            "include_in_direct_energy": "1",
            "notes": "test rule",
        },
        {
            "order": "3",
            "enabled": "1",
            "source_dataset": "Eurostat_Energy_Balances",
            "source_system": "nrg_bal",
            "source_version": "latest",
            "source_level": "balance-flow",
            "source_code": "FC_OTH_HH_E",
            "source_label": "Final consumption - other sectors - households - energy use",
            "source_balance_role": "final_use",
            "mapping_basis": "no_exio_production_sector",
            "selector_codes": "",
            "selector_label": "Households final-use row",
            "relation_type": "not-a-production-sector",
            "allocation_rule": "no-exio-target",
            "certainty": "definition-rule",
            "official_source": "test",
            "include_in_direct_energy": "1",
            "notes": "household final demand",
        },
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_nace_division_and_prefix_rules_build_independent_exio_links(tmp_path: Path) -> None:
    module = load_module()
    zip_path = tmp_path / "exiobase_concordances.zip"
    rules_path = tmp_path / "rules.csv"
    make_synthetic_zip(zip_path)
    make_rules(rules_path)

    result = module.build_mapping(zip_path, rules_path)

    assert set(result.collapsed["source_code"]) == {"FC_IND_IS_E", "FC_TRA_RAIL_E"}
    assert set(result.collapsed.loc[result.collapsed["source_code"] == "FC_IND_IS_E", "exio3_product_code"]) == {"p24.a"}
    assert set(result.collapsed.loc[result.collapsed["source_code"] == "FC_TRA_RAIL_E", "exio3_product_code"]) == {"p49.rail"}
    assert "p49.road" not in set(result.collapsed["exio3_product_code"])
    assert set(result.diagnostics["source_code"]) == {"FC_OTH_HH_E"}
    assert set(result.diagnostics["diagnostic_kind"]) == {"expected_no_exio_target"}


def test_outputs_write_manifest_and_diagnostics(tmp_path: Path) -> None:
    module = load_module()
    zip_path = tmp_path / "exiobase_concordances.zip"
    rules_path = tmp_path / "rules.csv"
    out_dir = tmp_path / "out"
    make_synthetic_zip(zip_path)
    make_rules(rules_path)

    result = module.build_mapping(zip_path, rules_path)
    manifest = module.write_outputs(result, concordances_zip=zip_path, rules_csv=rules_path, out_dir=out_dir)

    assert manifest["mapping_name"] == "eurostat_energy_balance_to_exiobase3_products"
    assert (out_dir / "eb_to_exio3_products_long.csv").exists()
    assert (out_dir / "eb_to_exio3_products_collapsed.csv").exists()
    assert (out_dir / "eb_unmatched_or_ambiguous.csv").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "README.md").exists()


def test_generator_source_does_not_read_manual_energy_balance_to_cpa_mapping() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "energy_balance_to_cpa65_products_mapping.csv" not in source
