from pathlib import Path

import pandas as pd


def test_compute_mapping_shares_from_outputs_and_disaggregate():
    from ecolecon_mrio.eurostat_energy_balances import (
        PRIMARY_SOURCES,
        compute_mapping_shares_from_outputs,
        disaggregate_unrolled_sources,
    )

    unrolled = pd.DataFrame(
        [
            {"Category": "A", **{s: 1.0 for s in PRIMARY_SOURCES}},
            {"Category": "B", **{s: 2.0 for s in PRIMARY_SOURCES}},
        ]
    )

    mapping = pd.DataFrame(
        [
            {"Category": "A", "TargetSector": "T1", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "A", "TargetSector": "T2", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "B", "TargetSector": "T2", "AllocationRule": "fixed-share", "DeclaredShare": 1.0},
        ]
    )

    outputs = {"T1": 1.0, "T2": 3.0}
    mapping_w = compute_mapping_shares_from_outputs(mapping, target_output_by_target=outputs)

    # Category A: shares should be 1/4 and 3/4
    a = mapping_w[mapping_w["Category"] == "A"].set_index("TargetSector")["Share_used"]
    assert abs(a.loc["T1"] - 0.25) < 1e-12
    assert abs(a.loc["T2"] - 0.75) < 1e-12

    out = disaggregate_unrolled_sources(unrolled, mapping_w)
    assert set(out.index) == {"T1", "T2"}
    # All sources identical; check one
    assert abs(out.loc["T1", PRIMARY_SOURCES[0]] - 0.25 * 1.0) < 1e-12
    assert abs(out.loc["T2", PRIMARY_SOURCES[0]] - (0.75 * 1.0 + 1.0 * 2.0)) < 1e-12


def test_norm_label_allows_hyphen():
    from ecolecon_mrio import eurostat_energy_balances as eb

    assert eb._norm_label("Final-consumption") == "final-consumption"


def test_nrg_bal_label_alias_resolves():
    from ecolecon_mrio import eurostat_energy_balances as eb

    label_to_code = {eb._norm_label("Final consumption - other sectors - households - energy use"): "FC_OTH_HH_E"}
    code = eb._resolve_nrg_bal_code("Final consumption - households - energy use", label_to_code=label_to_code)
    assert code == "FC_OTH_HH_E"


def test_read_category_target_mapping_csv_accepts_canonical_schema(tmp_path):
    from ecolecon_mrio.eurostat_energy_balances import read_category_target_mapping_csv

    mapping_path = tmp_path / "energy_balance_to_cpa65_products_mapping.csv"
    pd.DataFrame(
        [
            {
                "kind": "category-target",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "nrg_bal",
                "source_version": "current",
                "source_level": "category",
                "target_dataset": "Eurostat_IO",
                "target_system": "pxp",
                "target_version": "CPA_65",
                "target_level": "target-label",
                "source_code": "FC_IND_IS_E",
                "source_label": "Final consumption - industry sector - iron and steel - energy use",
                "target_code": "CPA_C24",
                "target_label": "Basic metals",
                "subtarget_code": "",
                "order": "1",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-concordance",
                "transformation_rule": "split-by-energy-product-prd-use",
                "source_name": "energy_balance_to_cpa65_products_mapping.csv",
                "official_source": "https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm",
                "notes": "",
                "source_balance_role": "final_use",
                "target_sector_role": "end_use_sector",
                "include_in_direct_energy": "1",
            },
            {
                "kind": "category-target",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "nrg_bal",
                "source_version": "current",
                "source_level": "category",
                "target_dataset": "Eurostat_IO",
                "target_system": "pxp",
                "target_version": "CPA_65",
                "target_level": "target-label",
                "source_code": "FC_IND_IS_E",
                "source_label": "Final consumption - industry sector - iron and steel - energy use",
                "target_code": "CPA_C23",
                "target_label": "Other non-metallic mineral products",
                "subtarget_code": "",
                "order": "2",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-concordance",
                "transformation_rule": "split-by-energy-product-prd-use",
                "source_name": "energy_balance_to_cpa65_products_mapping.csv",
                "official_source": "https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm",
                "notes": "",
                "source_balance_role": "final_use",
                "target_sector_role": "end_use_sector",
                "include_in_direct_energy": "1",
            },
        ]
    ).to_csv(mapping_path, index=False)

    out = read_category_target_mapping_csv(mapping_path)
    assert list(out.columns) == ["Category", "TargetSector"]
    assert out["Category"].tolist() == [
        "Final consumption - industry sector - iron and steel - energy use",
        "Final consumption - industry sector - iron and steel - energy use",
    ]
    assert out["TargetSector"].tolist() == ["Basic metals", "Other non-metallic mineral products"]


def test_read_category_mapping_frame_exposes_target_codes_and_rules(tmp_path):
    from ecolecon_mrio.mapping_csv import read_category_mapping_frame

    mapping_path = tmp_path / "energy_balance_to_cpa65_products_mapping.csv"
    pd.DataFrame(
        [
            {
                "kind": "category-target",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "nrg_bal",
                "source_version": "current",
                "source_level": "category",
                "target_dataset": "Eurostat_IO",
                "target_system": "pxp",
                "target_version": "CPA_65",
                "target_level": "target-label",
                "source_code": "FC_IND_IS_E",
                "source_label": "Final consumption - industry sector - iron and steel - energy use",
                "target_code": "CPA_C24",
                "target_label": "Basic metals",
                "subtarget_code": "",
                "order": "1",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-concordance",
                "transformation_rule": "split-by-energy-product-prd-use",
                "source_name": "energy_balance_to_cpa65_products_mapping.csv",
                "official_source": "https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm",
                "notes": "",
                "source_balance_role": "final_use",
                "target_sector_role": "end_use_sector",
                "include_in_direct_energy": "1",
            },
            {
                "kind": "category-target",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "nrg_bal",
                "source_version": "current",
                "source_level": "category",
                "target_dataset": "Eurostat_IO",
                "target_system": "pxp",
                "target_version": "CPA_65",
                "target_level": "target-label",
                "source_code": "FC_IND_IS_E",
                "source_label": "Final consumption - industry sector - iron and steel - energy use",
                "target_code": "CPA_C23",
                "target_label": "Other non-metallic mineral products",
                "subtarget_code": "",
                "order": "2",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-concordance",
                "transformation_rule": "split-by-energy-product-prd-use",
                "source_name": "energy_balance_to_cpa65_products_mapping.csv",
                "official_source": "https://ec.europa.eu/eurostat/cache/metadata/en/nrg_bal_esms.htm",
                "notes": "",
                "source_balance_role": "final_use",
                "target_sector_role": "end_use_sector",
                "include_in_direct_energy": "1",
            },
        ]
    ).to_csv(mapping_path, index=False)

    out = read_category_mapping_frame(mapping_path)
    assert out["TargetCode"].tolist() == ["CPA_C24", "CPA_C23"]
    assert out["AllocationRule"].tolist() == ["split-by-energy-product-prd-use", "split-by-energy-product-prd-use"]
    assert out["SourceBalanceRole"].tolist() == ["final_use", "final_use"]
    assert out["TargetSectorRole"].tolist() == ["end_use_sector", "end_use_sector"]
    assert out["IncludeInDirectEnergy"].tolist() == [True, True]


def test_read_source_bucket_mapping_preserves_declared_order(tmp_path):
    import pandas as pd

    from ecolecon_mrio.mapping_csv import read_source_bucket_mapping

    mapping_path = tmp_path / "energy_primary_source_buckets.csv"
    pd.DataFrame(
        [
            {
                "kind": "source-bucket",
                "source_dataset": "Repo_Energy_Taxonomy",
                "source_system": "primary_source",
                "source_version": "current",
                "source_level": "bucket",
                "target_dataset": "Repo_Energy_Taxonomy",
                "target_system": "reporting_group",
                "target_version": "current",
                "target_level": "group",
                "source_code": "Bucket B",
                "source_label": "Bucket B",
                "target_code": "Group",
                "target_label": "Group",
                "subtarget_code": "",
                "order": "2",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-taxonomy",
                "transformation_rule": "primary-energy-source-registry",
                "source_name": "energy_primary_source_buckets.csv",
                "official_source": "",
                "notes": "",
                "source_balance_role": "",
                "target_sector_role": "",
                "include_in_direct_energy": "",
            },
            {
                "kind": "source-bucket",
                "source_dataset": "Repo_Energy_Taxonomy",
                "source_system": "primary_source",
                "source_version": "current",
                "source_level": "bucket",
                "target_dataset": "Repo_Energy_Taxonomy",
                "target_system": "reporting_group",
                "target_version": "current",
                "target_level": "group",
                "source_code": "Bucket A",
                "source_label": "Bucket A",
                "target_code": "Group",
                "target_label": "Group",
                "subtarget_code": "",
                "order": "1",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-taxonomy",
                "transformation_rule": "primary-energy-source-registry",
                "source_name": "energy_primary_source_buckets.csv",
                "official_source": "",
                "notes": "",
                "source_balance_role": "",
                "target_sector_role": "",
                "include_in_direct_energy": "",
            },
        ]
    ).to_csv(mapping_path, index=False)

    assert read_source_bucket_mapping(mapping_path) == ["Bucket A", "Bucket B"]


def test_direct_raw_category_mapping_matches_aggregate_then_split():
    from ecolecon_mrio.eurostat_energy_balances import (
        PRIMARY_SOURCES,
        compute_mapping_shares_from_outputs,
        disaggregate_unrolled_sources,
    )

    unrolled_direct = pd.DataFrame(
        [
            {"Category": "Final consumption - transport sector - road - energy use", **{s: 10.0 for s in PRIMARY_SOURCES}},
            {"Category": "Final consumption - transport sector - rail - energy use", **{s: 2.0 for s in PRIMARY_SOURCES}},
        ]
    )
    mapping_direct = pd.DataFrame(
        [
            {"Category": "Final consumption - transport sector - road - energy use", "TargetSector": "Land transport services and transport services via pipelines", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "Final consumption - transport sector - road - energy use", "TargetSector": "Postal and courier services", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "Final consumption - transport sector - rail - energy use", "TargetSector": "Land transport services and transport services via pipelines", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "Final consumption - transport sector - rail - energy use", "TargetSector": "Postal and courier services", "AllocationRule": "split-by-energy-product-prd-use"},
        ]
    )
    unrolled_aggregate = pd.DataFrame(
        [
            {"Category": "transport sector - land", **{s: 12.0 for s in PRIMARY_SOURCES}},
        ]
    )
    mapping_aggregate = pd.DataFrame(
        [
            {"Category": "transport sector - land", "TargetSector": "Land transport services and transport services via pipelines", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "transport sector - land", "TargetSector": "Postal and courier services", "AllocationRule": "split-by-energy-product-prd-use"},
        ]
    )

    outputs = {
        "Land transport services and transport services via pipelines": 9.0,
        "Postal and courier services": 3.0,
    }

    direct = disaggregate_unrolled_sources(
        unrolled_direct,
        compute_mapping_shares_from_outputs(mapping_direct, target_output_by_target=outputs),
    )
    aggregate = disaggregate_unrolled_sources(
        unrolled_aggregate,
        compute_mapping_shares_from_outputs(mapping_aggregate, target_output_by_target=outputs),
    )

    pd.testing.assert_frame_equal(direct.sort_index(), aggregate.sort_index())


def test_read_product_source_mapping_groups_codes_by_bucket(tmp_path):
    import pandas as pd

    from ecolecon_mrio.mapping_csv import read_product_source_mapping

    mapping_path = tmp_path / "siec_to_primary_energy_sources.csv"
    pd.DataFrame(
        [
            {
                "kind": "product-source",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "siec",
                "source_version": "current",
                "source_level": "product",
                "target_dataset": "Repo_Energy_Taxonomy",
                "target_system": "primary_source",
                "target_version": "current",
                "target_level": "bucket",
                "source_code": "CODE_1",
                "source_label": "CODE_1",
                "target_code": "Source A",
                "target_label": "Source A",
                "subtarget_code": "",
                "order": "1",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-classification",
                "transformation_rule": "siec-to-primary-energy-source",
                "source_name": "siec_to_primary_energy_sources.csv",
                "official_source": "",
                "notes": "",
                "source_balance_role": "",
                "target_sector_role": "",
                "include_in_direct_energy": "",
            },
            {
                "kind": "product-source",
                "source_dataset": "Eurostat_Energy_Balances",
                "source_system": "siec",
                "source_version": "current",
                "source_level": "product",
                "target_dataset": "Repo_Energy_Taxonomy",
                "target_system": "primary_source",
                "target_version": "current",
                "target_level": "bucket",
                "source_code": "CODE_2",
                "source_label": "CODE_2",
                "target_code": "Source A",
                "target_label": "Source A",
                "subtarget_code": "",
                "order": "2",
                "enabled": "1",
                "provenance": "methodological",
                "certainty": "repo-authored-classification",
                "transformation_rule": "siec-to-primary-energy-source",
                "source_name": "siec_to_primary_energy_sources.csv",
                "official_source": "",
                "notes": "",
                "source_balance_role": "",
                "target_sector_role": "",
                "include_in_direct_energy": "",
            },
        ]
    ).to_csv(mapping_path, index=False)

    assert read_product_source_mapping(mapping_path) == {"Source A": ("CODE_1", "CODE_2")}


def test_primary_sources_are_loaded_from_declared_registry():
    from ecolecon_mrio.eurostat_energy_balances import PRIMARY_SOURCES, PRODUCT_SOURCE_MAPPING

    assert PRIMARY_SOURCES == (
        "Solid fossil fuels",
        "Fossil gas",
        "Oil and petroleum products (excluding biofuel portion)",
        "Hydro",
        "Wind",
        "Solar",
        "Biofuels",
        "Non-renewable waste",
        "Renewable waste and mixed waste",
        "Other Renewables",
    )
    assert PRODUCT_SOURCE_MAPPING["Fossil gas"] == ("G3000", "C0350-0370")


def test_compute_mapping_shares_requires_positive_weighting_reference():
    from ecolecon_mrio.eurostat_energy_balances import compute_mapping_shares_from_outputs

    mapping = pd.DataFrame(
        [
            {"Category": "A", "TargetSector": "T1", "AllocationRule": "split-by-energy-product-prd-use"},
            {"Category": "A", "TargetSector": "T2", "AllocationRule": "split-by-energy-product-prd-use"},
        ]
    )

    try:
        compute_mapping_shares_from_outputs(
            mapping,
            target_output_by_target={"T1": 0.0, "T2": 0.0},
        )
    except ValueError as exc:
        assert "No positive weighting reference" in str(exc)
    else:
        raise AssertionError("Expected strict allocation to fail without positive weighting reference")


def test_compute_mapping_shares_uses_declared_shares_without_runtime_weights():
    from ecolecon_mrio.eurostat_energy_balances import compute_mapping_shares_from_outputs

    mapping = pd.DataFrame(
        [
            {"Category": "A", "TargetSector": "T1", "DeclaredShare": 0.2, "AllocationRule": "fixed-share", "TargetCode": "CPA_T1"},
            {"Category": "A", "TargetSector": "T2", "DeclaredShare": 0.8, "AllocationRule": "fixed-share", "TargetCode": "CPA_T2"},
        ]
    )

    out = compute_mapping_shares_from_outputs(mapping)
    assert out["Share_used"].tolist() == [0.2, 0.8]


def test_compute_mapping_shares_can_weight_by_target_code():
    from ecolecon_mrio.eurostat_energy_balances import compute_mapping_shares_from_outputs

    mapping = pd.DataFrame(
        [
            {
                "Category": "A",
                "TargetSector": "Chemicals and chemical products",
                "TargetCode": "CPA_C20",
                "AllocationRule": "split-by-energy-product-prd-use",
            },
            {
                "Category": "A",
                "TargetSector": "Basic pharmaceutical products and pharmaceutical preparations",
                "TargetCode": "CPA_C21",
                "AllocationRule": "split-by-energy-product-prd-use",
            },
        ]
    )

    out = compute_mapping_shares_from_outputs(
        mapping,
        target_output_by_code={"CPA_C20": 2.0, "CPA_C21": 6.0},
    )
    shares = out.set_index("TargetCode")["Share_used"]
    assert abs(shares.loc["CPA_C20"] - 0.25) < 1e-12
    assert abs(shares.loc["CPA_C21"] - 0.75) < 1e-12


def test_build_postgrowth_energy_balance_splits_energy_rows():
    from ecolecon_mrio.eurostat_energy_balances import build_postgrowth_energy_balance

    mapped = pd.DataFrame(
        [
            {
                "TargetSector": "Coke and refined petroleum products",
                "Solid fossil fuels": 1.0,
                "Fossil gas": 2.0,
                "Oil and petroleum products (excluding biofuel portion)": 3.0,
                "Hydro": 0.0,
                "Wind": 0.0,
                "Solar": 0.0,
                "Biofuels": 1.0,
                "Non-renewable waste": 1.0,
                "Renewable waste and mixed waste": 1.0,
                "Other Renewables": 3.0,
            },
            {
                "TargetSector": "Electricity, gas, steam and air conditioning",
                "Solid fossil fuels": 10.0,
                "Fossil gas": 20.0,
                "Oil and petroleum products (excluding biofuel portion)": 30.0,
                "Hydro": 4.0,
                "Wind": 5.0,
                "Solar": 6.0,
                "Biofuels": 7.0,
                "Non-renewable waste": 4.0,
                "Renewable waste and mixed waste": 4.0,
                "Other Renewables": 9.0,
            },
            {
                "TargetSector": "Households",
                "Solid fossil fuels": 0.5,
                "Fossil gas": 1.5,
                "Oil and petroleum products (excluding biofuel portion)": 2.5,
                "Hydro": 0.0,
                "Wind": 0.0,
                "Solar": 0.0,
                "Biofuels": 0.0,
                "Non-renewable waste": 0.0,
                "Renewable waste and mixed waste": 0.0,
                "Other Renewables": 0.0,
            },
        ]
    )

    out = build_postgrowth_energy_balance(
        mapped,
        source_group_by_source={
            "Solid fossil fuels": "Fossil",
            "Fossil gas": "Fossil",
            "Oil and petroleum products (excluding biofuel portion)": "Fossil",
            "Hydro": "Renewable",
            "Wind": "Renewable",
            "Solar": "Renewable",
            "Biofuels": "Renewable",
            "Non-renewable waste": "Fossil",
            "Renewable waste and mixed waste": "Renewable",
            "Other Renewables": "Renewable",
        },
        split_energy_green_brown=True,
    )

    sectors = out["Sector"].tolist()
    assert sectors == ["Energy Brown", "Energy Green", "Households"]

    brown = out[out["Sector"] == "Energy Brown"].iloc[0]
    green = out[out["Sector"] == "Energy Green"].iloc[0]

    assert brown["Solid fossil fuels"] == 11.0
    assert brown["Fossil gas"] == 22.0
    assert brown["Oil and petroleum products (excluding biofuel portion)"] == 33.0
    assert brown["Hydro"] == 0.0
    assert brown["Biofuels"] == 0.0
    assert brown["Non-renewable waste"] == 5.0
    assert brown["Total_energy_GWh"] == 71.0

    assert green["Solid fossil fuels"] == 0.0
    assert green["Hydro"] == 4.0
    assert green["Wind"] == 5.0
    assert green["Solar"] == 6.0
    assert green["Biofuels"] == 8.0
    assert green["Non-renewable waste"] == 0.0
    assert green["Renewable waste and mixed waste"] == 5.0
    assert green["Other Renewables"] == 12.0
    assert green["Total_energy_GWh"] == 40.0


def test_build_postgrowth_energy_balance_can_split_single_electricity_row():
    from ecolecon_mrio.eurostat_energy_balances import build_postgrowth_energy_balance

    mapped = pd.DataFrame(
        [
            {
                "TargetSector": "Agriculture",
                "Solid fossil fuels": 1.0,
                "Fossil gas": 0.0,
                "Oil and petroleum products (excluding biofuel portion)": 0.0,
                "Hydro": 0.0,
                "Wind": 0.0,
                "Solar": 0.0,
                "Biofuels": 0.0,
                "Non-renewable waste": 0.0,
                "Renewable waste and mixed waste": 0.0,
                "Other Renewables": 0.0,
            },
            {
                "TargetSector": "Electricity, gas, steam and air conditioning",
                "Solid fossil fuels": 10.0,
                "Fossil gas": 20.0,
                "Oil and petroleum products (excluding biofuel portion)": 30.0,
                "Hydro": 4.0,
                "Wind": 5.0,
                "Solar": 6.0,
                "Biofuels": 7.0,
                "Non-renewable waste": 8.0,
                "Renewable waste and mixed waste": 9.0,
                "Other Renewables": 10.0,
            },
            {
                "TargetSector": "Households",
                "Solid fossil fuels": 0.5,
                "Fossil gas": 1.5,
                "Oil and petroleum products (excluding biofuel portion)": 2.5,
                "Hydro": 0.0,
                "Wind": 0.0,
                "Solar": 0.0,
                "Biofuels": 0.0,
                "Non-renewable waste": 0.0,
                "Renewable waste and mixed waste": 0.0,
                "Other Renewables": 0.0,
            },
        ]
    )

    out = build_postgrowth_energy_balance(
        mapped,
        source_group_by_source={
            "Solid fossil fuels": "Fossil",
            "Fossil gas": "Fossil",
            "Oil and petroleum products (excluding biofuel portion)": "Fossil",
            "Hydro": "Renewable",
            "Wind": "Renewable",
            "Solar": "Renewable",
            "Biofuels": "Renewable",
            "Non-renewable waste": "Fossil",
            "Renewable waste and mixed waste": "Renewable",
            "Other Renewables": "Renewable",
        },
        split_energy_green_brown=True,
    )

    assert out["Sector"].tolist() == ["Agriculture", "Energy Brown", "Energy Green", "Households"]

    brown = out[out["Sector"] == "Energy Brown"].iloc[0]
    green = out[out["Sector"] == "Energy Green"].iloc[0]

    assert brown["Solid fossil fuels"] == 10.0
    assert brown["Fossil gas"] == 20.0
    assert brown["Oil and petroleum products (excluding biofuel portion)"] == 30.0
    assert brown["Non-renewable waste"] == 8.0
    assert brown["Renewable waste and mixed waste"] == 0.0
    assert brown["Total_energy_GWh"] == 68.0

    assert green["Solid fossil fuels"] == 0.0
    assert green["Hydro"] == 4.0
    assert green["Wind"] == 5.0
    assert green["Solar"] == 6.0
    assert green["Biofuels"] == 7.0
    assert green["Non-renewable waste"] == 0.0
    assert green["Renewable waste and mixed waste"] == 9.0
    assert green["Other Renewables"] == 10.0
    assert green["Total_energy_GWh"] == 41.0


def test_energy_balance_mapping_covers_cached_nrg_bal_codes():
    import json

    repo_root = Path(__file__).resolve().parents[1]
    mapping_path = repo_root / "scripts" / "mappings" / "energy_balance_to_cpa65_products_mapping.csv"
    metadata_path = next((repo_root / "data" / "eurostat" / "nrg_bal_c").glob("*/raw.json"))

    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    nrg_bal_dim = metadata["dimension"]["nrg_bal"]["category"]["label"]
    official_codes = set(map(str, nrg_bal_dim.keys()))
    mapped_codes = set(mapping["source_code"].astype(str).str.strip())

    assert official_codes.issubset(mapped_codes)
