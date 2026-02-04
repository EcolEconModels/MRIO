from __future__ import annotations

import pandas as pd


def test_build_exio_to_eurostat_mapping_rules_and_labels():
    from ecolecon_mrio.exio_to_eurostat_mapping import build_exio_to_eurostat_mapping

    exio_names = [
        "Coke Oven Coke",
        "Petroleum Coke",
        "Electricity by coal",
        "Food products nec",
        "Leather products",
        "Unmatched sector",
    ]
    cpa_codes = [
        "CPA_C19",
        "CPA_D",
        "CPA_C10",
        "CPA_C15",
    ]
    cpa_label_map = {
        "CPA_C19": "Coke and refined petroleum products",
        "CPA_D": "Electricity, gas, steam and air conditioning",
        "CPA_C10": "Food products",
        "CPA_C15": "Leather and related products",
    }

    df = build_exio_to_eurostat_mapping(exio_names, cpa_codes, cpa_label_map, min_similarity=0.1)

    got = {(r["aggregate_sector"], r["exio_sector"]) for _, r in df.iterrows()}
    assert ("CPA_C19", "Coke Oven Coke") in got
    assert ("CPA_C19", "Petroleum Coke") in got
    assert ("CPA_D", "Electricity by coal") in got
    assert ("CPA_C10", "Food products nec") in got
    assert ("CPA_C15", "Leather products") in got
    assert ("CPA_C19", "Unmatched sector") not in got


def test_build_exio_to_eurostat_mapping_min_similarity_filters():
    from ecolecon_mrio.exio_to_eurostat_mapping import build_exio_to_eurostat_mapping

    exio_names = ["Food products nec", "Misc services"]
    cpa_codes = ["CPA_C10", "CPA_S96"]
    cpa_label_map = {
        "CPA_C10": "Food products",
        "CPA_S96": "Other personal services",
    }

    df = build_exio_to_eurostat_mapping(
        exio_names,
        cpa_codes,
        cpa_label_map,
        min_similarity=0.6,
    )

    assert (df["exio_sector"] == "Food products nec").any()
    assert not (df["exio_sector"] == "Misc services").any()
