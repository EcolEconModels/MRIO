from __future__ import annotations

import pandas as pd


def test_build_exio_to_eurostat_mapping_rules_and_labels():
    from ecolecon_mrio.exio_to_eurostat_mapping import build_exio_to_eurostat_mapping

    exio_names = [
        "Paper and paper products",
        "Food products",
        "Unmatched sector",
    ]
    cpa_codes = [
        "CPA_C17",
        "CPA_C10",
    ]
    cpa_label_map = {
        "CPA_C17": "Paper and paper products",
        "CPA_C10": "Food products",
    }

    df = build_exio_to_eurostat_mapping(exio_names, cpa_codes, cpa_label_map)

    got = {(r["aggregate_sector"], r["exio_sector"]) for _, r in df.iterrows()}
    assert ("CPA_C17", "Paper and paper products") in got
    assert ("CPA_C10", "Food products") in got
    assert ("CPA_C17", "Unmatched sector") not in got


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
        min_similarity=0.7,
        min_tfidf_similarity=1.01,
    )

    assert (df["exio_sector"] == "Food products nec").any()
    assert not (df["exio_sector"] == "Misc services").any()
