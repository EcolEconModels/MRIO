from __future__ import annotations

import pandas as pd
import pytest


def test_normalize_weighted_mapping_frame_validates_and_groups_duplicates():
    from ecolecon_mrio.diagnostic_concordance import normalize_weighted_mapping_frame

    raw = pd.DataFrame(
        [
            {"target": "CPA_A", "exio": "Sector 1", "w": 0.25, "note": "official"},
            {"target": "CPA_A", "exio": "Sector 1", "w": 0.25, "note": "duplicate"},
            {"target": "CPA_B", "exio": "Sector 1", "w": 0.50, "note": "official"},
            {"target": "CPA_A", "exio": "Sector 2", "w": 1.00, "note": "official"},
        ]
    )

    got = normalize_weighted_mapping_frame(
        raw,
        aggregate_col="target",
        exio_col="exio",
        weight_col="w",
        notes_col="note",
        source_name="official.csv",
        provenance="official",
        transformation_rule="identity",
    )

    assert list(got.columns) == [
        "aggregate_sector",
        "exio_sector",
        "weight",
        "enabled",
        "notes",
        "source_name",
        "provenance",
        "transformation_rule",
    ]
    row = got[(got["aggregate_sector"] == "CPA_A") & (got["exio_sector"] == "Sector 1")].iloc[0]
    assert abs(float(row["weight"]) - 0.5) < 1e-12
    assert "official" in row["notes"]
    assert "duplicate" in row["notes"]


def test_normalize_weighted_mapping_frame_rejects_bad_weight_sums():
    from ecolecon_mrio.diagnostic_concordance import normalize_weighted_mapping_frame

    raw = pd.DataFrame(
        [
            {"target": "CPA_A", "exio": "Sector 1", "w": 0.8},
            {"target": "CPA_B", "exio": "Sector 1", "w": 0.1},
        ]
    )

    with pytest.raises(ValueError):
        normalize_weighted_mapping_frame(
            raw,
            aggregate_col="target",
            exio_col="exio",
            weight_col="w",
            source_name="official.csv",
            provenance="official",
            transformation_rule="identity",
        )


def test_allocate_weighted_values_reconstructs_source_totals():
    from ecolecon_mrio.diagnostic_concordance import allocate_weighted_values

    mapping = pd.DataFrame(
        [
            {"aggregate_sector": "CPA_A", "exio_sector": "S1", "weight": 0.4, "enabled": "1"},
            {"aggregate_sector": "CPA_B", "exio_sector": "S1", "weight": 0.6, "enabled": "1"},
            {"aggregate_sector": "CPA_A", "exio_sector": "S2", "weight": 1.0, "enabled": "1"},
        ]
    )
    source = pd.Series({"S1": 100.0, "S2": 50.0})

    result = allocate_weighted_values(source, mapping)

    assert abs(result.aggregate_totals.loc["CPA_A"] - 90.0) < 1e-12
    assert abs(result.aggregate_totals.loc["CPA_B"] - 60.0) < 1e-12
    assert abs(result.source_reconstructed.loc["S1"] - 100.0) < 1e-12
    assert abs(result.source_reconstructed.loc["S2"] - 50.0) < 1e-12


def test_allocate_category_values_by_share_reconstructs_categories():
    from ecolecon_mrio.diagnostic_concordance import allocate_category_values_by_share

    mapping = pd.DataFrame(
        [
            {"Category": "C1", "TargetSector": "T1", "Share_used": 0.25},
            {"Category": "C1", "TargetSector": "T2", "Share_used": 0.75},
            {"Category": "C2", "TargetSector": "T2", "Share_used": 1.00},
        ]
    )
    values = pd.Series({"C1": 40.0, "C2": 10.0})

    result = allocate_category_values_by_share(values, mapping)

    assert abs(result.target_totals.loc["T1"] - 10.0) < 1e-12
    assert abs(result.target_totals.loc["T2"] - 40.0) < 1e-12
    assert abs(result.category_reconstructed.loc["C1"] - 40.0) < 1e-12
    assert abs(result.category_reconstructed.loc["C2"] - 10.0) < 1e-12


def test_build_exio_nace_audit_flags_one_to_many_mappings():
    from ecolecon_mrio.diagnostic_concordance import build_exio_nace_audit

    df = pd.DataFrame(
        [
            {"exio_sector": "Electricity by coal", "nace_code": "D.35.11", "source_name": "c.xlsx", "provenance": "semi-official", "transformation_rule": "raw"},
            {"exio_sector": "Electricity by coal", "nace_code": "D.35.13", "source_name": "c.xlsx", "provenance": "semi-official", "transformation_rule": "raw"},
            {"exio_sector": "refined petroleum products", "nace_code": "C.19.2", "source_name": "c.xlsx", "provenance": "semi-official", "transformation_rule": "raw"},
        ]
    )

    got = build_exio_nace_audit(df)
    coal = got[got["exio_sector"] == "Electricity by coal"]
    refined = got[got["exio_sector"] == "refined petroleum products"]

    assert (coal["nace_count"] == 2).all()
    assert not coal["is_one_to_one"].any()
    assert (refined["nace_count"] == 1).all()
    assert refined["is_one_to_one"].all()