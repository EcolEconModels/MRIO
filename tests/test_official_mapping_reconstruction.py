from __future__ import annotations

import pandas as pd


def test_build_weighted_value_comparison_scales_to_reference():
    from ecolecon_mrio.diagnostic_concordance import build_weighted_value_comparison

    mapping = pd.DataFrame(
        [
            {"aggregate_sector": "CPA_A", "exio_sector": "S1", "weight": 1.0, "enabled": "1", "notes": "", "source_name": "m.csv", "provenance": "official", "allocation_rule": "identity"},
            {"aggregate_sector": "CPA_B", "exio_sector": "S2", "weight": 1.0, "enabled": "1", "notes": "", "source_name": "m.csv", "provenance": "official", "allocation_rule": "identity"},
        ]
    )
    source = pd.Series({"S1": 20.0, "S2": 30.0})
    reference = pd.Series({"CPA_A": 40.0, "CPA_B": 60.0})

    got = build_weighted_value_comparison(source, reference, mapping, scale_to_reference=True)
    got = got.set_index("target_sector")

    assert abs(got.loc["CPA_A", "mapped_value_scaled"] - 40.0) < 1e-12
    assert abs(got.loc["CPA_B", "mapped_value_scaled"] - 60.0) < 1e-12
    assert abs(got.loc["CPA_A", "abs_gap"]) < 1e-12
    assert abs(got.loc["CPA_B", "abs_gap"]) < 1e-12


def test_build_mapping_audit_reports_weight_sums():
    from ecolecon_mrio.diagnostic_concordance import build_mapping_audit

    mapping = pd.DataFrame(
        [
            {"aggregate_sector": "CPA_A", "exio_sector": "S1", "weight": 0.5, "enabled": "1", "notes": "", "source_name": "m.csv", "provenance": "official", "allocation_rule": "identity"},
            {"aggregate_sector": "CPA_B", "exio_sector": "S1", "weight": 0.5, "enabled": "1", "notes": "x", "source_name": "m.csv", "provenance": "official", "allocation_rule": "identity"},
        ]
    )

    got = build_mapping_audit(mapping, mapping_family="weighted")

    assert (got["mapping_family"] == "weighted").all()
    assert got["weight_sum_ok"].all()
    assert got["has_notes"].sum() == 1


def test_allocate_category_matrix_by_share_reconstructs_energy_and_co2_components():
    from ecolecon_mrio.diagnostic_concordance import allocate_category_matrix_by_share

    mapping = pd.DataFrame(
        [
            {"Category": "Industry", "TargetSector": "T1", "Share_used": 0.4},
            {"Category": "Industry", "TargetSector": "T2", "Share_used": 0.6},
            {"Category": "Services", "TargetSector": "T2", "Share_used": 1.0},
        ]
    )
    category_matrix = pd.DataFrame(
        {
            "solid_gwh": {"Industry": 100.0, "Services": 0.0},
            "gas_gwh": {"Industry": 20.0, "Services": 10.0},
            "co2_tonnes": {"Industry": 50.0, "Services": 5.0},
        }
    )

    result = allocate_category_matrix_by_share(category_matrix, mapping)

    assert abs(result.target_matrix.loc["T1", "solid_gwh"] - 40.0) < 1e-12
    assert abs(result.target_matrix.loc["T2", "solid_gwh"] - 60.0) < 1e-12
    assert abs(result.target_matrix.loc["T1", "co2_tonnes"] - 20.0) < 1e-12
    assert abs(result.target_matrix.loc["T2", "co2_tonnes"] - 35.0) < 1e-12
    assert abs(result.category_reconstructed.loc["Industry", "solid_gwh"] - 100.0) < 1e-12
    assert abs(result.category_reconstructed.loc["Industry", "co2_tonnes"] - 50.0) < 1e-12
    assert abs(result.category_reconstructed.loc["Services", "gas_gwh"] - 10.0) < 1e-12