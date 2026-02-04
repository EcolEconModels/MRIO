from __future__ import annotations

import pandas as pd


def test_aggregate_intensity_by_mapping_ratio_of_sums():
    from ecolecon_mrio.sector_intensity import aggregate_intensity_by_mapping

    exio_df = pd.DataFrame(
        {
            "exio_sector": ["S1", "S2", "S3"],
            "extension": ["air", "air", "air"],
            "stressor": ["CO2", "CO2", "CO2"],
            "unit": ["kg", "kg", "kg"],
            "emission": [10.0, 30.0, 60.0],
            "monetary_output": [5.0, 15.0, 30.0],
        }
    )
    mapping = {"CPA_X": ["S1", "S2"], "CPA_Y": ["S3"]}

    res = aggregate_intensity_by_mapping(exio_df, mapping)

    res = res.set_index("aggregated_sector")
    assert res.loc["CPA_X", "emission"] == 40.0
    assert res.loc["CPA_X", "monetary_output"] == 20.0
    assert res.loc["CPA_X", "emission_intensity"] == 2.0
    assert res.loc["CPA_X", "n_exio_sectors"] == 2
    assert res.loc["CPA_Y", "emission_intensity"] == 2.0


def test_aggregate_intensity_by_mapping_without_exio_list():
    from ecolecon_mrio.sector_intensity import aggregate_intensity_by_mapping

    exio_df = pd.DataFrame(
        {
            "exio_sector": ["S1", "S2"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "emission": [10.0, 30.0],
            "monetary_output": [5.0, 15.0],
        }
    )
    mapping = {"CPA_X": ["S1", "S2"]}

    res = aggregate_intensity_by_mapping(exio_df, mapping, include_exio_list=False)

    assert "exio_sectors" not in res.columns
    assert "n_exio_sectors" not in res.columns


def test_aggregate_intensity_by_mapping_with_weights():
    from ecolecon_mrio.sector_intensity import aggregate_intensity_by_mapping

    exio_df = pd.DataFrame(
        {
            "exio_sector": ["S1", "S2"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "emission": [10.0, 30.0],
            "monetary_output": [5.0, 15.0],
        }
    )
    mapping = {"CPA_X": {"S1": 0.25}, "CPA_Y": {"S1": 0.75}, "CPA_Z": {"S2": 1.0}}

    res = aggregate_intensity_by_mapping(exio_df, mapping)

    res = res.set_index("aggregated_sector")
    assert res.loc["CPA_X", "emission"] == 2.5
    assert res.loc["CPA_X", "monetary_output"] == 1.25
    assert res.loc["CPA_X", "emission_intensity"] == 2.0
    assert "exio_weights" in res.columns
