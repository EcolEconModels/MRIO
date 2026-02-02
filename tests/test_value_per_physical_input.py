from __future__ import annotations

import pandas as pd


def test_map_exio_value_per_physical_input_uses_subsector_map():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import map_exio_value_per_physical_input

    split_profiles = {
        "D35": SplitProfiles(
            aggregate_sector="D35",
            subsectors=("D35::Coal", "D35::Solar"),
            output_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            intermediate_sales_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            intermediate_input_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            final_demand_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            subsector_to_exio_sectors={"D35::Coal": ["Coal"], "D35::Solar": ["Solar"]},
        )
    }

    exio_qp = pd.DataFrame(
        {
            "exio_sector": ["Coal", "Solar"],
            "extension": ["energy", "energy"],
            "stressor": ["Energy use - Gross", "Energy use - Gross"],
            "unit": ["TJ", "TJ"],
            "quantity": [10.0, 5.0],
            "monetary_output": [100.0, 80.0],
            "value_per_physical_input": [10.0, 16.0],
        }
    )

    res = map_exio_value_per_physical_input(exio_qp, split_profiles)
    assert res is not None
    assert set(res["disaggregated_sector"]) == {"D35::Coal", "D35::Solar"}

    values = dict(zip(res["disaggregated_sector"], res["value_per_physical_input"]))
    assert values["D35::Coal"] == 10.0
    assert values["D35::Solar"] == 16.0


def test_compute_aggregated_sector_value_per_physical_input_uses_ratio_of_sums():
    from ecolecon_mrio.iot_aggregation import compute_aggregated_sector_value_per_physical_input

    disagg = pd.DataFrame(
        {
            "disaggregated_sector": ["A", "B"],
            "extension": ["energy", "energy"],
            "stressor": ["Energy use - Gross", "Energy use - Gross"],
            "unit": ["TJ", "TJ"],
            "quantity": [10.0, 30.0],
            "monetary_output": [50.0, 60.0],
            "value_per_physical_input": [5.0, 2.0],
        }
    )
    mapping = {"A": "X", "B": "X"}

    res = compute_aggregated_sector_value_per_physical_input(disagg, mapping)
    assert res is not None
    assert len(res) == 1
    row = res.iloc[0]
    assert row["aggregated_sector"] == "X"
    assert abs(float(row["value_per_physical_input"]) - (110.0 / 40.0)) < 1e-12
    assert row["n_components"] == 2


def test_map_exio_emission_intensity_uses_ratio_of_sums():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import map_exio_emission_intensity

    split_profiles = {
        "D35": SplitProfiles(
            aggregate_sector="D35",
            subsectors=("D35::Coal", "D35::Solar"),
            output_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            intermediate_sales_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            intermediate_input_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            final_demand_share=pd.Series({"D35::Coal": 0.7, "D35::Solar": 0.3}),
            subsector_to_exio_sectors={"D35::Coal": ["Coal"], "D35::Solar": ["Solar"]},
        )
    }

    exio_co2 = pd.DataFrame(
        {
            "exio_sector": ["Coal", "Solar"],
            "extension": ["air_emissions", "air_emissions"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "emission": [200.0, 10.0],
            "monetary_output": [100.0, 50.0],
            "emission_intensity": [2.0, 0.2],
        }
    )

    res = map_exio_emission_intensity(exio_co2, split_profiles)
    assert res is not None
    vals = dict(zip(res["disaggregated_sector"], res["emission_intensity"]))
    assert abs(float(vals["D35::Coal"]) - (200.0 / 100.0)) < 1e-12
    assert abs(float(vals["D35::Solar"]) - (10.0 / 50.0)) < 1e-12


def test_compute_aggregated_sector_emission_intensity_uses_ratio_of_sums():
    from ecolecon_mrio.iot_aggregation import compute_aggregated_sector_emission_intensity

    disagg = pd.DataFrame(
        {
            "disaggregated_sector": ["A", "B"],
            "extension": ["air_emissions", "air_emissions"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "emission": [200.0, 100.0],
            "monetary_output": [100.0, 50.0],
            "emission_intensity": [2.0, 2.0],
        }
    )
    mapping = {"A": "X", "B": "X"}

    res = compute_aggregated_sector_emission_intensity(disagg, mapping)
    assert res is not None
    assert len(res) == 1
    row = res.iloc[0]
    assert row["aggregated_sector"] == "X"
    assert abs(float(row["emission_intensity"]) - (300.0 / 150.0)) < 1e-12
    assert abs(float(row["emission"]) - 300.0) < 1e-12
    assert abs(float(row["monetary_output"]) - 150.0) < 1e-12
    assert row["n_components"] == 2
