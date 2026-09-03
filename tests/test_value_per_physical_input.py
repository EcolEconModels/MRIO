from __future__ import annotations

import pandas as pd
import pytest


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


def test_project_environmental_extensions_applies_donor_intensity_to_eurostat_output():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import (
        project_environmental_extensions_to_split_sectors,
    )

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
    exio = pd.DataFrame(
        {
            "source_label": ["Coal", "Solar"],
            "extension": ["air_emissions", "air_emissions"],
            "kind": ["F", "F"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "value": [200.0, 10.0],
            "monetary_denominator": [100.0, 50.0],
            "intensity": [2.0, 0.2],
        }
    )
    eurostat_output = pd.Series(
        {"D35::Coal": 40.0, "D35::Solar": 100.0}, dtype="float64"
    )

    result = project_environmental_extensions_to_split_sectors(
        exio_extensions=exio,
        split_profiles=split_profiles,
        eurostat_output=eurostat_output,
    )
    assert result is not None
    by_sector = result.set_index("sector")
    assert float(by_sector.loc["D35::Coal", "value"]) == 80.0
    assert float(by_sector.loc["D35::Coal", "monetary_output"]) == 40.0
    assert float(by_sector.loc["D35::Coal", "intensity"]) == 2.0
    assert float(by_sector.loc["D35::Solar", "value"]) == 20.0
    assert float(by_sector.loc["D35::Solar", "monetary_output"]) == 100.0
    assert float(by_sector.loc["D35::Solar", "intensity"]) == 0.2

def test_aggregate_environmental_extensions_sums_totals_then_derives_intensity():
    from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions

    disaggregated = pd.DataFrame(
        {
            "sector": ["A", "B"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["air_emissions", "air_emissions"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "value": [200.0, 100.0],
            "monetary_output": [100.0, 50.0],
            "intensity": [2.0, 2.0],
        }
    )

    result = aggregate_environmental_extensions(
        disaggregated,
        {"A": "X", "B": "X"},
        output=pd.Series({"A": 100.0, "B": 50.0}, dtype="float64"),
    )
    assert result is not None
    assert len(result) == 1
    row = result.iloc[0]
    assert row["sector"] == "X"
    assert float(row["value"]) == 300.0
    assert float(row["monetary_output"]) == 150.0
    assert float(row["intensity"]) == 2.0


def test_aggregate_environmental_extensions_carries_unmapped_sector_identity():
    """Sparse reaggregation changes named sectors while retaining all other estimates."""
    from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions

    actual = pd.DataFrame(
        {
            "sector": ["A", "B", "C"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["synthetic"] * 3,
            "stressor": ["Synthetic indicator"] * 3,
            "unit": ["synthetic unit"] * 3,
            "value": [10.0, 20.0, 30.0],
            "monetary_output": [5.0, 10.0, 15.0],
            "intensity": [2.0, 2.0, 2.0],
        }
    )

    result = aggregate_environmental_extensions(
        actual,
        {"A": "A_NEW"},
        output=pd.Series({"A": 5.0, "B": 10.0, "C": 15.0}),
    )

    assert result is not None
    assert result["sector"].tolist() == ["A_NEW", "B", "C"]
    assert result.set_index("sector").loc["B", "value"] == 20.0
    assert result.set_index("sector").loc["C", "value"] == 30.0

def test_project_environmental_extensions_true_many_many_uses_projected_ratio_of_sums() -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import (
        project_environmental_extensions_to_split_sectors,
    )

    children = pd.Index(["E1@CPA_C1", "E2@CPA_C1"])
    shares = pd.Series([0.5, 0.5], index=children, dtype="float64")
    split_profiles = {
        "CPA_C1": SplitProfiles(
            aggregate_sector="CPA_C1",
            subsectors=tuple(children),
            output_share=shares.copy(),
            intermediate_sales_share=shares.copy(),
            intermediate_input_share=shares.copy(),
            final_demand_share=shares.copy(),
            subsector_to_exio_sectors={
                "E1@CPA_C1": ["E1"],
                "E2@CPA_C1": ["E2"],
            },
        )
    }
    exio = pd.DataFrame(
        {
            "source_label": ["E1", "E2"],
            "extension": ["air_emissions", "air_emissions"],
            "kind": ["F", "F"],
            "stressor": ["CO2 - combustion - air", "CO2 - combustion - air"],
            "unit": ["kg", "kg"],
            "value": [100.0, 240.0],
            "monetary_denominator": [200.0, 120.0],
            "intensity": [0.5, 2.0],
        }
    )
    mapping = {
        "CPA_C1": {"E1": 0.25, "E2": 0.50},
        "CPA_C2": {"E1": 0.75, "E2": 0.50},
    }
    eurostat_output = pd.Series(
        {"E1@CPA_C1": 40.0, "E2@CPA_C1": 30.0, "CPA_C2": 84.0},
        dtype="float64",
    )

    result = project_environmental_extensions_to_split_sectors(
        exio_extensions=exio,
        split_profiles=split_profiles,
        eurostat_output=eurostat_output,
        eurostat_sector_mapping=mapping,
    )
    assert result is not None
    by_sector = result.set_index("sector")
    assert float(by_sector.loc["E1@CPA_C1", "value"]) == pytest.approx(20.0)
    assert float(by_sector.loc["E2@CPA_C1", "value"]) == pytest.approx(60.0)
    # CPA_C2 receives both EXIO sources: (0.75*100 + 0.50*240) /
    # (0.75*200 + 0.50*120) = 195/210; applying that to 84 gives 78.
    assert float(by_sector.loc["CPA_C2", "intensity"]) == pytest.approx(195.0 / 210.0)
    assert float(by_sector.loc["CPA_C2", "value"]) == pytest.approx(78.0)


def test_project_environmental_extensions_fails_on_uncovered_positive_output_split_child() -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import (
        project_environmental_extensions_to_split_sectors,
    )

    share = pd.Series(
        {"E1@CPA_C1": 0.5, "E2@CPA_C1": 0.5}, dtype="float64"
    )
    profiles = {
        "CPA_C1": SplitProfiles(
            aggregate_sector="CPA_C1",
            subsectors=("E1@CPA_C1", "E2@CPA_C1"),
            output_share=share.copy(),
            intermediate_sales_share=share.copy(),
            intermediate_input_share=share.copy(),
            final_demand_share=share.copy(),
            subsector_to_exio_sectors={
                "E1@CPA_C1": ["E1"],
                "E2@CPA_C1": ["E2"],
            },
        )
    }
    exio = pd.DataFrame(
        {
            "source_label": ["E1"],
            "extension": ["air_emissions"],
            "kind": ["F"],
            "stressor": ["CO2 - combustion - air"],
            "unit": ["kg"],
            "value": [10.0],
            "monetary_denominator": [20.0],
            "intensity": [0.5],
        }
    )

    with pytest.raises(ValueError, match="Positive-output environmental extension coverage"):
        project_environmental_extensions_to_split_sectors(
            exio_extensions=exio,
            split_profiles=profiles,
            eurostat_output=pd.Series(
                {
                    "E1@CPA_C1": 10.0,
                    "E2@CPA_C1": 5.0,
                    "CPA_C2": 5.0,
                },
                dtype="float64",
            ),
            eurostat_sector_mapping={"CPA_C1": {"E1": 1.0}},
        )


def test_aggregate_environmental_extensions_fails_on_partial_positive_output_coverage() -> None:
    from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions

    partial = pd.DataFrame(
        {
            "sector": ["A"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["air_emissions"],
            "stressor": ["CO2 - combustion - air"],
            "unit": ["kg"],
            "value": [100.0],
            "monetary_output": [50.0],
            "intensity": [2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="target='X'.*sectors=\\['B'\\]",
    ):
        aggregate_environmental_extensions(
            partial,
            {"A": "X", "B": "X"},
            output=pd.Series({"A": 50.0, "B": 25.0}, dtype="float64"),
        )
