import pandas as pd

from ecolecon_mrio.counterfactual import run_panel_scenarios
from ecolecon_mrio.decomposition import decompose_slack_change
from ecolecon_mrio.temporal import compute_viability_panel, load_mrio_panel
from ecolecon_mrio.transition import classify_transition_years, principal_direction_index


CONSTRAINTS_CFG = {
    "ecological": [
        {"name": "co2_proxy", "mode": "output_rank_high", "ceiling_share_of_current": 0.92},
        {"name": "materials_proxy", "mode": "output_weighted", "ceiling_share_of_current": 0.95},
    ],
    "social": [
        {"name": "foundational_services", "mode": "output_rank_low", "floor_share_of_current": 0.88},
        {"name": "employment_proxy", "mode": "uniform", "floor_share_of_current": 0.82},
    ],
}

SCENARIOS_CFG = {
    "scenarios": [
        {"id": "baseline", "label": "Baseline", "operations": []},
        {
            "id": "efficiency",
            "label": "Efficiency",
            "operations": [{"type": "technical_coefficient_scaling", "factor": 0.95}],
        },
    ]
}


def test_load_mrio_panel_test_dataset():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021], calc_all=True, include_ghosh=True)
    assert set(panel.keys()) == {2020, 2021}
    for _, (_, y_full) in panel.items():
        assert isinstance(y_full, pd.Series)
        assert len(y_full) > 0


def test_compute_viability_panel_outputs():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20)
    assert not out.annual.empty
    assert {"year", "rho", "tech_slack", "delta_viability"}.issubset(out.annual.columns)
    assert not out.sector_scores.empty
    assert set(out.x_by_year.keys()) == {2020, 2021}


def test_decomposition_additive_identity():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20)
    decomp = decompose_slack_change(out.annual)
    assert len(decomp) == 1
    row = decomp.iloc[0]
    comp_sum = row["technology_component"] + row["demand_component"] + row["intensity_component"] + row["trade_component"] + row["residual"]
    assert abs(comp_sum - row["total_change"]) < 1e-8


def test_transition_index_and_classification_bounds():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021, 2022], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20)
    align = principal_direction_index(out.x_by_year, out.sector_scores)
    assert not align.empty
    assert ((align["align"].dropna() <= 1.0) & (align["align"].dropna() >= -1.0)).all()
    classified = classify_transition_years(out.annual, align, align_high_threshold=0.2, align_low_threshold=-0.2)
    assert "transition_class" in classified.columns


def test_counterfactual_identity_for_baseline():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20)
    align = principal_direction_index(out.x_by_year, out.sector_scores)
    annual = out.annual.merge(align, on="year", how="left")
    scenarios = run_panel_scenarios(annual, SCENARIOS_CFG)
    baseline = scenarios[scenarios["scenario_id"] == "baseline"].sort_values("year")
    merged = baseline.merge(annual[["year", "tech_slack"]], on="year", how="left")
    assert ((merged["tech_slack_cf"] - merged["tech_slack"]).abs() < 1e-12).all()
