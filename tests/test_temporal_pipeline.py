import numpy as np
import pandas as pd

from ecolecon_mrio.counterfactual import run_panel_scenarios
from ecolecon_mrio.diagnostics import calc_tech_slack
from ecolecon_mrio.decomposition import decompose_slack_change
from ecolecon_mrio.growth_direction import principal_growth_direction, principal_growth_panel
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
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, dataset="test", max_sectors=20, allow_relaxation=True)
    assert not out.annual.empty
    assert {"year", "rho", "tech_slack", "delta_viability"}.issubset(out.annual.columns)
    assert not out.sector_scores.empty
    assert set(out.x_by_year.keys()) == {2020, 2021}
    assert set(out.A_by_year.keys()) == {2020, 2021}
    assert set(out.state_by_year.keys()) == {2020, 2021}
    for year in [2020, 2021]:
        state = out.state_by_year[year]
        for key in ["eco_source_mode", "eco_selector_note", "eco_budget_factor", "eco_budget_ref_year", "eco_row_label"]:
            assert key in state


def test_decomposition_additive_identity():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021, 2022], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20, allow_relaxation=True)

    def viability_fn(state):
        _, slack = calc_tech_slack(state["A"])
        return float(slack)

    decomp = decompose_slack_change(out.state_by_year, viability_fn, blocks=("A", "y", "E", "S"))
    assert not decomp.empty
    for _, row in decomp.iterrows():
        comp_sum = row["A_component"] + row["y_component"] + row["E_component"] + row["S_component"] + row["residual"]
        assert abs(comp_sum - row["total_change"]) < 1e-8


def test_transition_index_and_classification_bounds():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021, 2022], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20, allow_relaxation=True)
    align = principal_direction_index(out.x_by_year, out.sector_scores)
    assert not align.empty
    assert ((align["align"].dropna() <= 1.0) & (align["align"].dropna() >= -1.0)).all()
    classified = classify_transition_years(out.annual, align, align_high_threshold=0.2, align_low_threshold=-0.2)
    assert "transition_class" in classified.columns


def test_counterfactual_identity_for_baseline():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20, allow_relaxation=True)

    def solve_fn(state):
        _, slack = calc_tech_slack(state["A"])
        return {"tech_slack_cf": float(slack)}

    scenarios = run_panel_scenarios(out.state_by_year, SCENARIOS_CFG, solve_fn)
    baseline = scenarios[scenarios["scenario_id"] == "baseline"].sort_values("year")
    annual = out.annual[["year", "tech_slack"]].copy()
    merged = baseline.merge(annual, on="year", how="left")
    assert ((merged["tech_slack_cf"] - merged["tech_slack"]).abs() < 1e-12).all()


def test_vn_identity_case_b_eq_i_matches_pf():
    panel = load_mrio_panel(dataset="test", years=[2020], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20, allow_relaxation=True)
    A = out.A_by_year[2020]
    B = pd.DataFrame(np.eye(len(A)), index=A.index, columns=A.columns)
    vn = principal_growth_direction(A, B=B)
    pf = principal_growth_direction(A, B=None)
    assert abs(vn.alpha - pf.alpha) < 1e-8
    assert abs(vn.q.sum() - 1.0) < 1e-10
    assert (vn.q >= -1e-12).all()


def test_principal_growth_panel_metrics():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021, 2022], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(panel, CONSTRAINTS_CFG, max_sectors=20, allow_relaxation=True)
    summary, pairs, details = principal_growth_panel(out.x_by_year, out.A_by_year, out.B_by_year)
    assert len(summary) == 3
    assert {"year", "alpha", "method", "fallback_used"}.issubset(summary.columns)
    assert {"year_from", "year_to", "theta_forward", "alignment_with_dx_forward"}.issubset(pairs.columns)
    assert len(pairs) == 2
    assert all(y in details for y in [2020, 2021, 2022])
    aligns = pairs["alignment_with_dx_forward"].dropna()
    assert ((aligns <= 1.0) & (aligns >= -1.0)).all()


def test_panel_union_zero_pad_top_n_has_fixed_universe_manifest():
    panel = load_mrio_panel(dataset="test", years=[2020, 2021, 2022], calc_all=True, include_ghosh=True)
    out = compute_viability_panel(
        panel,
        CONSTRAINTS_CFG,
        max_sectors=20,
        sector_universe_mode="panel_union_zero_pad_top_n",
        sector_universe_n=20,
        allow_relaxation=True,
    )
    assert not out.sector_universe_manifest.empty
    assert {"sector", "score", "selected", "rank", "sector_universe_mode", "sector_universe_n"}.issubset(
        set(out.sector_universe_manifest.columns)
    )
    idx_sets = [tuple(pd.Index(x.index).astype(str)) for x in out.x_by_year.values()]
    assert len(set(idx_sets)) == 1
