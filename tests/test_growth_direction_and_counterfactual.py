import numpy as np
import pandas as pd

from ecolecon_mrio.counterfactual import run_panel_scenarios
from ecolecon_mrio.diagnostics import calc_tech_slack
from ecolecon_mrio.growth_direction import principal_growth_direction, principal_growth_panel


def _toy_state() -> dict:
    idx = pd.Index(["s1", "s2"])
    A = pd.DataFrame([[0.2, 0.1], [0.05, 0.15]], index=idx, columns=idx, dtype=float)
    y = pd.Series([10.0, 6.0], index=idx, dtype=float)
    E = pd.DataFrame([[1.0, 0.5]], index=["co2"], columns=idx, dtype=float)
    S = pd.DataFrame([[0.6, 0.8]], index=["jobs"], columns=idx, dtype=float)
    e_bar = pd.Series([20.0], index=["co2"], dtype=float)
    s_floor = pd.Series([8.0], index=["jobs"], dtype=float)
    return {
        "A": A,
        "B": pd.DataFrame(np.eye(2), index=idx, columns=idx),
        "y": y,
        "E": E,
        "S": S,
        "e_bar": e_bar,
        "s_floor": s_floor,
        "groups": {"brown": ["s1"], "green": ["s2"]},
    }


def _solve_stub(state):
    _, slack = calc_tech_slack(state["A"])
    return {"tech_slack_cf": float(slack), "y_sum": float(state["y"].sum())}


def test_principal_direction_nonnegative_normalized():
    state = _toy_state()
    res = principal_growth_direction(state["A"], B=state["B"])
    assert abs(res.q.sum() - 1.0) < 1e-10
    assert abs(res.p.sum() - 1.0) < 1e-10
    assert (res.q >= -1e-12).all()
    assert (res.p >= -1e-12).all()


def test_vn_identity_case_b_eq_i_matches_pf():
    state = _toy_state()
    vn = principal_growth_direction(state["A"], B=state["B"])
    pf = principal_growth_direction(state["A"], B=None)
    assert abs(vn.alpha - pf.alpha) < 1e-8


def test_rotation_metric_bounds():
    s0 = _toy_state()
    s1 = _toy_state()
    s1["A"] = s1["A"] * 1.01
    x_by_year = {
        2020: pd.Series([100.0, 50.0], index=s0["A"].index),
        2021: pd.Series([102.0, 50.5], index=s0["A"].index),
    }
    A_by_year = {2020: s0["A"], 2021: s1["A"]}
    B_by_year = {2020: s0["B"], 2021: s1["B"]}
    summary, _ = principal_growth_panel(x_by_year, A_by_year, B_by_year)
    rot = summary["rotation_to_prev"].dropna()
    if not rot.empty:
        assert ((rot >= 0.0) & (rot <= np.pi)).all()


def test_noop_scenario_identity():
    states = {2020: _toy_state(), 2021: _toy_state()}
    scenarios_cfg = {"scenarios": [{"id": "baseline", "operations": []}]}
    out = run_panel_scenarios(states, scenarios_cfg, _solve_stub)
    assert len(out) == 2
    assert (out["tech_slack_cf"] > 0).all()


def test_efficiency_operator_changes_a_and_recomputes():
    states = {2020: _toy_state()}
    scenarios_cfg = {
        "scenarios": [
            {"id": "baseline", "operations": []},
            {"id": "eff", "operations": [{"type": "technical_coefficient_scaling", "factor": 0.9}]},
        ]
    }
    out = run_panel_scenarios(states, scenarios_cfg, _solve_stub)
    base = float(out[out["scenario_id"] == "baseline"]["tech_slack_cf"].iloc[0])
    eff = float(out[out["scenario_id"] == "eff"]["tech_slack_cf"].iloc[0])
    assert eff >= base


def test_demand_reallocation_mass_conserved():
    states = {2020: _toy_state()}
    scenarios_cfg = {
        "scenarios": [
            {"id": "baseline", "operations": []},
            {
                "id": "shift",
                "operations": [
                    {
                        "type": "demand_reallocation",
                        "from_group": "brown",
                        "to_group": "green",
                        "share": 0.25,
                    }
                ],
            },
        ]
    }
    out = run_panel_scenarios(states, scenarios_cfg, _solve_stub)
    base = float(out[out["scenario_id"] == "baseline"]["y_sum"].iloc[0])
    shift = float(out[out["scenario_id"] == "shift"]["y_sum"].iloc[0])
    assert abs(base - shift) < 1e-9



def test_pf_metadata_exposed_in_identity_fallback():
    state = _toy_state()
    res = principal_growth_direction(state["A"], B=None)
    assert isinstance(res.pf_metadata, dict)
    for key in ["reducible", "n_strong_components", "rho"]:
        assert key in res.pf_metadata
