import numpy as np
import pandas as pd

from ecolecon_mrio.constraints import build_constraints, build_constraints_with_context, solve_beta
from ecolecon_mrio.counterfactual import run_panel_scenarios, run_shock_suite
from ecolecon_mrio.diagnostics import calc_coefficient_diagnostics, calc_tech_slack


def _toy_state() -> dict:
    idx = pd.Index(["US_s1", "CN_s2"])
    A = pd.DataFrame([[0.20, 0.10], [0.05, 0.15]], index=idx, columns=idx, dtype=float)
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
        "groups": {"us": ["US_s1"], "cn": ["CN_s2"]},
    }


def test_build_constraints_shapes_and_names():
    x = pd.Series([100.0, 50.0], index=["a", "b"])
    cfg = {
        "ecological": [{"name": "eco_u", "mode": "uniform", "ceiling_share_of_current": 0.9}],
        "social": [{"name": "soc_u", "mode": "uniform", "floor_share_of_current": 0.8}],
    }
    E, S, e_bar, s_floor = build_constraints(x, cfg)
    assert list(E.index) == ["eco_u"]
    assert list(S.index) == ["soc_u"]
    assert list(E.columns) == ["a", "b"]
    assert list(S.columns) == ["a", "b"]
    assert float(e_bar.iloc[0]) > 0.0
    assert float(s_floor.iloc[0]) > 0.0


def _toy_mrio_with_satellite(columns: list[str]) -> object:
    sat = type("Satellite", (), {})()
    sat.F = pd.DataFrame(
        [[2.0, 1.0], [0.2, 0.4]],
        index=["CO2 combustion", "other_extension"],
        columns=columns,
        dtype=float,
    )
    mrio = type("MRIOObj", (), {})()
    mrio.satellite = sat
    return mrio


def test_build_constraints_extension_row_budget_factor():
    x = pd.Series([100.0, 50.0], index=["a", "b"], dtype=float)
    mrio = _toy_mrio_with_satellite(["a", "b"])
    cfg = {
        "ecological": [
            {
                "name": "eco_co2",
                "source": "extension_row",
                "selector": {"mode": "row_name_contains_any", "row_name_contains_any": ["co2"]},
                "ceiling_mode": "budget_share_of_current",
                "budget_path_ref": "paris_co2",
            }
        ],
        "social": [{"name": "soc_u", "source": "proxy", "mode": "uniform", "floor_share_of_current": 0.8}],
        "_budget_paths_data": {
            "paths": {"paris_co2": {"test": {"2020": 0.9}}},
            "reference_year": {"test": 2020},
        },
    }
    E, S, e_bar, s_floor, meta = build_constraints_with_context(
        x,
        cfg,
        mrio_obj=mrio,
        year=2020,
        dataset="test",
        return_metadata=True,
    )
    assert list(E.index) == ["eco_co2"]
    assert float(E.loc["eco_co2", "a"]) == 2.0
    assert float(E.loc["eco_co2", "b"]) == 1.0
    assert abs(float(e_bar.loc["eco_co2"]) - (0.9 * (2.0 * 100.0 + 1.0 * 50.0))) < 1e-9
    assert list(S.index) == ["soc_u"]
    assert float(s_floor.iloc[0]) > 0.0
    assert isinstance(meta, dict)
    assert meta.get("eco_source_mode") == "extension_row"
    assert "satellite.F:match:CO2 combustion" in str(meta.get("eco_selector_note"))


def test_build_constraints_extension_row_missing_budget_factor_fails():
    x = pd.Series([100.0, 50.0], index=["a", "b"], dtype=float)
    mrio = _toy_mrio_with_satellite(["a", "b"])
    cfg = {
        "ecological": [
            {
                "name": "eco_co2",
                "source": "extension_row",
                "selector": {"mode": "row_name_contains_any", "row_name_contains_any": ["co2"]},
                "ceiling_mode": "budget_share_of_current",
                "budget_path_ref": "paris_co2",
            }
        ],
        "social": [{"name": "soc_u", "source": "proxy", "mode": "uniform", "floor_share_of_current": 0.8}],
        "_budget_paths_data": {"paths": {"paris_co2": {"test": {"2021": 0.9}}}},
    }
    try:
        build_constraints(
            x,
            cfg,
            mrio_obj=mrio,
            year=2020,
            dataset="test",
        )
    except ValueError as exc:
        assert "Missing budget factor" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing budget factor.")


def test_solve_beta_feasible_and_binding_eco():
    q = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    E = pd.DataFrame([[1.0, 2.0]], index=["eco_cap"], columns=q.index, dtype=float)
    S = pd.DataFrame([[1.0, 0.0]], index=["soc_floor"], columns=q.index, dtype=float)
    e_bar = pd.Series([1.0], index=E.index, dtype=float)
    s_floor = pd.Series([0.2], index=S.index, dtype=float)

    res = solve_beta(q, E, S, e_bar, s_floor)
    assert res.feasible
    assert abs(res.beta - (1.0 / 1.4)) < 1e-10
    assert "eco_cap" in res.binding_eco


def test_solve_beta_infeasible_zero_denominator_contradiction():
    q = pd.Series([1.0, 0.0], index=["a", "b"], dtype=float)
    E = pd.DataFrame([[0.0, 0.0]], index=["eco_bad"], columns=q.index, dtype=float)
    S = pd.DataFrame([[1.0, 0.0]], index=["soc_ok"], columns=q.index, dtype=float)
    e_bar = pd.Series([-1.0], index=E.index, dtype=float)
    s_floor = pd.Series([0.0], index=S.index, dtype=float)

    res = solve_beta(q, E, S, e_bar, s_floor)
    assert not res.feasible
    assert "zero denominator" in res.message


def test_solve_beta_binding_social_case():
    q = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    E = pd.DataFrame([[1.0, 1.0]], index=["eco_cap"], columns=q.index, dtype=float)
    S = pd.DataFrame([[2.0, 0.0]], index=["soc_tight"], columns=q.index, dtype=float)
    e_bar = pd.Series([1.2], index=E.index, dtype=float)
    s_floor = pd.Series([1.2], index=S.index, dtype=float)

    res = solve_beta(q, E, S, e_bar, s_floor)
    assert res.feasible
    assert abs(res.beta - 1.2) < 1e-10
    assert "soc_tight" in res.binding_soc


def test_matrix_block_scaling_with_indices_and_groups():
    states = {2020: _toy_state()}

    def solve_fn(state):
        A = pd.DataFrame(state["A"])
        return {
            "a01": float(A.iloc[0, 1]),
            "a10": float(A.iloc[1, 0]),
        }

    scenarios_cfg = {
        "scenarios": [
            {"id": "baseline", "operations": []},
            {
                "id": "by_indices",
                "operations": [
                    {
                        "type": "matrix_block_scaling",
                        "row_indices": ["US_s1"],
                        "col_indices": ["CN_s2"],
                        "factor": 2.0,
                    }
                ],
            },
            {
                "id": "by_groups",
                "operations": [
                    {
                        "type": "matrix_block_scaling",
                        "row_group": "cn",
                        "col_group": "us",
                        "factor": 3.0,
                    }
                ],
            },
        ]
    }

    out = run_panel_scenarios(states, scenarios_cfg, solve_fn)
    base = out[out["scenario_id"] == "baseline"].iloc[0]
    idx_case = out[out["scenario_id"] == "by_indices"].iloc[0]
    grp_case = out[out["scenario_id"] == "by_groups"].iloc[0]

    assert abs(float(idx_case["a01"]) - float(base["a01"]) * 2.0) < 1e-12
    assert abs(float(grp_case["a10"]) - float(base["a10"]) * 3.0) < 1e-12


def test_run_shock_suite_schema_and_row_count():
    year_states = {2020: _toy_state(), 2021: _toy_state()}
    shocks_cfg = {
        "shock_families": [
            {
                "id": "capacity_cut",
                "parameters": {"targets": ["US_s1"], "factor_grid": [0.95, 0.90]},
            },
            {
                "id": "trade_partner_cut",
                "parameters": {"partner_ids": ["CN"], "factor_grid": [0.90]},
            },
        ]
    }

    def solve_fn(state):
        _, slack = calc_tech_slack(pd.DataFrame(state["A"]))
        return {
            "alpha": 1.0 + slack,
            "beta": 0.9 + slack,
            "mu_tech": slack,
        }

    out = run_shock_suite(
        year_states,
        shocks_cfg,
        solve_fn,
        families=["capacity_cut", "trade_partner_cut"],
    )

    expected_rows = 2 * (2 + 1)
    assert len(out) == expected_rows

    required = {
        "year",
        "shock_family",
        "shock_case_id",
        "alpha_shock",
        "beta_shock",
        "mu_tech_shock",
        "scenario_factor",
        "metadata",
    }
    assert required.issubset(set(out.columns))
    assert {"capacity_cut", "trade_partner_cut"} == set(out["shock_family"].unique())



def test_calc_coefficient_diagnostics_basic_bounds():
    A = pd.DataFrame([[0.2, 0.0], [0.4, 0.9]], dtype=float)
    out = calc_coefficient_diagnostics(A)
    assert abs(out["a_min"] - 0.0) < 1e-12
    assert abs(out["a_max"] - 0.9) < 1e-12
    assert out["share_neg"] == 0.0
    assert out["share_gt_upper"] == 0.0
    assert out["share_zero_or_near_zero"] >= 0.25
    assert abs(out["upper_margin_min"] - 0.1) < 1e-12


def test_calc_coefficient_diagnostics_detects_neg_and_gt1():
    A = pd.DataFrame([[1.2, -0.1], [0.2, 0.3]], dtype=float)
    out = calc_coefficient_diagnostics(A)
    assert out["share_neg"] > 0.0
    assert out["share_gt_upper"] > 0.0
    assert out["a_max"] > 1.0
    assert out["upper_margin_min"] < 0.0


def test_calc_coefficient_diagnostics_sparse_matrix():
    A = pd.DataFrame(np.zeros((4, 4), dtype=float))
    A.iloc[0, 1] = 0.25
    out = calc_coefficient_diagnostics(A)
    assert out["share_neg"] == 0.0
    assert out["share_gt_upper"] == 0.0
    assert out["share_zero_or_near_zero"] > 0.9
    assert abs(out["a_max"] - 0.25) < 1e-12
