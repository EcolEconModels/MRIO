import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.diagnostics import (
    aggregate_diagnostics,
    calc_boundary_slack,
    calc_pf_sensitivities,
    calc_spectral_radius,
    calc_tech_slack,
    solve_safe_just_lp,
)


def make_test_matrix() -> pd.DataFrame:
    data = np.array([[0.2, 0.1], [0.05, 0.25]])
    idx = pd.Index(["r1", "r2"])
    return pd.DataFrame(data, index=idx, columns=idx)


def test_calc_tech_slack_identity():
    axis = pd.Index(["a"])
    matrix = pd.DataFrame([[0.3]], index=axis, columns=axis)
    rho, slack = calc_tech_slack(matrix)
    assert pytest.approx(rho, rel=1e-6) == 0.3
    assert pytest.approx(slack, rel=1e-6) == 0.7


def test_pf_sensitivity_matches_fd():
    matrix = make_test_matrix()
    base_rho = calc_spectral_radius(matrix)
    sens = calc_pf_sensitivities(matrix)
    delta = 1e-6
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            perturbed = matrix.copy()
            perturbed.iloc[i, j] += delta
            perturbed_rho = calc_spectral_radius(perturbed)
            derivative = (perturbed_rho - base_rho) / delta
            assert pytest.approx(sens.iloc[i, j], rel=1e-3) == derivative


def test_safe_and_just_lp_returns_identity_solution():
    x = pd.Series([10.0, 5.0], index=pd.Index(["a", "b"]))
    y = pd.Series([2.0, 1.0], index=x.index)
    A = pd.DataFrame(np.zeros((2, 2)), index=x.index, columns=x.index)
    E = pd.DataFrame(np.full((1, 2), 0.01), index=pd.Index(["eco"]), columns=x.index)
    e_bar = pd.Series([1.0], index=E.index)
    S = pd.DataFrame(np.full((1, 2), 0.02), index=pd.Index(["soc"]), columns=x.index)
    s_floor = pd.Series([0.0], index=S.index)
    result = solve_safe_just_lp(x, y, A, E, S, e_bar, s_floor)
    assert result.x_opt.equals(x)
    assert (result.delta_plus == 0).all()
    assert (result.delta_minus == 0).all()
    assert result.inequality_duals is not None
    assert result.equality_duals is not None
    assert len(result.inequality_duals) == len(x) + len(E.index) + len(S.index)
    assert len(result.equality_duals) == len(x)


def test_calc_boundary_slack_simple():
    x = pd.Series([4.0, 6.0], index=pd.Index(["a", "b"]))
    E = pd.DataFrame([[0.1, 0.2]], index=pd.Index(["eco"]), columns=x.index)
    e_bar = pd.Series([2.0], index=E.index)
    S = pd.DataFrame([[0.05, 0.05]], index=pd.Index(["soc"]), columns=x.index)
    s_floor = pd.Series([1.0], index=S.index)
    eco_slack, soc_slack = calc_boundary_slack(E, S, x, e_bar, s_floor)
    assert pytest.approx(eco_slack.iloc[0], rel=1e-6) == 2.0 - (0.1 * 4 + 0.2 * 6)
    assert pytest.approx(soc_slack.iloc[0], rel=1e-6) == (0.05 * 4 + 0.05 * 6) - 1.0


def test_aggregate_diagnostics_handles_negative_slack_without_blowup():
    basis = pd.Index(["a", "b"])
    pf = pd.DataFrame([[0.2, 0.3], [0.1, 0.4]], index=pd.Index(["r1", "r2"]), columns=basis)
    E = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["eco"]), columns=basis)
    S = pd.DataFrame([[0.5, 0.5]], index=pd.Index(["soc"]), columns=basis)
    eco_slack = pd.Series([-1e-3], index=["eco"])
    soc_slack = pd.Series([1e-12], index=["soc"])
    out = aggregate_diagnostics(pf, E, S, eco_slack, soc_slack)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
