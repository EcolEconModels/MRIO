import numpy as np
import pandas as pd
import pytest
from typing import cast
from scipy.sparse import csr_matrix

from ecolecon_mrio.counterfactual import run_panel_scenarios
from ecolecon_mrio.diagnostics import calc_pf_eigenvectors, calc_tech_slack
from ecolecon_mrio.growth_direction import principal_growth_direction, principal_growth_panel


def _toy_state() -> dict:
    idx = pd.Index(["s1", "s2"])
    A = pd.DataFrame([[0.2, 0.1], [0.05, 0.15]], index=idx, columns=idx, dtype=float)
    y = pd.Series([10.0, 6.0], index=idx, dtype=float)
    E = pd.DataFrame([[1.0, 0.5]], index=pd.Index(["co2"]), columns=idx, dtype=float)
    S = pd.DataFrame([[0.6, 0.8]], index=pd.Index(["jobs"]), columns=idx, dtype=float)
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


def test_pf_eigenvectors_preserve_multiindex_labels():
    idx = pd.MultiIndex.from_tuples([("r1", "s1"), ("r1", "s2")], names=["region", "sector"])
    A = pd.DataFrame([[0.2, 0.1], [0.05, 0.15]], index=idx, columns=idx, dtype=float)

    p, q, _rho = cast(tuple[pd.Series, pd.Series, float], calc_pf_eigenvectors(A))

    assert q.index.equals(idx)
    assert p.index.equals(idx)


def test_principal_growth_direction_preserves_dense_multiindex_labels():
    idx = pd.MultiIndex.from_tuples([("r1", "s1"), ("r1", "s2")], names=["region", "sector"])
    A = pd.DataFrame([[0.2, 0.1], [0.05, 0.15]], index=idx, columns=idx, dtype=float)

    result = principal_growth_direction(A)

    assert result.q.index.equals(idx)
    assert result.p.index.equals(idx)


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
    summary, pairs, _ = principal_growth_panel(x_by_year, A_by_year, B_by_year)
    assert len(summary) == 2
    rot = pairs["theta_forward"].dropna()
    if not rot.empty:
        assert ((rot >= 0.0) & (rot <= np.pi)).all()


def test_principal_growth_panel_preserves_multiindex_basis_for_sparse_a():
    idx = pd.MultiIndex.from_tuples([("r1", "s1"), ("r1", "s2")], names=["region", "sector"])
    x_by_year = {
        2020: pd.Series([100.0, 50.0], index=idx, dtype=float),
        2021: pd.Series([102.0, 50.5], index=idx, dtype=float),
    }
    dense_a = np.array([[0.2, 0.1], [0.05, 0.15]], dtype=float)
    A_by_year = {2020: csr_matrix(dense_a), 2021: csr_matrix(dense_a * 1.01)}

    summary, pairs, results = principal_growth_panel(x_by_year, A_by_year, None)

    assert len(summary) == 2
    assert len(pairs) == 1
    assert isinstance(results[2020].q.index, pd.MultiIndex)
    assert results[2020].q.index.equals(idx)
    assert results[2020].p.index.equals(idx)


def test_noop_scenario_identity():
    states = {2020: _toy_state(), 2021: _toy_state()}
    scenarios_cfg = {"scenarios": [{"id": "baseline", "operations": []}]}
    out = cast(pd.DataFrame, run_panel_scenarios(states, scenarios_cfg, _solve_stub))
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
    out = cast(pd.DataFrame, run_panel_scenarios(states, scenarios_cfg, _solve_stub))
    base_rows = cast(pd.DataFrame, out[out["scenario_id"] == "baseline"])
    eff_rows = cast(pd.DataFrame, out[out["scenario_id"] == "eff"])
    base = float(cast(pd.Series, base_rows["tech_slack_cf"]).iloc[0])
    eff = float(cast(pd.Series, eff_rows["tech_slack_cf"]).iloc[0])
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
    base_rows = cast(pd.DataFrame, out[out["scenario_id"] == "baseline"])
    shift_rows = cast(pd.DataFrame, out[out["scenario_id"] == "shift"])
    base = float(cast(pd.Series, base_rows["y_sum"]).iloc[0])
    shift = float(cast(pd.Series, shift_rows["y_sum"]).iloc[0])
    assert abs(base - shift) < 1e-9



def test_pf_metadata_exposed_in_identity_fallback():
    state = _toy_state()
    res = principal_growth_direction(state["A"], B=None)
    assert isinstance(res.pf_metadata, dict)
    for key in ["reducible", "n_strong_components", "rho"]:
        assert key in res.pf_metadata


# --- Stage-2 PF falsification regressions ---
def _large_irreducible_pf_matrix(n: int = 256) -> csr_matrix:
    """Return a sparse nonnegative strongly-connected toy matrix for ARPACK tests."""
    rows = np.arange(n, dtype=int)
    cols = (rows + 1) % n
    cycle = csr_matrix((np.full(n, 0.01), (rows, cols)), shape=(n, n))
    return csr_matrix(cycle + csr_matrix((np.full(n, 0.2), (rows, rows)), shape=(n, n)))


def test_sparse_and_dense_pf_are_numerically_equivalent_on_same_multiindex() -> None:
    """Sparse label preservation must preserve the PF vector itself, not only its index."""
    idx = pd.MultiIndex.from_tuples(
        [("R", "a"), ("R", "b"), ("R", "c")], names=["region", "sector"]
    )
    values = np.array([[0.20, 0.03, 0.02], [0.04, 0.18, 0.03], [0.02, 0.05, 0.17]])
    dense = principal_growth_direction(pd.DataFrame(values, index=idx, columns=idx), index=idx)
    sparse = principal_growth_direction(csr_matrix(values), index=idx)
    assert sparse.q.index.equals(idx) and sparse.p.index.equals(idx)
    assert np.isfinite(sparse.q.to_numpy()).all() and np.isfinite(sparse.p.to_numpy()).all()
    assert float(sparse.q.sum()) == pytest.approx(1.0)
    assert float(sparse.p.sum()) == pytest.approx(1.0)
    assert sparse.alpha == pytest.approx(dense.alpha, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(sparse.q.to_numpy(), dense.q.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sparse.p.to_numpy(), dense.p.to_numpy(), rtol=1e-10, atol=1e-12)


def test_dense_pf_rejects_mismatched_axes() -> None:
    """A dense PF solve cannot silently reinterpret different buyer/seller labels."""
    rows = pd.Index(["a", "b"])
    cols = pd.Index(["b", "a"])
    with pytest.raises(ValueError, match="identical row and column axes"):
        principal_growth_direction(pd.DataFrame([[0.2, 0.1], [0.1, 0.2]], index=rows, columns=cols))


def test_pf_panel_rejects_missing_year_and_nonfinite_output() -> None:
    """Panel diagnostics require a complete finite gross-output vector for every A year."""
    idx = pd.Index(["a", "b"])
    A = pd.DataFrame([[0.2, 0.1], [0.1, 0.2]], index=idx, columns=idx)
    with pytest.raises(ValueError, match="exactly the same years"):
        principal_growth_panel({2020: pd.Series([1.0, 1.0], index=idx)}, {2020: A, 2021: A})
    with pytest.raises(ValueError, match="finite"):
        principal_growth_panel(
            {2020: pd.Series([1.0, np.nan], index=idx)},
            {2020: A},
        )


def test_left_arpack_partial_pair_cannot_overwrite_right_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial left solve must remain the left eigenpair and preserve the right solve."""
    import ecolecon_mrio.diagnostics as diagnostics
    from scipy.sparse.linalg import ArpackNoConvergence

    n = 256
    A = _large_irreducible_pf_matrix(n)
    idx = pd.Index([f"s{i}" for i in range(n)])
    calls = 0

    def fake_eigs(_operator, **_kwargs):
        nonlocal calls
        calls += 1
        vals = np.array([0.21 + 0.0j])
        vec = np.ones((n, 1), dtype=complex)
        if calls == 2:
            raise ArpackNoConvergence("left partial", vals, vec)
        return vals, vec

    monkeypatch.setattr(diagnostics, "eigs", fake_eigs)
    p, q, rho = cast(tuple[pd.Series, pd.Series, float], diagnostics.calc_pf_eigenvectors(A, index=idx))
    assert calls == 2
    assert rho == pytest.approx(0.21)
    assert p.index.equals(idx) and q.index.equals(idx)
    assert np.isfinite(p.to_numpy()).all() and np.isfinite(q.to_numpy()).all()
    assert float(p.sum()) > 0.0 and float(q.sum()) > 0.0


def test_large_sparse_pf_failure_never_densifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unusable ARPACK result on an EXIO-sized sparse branch must fail, not call dense eig."""
    import ecolecon_mrio.diagnostics as diagnostics
    from scipy.sparse.linalg import ArpackNoConvergence

    n = 256
    A = _large_irreducible_pf_matrix(n)
    monkeypatch.setattr(
        diagnostics,
        "eigs",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ArpackNoConvergence("no partial", np.array([]), np.empty((n, 0)))
        ),
    )
    monkeypatch.setattr(
        diagnostics.linalg,
        "eig",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dense eig fallback was used")),
    )
    with pytest.raises(ValueError, match="valid sparse eigenpair"):
        diagnostics.calc_pf_eigenvectors(A, index=pd.RangeIndex(n))


def test_large_sparse_spectral_radius_failure_never_densifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dominant-block screening has the same no-densification contract as the PF solve."""
    import ecolecon_mrio.diagnostics as diagnostics
    from scipy.sparse.linalg import ArpackNoConvergence

    n = 256
    A = _large_irreducible_pf_matrix(n)
    monkeypatch.setattr(
        diagnostics,
        "eigs",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ArpackNoConvergence("no partial", np.array([]), np.empty((n, 0)))
        ),
    )
    monkeypatch.setattr(
        diagnostics.linalg,
        "eigvals",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dense eigvals fallback was used")),
    )
    with pytest.raises(ValueError, match="no usable eigenvalue"):
        diagnostics._spectral_radius_from_sparse(A)


def test_pf_rejects_nonfinite_sparse_coefficients() -> None:
    """NaN/Inf technical coefficients must fail at the labelled PF boundary."""
    A = csr_matrix(np.array([[0.2, np.nan], [0.1, 0.2]], dtype=float))
    with pytest.raises(ValueError, match="finite"):
        principal_growth_direction(A, index=pd.Index(["a", "b"]))


def test_lp_working_set_rejects_malformed_output_and_role_mask() -> None:
    """Sector-universe selection cannot manufacture zero output or missing role membership."""
    import ecolecon_mrio.diagnostics as diagnostics

    idx = pd.Index(["a", "b", "c"])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        diagnostics._select_lp_working_set_index(
            pd.Series([1.0, np.nan, 2.0], index=idx),
            working_set_n=2,
            mode="output_weighted",
        )
    wrong = pd.Series([True, False, False], index=pd.Index(["c", "b", "a"]))
    with pytest.raises(ValueError, match="exactly match"):
        diagnostics._select_lp_working_set_index(
            pd.Series([1.0, 2.0, 3.0], index=idx),
            working_set_n=2,
            mode="stratified_transition_roles",
            role_masks={"expand": wrong},
        )
