from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio import InverseMrioMechanismProblem


def _problem(*, capacity: float = 0.1, signed_f0: bool = False) -> InverseMrioMechanismProblem:
    A0 = csr_matrix(np.diag([0.2, 0.2]))
    x0 = np.array([10.0, 10.0])
    f0 = x0 - A0.dot(x0)
    if signed_f0:
        # The accounting API permits signed final-use items; the L1 denominator
        # remains gross. This fixture only tests the boundary, not source balance.
        f0 = np.array([8.0, -8.0])
    lower = csr_matrix(np.diag([-capacity, -capacity]))
    upper = csr_matrix(np.diag([capacity, capacity]))
    target = csr_matrix([[1.0, 0.0]])
    return InverseMrioMechanismProblem(
        A0,
        x0,
        f0,
        lower,
        upper,
        target,
        accompaniment_indices=np.array([1], dtype=np.int32),
        feasibility_tolerance=1.0e-9,
        objective_relative_tolerance=1.0e-10,
        objective_absolute_tolerance=1.0e-9,
    )


def test_inverse_mrio_known_pure_endpoints_and_interior_epsilon() -> None:
    problem = _problem()
    rhs = np.array([1.0])

    pure_final = problem.solve_pure_final_use(rhs)
    assert pure_final.optimal
    assert np.isclose(pure_final.j_structure, 0.0, atol=1.0e-10)
    assert np.isclose(pure_final.j_final_use, 0.8 / 16.0, atol=1.0e-9)

    pure_structure = problem.solve_pure_structure(rhs)
    assert pure_structure.optimal
    assert np.isclose(pure_structure.j_final_use, 0.0, atol=1.0e-9)
    assert np.isclose(pure_structure.j_structure, 0.8 / 4.0, atol=1.0e-9)

    interior = problem.solve_epsilon(rhs, 0.025)
    assert interior.optimal
    assert np.isclose(interior.j_final_use, 0.025, atol=2.0e-8)
    assert np.isclose(interior.j_structure, 0.1, atol=2.0e-8)
    assert interior.max_accounting_abs_residual <= 1.0e-9
    assert interior.max_target_abs_residual <= 1.0e-9
    assert interior.min_lower_capacity_slack >= -1.0e-9
    assert interior.min_upper_capacity_slack >= -1.0e-9


def test_inverse_mrio_pure_structure_infeasible_but_final_use_feasible() -> None:
    problem = _problem(capacity=0.05)
    rhs = np.array([1.0])
    assert problem.solve_pure_structure(rhs).status == "infeasible"
    assert problem.solve_epsilon(rhs, 0.0).status == "infeasible"
    assert problem.solve_pure_final_use(rhs).optimal
    assert problem.solve_epsilon(rhs, 0.05).optimal


def test_inverse_mrio_target_can_reach_zero_output_without_false_infeasibility() -> None:
    A0 = csr_matrix([[0.2]])
    x0 = np.array([10.0])
    f0 = np.array([8.0])
    lower = csr_matrix([[-0.2]])
    upper = csr_matrix([[0.2]])
    target = csr_matrix([[1.0]])
    problem = InverseMrioMechanismProblem(A0, x0, f0, lower, upper, target)
    result = problem.solve_pure_final_use(np.array([-10.0]))
    assert result.optimal
    assert result.delta_x is not None
    assert np.isclose(result.delta_x[0], -10.0, atol=1.0e-9)
    assert result.min_output >= -1.0e-9


def test_inverse_mrio_inherits_baseline_source_residual_without_rebalancing() -> None:
    A0 = csr_matrix([[0.2]])
    x0 = np.array([10.0])
    # Deliberately retain a tiny finite source-account residual rho=1e-7.
    f0 = np.array([7.9999999])
    lower = csr_matrix([[-0.1]])
    upper = csr_matrix([[0.1]])
    target = csr_matrix([[1.0]])
    problem = InverseMrioMechanismProblem(A0, x0, f0, lower, upper, target)
    result = problem.solve_pure_final_use(np.array([1.0]))
    assert result.optimal
    assert np.isclose(problem.baseline_accounting_residual[0], 1.0e-7, atol=1.0e-12)
    assert result.delta_x is not None and result.delta_final_use is not None
    x1 = x0 + result.delta_x
    f1 = f0 + result.delta_final_use
    counterfactual_residual = x1 - A0.dot(x1) - f1
    np.testing.assert_allclose(counterfactual_residual, problem.baseline_accounting_residual, atol=1.0e-9)


def test_inverse_mrio_accepts_signed_complete_final_use_boundary() -> None:
    problem = _problem(signed_f0=True)
    result = problem.solve_pure_final_use(np.array([1.0]))
    assert result.optimal
    assert np.isclose(problem.F0, 16.0)

def test_inverse_mrio_defers_capacity_rows_until_structure_is_allowed() -> None:
    """Pure final use must not pay the EXIO-scale L/U construction cost."""

    problem = _problem()
    assert problem.coefficient_capacity_materialized is False
    pure_final = problem.solve_pure_final_use(np.array([1.0]))
    assert pure_final.optimal
    assert problem.coefficient_capacity_materialized is False
    joint = problem.solve_epsilon(np.array([1.0]), 0.025)
    assert joint.optimal
    assert problem.coefficient_capacity_materialized is True


def test_inverse_mrio_scaled_floor_avoids_huge_row_amplification() -> None:
    """Tiny scaled coefficients may be omitted without whole-row amplification."""

    tiny = 1.0e-20
    A0 = csr_matrix([[0.2, tiny], [0.0, 0.2]])
    x0 = np.array([10.0, 10.0])
    f0 = x0 - A0.dot(x0)
    lower = csr_matrix([[-0.1, -tiny], [0.0, -0.1]])
    upper = csr_matrix([[0.1, tiny], [0.0, 0.1]])
    target = csr_matrix([[1.0, 0.0]])
    problem = InverseMrioMechanismProblem(
        A0,
        x0,
        f0,
        lower,
        upper,
        target,
        solver_matrix_floor=1.0e-12,
        postcheck_tolerance=1.0e-9,
        cold_start_solver="ipm",
    )
    result = problem.solve_pure_final_use(np.array([1.0]))
    assert result.optimal
    diag = problem.numerical_matrix_diagnostics
    assert int(diag["scaled_matrix_pruned_nonzeros"]) >= 1
    assert float(diag["scaled_matrix_max_abs_pruned"]) <= 1.0e-12
    assert float(diag["scaled_matrix_dynamic_range"]) < 1.0e6
    assert int(diag["solver_run_count"]) >= 1


def test_inverse_mrio_original_coordinate_postcheck_rejects_material_pruning() -> None:
    """A backend-optimal point is rejected when an omitted scaled term matters."""

    A0 = csr_matrix(np.diag([0.2, 0.2]))
    x0 = np.array([10.0, 10.0])
    f0 = x0 - A0.dot(x0)
    lower = csr_matrix(np.diag([-0.1, -0.1]))
    upper = csr_matrix(np.diag([0.1, 0.1]))
    target = csr_matrix(np.eye(2))
    tiny = 5.0e-10
    guard = csr_matrix([[1.0, tiny]])
    problem = InverseMrioMechanismProblem(
        A0,
        x0,
        f0,
        lower,
        upper,
        target,
        output_guard_matrix=guard,
        solver_matrix_floor=1.0e-9,
        postcheck_tolerance=1.0e-12,
        cold_start_solver="simplex",
    )
    result = problem.solve_pure_final_use(
        np.array([1.0, 1.0]),
        guard_lower=np.array([-np.inf]),
        guard_upper=np.array([1.0]),
    )
    assert result.status == "numerical_failure"
    assert "original-coordinate postcheck failed" in result.message
    assert int(problem.numerical_matrix_diagnostics["scaled_matrix_pruned_nonzeros"]) >= 1


def test_inverse_mrio_linear_output_guard_is_enforced() -> None:
    """Caller-owned output guards constrain dx without acquiring scientific meaning."""

    A0 = csr_matrix(np.diag([0.2, 0.2]))
    x0 = np.array([10.0, 10.0])
    f0 = x0 - A0.dot(x0)
    lower = csr_matrix(np.diag([-0.1, -0.1]))
    upper = csr_matrix(np.diag([0.1, 0.1]))
    target = csr_matrix([[1.0, 0.0]])
    guard = csr_matrix([[0.0, 1.0]])
    problem = InverseMrioMechanismProblem(
        A0,
        x0,
        f0,
        lower,
        upper,
        target,
        output_guard_matrix=guard,
        postcheck_tolerance=1.0e-9,
    )
    result = problem.solve_pure_final_use(
        np.array([1.0]),
        guard_lower=np.array([-np.inf]),
        guard_upper=np.array([0.0]),
    )
    assert result.optimal
    assert result.delta_x is not None
    assert result.delta_x[1] <= 1.0e-9
    assert result.max_guard_scaled_violation <= 1.0e-9

def test_inverse_mrio_rejects_negative_lower_coefficient_corner() -> None:
    """The generic coefficient-change box may never admit negative technical coefficients."""

    with np.testing.assert_raises_regex(ValueError, "A0 \\+ L negative"):
        InverseMrioMechanismProblem(
            csr_matrix([[0.2]]),
            np.array([1.0]),
            np.array([0.8]),
            csr_matrix([[-0.3]]),
            csr_matrix([[0.1]]),
            csr_matrix([[1.0]]),
        )

def test_inverse_mrio_two_objective_frontier_uses_exactly_two_highs_runs_without_tertiary() -> None:
    """A Pareto point needs primary+secondary HiGHS runs, not a zero-objective third run."""

    base = _problem()
    problem = InverseMrioMechanismProblem(
        base.A0,
        base.x0,
        base.f0,
        base.L,
        base.U,
        base.C,
        accompaniment_indices=np.empty(0, dtype=np.int32),
        feasibility_tolerance=1.0e-9,
        objective_relative_tolerance=1.0e-10,
        objective_absolute_tolerance=1.0e-9,
    )
    before = int(problem.numerical_matrix_diagnostics["solver_run_count"])
    result = problem.solve_epsilon(np.array([1.0]), 0.025)
    after = int(problem.numerical_matrix_diagnostics["solver_run_count"])
    assert result.optimal
    assert after - before == 2

# PUBLIC_HIGHSPY_BACKEND_TESTS_V1
def test_inverse_mrio_rejects_public_only_solver_on_scipy_backend() -> None:
    """Explicit HiPO/IPX names must not be sent to the older bundled backend."""

    base = _problem()
    with pytest.raises(ValueError, match="requires highs_backend='highspy'"):
        InverseMrioMechanismProblem(
            base.A0,
            base.x0,
            base.f0,
            base.L,
            base.U,
            base.C,
            highs_backend="scipy",
            cold_start_solver="hipo",
        )


def test_inverse_mrio_public_highspy_hipo_preserves_audited_solution() -> None:
    """Public HiPO solves the same tiny economic problem and passes postchecks."""

    highspy = pytest.importorskip("highspy")
    pytest.importorskip("highspy_extras")
    A0 = csr_matrix(np.diag([0.2, 0.2]))
    x0 = np.array([10.0, 10.0])
    f0 = x0 - A0.dot(x0)
    lower = csr_matrix(np.diag([-0.1, -0.1]))
    upper = csr_matrix(np.diag([0.1, 0.1]))
    target = csr_matrix([[1.0, 0.0]])
    problem = InverseMrioMechanismProblem(
        A0,
        x0,
        f0,
        lower,
        upper,
        target,
        accompaniment_indices=np.empty(0, dtype=np.int32),
        highs_backend="highspy",
        cold_start_solver="hipo",
        cold_start_crossover="off",
        postcheck_tolerance=1.0e-9,
    )
    result = problem.solve_pure_final_use(np.array([1.0]))
    assert result.optimal
    assert np.isclose(result.j_structure, 0.0, atol=1.0e-10)
    diag = problem.numerical_matrix_diagnostics
    assert diag["solver_backend"] == "highspy"
    assert diag["cold_start_solver"] == "hipo"
    assert diag["cold_start_crossover"] == "off"
    assert str(diag["highs_version"]) == str(highspy.Highs().version())
