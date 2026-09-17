# GENERIC_INVERSE_MRIO_MECHANISM_TESTS_V1
from __future__ import annotations

import numpy as np
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
