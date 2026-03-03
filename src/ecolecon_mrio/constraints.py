"""Generic linear-constraint helpers for transition-viability workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class BetaResult:
    beta: float
    x_beta: pd.Series
    beta_lower_bound: float
    beta_upper_bound: float
    binding_eco: list[str]
    binding_soc: list[str]
    feasible: bool
    message: str


def _normalize_positive(values: np.ndarray, tol: float) -> np.ndarray:
    clipped = np.clip(values.astype(float, copy=False), tol, None)
    total = float(clipped.sum())
    if total <= tol:
        raise ValueError("Cannot normalize weights with nonpositive total.")
    return clipped / total


def _weights_from_mode(mode: str, x: pd.Series, *, tol: float) -> np.ndarray:
    n = len(x)
    if n == 0:
        raise ValueError("Cannot build constraints on empty output vector.")

    if mode == "uniform":
        return np.full(n, 1.0 / float(n), dtype=float)
    if mode == "output_weighted":
        return _normalize_positive(x.to_numpy(dtype=float), tol)

    rank = x.rank(pct=True).to_numpy(dtype=float)
    if mode == "output_rank_high":
        return _normalize_positive(rank, tol)
    if mode == "output_rank_low":
        return _normalize_positive(1.0 - rank + 1e-3, tol)

    raise ValueError(f"Unknown constraint mode: {mode}")


def build_constraints(
    x: pd.Series,
    constraints_cfg: Mapping[str, Any],
    *,
    tol: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build ecological/social linear proxy constraints for a baseline output vector."""
    x_series = pd.Series(x, copy=True).astype(float)
    if (x_series < -tol).any():
        raise ValueError("Output vector x must be nonnegative to build proxy constraints.")

    eco_rows: list[np.ndarray] = []
    eco_names: list[str] = []
    eco_bounds: list[float] = []
    for item in constraints_cfg.get("ecological", []):
        w = _weights_from_mode(str(item["mode"]), x_series, tol=tol)
        cur = float(np.dot(w, x_series.to_numpy(dtype=float)))
        eco_rows.append(w)
        eco_names.append(str(item["name"]))
        eco_bounds.append(float(item["ceiling_share_of_current"]) * cur)

    soc_rows: list[np.ndarray] = []
    soc_names: list[str] = []
    soc_bounds: list[float] = []
    for item in constraints_cfg.get("social", []):
        w = _weights_from_mode(str(item["mode"]), x_series, tol=tol)
        cur = float(np.dot(w, x_series.to_numpy(dtype=float)))
        soc_rows.append(w)
        soc_names.append(str(item["name"]))
        soc_bounds.append(float(item["floor_share_of_current"]) * cur)

    E = pd.DataFrame(eco_rows, index=eco_names, columns=x_series.index, dtype=float)
    S = pd.DataFrame(soc_rows, index=soc_names, columns=x_series.index, dtype=float)
    e_bar = pd.Series(eco_bounds, index=eco_names, dtype=float)
    s_floor = pd.Series(soc_bounds, index=soc_names, dtype=float)
    return E, S, e_bar, s_floor


def solve_beta(
    q: pd.Series,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    tol: float = 1e-12,
) -> BetaResult:
    """Solve max scalar beta along direction q under linear eco/social constraints.

    The feasible set is
    x = beta * q,
    x >= 0,
    E x <= e_bar,
    S x >= s_floor.
    """
    q_series = pd.Series(q, copy=True).astype(float)
    if q_series.empty:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q is empty.",
        )

    if (q_series < -tol).any():
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q has negative entries; x = beta*q with x>=0 is not admissible.",
        )

    q_vals = q_series.to_numpy(dtype=float)
    if float(np.clip(q_vals, 0.0, None).sum()) <= tol:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q has zero mass; direction scaling is undefined.",
        )

    E_mat = pd.DataFrame(E, copy=True).reindex(columns=q_series.index).fillna(0.0)
    S_mat = pd.DataFrame(S, copy=True).reindex(columns=q_series.index).fillna(0.0)
    e_vec = pd.Series(e_bar, copy=True).reindex(E_mat.index).astype(float)
    s_vec = pd.Series(s_floor, copy=True).reindex(S_mat.index).astype(float)

    lower = 0.0
    upper = float("inf")
    binding_eco: list[str] = []
    binding_soc: list[str] = []

    # Ecological ceilings: (E_i q) beta <= e_bar_i
    for name, row in E_mat.iterrows():
        denom = float(np.dot(row.to_numpy(dtype=float), q_vals))
        rhs = float(e_vec.loc[name])
        if abs(denom) <= tol:
            if rhs < -tol:
                return BetaResult(
                    beta=np.nan,
                    x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
                    beta_lower_bound=lower,
                    beta_upper_bound=upper,
                    binding_eco=[str(name)],
                    binding_soc=binding_soc,
                    feasible=False,
                    message=(
                        f"Infeasible ecological constraint '{name}': zero denominator with negative ceiling."
                    ),
                )
            continue
        candidate = rhs / denom
        if denom > 0:
            if candidate < upper:
                upper = candidate
        else:
            lower = max(lower, candidate)

    # Social floors: (S_j q) beta >= s_floor_j
    for name, row in S_mat.iterrows():
        denom = float(np.dot(row.to_numpy(dtype=float), q_vals))
        rhs = float(s_vec.loc[name])
        if abs(denom) <= tol:
            if rhs > tol:
                return BetaResult(
                    beta=np.nan,
                    x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
                    beta_lower_bound=lower,
                    beta_upper_bound=upper,
                    binding_eco=binding_eco,
                    binding_soc=[str(name)],
                    feasible=False,
                    message=(
                        f"Infeasible social constraint '{name}': zero denominator with positive floor."
                    ),
                )
            continue
        candidate = rhs / denom
        if denom > 0:
            lower = max(lower, candidate)
        else:
            if candidate < upper:
                upper = candidate

    if lower - upper > tol:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=lower,
            beta_upper_bound=upper,
            binding_eco=sorted(set(binding_eco)),
            binding_soc=sorted(set(binding_soc)),
            feasible=False,
            message="No feasible beta satisfies both lower and upper bounds.",
        )

    if np.isfinite(upper):
        beta = float(upper)
        msg = "Optimal beta found at upper bound."
    else:
        beta = float("inf")
        msg = "Beta is unbounded above under supplied constraints."

    x_beta = q_series * beta

    binding_eco = []
    for name, row in E_mat.iterrows():
        lhs = float(np.dot(row.to_numpy(dtype=float), x_beta.to_numpy(dtype=float)))
        rhs = float(e_vec.loc[name])
        if abs(lhs - rhs) <= max(tol, tol * max(1.0, abs(rhs))):
            binding_eco.append(str(name))

    binding_soc = []
    for name, row in S_mat.iterrows():
        lhs = float(np.dot(row.to_numpy(dtype=float), x_beta.to_numpy(dtype=float)))
        rhs = float(s_vec.loc[name])
        if abs(lhs - rhs) <= max(tol, tol * max(1.0, abs(rhs))):
            binding_soc.append(str(name))

    return BetaResult(
        beta=beta,
        x_beta=x_beta,
        beta_lower_bound=float(lower),
        beta_upper_bound=float(upper),
        binding_eco=sorted(set(binding_eco)),
        binding_soc=sorted(set(binding_soc)),
        feasible=True,
        message=msg,
    )
