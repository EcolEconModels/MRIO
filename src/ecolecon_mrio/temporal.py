"""Temporal utilities for multi-year MRIO viability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import load_mrio
from .diagnostics import (
    aggregate_diagnostics,
    calc_boundary_slack,
    calc_pf_sensitivities,
    calc_tech_slack,
    solve_safe_just_lp,
)


@dataclass
class PanelDiagnostics:
    annual: pd.DataFrame
    sector_scores: pd.DataFrame
    x_by_year: dict[int, pd.Series]


def _series_from_any(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.astype(float)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].astype(float)
        return obj.sum(axis=1).astype(float)
    raise TypeError(f"Unsupported type for series conversion: {type(obj)}")


def _coerce_square(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(df, copy=True).reindex(index=index, columns=index).fillna(0.0).astype(float)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, None)
    return clipped / clipped.sum()


def _weights_from_mode(mode: str, x: pd.Series) -> np.ndarray:
    n = len(x)
    if mode == "uniform":
        return np.full(n, 1.0 / n)
    if mode == "output_weighted":
        return _normalize_positive(x.to_numpy())
    rank = x.rank(pct=True).to_numpy()
    if mode == "output_rank_high":
        return _normalize_positive(rank)
    if mode == "output_rank_low":
        return _normalize_positive(1.0 - rank + 1e-3)
    raise ValueError(f"Unknown constraint mode: {mode}")


def _build_constraints(x: pd.Series, constraints_cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    eco_rows = []
    eco_names = []
    eco_bounds = []
    for item in constraints_cfg["ecological"]:
        w = _weights_from_mode(item["mode"], x)
        cur = float(np.dot(w, x.to_numpy()))
        eco_rows.append(w)
        eco_names.append(item["name"])
        eco_bounds.append(float(item["ceiling_share_of_current"]) * cur)

    soc_rows = []
    soc_names = []
    soc_bounds = []
    for item in constraints_cfg["social"]:
        w = _weights_from_mode(item["mode"], x)
        cur = float(np.dot(w, x.to_numpy()))
        soc_rows.append(w)
        soc_names.append(item["name"])
        soc_bounds.append(float(item["floor_share_of_current"]) * cur)

    E = pd.DataFrame(eco_rows, index=eco_names, columns=x.index, dtype=float)
    S = pd.DataFrame(soc_rows, index=soc_names, columns=x.index, dtype=float)
    e_bar = pd.Series(eco_bounds, index=eco_names, dtype=float)
    s_floor = pd.Series(soc_bounds, index=soc_names, dtype=float)
    return E, S, e_bar, s_floor


def _solve_with_relaxation(
    x: pd.Series,
    y: pd.Series,
    A: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
) -> tuple[Any, float, pd.Series, pd.Series]:
    for relax in [1.0, 1.02, 1.05, 1.1, 1.2, 1.35, 1.5]:
        try:
            result = solve_safe_just_lp(x, y, A, E, S, e_bar * relax, s_floor / relax)
            return result, relax, e_bar * relax, s_floor / relax
        except RuntimeError:
            pass
    raise RuntimeError("Safe-and-just LP infeasible even after relaxation ladder.")


def load_mrio_panel(
    dataset: str,
    years: list[int],
    *,
    force_download: bool = False,
    data_root: str | None = None,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
) -> dict[int, tuple[Any, pd.Series]]:
    panel: dict[int, tuple[Any, pd.Series]] = {}
    for year in years:
        mrio, y_full = load_mrio(
            dataset=dataset,
            year=year,
            force_download=force_download,
            data_root=data_root,
            calc_all=calc_all,
            include_ghosh=include_ghosh,
            verbose=verbose,
        )
        panel[int(year)] = (mrio, _series_from_any(y_full))
    return panel


def compute_viability_panel(
    panel: Mapping[int, tuple[Any, pd.Series]],
    constraints_cfg: Mapping[str, Any],
    *,
    max_sectors: int = 80,
) -> PanelDiagnostics:
    annual_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    x_by_year: dict[int, pd.Series] = {}

    for year in sorted(panel.keys()):
        mrio, y_full = panel[year]
        x = _series_from_any(mrio.x)
        y = _series_from_any(y_full).reindex(x.index).fillna(0.0)

        top_idx = x.sort_values(ascending=False).head(max_sectors).index
        x = x.reindex(top_idx)
        y = y.reindex(top_idx).fillna(0.0)
        A = _coerce_square(mrio.A, top_idx)

        rho, tech_slack = calc_tech_slack(A)
        pf_sens = calc_pf_sensitivities(A)

        E, S, e_bar, s_floor = _build_constraints(x, constraints_cfg)
        lp, relax, e_used, s_used = _solve_with_relaxation(x, y, A, E, S, e_bar, s_floor)
        eco_slack, soc_slack = calc_boundary_slack(E, S, lp.x_opt, e_used, s_used)
        composite = aggregate_diagnostics(pf_sens, E, S, eco_slack, soc_slack)

        annual_rows.append(
            {
                "year": int(year),
                "rho": float(rho),
                "tech_slack": float(tech_slack),
                "eco_slack_mean": float(eco_slack.mean()),
                "soc_slack_mean": float(soc_slack.mean()),
                "safe_just_objective": float(lp.objective),
                "constraint_relaxation_factor": float(relax),
            }
        )

        for idx, value in composite.sort_values(ascending=False).items():
            score_rows.append(
                {
                    "year": int(year),
                    "sector": str(idx),
                    "composite_score": float(value),
                }
            )
        x_by_year[int(year)] = x

    annual = pd.DataFrame(annual_rows).sort_values("year").reset_index(drop=True)
    annual["delta_viability"] = annual["tech_slack"].diff()
    sector_scores = pd.DataFrame(score_rows)

    return PanelDiagnostics(annual=annual, sector_scores=sector_scores, x_by_year=x_by_year)
