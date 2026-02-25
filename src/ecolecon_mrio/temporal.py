"""Temporal utilities for multi-year MRIO viability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .constraints import build_constraints
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
    A_by_year: dict[int, pd.DataFrame]
    B_by_year: dict[int, pd.DataFrame]
    state_by_year: dict[int, dict[str, Any]]


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


def _solve_safe_just(
    x: pd.Series,
    y: pd.Series,
    A: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
) -> tuple[Any, float, pd.Series, pd.Series]:
    result = solve_safe_just_lp(x, y, A, E, S, e_bar, s_floor)
    return result, 1.0, e_bar, s_floor


def _solve_with_optional_relaxation(
    x: pd.Series,
    y: pd.Series,
    A: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    allow_relaxation: bool = False,
) -> tuple[Any, float, pd.Series, pd.Series]:
    if not allow_relaxation:
        return _solve_safe_just(x, y, A, E, S, e_bar, s_floor)
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
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
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
            allow_download_missing=allow_download_missing,
            resume_downloads=resume_downloads,
            show_download_progress=show_download_progress,
            download_chunk_mb=download_chunk_mb,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
        )
        panel[int(year)] = (mrio, _series_from_any(y_full))
    return panel


def compute_viability_panel(
    panel: Mapping[int, tuple[Any, pd.Series]],
    constraints_cfg: Mapping[str, Any],
    *,
    max_sectors: int = 80,
    allow_relaxation: bool = False,
) -> PanelDiagnostics:
    annual_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    x_by_year: dict[int, pd.Series] = {}
    A_by_year: dict[int, pd.DataFrame] = {}
    B_by_year: dict[int, pd.DataFrame] = {}
    state_by_year: dict[int, dict[str, Any]] = {}

    for year in sorted(panel.keys()):
        mrio, y_full = panel[year]
        x = _series_from_any(mrio.x)
        y = _series_from_any(y_full).reindex(x.index).fillna(0.0)

        top_idx = x.sort_values(ascending=False).head(max_sectors).index
        x = x.reindex(top_idx)
        y = y.reindex(top_idx).fillna(0.0)
        A = _coerce_square(mrio.A, top_idx)

        rho, tech_slack = calc_tech_slack(A)
        pf_sens = calc_pf_sensitivities(A, strict=False)

        E, S, e_bar, s_floor = build_constraints(x, constraints_cfg)
        lp, relax, e_used, s_used = _solve_with_optional_relaxation(
            x,
            y,
            A,
            E,
            S,
            e_bar,
            s_floor,
            allow_relaxation=allow_relaxation,
        )
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
        A_by_year[int(year)] = A
        B_by_year[int(year)] = pd.DataFrame(np.eye(len(A)), index=A.index, columns=A.columns)
        state_by_year[int(year)] = {
            "A": A,
            "B": B_by_year[int(year)],
            "y": y,
            "E": E,
            "S": S,
            "e_bar": e_used,
            "s_floor": s_used,
            "x": x,
        }

    annual = pd.DataFrame(annual_rows).sort_values("year").reset_index(drop=True)
    annual["delta_viability"] = annual["tech_slack"].diff()
    sector_scores = pd.DataFrame(score_rows)

    return PanelDiagnostics(
        annual=annual,
        sector_scores=sector_scores,
        x_by_year=x_by_year,
        A_by_year=A_by_year,
        B_by_year=B_by_year,
        state_by_year=state_by_year,
    )
