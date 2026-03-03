"""Temporal utilities for multi-year MRIO viability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import time

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
    sector_universe_manifest: pd.DataFrame


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


def _select_panel_union_zero_pad_top_n(
    x_raw_by_year: Mapping[int, pd.Series],
    *,
    n: int,
) -> tuple[pd.Index, pd.DataFrame]:
    if not x_raw_by_year:
        return pd.Index([], dtype=object), pd.DataFrame(columns=["sector", "score", "years_present", "selected", "rank", "sector_universe_mode", "sector_universe_n"])
    union_labels: list[Any] = []
    seen: set[Any] = set()
    for y in sorted(x_raw_by_year.keys()):
        for label in list(x_raw_by_year[y].index):
            if label not in seen:
                seen.add(label)
                union_labels.append(label)
    idx = pd.Index(union_labels, dtype=object)

    shares_by_year: list[pd.Series] = []
    for y in sorted(x_raw_by_year.keys()):
        x = pd.to_numeric(x_raw_by_year[y], errors="coerce").fillna(0.0).astype(float)
        denom = float(x.sum())
        s = (x / denom) if denom > 0 else (x * 0.0)
        shares_by_year.append(s.reindex(idx).fillna(0.0))
    share_df = pd.DataFrame(shares_by_year, index=sorted(x_raw_by_year.keys()), columns=idx)
    score = share_df.mean(axis=0)
    years_present = pd.DataFrame(
        [pd.to_numeric(x_raw_by_year[y], errors="coerce").fillna(0.0).reindex(idx).fillna(0.0) > 0 for y in sorted(x_raw_by_year.keys())],
        index=sorted(x_raw_by_year.keys()),
        columns=idx,
    ).sum(axis=0)
    ranked = (
        pd.DataFrame(
            {
                "sector_obj": list(idx),
                "sector": [str(x) for x in idx],
                "score": score.to_numpy(dtype=float),
                "years_present": years_present.to_numpy(dtype=float),
            }
        )
        .sort_values(["score", "sector"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selected"] = ranked["rank"] <= int(max(1, n))
    ranked["sector_universe_mode"] = "panel_union_zero_pad_top_n"
    ranked["sector_universe_n"] = int(max(1, n))
    selected = ranked.loc[ranked["selected"], "sector_obj"].tolist()
    return pd.Index(selected, dtype=object), ranked[["sector", "score", "years_present", "selected", "rank", "sector_universe_mode", "sector_universe_n"]].copy()


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
    delete_source_zip_after_parse: bool = False,
) -> dict[int, tuple[Any, pd.Series]]:
    panel: dict[int, tuple[Any, pd.Series]] = {}
    for i, year in enumerate(years, start=1):
        t0 = time.time()
        if verbose:
            print(f"[MRIO] loading {dataset} year={year} ({i}/{len(years)})...", flush=True)
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
            delete_source_zip_after_parse=delete_source_zip_after_parse,
        )
        panel[int(year)] = (mrio, _series_from_any(y_full))
        if verbose:
            print(f"[MRIO] loaded {dataset} year={year} ({i}/{len(years)}) in {time.time()-t0:.1f}s", flush=True)
    return panel


def compute_viability_panel(
    panel: Mapping[int, tuple[Any, pd.Series]],
    constraints_cfg: Mapping[str, Any],
    *,
    max_sectors: int = 80,
    sector_universe_mode: str = "per_year_top_n",
    sector_universe_n: int | None = None,
    allow_relaxation: bool = False,
) -> PanelDiagnostics:
    annual_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    x_by_year: dict[int, pd.Series] = {}
    A_by_year: dict[int, pd.DataFrame] = {}
    B_by_year: dict[int, pd.DataFrame] = {}
    state_by_year: dict[int, dict[str, Any]] = {}
    x_raw_by_year: dict[int, pd.Series] = {}
    y_raw_by_year: dict[int, pd.Series] = {}
    for year in sorted(panel.keys()):
        mrio, y_full = panel[year]
        x_raw = _series_from_any(mrio.x)
        y_raw = _series_from_any(y_full).reindex(x_raw.index).fillna(0.0)
        x_raw_by_year[int(year)] = x_raw.astype(float)
        y_raw_by_year[int(year)] = y_raw.astype(float)

    n_sel = int(sector_universe_n or max_sectors)
    mode = str(sector_universe_mode or "per_year_top_n").strip().lower()
    fixed_index: pd.Index | None = None
    sector_universe_manifest = pd.DataFrame()
    if mode == "panel_union_zero_pad_top_n":
        fixed_index, sector_universe_manifest = _select_panel_union_zero_pad_top_n(
            x_raw_by_year,
            n=n_sel,
        )

    for year in sorted(panel.keys()):
        mrio, _yfull = panel[year]
        x = x_raw_by_year[int(year)]
        y = y_raw_by_year[int(year)]
        if fixed_index is not None:
            top_idx = fixed_index
        else:
            top_idx = x.sort_values(ascending=False).head(n_sel).index
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
    annual.loc[annual["year"].diff() != 1, "delta_viability"] = np.nan
    sector_scores = pd.DataFrame(score_rows)

    return PanelDiagnostics(
        annual=annual,
        sector_scores=sector_scores,
        x_by_year=x_by_year,
        A_by_year=A_by_year,
        B_by_year=B_by_year,
        state_by_year=state_by_year,
        sector_universe_manifest=sector_universe_manifest,
    )
