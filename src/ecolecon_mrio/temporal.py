"""Temporal utilities for multi-year MRIO viability analysis."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Mapping
import time

import numpy as np
import pandas as pd

from .constraints import build_constraints_with_context
from .data import load_mrio
from .diagnostics import (
    aggregate_diagnostics_from_pf_vectors,
    calc_boundary_slack,
    calc_pf_eigenvectors,
    solve_safe_just_lp,
)


@dataclass
class PanelDiagnostics:
    annual: pd.DataFrame
    sector_scores: pd.DataFrame
    x_by_year: dict[int, pd.Series]
    A_by_year: dict[int, pd.DataFrame]
    B_by_year: dict[int, Any]
    state_by_year: dict[int, dict[str, Any]]
    sector_universe_manifest: pd.DataFrame


def _rss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX fallback
        return float("nan")
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    import sys

    if sys.platform == "darwin":
        rss /= 1024.0 * 1024.0
    else:
        rss /= 1024.0
    return float(rss)


def _as_float_series(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float)
    if index is not None:
        series = pd.to_numeric(series.reindex(index), errors="coerce").fillna(0.0).astype(float)
    return series


def _as_float_frame(values: Any, *, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values).reindex(index=index, columns=columns).fillna(0.0).astype(float)


def compute_output_change_decomposition(
    A_by_year: Mapping[int, pd.DataFrame],
    x_by_year: Mapping[int, pd.Series],
    y_by_year: Mapping[int, pd.Series],
) -> pd.DataFrame:
    """Compute the exact adjacent-year output-change decomposition.

    For adjacent years ``t-1`` and ``t`` this uses the current-year Leontief
    inverse ``L_t = (I - A_t)^{-1}`` and returns

    ``tech_component = L_t DeltaA_t x_{t-1}``
    ``demand_component = L_t Delta y_t``

    with ``total_change = x_t - x_{t-1}``.
    """
    rows: list[dict[str, Any]] = []
    years = sorted(
        set(int(y) for y in A_by_year.keys())
        & set(int(y) for y in x_by_year.keys())
        & set(int(y) for y in y_by_year.keys())
    )
    for i, year_prev in enumerate(years[:-1]):
        year = years[i + 1]
        idx = pd.Index(x_by_year[year_prev].index)
        A_prev = _as_float_frame(A_by_year[year_prev], index=idx, columns=idx)
        A_t = _as_float_frame(A_by_year[year], index=idx, columns=idx)
        x_prev = _as_float_series(x_by_year[year_prev], index=idx)
        x_t = _as_float_series(x_by_year[year], index=idx)
        y_prev = _as_float_series(y_by_year[year_prev], index=idx)
        y_t = _as_float_series(y_by_year[year], index=idx)

        I_minus_A_t = np.eye(len(idx), dtype=float) - A_t.to_numpy(dtype=float)
        try:
            L_t = np.linalg.inv(I_minus_A_t)
        except np.linalg.LinAlgError:
            continue

        delta_A = A_t - A_prev
        delta_y = y_t - y_prev
        tech_component = pd.Series(
            L_t.dot(delta_A.to_numpy(dtype=float).dot(x_prev.to_numpy(dtype=float))),
            index=idx,
            dtype=float,
        )
        demand_component = pd.Series(L_t.dot(delta_y.to_numpy(dtype=float)), index=idx, dtype=float)
        total_change = (x_t - x_prev).astype(float)

        for sector, tech_value, demand_value, total_value in zip(
            idx,
            tech_component.to_numpy(dtype=float),
            demand_component.to_numpy(dtype=float),
            total_change.to_numpy(dtype=float),
            strict=False,
        ):
            rows.append(
                {
                    "year_from": int(year_prev),
                    "year_to": int(year),
                    "sector": str(sector),
                    "tech_component": float(tech_value),
                    "demand_component": float(demand_value),
                    "total_change": float(total_value),
                }
            )
    return pd.DataFrame(rows)


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


def _preserve_index_order(reference: pd.Index, selected: pd.Index) -> pd.Index:
    reference_index = pd.Index(reference)
    selected_index = pd.Index(selected)
    if selected_index.empty:
        return selected_index
    return reference_index[reference_index.isin(selected_index)]


def _solve_safe_just(
    x: pd.Series,
    y: pd.Series,
    A: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    lp_working_set_n: int | None = None,
    lp_working_set_mode: str = "output_weighted",
    lp_working_set_role_masks: Mapping[str, pd.Series] | None = None,
    lp_working_set_role_mix: Mapping[str, float] | None = None,
) -> tuple[Any, float, pd.Series, pd.Series]:
    result = solve_safe_just_lp(
        x,
        y,
        A,
        E,
        S,
        e_bar,
        s_floor,
        working_set_n=lp_working_set_n,
        working_set_mode=lp_working_set_mode,
        working_set_role_masks=lp_working_set_role_masks,
        working_set_role_mix=lp_working_set_role_mix,
    )
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
    lp_working_set_n: int | None = None,
    lp_working_set_mode: str = "output_weighted",
    lp_working_set_role_masks: Mapping[str, pd.Series] | None = None,
    lp_working_set_role_mix: Mapping[str, float] | None = None,
) -> tuple[Any, float, pd.Series, pd.Series]:
    if not allow_relaxation:
        return _solve_safe_just(
            x,
            y,
            A,
            E,
            S,
            e_bar,
            s_floor,
            lp_working_set_n=lp_working_set_n,
            lp_working_set_mode=lp_working_set_mode,
            lp_working_set_role_masks=lp_working_set_role_masks,
            lp_working_set_role_mix=lp_working_set_role_mix,
        )
    for relax in [1.0, 1.02, 1.05, 1.1, 1.2, 1.35, 1.5]:
        try:
            result = solve_safe_just_lp(
                x,
                y,
                A,
                E,
                S,
                e_bar * relax,
                s_floor / relax,
                working_set_n=lp_working_set_n,
                working_set_mode=lp_working_set_mode,
                working_set_role_masks=lp_working_set_role_masks,
                working_set_role_mix=lp_working_set_role_mix,
            )
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
    dataset: str | None = None,
    max_sectors: int = 80,
    sector_universe_mode: str = "full_basis",
    sector_universe_n: int | None = None,
    solver_mode: str = "sparse_exact",
    low_memory_sector_universe_n: int = 500,
    lp_working_set_n: int | None = None,
    lp_working_set_mode: str = "output_weighted",
    lp_working_set_role_masks_by_year: Mapping[int, Mapping[str, pd.Series]] | None = None,
    lp_working_set_role_mix: Mapping[str, float] | None = None,
    repair_lp_enabled: bool = True,
    allow_relaxation: bool = False,
    progress_emit_stdout: bool = False,
    progress_heartbeat_seconds: int = 30,
) -> PanelDiagnostics:
    annual_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    x_by_year: dict[int, pd.Series] = {}
    A_by_year: dict[int, pd.DataFrame] = {}
    B_by_year: dict[int, pd.DataFrame] = {}
    state_by_year: dict[int, dict[str, Any]] = {}
    sector_universe_parts: list[pd.DataFrame] = []
    heartbeat_s = max(5, int(progress_heartbeat_seconds))
    solver_mode_norm = str(solver_mode or "sparse_exact").strip().lower()
    low_memory_full_basis_mode = solver_mode_norm in {
        "low_memory_full_basis",
        "full_basis_approx",
        "approx_full_basis",
        "sparse_approx_full_basis",
        "sparse_iterative_full_basis",
    }
    low_memory_n = max(1, int(low_memory_sector_universe_n))
    n_sel = int(sector_universe_n or max_sectors)
    lp_working_set_n_effective = int(lp_working_set_n) if lp_working_set_n is not None else None
    mode = str(sector_universe_mode or "full_basis").strip().lower()
    full_basis_mode = mode in {"full_basis", "full", "full_basis_ordered"} or low_memory_full_basis_mode
    sector_universe_manifest = pd.DataFrame()
    if low_memory_full_basis_mode and lp_working_set_n_effective is None:
        lp_working_set_n_effective = int(low_memory_n)

    for year in sorted(panel.keys()):
        mrio, _yfull = panel[year]
        t_year = time.time()
        if progress_emit_stdout:
            print(
                f"[MRIO] viability year={int(year)} start solver_mode={solver_mode_norm} "
                f"rss={float(_rss_mb()):.1f}MB",
                flush=True,
            )
        x_raw = _series_from_any(mrio.x)
        y_raw = _series_from_any(_yfull).reindex(x_raw.index).fillna(0.0)
        x_full = x_raw.astype(float)
        y_full = y_raw.astype(float)
        top_idx = x_full.index.copy()
        if full_basis_mode:
            A_year = pd.DataFrame(mrio.A, copy=False)
            if not A_year.index.equals(top_idx) or not A_year.columns.equals(top_idx):
                A_year = _coerce_square(A_year, top_idx)
            basis_scores = pd.to_numeric(x_full, errors="coerce").fillna(0.0).astype(float)
            denom = float(basis_scores.sum())
            if denom > 0:
                basis_scores = basis_scores / denom
            manifest = pd.DataFrame(
                {
                    "sector": [str(idx) for idx in top_idx],
                    "score": basis_scores.to_numpy(dtype=float),
                    "years_present": np.ones(len(top_idx), dtype=float),
                    "selected": np.ones(len(top_idx), dtype=bool),
                    "rank": np.arange(1, len(top_idx) + 1, dtype=int),
                    "sector_universe_mode": "full_basis",
                    "sector_universe_n": int(len(top_idx)),
                    "year": int(year),
                }
            )
            sector_universe_parts.append(manifest)
        else:
            A_year = _coerce_square(mrio.A, top_idx)
        x_work = x_full
        y_work = y_full
        A_work = A_year
        if progress_emit_stdout:
            print(
                f"[MRIO] viability year={int(year)} basis_size={len(top_idx)} "
                f"A_shape={A_year.shape} x_nnz={int((x_full != 0).sum())} rss={float(_rss_mb()):.1f}MB",
                flush=True,
        )

        t_pf = time.time()
        pf_left, pf_right, rho, pf_meta = calc_pf_eigenvectors(A_work, return_metadata=True)
        tech_slack = max(0.0, 1.0 - rho)
        if progress_emit_stdout:
            print(
                f"[MRIO] viability year={int(year)} tech_slack rho={rho:.6g} "
                f"mode={solver_mode_norm} elapsed={time.time()-t_pf:.1f}s rss={float(_rss_mb()):.1f}MB",
                flush=True,
            )

        t_constraints = time.time()
        E, S, e_bar, s_floor, cmeta = build_constraints_with_context(
            x_work,
            constraints_cfg,
            mrio_obj=mrio,
            year=int(year),
            dataset=dataset,
            selected_index=top_idx,
            extra_cfg={"budget_paths_data": dict(constraints_cfg.get("_budget_paths_data", {}) or {})},
            return_metadata=True,
        )
        if progress_emit_stdout:
            print(
                f"[MRIO] viability year={int(year)} constraints built E={E.shape} S={S.shape} "
                f"elapsed={time.time()-t_constraints:.1f}s rss={float(_rss_mb()):.1f}MB",
                flush=True,
            )
        t_lp = time.time()
        lp_role_masks = (
            dict(lp_working_set_role_masks_by_year.get(int(year), {}))
            if lp_working_set_role_masks_by_year is not None and int(year) in lp_working_set_role_masks_by_year
            else None
        )
        if repair_lp_enabled:
            lp, relax, e_used, s_used = _solve_with_optional_relaxation(
                x_work,
                y_work,
                A_work,
                E,
                S,
                e_bar,
                s_floor,
                allow_relaxation=allow_relaxation,
                lp_working_set_n=lp_working_set_n_effective,
                lp_working_set_mode=lp_working_set_mode,
                lp_working_set_role_masks=lp_role_masks,
                lp_working_set_role_mix=lp_working_set_role_mix,
            )
            if progress_emit_stdout:
                print(
                    f"[MRIO] viability year={int(year)} lp solved relax={relax:.3g} "
                    f"elapsed={time.time()-t_lp:.1f}s rss={float(_rss_mb()):.1f}MB",
                    flush=True,
            )
        else:
            relax = 1.0
            e_used = e_bar
            s_used = s_floor
            lp = type(
                "SafeJustResult",
                (),
                {
                    "x_opt": x_work.copy(),
                    "delta_plus": pd.Series(0.0, index=x_work.index, dtype=float),
                    "delta_minus": pd.Series(0.0, index=x_work.index, dtype=float),
                    "objective": 0.0,
                    "inequality_duals": None,
                    "equality_duals": None,
                    "eco_violation": None,
                    "soc_violation": None,
                    "status": "skipped",
                    "message": "repair_lp_disabled_by_config",
                },
            )()
            if progress_emit_stdout:
                print(
                    f"[MRIO] viability year={int(year)} repair LP skipped by config "
                    f"elapsed={time.time()-t_lp:.1f}s rss={float(_rss_mb()):.1f}MB",
                    flush=True,
                )
        x_opt_full = pd.Series(0.0, index=x_full.index, dtype=float)
        x_opt_full.loc[lp.x_opt.index] = pd.to_numeric(lp.x_opt, errors="coerce").fillna(0.0).astype(float)
        eco_slack, soc_slack = calc_boundary_slack(E, S, lp.x_opt, e_used, s_used)
        composite_work = aggregate_diagnostics_from_pf_vectors(pf_left, pf_right, E, S, eco_slack, soc_slack)
        composite = pd.Series(0.0, index=x_full.index, dtype=float)
        composite.loc[composite_work.index] = composite_work.astype(float)

        annual_rows.append(
            {
                "year": int(year),
                "rho": float(rho),
                "tech_slack": float(tech_slack),
                "eco_slack_mean": float(eco_slack.mean()),
                "soc_slack_mean": float(soc_slack.mean()),
                "safe_just_objective": float(lp.objective),
                "constraint_relaxation_factor": float(relax),
                "basis_size": int(len(top_idx)),
                "solver_mode": str(solver_mode_norm),
                "lp_working_set_n": int(lp_working_set_n_effective) if lp_working_set_n_effective is not None else int(len(top_idx)),
                "lp_working_set_mode": str(lp_working_set_mode),
                "lp_working_set_role_mix": dict(lp_working_set_role_mix or {}),
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
        x_by_year[int(year)] = x_full.astype(float)
        A_by_year[int(year)] = pd.DataFrame(mrio.A, copy=False).reindex(index=x_full.index, columns=x_full.index).fillna(0.0).astype(float)
        B_by_year[int(year)] = None
        state_by_year[int(year)] = {
            "A": A_by_year[int(year)],
            "B": None,
            "y": y_full.astype(float),
            "E": E,
            "S": S,
            "e_bar": e_used,
            "s_floor": s_used,
            "x": x_full.astype(float),
            "eco_source_mode": (cmeta or {}).get("eco_source_mode", "unknown"),
            "eco_selector_note": (cmeta or {}).get("eco_selector_note", ""),
            "eco_budget_factor": (cmeta or {}).get("eco_budget_factor", np.nan),
            "eco_budget_ref_year": (cmeta or {}).get("eco_budget_ref_year", ""),
            "eco_row_label": (cmeta or {}).get("eco_row_label", ""),
            "solver_mode": str(solver_mode_norm),
            "pf_method": "low_memory_full_basis" if low_memory_full_basis_mode else "sparse_exact",
            "pf_metadata": pf_meta,
            "basis_size": int(len(top_idx)),
            "basis_mode": "full_basis" if full_basis_mode else str(mode),
            "lp_working_set_n": int(lp_working_set_n_effective) if lp_working_set_n_effective is not None else int(len(top_idx)),
            "lp_working_set_mode": str(lp_working_set_mode),
            "lp_working_set_role_mix": dict(lp_working_set_role_mix or {}),
            "repair_lp_enabled": bool(repair_lp_enabled),
        }
        if progress_emit_stdout:
            print(
                f"[MRIO] viability year={int(year)} complete elapsed={time.time()-t_year:.1f}s "
                f"annual_rows={len(annual_rows)} scores={len(score_rows)} rss={float(_rss_mb()):.1f}MB",
                flush=True,
            )
        del lp, eco_slack, soc_slack, composite, composite_work, pf_left, pf_right, pf_meta, x_raw, y_raw, x_work, y_work, A_work, A_year
        gc.collect()

    annual = pd.DataFrame(annual_rows).sort_values("year").reset_index(drop=True)
    annual["delta_viability"] = annual["tech_slack"].diff()
    annual.loc[annual["year"].diff() != 1, "delta_viability"] = np.nan
    sector_scores = pd.DataFrame(score_rows)
    if full_basis_mode and sector_universe_parts:
        sector_universe_manifest = pd.concat(sector_universe_parts, ignore_index=True)
    elif not sector_universe_manifest.empty:
        sector_universe_manifest = sector_universe_manifest.reset_index(drop=True)

    return PanelDiagnostics(
        annual=annual,
        sector_scores=sector_scores,
        x_by_year=x_by_year,
        A_by_year=A_by_year,
        B_by_year=B_by_year,
        state_by_year=state_by_year,
        sector_universe_manifest=sector_universe_manifest,
    )
