"""Transition-direction diagnostics for multi-year MRIO analysis."""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np
import pandas as pd


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, None)
    return clipped / clipped.sum()


def principal_direction_index(
    x_by_year: Mapping[int, pd.Series],
    score_by_year: pd.DataFrame | None = None,
) -> pd.DataFrame:
    warnings.warn(
        "principal_direction_index is deprecated; use growth_direction.principal_growth_panel "
        "for eigenvector-based principal growth diagnostics.",
        DeprecationWarning,
        stacklevel=2,
    )
    """Compute annual alignment between observed structural change and reference direction."""
    years = sorted(x_by_year.keys())
    rows: list[dict[str, float | int]] = []

    for i in range(1, len(years)):
        y0 = years[i - 1]
        y1 = years[i]
        x0 = x_by_year[y0]
        x1 = x_by_year[y1].reindex(x0.index).fillna(0.0)
        delta = (x1 - x0).to_numpy(dtype=float)

        if score_by_year is not None and not score_by_year.empty:
            subset = score_by_year[score_by_year["year"] == y1].set_index("sector")
            q = subset.reindex([str(idx) for idx in x0.index])["composite_score"].fillna(0.0).to_numpy(dtype=float)
            if np.allclose(q.sum(), 0.0):
                q = np.ones_like(delta)
        else:
            q = np.ones_like(delta)

        qn = _normalize_positive(q)
        denom = np.linalg.norm(delta) * np.linalg.norm(qn)
        align = float(np.dot(delta, qn) / denom) if denom > 0 else np.nan
        rows.append({"year": int(y1), "align": align})

    return pd.DataFrame(rows)


def classify_transition_years(
    annual: pd.DataFrame,
    alignment: pd.DataFrame,
    *,
    align_high_threshold: float = 0.2,
    align_low_threshold: float = -0.2,
) -> pd.DataFrame:
    required_annual = {"year", "delta_viability"}
    if not required_annual.issubset(annual.columns):
        raise ValueError(f"annual missing required columns: {sorted(required_annual - set(annual.columns))}")
    if not {"year", "align"}.issubset(alignment.columns):
        raise ValueError("alignment must contain columns ['year', 'align']")

    merged = annual.merge(alignment, on="year", how="left")

    classes = []
    for _, row in merged.iterrows():
        dv = 0.0 if pd.isna(row["delta_viability"]) else float(row["delta_viability"])
        al = 0.0 if pd.isna(row["align"]) else float(row["align"])
        if dv > 0 and al > align_high_threshold:
            classes.append("advancing")
        elif dv < 0 and al < align_low_threshold:
            classes.append("delaying")
        else:
            classes.append("ambiguous")

    merged["transition_class"] = classes
    return merged
