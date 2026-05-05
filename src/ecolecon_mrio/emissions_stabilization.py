"""Utilities for stabilizing direct emissions intensities.

These helpers operate on intensity vectors e = F / x and mitigate instability
when output entries are very small.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def threshold_intensity(
    e: pd.Series,
    x: pd.Series,
    tau: float = 1.0e-8,
) -> pd.Series:
    """Replace intensities with a reference mean when output is tiny.

    Cells with |x_i| <= tau * sum(|x|) are considered small-output cells.
    Their intensity is replaced by the mean intensity of the non-small cells.
    """
    if float(tau) < 0.0:
        raise ValueError("tau must be non-negative")

    coeff = pd.to_numeric(pd.Series(e), errors="coerce").fillna(0.0).astype(float)
    output = pd.to_numeric(pd.Series(x).reindex(coeff.index), errors="coerce").fillna(0.0).astype(float)

    total_output = float(output.abs().sum())
    threshold = float(tau) * total_output
    mask_small = output.abs() <= threshold
    if bool(mask_small.any()):
        mask_large = ~mask_small
        reference = float(coeff.loc[mask_large].mean()) if bool(mask_large.any()) else 0.0
        coeff.loc[mask_small] = reference
    return coeff


def winsorize_intensity(
    e: pd.Series,
    p_low: float = 0.0,
    p_high: float = 0.99,
) -> pd.Series:
    """Clip intensities at the selected quantile range."""
    if not (0.0 <= float(p_low) < float(p_high) <= 1.0):
        raise ValueError("percentiles must satisfy 0 <= p_low < p_high <= 1")

    coeff = pd.to_numeric(pd.Series(e), errors="coerce").fillna(0.0).astype(float)
    lower = float(coeff.quantile(float(p_low), interpolation="higher"))
    upper = float(coeff.quantile(float(p_high), interpolation="lower"))
    return coeff.clip(lower=lower, upper=upper).astype(float)


def shrink_intensity_to_group_mean(
    e: pd.Series,
    x: pd.Series,
    group_masks: Mapping[str, pd.Series],
    lambda_scale: float = 1.0,
) -> pd.Series:
    """Shrink cell intensities toward their group mean based on output share.

    For group g and cell i in g, lambda_i = clip(lambda_scale * x_i / sum_{j in g} x_j, 0, 1).
    Then e_tilde_i = lambda_i * e_i + (1 - lambda_i) * mean(e_g).
    """
    if float(lambda_scale) < 0.0:
        raise ValueError("lambda_scale must be non-negative")

    coeff = pd.to_numeric(pd.Series(e), errors="coerce").fillna(0.0).astype(float)
    output = pd.to_numeric(pd.Series(x).reindex(coeff.index), errors="coerce").fillna(0.0).astype(float)

    out = coeff.copy()
    for _gid, mask in dict(group_masks or {}).items():
        active = pd.Series(mask, index=coeff.index).reindex(coeff.index).fillna(False).astype(bool)
        if not bool(active.any()):
            continue
        group_e = coeff.loc[active]
        group_x = output.loc[active]
        total_x = float(group_x.sum())
        if total_x <= 0.0 or not np.isfinite(total_x):
            continue
        group_mean = float(group_e.mean())
        lambdas = (float(lambda_scale) * group_x / total_x).clip(lower=0.0, upper=1.0)
        out.loc[active] = (lambdas * group_e + (1.0 - lambdas) * group_mean).astype(float)
    return out


def smooth_intensity_across_years(
    intensities_by_year: Mapping[int, pd.Series],
    window: int = 3,
) -> dict[int, pd.Series]:
    """Return centered moving-average intensity vectors by year."""
    if int(window) <= 0 or int(window) % 2 != 1:
        raise ValueError("window must be a positive odd integer")

    if not intensities_by_year:
        return {}

    years = sorted(int(y) for y in intensities_by_year.keys())
    half = int(window) // 2
    smoothed: dict[int, pd.Series] = {}

    for pos, year in enumerate(years):
        start = max(0, pos - half)
        end = min(len(years), pos + half + 1)
        window_years = years[start:end]
        vals: list[pd.Series] = []
        for yy in window_years:
            vec = pd.to_numeric(pd.Series(intensities_by_year[int(yy)]), errors="coerce").fillna(0.0).astype(float)
            vals.append(vec)
        if not vals:
            smoothed[int(year)] = pd.to_numeric(
                pd.Series(intensities_by_year[int(year)]), errors="coerce"
            ).fillna(0.0).astype(float)
            continue
        mean_vec = pd.concat(vals, axis=1).mean(axis=1)
        smoothed[int(year)] = pd.to_numeric(mean_vec, errors="coerce").fillna(0.0).astype(float)
    return smoothed
