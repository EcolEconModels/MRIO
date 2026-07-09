from __future__ import annotations

import numpy as np
import pandas as pd

from ecolecon_mrio.emissions_stabilization import (
    shrink_intensity_to_group_mean,
    smooth_intensity_across_years,
    threshold_intensity,
    winsorize_intensity,
)


def test_threshold_intensity_replaces_small_outputs() -> None:
    e = pd.Series([10.0, 1.0, 5.0], index=["a", "b", "c"], dtype=float)
    x = pd.Series([1000.0, 0.000001, 50.0], index=["a", "b", "c"], dtype=float)
    out = threshold_intensity(e, x, tau=1.0e-6)

    assert np.isclose(float(out["b"]), (10.0 * 1000.0 + 5.0 * 50.0) / (1000.0 + 50.0))
    assert np.isclose(float(out["a"]), 10.0)
    assert np.isclose(float(out["c"]), 5.0)


def test_threshold_intensity_uses_extension_values_for_weighted_reference() -> None:
    e = pd.Series([10.0, 1.0, 5.0], index=["a", "b", "c"], dtype=float)
    x = pd.Series([1000.0, 0.000001, 50.0], index=["a", "b", "c"], dtype=float)
    values = pd.Series([8000.0, 1.0, 500.0], index=["a", "b", "c"], dtype=float)
    out = threshold_intensity(e, x, tau=1.0e-6, values=values)

    assert np.isclose(float(out["b"]), (8000.0 + 500.0) / (1000.0 + 50.0))


def test_winsorize_clips_extremes() -> None:
    e = pd.Series([1.0, 2.0, 100.0, 3.0, 4.0], dtype=float)
    out = winsorize_intensity(e, p_low=0.0, p_high=0.8)

    assert float(out.max()) <= 4.0 + 1.0e-9
    assert float(out.min()) >= 1.0 - 1.0e-9


def test_shrink_intensity_to_group_mean() -> None:
    e = pd.Series([1.0, 10.0, 5.0], index=["a", "b", "c"], dtype=float)
    x = pd.Series([1.0, 9.0, 10.0], index=["a", "b", "c"], dtype=float)
    group_masks = {
        "g": pd.Series([True, True, True], index=["a", "b", "c"]),
    }
    out = shrink_intensity_to_group_mean(e, x, group_masks, lambda_scale=1.0)

    mean_e = (1.0 + 10.0 + 5.0) / 3.0
    expected = pd.Series(
        [
            0.05 * 1.0 + 0.95 * mean_e,
            0.45 * 10.0 + 0.55 * mean_e,
            0.5 * 5.0 + 0.5 * mean_e,
        ],
        index=["a", "b", "c"],
        dtype=float,
    )
    assert np.allclose(out.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_smooth_intensity_across_years() -> None:
    e_by_year = {
        2000: pd.Series([1.0, 2.0], index=["a", "b"], dtype=float),
        2001: pd.Series([3.0, 4.0], index=["a", "b"], dtype=float),
        2002: pd.Series([5.0, 6.0], index=["a", "b"], dtype=float),
    }
    smoothed = smooth_intensity_across_years(e_by_year, window=3)

    assert np.isclose(float(smoothed[2001]["a"]), 3.0)
    assert np.isclose(float(smoothed[2001]["b"]), 4.0)
