from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScopeTotals:
    all_total: float
    industry_total: float
    household_total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ClosureGrowthResult:
    g_ren: float
    g_nren: float
    g_other: float
    g_vec: np.ndarray
    weighted_total_prev: float
    weighted_total_next: float

    def to_dict(self) -> dict[str, object]:
        return {
            "g_ren": float(self.g_ren),
            "g_nren": float(self.g_nren),
            "g_other": float(self.g_other),
            "g_vec": [float(value) for value in self.g_vec.tolist()],
            "weighted_total_prev": float(self.weighted_total_prev),
            "weighted_total_next": float(self.weighted_total_next),
        }


def percent_deviation(observed: float, reference: float) -> float:
    if not math.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (float(observed) - float(reference)) / float(reference)


def compound_endpoint(start_value: float, annual_rate: float, years: int) -> float:
    if int(years) < 1:
        raise ValueError("years must be >= 1")
    return float(start_value) * (1.0 + float(annual_rate)) ** (int(years) - 1)


def annualized_growth_rate(start_value: float, end_value: float, transitions: int) -> float:
    if int(transitions) <= 0:
        raise ValueError("transitions must be > 0")
    start = float(start_value)
    end = float(end_value)
    if start <= 0 or end <= 0:
        raise ValueError("start_value and end_value must be > 0")
    return math.exp(math.log(end / start) / int(transitions)) - 1.0


def sum_group_total(frame: pd.DataFrame, columns: Sequence[str], *, scale_divisor: float = 1000.0) -> float:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for group total: {missing}")
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return float(numeric.to_numpy().sum()) / float(scale_divisor)


def summarize_scope_totals(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    sector_col: str = "Sector",
    household_label: str = "Households",
    scale_divisor: float = 1000.0,
) -> ScopeTotals:
    if sector_col not in frame.columns:
        raise ValueError(f"Missing sector column: {sector_col}")
    all_total = sum_group_total(frame, columns, scale_divisor=scale_divisor)
    household_mask = frame[sector_col].astype(str) == str(household_label)
    household = frame.loc[household_mask].copy()
    industry = frame.loc[~household_mask].copy()
    return ScopeTotals(
        all_total=all_total,
        industry_total=sum_group_total(industry, columns, scale_divisor=scale_divisor),
        household_total=sum_group_total(household, columns, scale_divisor=scale_divisor),
    )


def compute_closure_growth(
    *,
    option: str,
    eps_ren: float,
    eps_nren: float,
    aggregate_growth: float,
    weights: Iterable[float],
    renewable_indices: Sequence[int],
    nonrenewable_indices: Sequence[int],
) -> ClosureGrowthResult:
    weight_vec = np.asarray(list(weights), dtype=float)
    if weight_vec.ndim != 1 or weight_vec.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")

    ren_idx = np.asarray(list(renewable_indices), dtype=int)
    nren_idx = np.asarray(list(nonrenewable_indices), dtype=int)
    if ren_idx.size == 0 or nren_idx.size == 0:
        raise ValueError("renewable_indices and nonrenewable_indices must both be non-empty")

    all_idx = np.arange(weight_vec.size, dtype=int)
    other_idx = np.setdiff1d(all_idx, np.concatenate([ren_idx, nren_idx]), assume_unique=False)

    x_ren_prev = float(weight_vec[ren_idx].sum())
    x_nren_prev = float(weight_vec[nren_idx].sum())
    x_other_prev = float(weight_vec[other_idx].sum())
    x_total_prev = float(weight_vec.sum())

    def _safe_growth(num: float, den: float, default_value: float) -> float:
        if not math.isfinite(den) or den == 0:
            return float(default_value)
        return float(num) / float(den) - 1.0

    option_norm = str(option).strip().lower()
    if option_norm == "residual-others":
        g_ren = float(eps_ren)
        g_nren = float(eps_nren)
        g_other = _safe_growth(
            (1.0 + float(aggregate_growth)) * x_total_prev - (1.0 + g_ren) * x_ren_prev - (1.0 + g_nren) * x_nren_prev,
            x_other_prev,
            float(aggregate_growth),
        )
    elif option_norm == "fixed-others":
        g_other = float(aggregate_growth)
        g_ren = float(eps_ren)
        g_nren = _safe_growth(
            (1.0 + float(aggregate_growth)) * x_total_prev - (1.0 + g_ren) * x_ren_prev - (1.0 + g_other) * x_other_prev,
            x_nren_prev,
            float(aggregate_growth),
        )
    elif option_norm == "uniform-demand":
        g_ren = float(aggregate_growth)
        g_nren = float(aggregate_growth)
        g_other = _safe_growth(
            (1.0 + float(aggregate_growth)) * x_total_prev - (1.0 + g_ren) * x_ren_prev - (1.0 + g_nren) * x_nren_prev,
            x_other_prev,
            float(aggregate_growth),
        )
    elif option_norm == "eps-only":
        g_ren = float(aggregate_growth)
        g_nren = float(aggregate_growth)
        g_other = float(aggregate_growth)
    else:
        raise ValueError(f"Unknown closure option: {option}")

    g_vec = np.full(weight_vec.shape, g_other, dtype=float)
    g_vec[ren_idx] = g_ren
    g_vec[nren_idx] = g_nren
    weighted_total_next = float((weight_vec * (1.0 + g_vec)).sum())
    return ClosureGrowthResult(
        g_ren=g_ren,
        g_nren=g_nren,
        g_other=g_other,
        g_vec=g_vec,
        weighted_total_prev=x_total_prev,
        weighted_total_next=weighted_total_next,
    )