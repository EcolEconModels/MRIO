"""Decomposition utilities for multi-year viability margin changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_slack_change(
    annual: pd.DataFrame,
    *,
    method: str = "additive_sda",
    include_trade_effect: bool = True,
) -> pd.DataFrame:
    """Decompose year-to-year tech-slack changes into stylized additive components.

    The function uses observed annual changes in summary diagnostics as weights and
    returns an additive decomposition whose components sum to the total change.
    """
    if method != "additive_sda":
        raise ValueError("Only method='additive_sda' is currently supported.")
    required = {"year", "tech_slack", "eco_slack_mean", "soc_slack_mean", "safe_just_objective"}
    missing = required - set(annual.columns)
    if missing:
        raise ValueError(f"annual dataframe missing required columns: {sorted(missing)}")

    annual_sorted = annual.sort_values("year").reset_index(drop=True)
    rows: list[dict[str, float | int]] = []

    for idx in range(1, len(annual_sorted)):
        prev = annual_sorted.iloc[idx - 1]
        curr = annual_sorted.iloc[idx]
        total_change = float(curr["tech_slack"] - prev["tech_slack"])

        drivers = np.array(
            [
                abs(float(curr["eco_slack_mean"] - prev["eco_slack_mean"])),
                abs(float(curr["soc_slack_mean"] - prev["soc_slack_mean"])),
                abs(float(curr["safe_just_objective"] - prev["safe_just_objective"])),
            ],
            dtype=float,
        )
        if include_trade_effect:
            trade_proxy = abs(float(curr.get("constraint_relaxation_factor", 1.0) - prev.get("constraint_relaxation_factor", 1.0)))
            drivers = np.append(drivers, trade_proxy)

        if np.allclose(drivers.sum(), 0.0):
            weights = np.full(len(drivers), 1.0 / len(drivers))
        else:
            weights = drivers / drivers.sum()

        comps = weights * total_change
        tech_comp = float(comps[0])
        demand_comp = float(comps[1])
        intensity_comp = float(comps[2])
        trade_comp = float(comps[3]) if include_trade_effect and len(comps) > 3 else 0.0

        rows.append(
            {
                "year_from": int(prev["year"]),
                "year_to": int(curr["year"]),
                "total_change": total_change,
                "technology_component": tech_comp,
                "demand_component": demand_comp,
                "intensity_component": intensity_comp,
                "trade_component": trade_comp,
                "residual": float(total_change - (tech_comp + demand_comp + intensity_comp + trade_comp)),
            }
        )

    return pd.DataFrame(rows)
