"""Counterfactual scenario runner for panel diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _scenario_factor(operations: list[dict[str, Any]]) -> float:
    factor = 1.0
    for op in operations:
        op_type = op.get("type", "")
        if op_type == "technical_coefficient_scaling":
            factor *= float(op.get("factor", 1.0))
        elif op_type == "ceiling_tightening":
            factor *= float(op.get("factor", 1.0))
        elif op_type == "floor_raising":
            factor /= float(op.get("factor", 1.0))
        elif op_type == "demand_reallocation":
            factor *= 1.0 - 0.2 * float(op.get("share", 0.0))
    return factor


def run_panel_scenarios(
    annual: pd.DataFrame,
    scenarios_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    required = {"year", "tech_slack"}
    if not required.issubset(annual.columns):
        raise ValueError(f"annual missing required columns: {sorted(required - set(annual.columns))}")

    rows: list[dict[str, Any]] = []
    for scenario in scenarios_cfg.get("scenarios", []):
        sc_id = scenario["id"]
        sc_label = scenario.get("label", sc_id)
        factor = _scenario_factor(scenario.get("operations", []))

        for _, row in annual.iterrows():
            align = float(row.get("align", 0.0)) if not pd.isna(row.get("align", 0.0)) else 0.0
            rows.append(
                {
                    "scenario_id": sc_id,
                    "scenario_label": sc_label,
                    "year": int(row["year"]),
                    "tech_slack_cf": float(row["tech_slack"] * factor),
                    "align_cf": align * factor,
                    "factor": factor,
                }
            )

    return pd.DataFrame(rows)
