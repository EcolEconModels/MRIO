"""Counterfactual scenario runner with full operator re-solves."""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, pd.Series):
            out[key] = value.copy(deep=True)
        elif isinstance(value, pd.DataFrame):
            out[key] = value.copy(deep=True)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_group(
    keyspace: Sequence[Any],
    op: Mapping[str, Any],
    *,
    state_groups: Mapping[str, Sequence[Any]] | None = None,
    default_all: bool = True,
) -> list[Any]:
    if "indices" in op and op["indices"] is not None:
        idx = list(op["indices"])
        return [k for k in idx if k in keyspace]
    group = op.get("group")
    if group is not None and state_groups is not None and group in state_groups:
        return [k for k in state_groups[group] if k in keyspace]
    if default_all:
        return list(keyspace)
    return []


def _resolve_axis_selector(
    keyspace: Sequence[Any],
    *,
    indices: Sequence[Any] | None,
    group: str | None,
    state_groups: Mapping[str, Sequence[Any]] | None,
    default_all: bool,
) -> list[Any]:
    op = {"indices": indices, "group": group}
    return _resolve_group(keyspace, op, state_groups=state_groups, default_all=default_all)


def _apply_operation(state: dict[str, Any], op: Mapping[str, Any]) -> None:
    op_type = str(op.get("type", ""))
    if op_type == "technical_coefficient_scaling":
        if "A" not in state:
            raise ValueError("State must contain A for technical_coefficient_scaling.")
        A = pd.DataFrame(state["A"], copy=True)
        idx = _resolve_group(
            A.columns,
            op,
            state_groups=state.get("groups"),
            default_all=True,
        )
        factor = float(op.get("factor", 1.0))
        axis = str(op.get("axis", "columns"))
        if axis == "columns":
            A.loc[:, idx] = A.loc[:, idx] * factor
        elif axis == "rows":
            A.loc[idx, :] = A.loc[idx, :] * factor
        elif axis == "both":
            A.loc[:, idx] = A.loc[:, idx] * factor
            A.loc[idx, :] = A.loc[idx, :] * factor
        else:
            raise ValueError(f"Unknown axis for technical_coefficient_scaling: {axis}")
        state["A"] = A
        return

    if op_type == "matrix_block_scaling":
        if "A" not in state:
            raise ValueError("State must contain A for matrix_block_scaling.")
        A = pd.DataFrame(state["A"], copy=True)
        groups = state.get("groups")
        rows = _resolve_axis_selector(
            A.index,
            indices=op.get("row_indices"),
            group=op.get("row_group"),
            state_groups=groups,
            default_all=True,
        )
        cols = _resolve_axis_selector(
            A.columns,
            indices=op.get("col_indices"),
            group=op.get("col_group"),
            state_groups=groups,
            default_all=True,
        )
        factor = float(op.get("factor", 1.0))
        if rows and cols:
            A.loc[rows, cols] = A.loc[rows, cols] * factor
        state["A"] = A
        return

    if op_type == "demand_reallocation":
        if "y" not in state:
            raise ValueError("State must contain y for demand_reallocation.")
        y = pd.Series(state["y"], copy=True)
        groups = state.get("groups", {})
        from_group = str(op.get("from_group", ""))
        to_group = str(op.get("to_group", ""))
        if from_group not in groups or to_group not in groups:
            raise ValueError(f"Unknown groups in demand_reallocation: {from_group}, {to_group}")
        src_idx = [k for k in groups[from_group] if k in y.index]
        dst_idx = [k for k in groups[to_group] if k in y.index]
        if not src_idx or not dst_idx:
            return
        share = float(op.get("share", 0.0))
        share = min(max(share, 0.0), 1.0)
        transfer = float(y.loc[src_idx].sum()) * share
        if transfer <= 0:
            return
        src_total = float(y.loc[src_idx].sum())
        dst_total = float(y.loc[dst_idx].sum())
        if src_total > 0:
            y.loc[src_idx] = y.loc[src_idx] - transfer * (y.loc[src_idx] / src_total)
        else:
            y.loc[src_idx] = y.loc[src_idx] - transfer / float(len(src_idx))
        if dst_total > 0:
            y.loc[dst_idx] = y.loc[dst_idx] + transfer * (y.loc[dst_idx] / dst_total)
        else:
            y.loc[dst_idx] = y.loc[dst_idx] + transfer / float(len(dst_idx))
        state["y"] = y
        return

    if op_type == "ceiling_tightening":
        if "e_bar" not in state:
            raise ValueError("State must contain e_bar for ceiling_tightening.")
        e_bar = pd.Series(state["e_bar"], copy=True)
        factor = float(op.get("factor", 1.0))
        names = op.get("constraints")
        if names is None:
            e_bar = e_bar * factor
        else:
            names_list = [n for n in names if n in e_bar.index]
            e_bar.loc[names_list] = e_bar.loc[names_list] * factor
        state["e_bar"] = e_bar
        return

    if op_type == "floor_raising":
        if "s_floor" not in state:
            raise ValueError("State must contain s_floor for floor_raising.")
        s_floor = pd.Series(state["s_floor"], copy=True)
        factor = float(op.get("factor", 1.0))
        names = op.get("constraints")
        if names is None:
            s_floor = s_floor * factor
        else:
            names_list = [n for n in names if n in s_floor.index]
            s_floor.loc[names_list] = s_floor.loc[names_list] * factor
        state["s_floor"] = s_floor
        return

    raise ValueError(f"Unknown counterfactual operation type: {op_type}")


def run_panel_scenarios(
    year_states: Mapping[int, Mapping[str, Any]],
    scenarios_cfg: Mapping[str, Any],
    solve_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> pd.DataFrame:
    years = sorted(year_states.keys())
    if not years:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for scenario in scenarios_cfg.get("scenarios", []):
        sc_id = scenario["id"]
        sc_label = scenario.get("label", sc_id)
        operations = list(scenario.get("operations", []))
        for year in years:
            state = _clone_state(year_states[year])
            for op in operations:
                _apply_operation(state, op)
            metrics = dict(solve_fn(state))
            row = {
                "scenario_id": sc_id,
                "scenario_label": sc_label,
                "year": int(year),
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def _float_token(value: float) -> str:
    token = f"{float(value):.6g}"
    return token.replace("-", "m").replace(".", "p")


def _infer_partner_rows(index: pd.Index, partner_id: str) -> list[Any]:
    if isinstance(index, pd.MultiIndex):
        mask = np.zeros(len(index), dtype=bool)
        pid = str(partner_id)
        for level in range(index.nlevels):
            mask = mask | (index.get_level_values(level).astype(str) == pid)
        return index[mask].tolist()

    pid = str(partner_id)
    out: list[Any] = []
    for label in index.tolist():
        s = str(label)
        if (
            s == pid
            or s.startswith(f"{pid}|")
            or s.startswith(f"{pid}_")
            or s.startswith(f"{pid}::")
            or s.startswith(f"{pid} ")
        ):
            out.append(label)
    return out


def _build_capacity_cut_cases(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = list(params.get("targets", []))
    factor_grid = list(params.get("factor_grid", [1.0]))
    cases: list[dict[str, Any]] = []
    for factor_raw in factor_grid:
        factor = float(factor_raw)
        if factor <= 0.0:
            raise ValueError("capacity_cut factor must be positive.")
        scarcity_inflation = 1.0 / factor
        op: dict[str, Any] = {
            "type": "technical_coefficient_scaling",
            "axis": "columns",
            "factor": scarcity_inflation,
        }
        if targets:
            op["indices"] = targets
        case_id = f"capacity_cut__f_{_float_token(factor)}"
        cases.append(
            {
                "id": case_id,
                "operations": [op],
                "scenario_factor": factor,
                "metadata": {
                    "targets": [str(t) for t in targets],
                    "scarcity_inflation": scarcity_inflation,
                },
            }
        )
    return cases


def _build_trade_partner_cut_cases(params: Mapping[str, Any], index: pd.Index) -> list[dict[str, Any]]:
    partner_ids = list(params.get("partner_ids", []))
    factor_grid = list(params.get("factor_grid", [1.0]))
    if not partner_ids:
        partner_ids = ["__all__"]

    cases: list[dict[str, Any]] = []
    all_cols = index.tolist()
    for partner in partner_ids:
        resolved_rows = all_cols if partner == "__all__" else _infer_partner_rows(index, str(partner))
        for factor_raw in factor_grid:
            factor = float(factor_raw)
            if factor <= 0.0:
                raise ValueError("trade_partner_cut factor must be positive.")
            scarcity_inflation = 1.0 / factor
            op: dict[str, Any] = {
                "type": "matrix_block_scaling",
                "row_indices": resolved_rows,
                "col_indices": all_cols,
                "factor": scarcity_inflation,
            }
            case_id = f"trade_partner_cut__{partner}__f_{_float_token(factor)}"
            cases.append(
                {
                    "id": case_id,
                    "operations": [op],
                    "scenario_factor": factor,
                    "metadata": {
                        "partner_id": str(partner),
                        "matched_rows": int(len(resolved_rows)),
                        "scarcity_inflation": scarcity_inflation,
                    },
                }
            )
    return cases


def _build_family_cases(
    family_id: str,
    params: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    A = pd.DataFrame(state["A"])
    if family_id == "capacity_cut":
        return _build_capacity_cut_cases(params)
    if family_id == "trade_partner_cut":
        return _build_trade_partner_cut_cases(params, A.index)
    return []


def run_shock_suite(
    year_states: Mapping[int, Mapping[str, Any]],
    shocks_cfg: Mapping[str, Any],
    solve_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    families: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run a family/grid shock sweep over yearly states."""
    years = sorted(year_states.keys())
    required_cols = [
        "year",
        "shock_family",
        "shock_case_id",
        "alpha_shock",
        "beta_shock",
        "mu_tech_shock",
        "scenario_factor",
        "metadata",
    ]
    if not years:
        return pd.DataFrame(columns=required_cols)

    family_filter = None if families is None else {str(f) for f in families}
    rows: list[dict[str, Any]] = []

    for family in shocks_cfg.get("shock_families", []):
        family_id = str(family.get("id", ""))
        if not family_id:
            continue
        if family_filter is not None and family_id not in family_filter:
            continue

        params = dict(family.get("parameters", {}))
        for year in years:
            base_state = year_states[year]
            cases = _build_family_cases(family_id, params, base_state)
            for case in cases:
                state = _clone_state(base_state)
                for op in case.get("operations", []):
                    _apply_operation(state, op)
                metrics = dict(solve_fn(state))

                alpha_shock = metrics.get("alpha_shock", metrics.get("alpha", np.nan))
                beta_shock = metrics.get("beta_shock", metrics.get("beta", np.nan))
                mu_tech_shock = metrics.get(
                    "mu_tech_shock",
                    metrics.get("mu_tech", metrics.get("tech_slack", np.nan)),
                )

                row = {
                    "year": int(year),
                    "shock_family": family_id,
                    "shock_case_id": str(case.get("id", family_id)),
                    "alpha_shock": float(alpha_shock) if np.isfinite(alpha_shock) else np.nan,
                    "beta_shock": float(beta_shock) if np.isfinite(beta_shock) else np.nan,
                    "mu_tech_shock": float(mu_tech_shock) if np.isfinite(mu_tech_shock) else np.nan,
                    "scenario_factor": float(case.get("scenario_factor", np.nan)),
                    "metadata": json.dumps(case.get("metadata", {}), sort_keys=True),
                }
                for key, value in metrics.items():
                    if key not in row:
                        row[key] = value
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out.sort_values(["year", "shock_family", "shock_case_id"]).reset_index(drop=True)
