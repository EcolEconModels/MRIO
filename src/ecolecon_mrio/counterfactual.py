"""Counterfactual scenario runner with full operator re-solves."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Sequence

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
