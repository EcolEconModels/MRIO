"""Generic linear-constraint helpers for transition-viability workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class BetaResult:
    beta: float
    x_beta: pd.Series
    beta_lower_bound: float
    beta_upper_bound: float
    binding_eco: list[str]
    binding_soc: list[str]
    feasible: bool
    message: str


def _normalize_positive(values: np.ndarray, tol: float) -> np.ndarray:
    clipped = np.clip(values.astype(float, copy=False), tol, None)
    total = float(clipped.sum())
    if total <= tol:
        raise ValueError("Cannot normalize weights with nonpositive total.")
    return clipped / total


def _weights_from_mode(mode: str, x: pd.Series, *, tol: float) -> np.ndarray:
    n = len(x)
    if n == 0:
        raise ValueError("Cannot build constraints on empty output vector.")

    if mode == "uniform":
        return np.full(n, 1.0 / float(n), dtype=float)
    if mode == "output_weighted":
        return _normalize_positive(x.to_numpy(dtype=float), tol)

    rank = x.rank(pct=True).to_numpy(dtype=float)
    if mode == "output_rank_high":
        return _normalize_positive(rank, tol)
    if mode == "output_rank_low":
        return _normalize_positive(1.0 - rank + 1e-3, tol)

    raise ValueError(f"Unknown constraint mode: {mode}")


def _normalize_dataset_id(dataset: str | None) -> str:
    return str(dataset or "default").strip().lower()


def _resolve_nested_attr(obj: Any, dotted: str) -> Any:
    cur = obj
    for key in [p for p in str(dotted).split(".") if p]:
        if cur is None:
            return None
        cur = getattr(cur, key, None)
    return cur


def _coerce_extension_table(mrio_obj: Any, table_path: str) -> pd.DataFrame:
    if mrio_obj is None:
        raise ValueError("mrio_obj is required for source='extension_row' constraints.")
    raw = _resolve_nested_attr(mrio_obj, table_path)
    if raw is None:
        raise ValueError(f"Extension table not found at mrio_obj.{table_path}.")
    tbl = pd.DataFrame(raw).copy()
    if tbl.empty:
        raise ValueError(f"Extension table at mrio_obj.{table_path} is empty.")
    return tbl


def _select_extension_row(
    mrio_obj: Any,
    selector: Mapping[str, Any] | None,
) -> tuple[pd.Series, str]:
    sel = dict(selector or {})
    table_path = str(sel.get("table", "satellite.F"))
    table = _coerce_extension_table(mrio_obj, table_path)
    mode = str(sel.get("mode", "row_name_contains_any")).strip().lower()
    index_pairs = [(idx, str(idx)) for idx in table.index]
    index_labels = [lbl for _, lbl in index_pairs]

    if mode == "row_name_exact":
        row_name = str(sel.get("row_name", "")).strip()
        for raw_idx, label_txt in index_pairs:
            if label_txt == row_name:
                return pd.to_numeric(table.loc[raw_idx], errors="coerce"), f"{table_path}:exact:{label_txt}"
        raise ValueError(f"Exact extension row '{row_name}' not found in {table_path}.")

    if mode == "row_index":
        idx = int(sel.get("row_index", 0))
        if idx < 0 or idx >= len(table.index):
            raise ValueError(f"row_index={idx} is out of bounds for {table_path} (n_rows={len(table.index)}).")
        row_name = str(index_pairs[idx][1])
        return pd.to_numeric(table.iloc[idx], errors="coerce"), f"{table_path}:row_index:{idx}:{row_name}"

    tokens = [str(t).strip().lower() for t in (sel.get("row_name_contains_any", []) or []) if str(t).strip()]
    if not tokens:
        tokens = ["co2", "carbon", "ghg", "emission"]
    for raw_idx, label in index_pairs:
        low = label.lower()
        if any(tok in low for tok in tokens):
            return pd.to_numeric(table.loc[raw_idx], errors="coerce"), f"{table_path}:match:{label}"
    if bool(sel.get("fallback_first_row", False)):
        row_name = str(index_pairs[0][1])
        return pd.to_numeric(table.iloc[0], errors="coerce"), f"{table_path}:fallback_first_row:{row_name}"
    raise ValueError(
        f"No extension row matched tokens={tokens} in {table_path}; available sample={index_labels[:5]}."
    )


def _resolve_budget_factor(
    item: Mapping[str, Any],
    *,
    dataset: str | None,
    year: int | None,
    extra_cfg: Mapping[str, Any] | None,
) -> tuple[float, str | None]:
    if "budget_factor" in item and pd.notna(pd.to_numeric(item.get("budget_factor"), errors="coerce")):
        return float(item["budget_factor"]), None

    ref = str(item.get("budget_path_ref", "")).strip()
    if not ref:
        raise ValueError("ecological item with source='extension_row' requires budget_factor or budget_path_ref.")
    extra = dict(extra_cfg or {})
    data = dict(extra.get("budget_paths_data", {}) or {})
    paths = dict(data.get("paths", {}) or {})
    ref_block = dict(paths.get(ref, {}) or {})
    dkey = _normalize_dataset_id(dataset)
    year_key = str(int(year)) if year is not None else None

    candidates = [dkey, "default", "*"]
    for dk in candidates:
        per_ds = ref_block.get(dk, {})
        if isinstance(per_ds, Mapping):
            if year_key is not None and year_key in per_ds:
                val = pd.to_numeric(per_ds[year_key], errors="coerce")
                if np.isfinite(val):
                    ref_years = dict(data.get("reference_year", {}) or {})
                    ref_year = ref_years.get(dk, ref_years.get(dkey, ref_years.get("default")))
                    ref_year_txt = str(int(ref_year)) if pd.notna(pd.to_numeric(ref_year, errors="coerce")) else None
                    return float(val), ref_year_txt
    raise ValueError(
        f"Missing budget factor for budget_path_ref='{ref}', dataset='{dkey}', year='{year_key}'."
    )


def build_constraints(
    x: pd.Series,
    constraints_cfg: Mapping[str, Any],
    *,
    tol: float = 1e-12,
    mrio_obj: Any | None = None,
    year: int | None = None,
    dataset: str | None = None,
    selected_index: pd.Index | None = None,
    extra_cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    E, S, e_bar, s_floor, _meta = build_constraints_with_context(
        x,
        constraints_cfg,
        tol=tol,
        mrio_obj=mrio_obj,
        year=year,
        dataset=dataset,
        selected_index=selected_index,
        extra_cfg=extra_cfg,
        return_metadata=False,
    )
    return E, S, e_bar, s_floor


def build_constraints_with_context(
    x: pd.Series,
    constraints_cfg: Mapping[str, Any],
    *,
    tol: float = 1e-12,
    mrio_obj: Any | None = None,
    year: int | None = None,
    dataset: str | None = None,
    selected_index: pd.Index | None = None,
    extra_cfg: Mapping[str, Any] | None = None,
    return_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any] | None]:
    """Build ecological/social linear constraints for a baseline output vector.

    Backward-compatible behavior:
    - proxy ecological/social rows still supported as before.

    Extension behavior:
    - ecological rows can use `source: extension_row` and a row selector.
    - ecological ceilings can be tied to budget factors (fixed or path-based).
    """
    x_series = pd.Series(x, copy=True).astype(float)
    if (x_series < -tol).any():
        raise ValueError("Output vector x must be nonnegative to build constraints.")
    if selected_index is not None:
        x_series = x_series.reindex(pd.Index(selected_index)).fillna(0.0).astype(float)

    extra = dict(extra_cfg or {})
    if "budget_paths_data" not in extra:
        extra["budget_paths_data"] = dict(constraints_cfg.get("_budget_paths_data", {}) or {})

    eco_rows: list[np.ndarray] = []
    eco_names: list[str] = []
    eco_bounds: list[float] = []
    eco_source_modes: list[str] = []
    eco_selector_notes: list[str] = []
    eco_row_labels: list[str] = []
    eco_budget_factors: list[float] = []
    eco_budget_ref_years: list[str] = []

    for item in constraints_cfg.get("ecological", []):
        source = str(item.get("source", "proxy")).strip().lower()
        eco_source_modes.append(source)
        name = str(item.get("name", f"eco_{len(eco_names)}"))
        if source == "proxy":
            w = _weights_from_mode(str(item["mode"]), x_series, tol=tol)
            cur = float(np.dot(w, x_series.to_numpy(dtype=float)))
            eco_rows.append(w)
            eco_names.append(name)
            eco_bounds.append(float(item["ceiling_share_of_current"]) * cur)
            eco_selector_notes.append("proxy_mode")
            eco_row_labels.append(name)
            eco_budget_factors.append(float(item.get("ceiling_share_of_current", np.nan)))
            eco_budget_ref_years.append("")
            continue

        if source == "extension_row":
            row, note = _select_extension_row(mrio_obj, item.get("selector"))
            row = pd.to_numeric(row.reindex(x_series.index), errors="coerce").fillna(0.0).astype(float)
            row = row.clip(lower=0.0)
            if float(row.sum()) <= tol:
                raise ValueError(
                    f"Selected extension row has near-zero support on selected sectors for ecological item '{name}'."
                )
            ceiling_mode = str(item.get("ceiling_mode", "budget_share_of_current")).strip().lower()
            if ceiling_mode != "budget_share_of_current":
                raise ValueError(f"Unsupported ceiling_mode='{ceiling_mode}' for ecological item '{name}'.")
            factor, ref_year = _resolve_budget_factor(item, dataset=dataset, year=year, extra_cfg=extra)
            cur = float(np.dot(row.to_numpy(dtype=float), x_series.to_numpy(dtype=float)))
            eco_rows.append(row.to_numpy(dtype=float))
            eco_names.append(name)
            eco_bounds.append(float(factor) * cur)
            eco_selector_notes.append(note)
            eco_row_labels.append(note.split(":")[-1] if ":" in note else note)
            eco_budget_factors.append(float(factor))
            eco_budget_ref_years.append("" if ref_year is None else str(ref_year))
            continue

        raise ValueError(f"Unknown ecological source='{source}' for item '{name}'.")

    soc_rows: list[np.ndarray] = []
    soc_names: list[str] = []
    soc_bounds: list[float] = []
    for item in constraints_cfg.get("social", []):
        source = str(item.get("source", "proxy")).strip().lower()
        if source != "proxy":
            raise ValueError(f"Unsupported social source='{source}' in current constraints builder.")
        w = _weights_from_mode(str(item["mode"]), x_series, tol=tol)
        cur = float(np.dot(w, x_series.to_numpy(dtype=float)))
        soc_rows.append(w)
        soc_names.append(str(item["name"]))
        soc_bounds.append(float(item["floor_share_of_current"]) * cur)

    E = pd.DataFrame(eco_rows, index=eco_names, columns=x_series.index, dtype=float)
    S = pd.DataFrame(soc_rows, index=soc_names, columns=x_series.index, dtype=float)
    e_bar = pd.Series(eco_bounds, index=eco_names, dtype=float)
    s_floor = pd.Series(soc_bounds, index=soc_names, dtype=float)
    eco_mode_unique = sorted(set([m for m in eco_source_modes if m]))
    eco_mode = eco_mode_unique[0] if len(eco_mode_unique) == 1 else ("mixed" if eco_mode_unique else "unknown")
    meta = {
        "eco_source_mode": eco_mode,
        "eco_selector_note": "|".join(eco_selector_notes),
        "eco_budget_factor": float(np.nanmean(eco_budget_factors)) if eco_budget_factors else np.nan,
        "eco_budget_ref_year": "|".join([x for x in eco_budget_ref_years if x]),
        "eco_row_label": "|".join([x for x in eco_row_labels if x]),
    }
    E.attrs["constraint_metadata"] = dict(meta)
    if return_metadata:
        return E, S, e_bar, s_floor, meta
    return E, S, e_bar, s_floor, None


def solve_beta(
    q: pd.Series,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    tol: float = 1e-12,
) -> BetaResult:
    """Solve max scalar beta along direction q under linear eco/social constraints.

    The feasible set is
    x = beta * q,
    x >= 0,
    E x <= e_bar,
    S x >= s_floor.
    """
    q_series = pd.Series(q, copy=True).astype(float)
    if q_series.empty:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q is empty.",
        )

    if (q_series < -tol).any():
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q has negative entries; x = beta*q with x>=0 is not admissible.",
        )

    q_vals = q_series.to_numpy(dtype=float)
    if float(np.clip(q_vals, 0.0, None).sum()) <= tol:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=np.nan,
            beta_upper_bound=np.nan,
            binding_eco=[],
            binding_soc=[],
            feasible=False,
            message="q has zero mass; direction scaling is undefined.",
        )

    E_mat = pd.DataFrame(E, copy=True).reindex(columns=q_series.index).fillna(0.0)
    S_mat = pd.DataFrame(S, copy=True).reindex(columns=q_series.index).fillna(0.0)
    e_vec = pd.Series(e_bar, copy=True).reindex(E_mat.index).astype(float)
    s_vec = pd.Series(s_floor, copy=True).reindex(S_mat.index).astype(float)

    lower = 0.0
    upper = float("inf")
    binding_eco: list[str] = []
    binding_soc: list[str] = []

    # Ecological ceilings: (E_i q) beta <= e_bar_i
    for name, row in E_mat.iterrows():
        denom = float(np.dot(row.to_numpy(dtype=float), q_vals))
        rhs = float(e_vec.loc[name])
        if abs(denom) <= tol:
            if rhs < -tol:
                return BetaResult(
                    beta=np.nan,
                    x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
                    beta_lower_bound=lower,
                    beta_upper_bound=upper,
                    binding_eco=[str(name)],
                    binding_soc=binding_soc,
                    feasible=False,
                    message=(
                        f"Infeasible ecological constraint '{name}': zero denominator with negative ceiling."
                    ),
                )
            continue
        candidate = rhs / denom
        if denom > 0:
            if candidate < upper:
                upper = candidate
        else:
            lower = max(lower, candidate)

    # Social floors: (S_j q) beta >= s_floor_j
    for name, row in S_mat.iterrows():
        denom = float(np.dot(row.to_numpy(dtype=float), q_vals))
        rhs = float(s_vec.loc[name])
        if abs(denom) <= tol:
            if rhs > tol:
                return BetaResult(
                    beta=np.nan,
                    x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
                    beta_lower_bound=lower,
                    beta_upper_bound=upper,
                    binding_eco=binding_eco,
                    binding_soc=[str(name)],
                    feasible=False,
                    message=(
                        f"Infeasible social constraint '{name}': zero denominator with positive floor."
                    ),
                )
            continue
        candidate = rhs / denom
        if denom > 0:
            lower = max(lower, candidate)
        else:
            if candidate < upper:
                upper = candidate

    if lower - upper > tol:
        return BetaResult(
            beta=np.nan,
            x_beta=pd.Series(np.nan, index=q_series.index, dtype=float),
            beta_lower_bound=lower,
            beta_upper_bound=upper,
            binding_eco=sorted(set(binding_eco)),
            binding_soc=sorted(set(binding_soc)),
            feasible=False,
            message="No feasible beta satisfies both lower and upper bounds.",
        )

    if np.isfinite(upper):
        beta = float(upper)
        msg = "Optimal beta found at upper bound."
    else:
        beta = float("inf")
        msg = "Beta is unbounded above under supplied constraints."

    x_beta = q_series * beta

    binding_eco = []
    for name, row in E_mat.iterrows():
        lhs = float(np.dot(row.to_numpy(dtype=float), x_beta.to_numpy(dtype=float)))
        rhs = float(e_vec.loc[name])
        if abs(lhs - rhs) <= max(tol, tol * max(1.0, abs(rhs))):
            binding_eco.append(str(name))

    binding_soc = []
    for name, row in S_mat.iterrows():
        lhs = float(np.dot(row.to_numpy(dtype=float), x_beta.to_numpy(dtype=float)))
        rhs = float(s_vec.loc[name])
        if abs(lhs - rhs) <= max(tol, tol * max(1.0, abs(rhs))):
            binding_soc.append(str(name))

    return BetaResult(
        beta=beta,
        x_beta=x_beta,
        beta_lower_bound=float(lower),
        beta_upper_bound=float(upper),
        binding_eco=sorted(set(binding_eco)),
        binding_soc=sorted(set(binding_soc)),
        feasible=True,
        message=msg,
    )
