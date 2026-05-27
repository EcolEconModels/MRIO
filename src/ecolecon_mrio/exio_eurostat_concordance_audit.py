from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_REL_THRESHOLD = 0.20
DEFAULT_MIN_ABS_MEUR = 5.0
DEFAULT_SUBCOMPONENT_RATIO = 1.20
EPS = 1e-12

METRIC_X_OUTPUT = "X_OUTPUT"
METRIC_Y_FINAL_DEMAND_TOTAL = "Y_FINAL_DEMAND_TOTAL"
METRIC_Z_ROW_INTERMEDIATE_SALES = "Z_ROW_INTERMEDIATE_SALES"
METRIC_Z_COL_INTERMEDIATE_PURCHASES = "Z_COL_INTERMEDIATE_PURCHASES"
METRIC_TOTAL_USE = "TOTAL_USE_Z_PLUS_Y"
METRIC_Z_CELL = "Z_CELL"


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().replace("\u00a0", " ").split())


def _parse_boolish(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and np.isnan(value):
        return True
    s = str(value).strip().lower()
    if s == "":
        return True
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _to_float_series(values: pd.Series | Mapping[str, Any] | None, *, name: str) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64", name=name)
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values)
    series.index = series.index.astype(str)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float64")


def _to_float_frame(frame: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(dtype="float64")
    out = frame.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("inf") if numerator > 0 else 0.0
    return float(numerator / denominator)


def _relative_gap(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), EPS)


def _severity(rel_gap: float) -> str:
    if rel_gap >= 1.0:
        return "CRITICAL"
    if rel_gap >= 0.5:
        return "HIGH"
    if rel_gap >= DEFAULT_REL_THRESHOLD:
        return "MEDIUM"
    return "OK"


def _metric_issue_type(metric: str, *, is_flagged: bool, low_value_noise: bool = False) -> str:
    if low_value_noise:
        return "LOW_VALUE_RELATIVE_NOISE"
    if not is_flagged:
        return "LOW_VALUE_RELATIVE_NOISE"
    if metric == METRIC_Y_FINAL_DEMAND_TOTAL:
        return "FINAL_DEMAND_ALLOCATION_DIFFERENCE"
    if metric in {METRIC_Z_ROW_INTERMEDIATE_SALES, METRIC_Z_COL_INTERMEDIATE_PURCHASES}:
        return "INTERMEDIATE_USE_ALLOCATION_DIFFERENCE"
    if metric in {METRIC_X_OUTPUT, METRIC_TOTAL_USE}:
        return "OUTPUT_LEVEL_DIFFERENCE"
    if metric == METRIC_Z_CELL:
        return "DATABASE_CLASSIFICATION_DIFFERENCE"
    return "UNKNOWN_REQUIRES_MANUAL_REVIEW"


def _combine_texts(values: pd.Series) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str):
        text = value.strip()
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    return " | ".join(parts)


def _read_label_map(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.shape[1] < 2:
        raise ValueError(f"Label dictionary {path} must contain at least two columns.")
    code_col, label_col = frame.columns[:2]
    return dict(zip(frame[code_col].astype(str), frame[label_col].astype(str), strict=False))


def _top_contributors_string(
    contributions: pd.Series,
    *,
    label_map: Mapping[str, str],
    top_n: int = 3,
) -> str:
    if contributions.empty:
        return ""
    ordered = contributions.astype("float64").sort_values(ascending=False)
    total = float(ordered.sum())
    pieces: list[str] = []
    for code, value in ordered.head(top_n).items():
        label = label_map.get(str(code), str(code))
        share = _safe_ratio(float(value), total)
        pieces.append(f"{code} {label}: {value:.3f} ({share:.1%})")
    return " | ".join(pieces)


def _markdown_table(df: pd.DataFrame, *, columns: Sequence[str], max_rows: int | None = None) -> str:
    view = df.loc[:, list(columns)].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    rows: list[str] = []
    headers = [str(col) for col in view.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("")
                else:
                    cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _identity_metric_rows(
    *,
    database: str,
    country: str,
    year: int,
    x: pd.Series,
    z: pd.DataFrame,
    y: pd.DataFrame,
    identity_mode: str,
) -> pd.DataFrame:
    x = _to_float_series(x, name="x")
    z = _to_float_frame(z, name="Z")
    y = _to_float_frame(y, name="Y")

    row_total = z.sum(axis=1).reindex(x.index).fillna(0.0)
    y_total = y.sum(axis=1).reindex(x.index).fillna(0.0)
    z_plus_y = row_total + y_total

    rows: list[dict[str, Any]] = []
    for code in x.index.union(z.index).union(y.index):
        x_value = float(x.get(code, 0.0))
        row_value = float(row_total.get(code, 0.0))
        y_value = float(y_total.get(code, 0.0))
        total_value = float(z_plus_y.get(code, 0.0))
        abs_gap = abs(x_value - total_value)
        rel_gap = _relative_gap(x_value, total_value)
        rows.append(
            {
                "database": database,
                "country": country,
                "year": year,
                "code": code,
                "x": x_value,
                "z_row_total": row_value,
                "y_total": y_value,
                "z_plus_y": total_value,
                "signed_gap": x_value - total_value,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                "severity": _severity(rel_gap),
                "identity_mode": identity_mode,
                "status": "OK" if abs_gap <= EPS else "INTERNAL_IDENTITY_FAIL",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _build_mapping_weights(mapping_df: pd.DataFrame) -> pd.DataFrame:
    df = mapping_df.copy().fillna("")
    required = {"source_code", "target_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping frame missing required columns: {sorted(missing)}")

    enabled = df["enabled"].map(_parse_boolish) if "enabled" in df.columns else pd.Series(True, index=df.index)
    df = df[enabled].copy()
    if df.empty:
        raise ValueError("Mapping frame is empty after enabled filtering.")

    df["source_code"] = df["source_code"].astype(str).str.strip()
    df["target_code"] = df["target_code"].astype(str).str.strip()
    if "source_label" not in df.columns:
        df["source_label"] = ""
    if "target_label" not in df.columns:
        df["target_label"] = ""
    df["source_label"] = df["source_label"].astype(str).str.strip()
    df["target_label"] = df["target_label"].astype(str).str.strip()

    if "weight" in df.columns:
        df["raw_weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        explicit_weight = True
    elif "Share_used" in df.columns:
        df["raw_weight"] = pd.to_numeric(df["Share_used"], errors="coerce").fillna(0.0)
        explicit_weight = True
    else:
        df["raw_weight"] = 1.0
        explicit_weight = False

    agg_map: dict[str, Any] = {
        "raw_weight": ("raw_weight", "sum"),
        "source_label": ("source_label", _combine_texts),
        "target_label": ("target_label", _combine_texts),
    }
    if "notes" in df.columns:
        agg_map["notes"] = ("notes", _combine_texts)
    grouped = df.groupby(["source_code", "target_code"], as_index=False).agg(**agg_map).copy()

    if explicit_weight:
        grouped["weight"] = grouped["raw_weight"]
    else:
        totals = grouped.groupby("source_code")["raw_weight"].transform("sum")
        grouped["weight"] = np.where(totals > 0, grouped["raw_weight"] / totals, 0.0)

    if "notes" not in grouped.columns:
        grouped["notes"] = ""
    return grouped.loc[:, ["source_code", "target_code", "weight", "source_label", "target_label", "notes"]].copy()


def _build_mapping_matrix(
    mapping_weights: pd.DataFrame,
    *,
    source_codes: Sequence[str],
    target_codes: Sequence[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=pd.Index(list(target_codes), name="target_code"), columns=pd.Index(list(source_codes), name="source_code"))
    for row in mapping_weights.itertuples(index=False):
        target = str(row.target_code)
        source = str(row.source_code)
        if target in matrix.index and source in matrix.columns:
            matrix.loc[target, source] = float(row.weight)
    return matrix


def _source_metrics(z: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    z = _to_float_frame(z, name="Z")
    y = _to_float_frame(y, name="Y")
    row_total = z.sum(axis=1)
    col_total = z.sum(axis=0)
    y_total = y.sum(axis=1)
    return row_total, col_total, y_total


def _mapped_series(mapping_matrix: pd.DataFrame, source_values: pd.Series) -> pd.Series:
    source_values = _to_float_series(source_values, name="source_values")
    source_values = source_values.reindex(mapping_matrix.columns).fillna(0.0)
    mapped = mapping_matrix.dot(source_values)
    mapped.index = mapped.index.astype(str)
    return mapped.astype("float64")


def _mapped_matrix(mapping_matrix: pd.DataFrame, source_matrix: pd.DataFrame) -> pd.DataFrame:
    source_matrix = _to_float_frame(source_matrix, name="source_matrix")
    source_matrix = source_matrix.reindex(index=mapping_matrix.columns, columns=mapping_matrix.columns).fillna(0.0)
    mapped = mapping_matrix.dot(source_matrix).dot(mapping_matrix.T)
    mapped.index = mapped.index.astype(str)
    mapped.columns = mapped.columns.astype(str)
    return mapped.astype("float64")


def _series_contributions(
    mapping_weights: pd.DataFrame,
    source_values: pd.Series,
    *,
    target_code: str,
) -> pd.Series:
    df = mapping_weights[mapping_weights["target_code"] == target_code].copy()
    if df.empty:
        return pd.Series(dtype="float64")
    source_values = _to_float_series(source_values, name="source_values")
    df["source_value"] = df["source_code"].map(source_values).fillna(0.0)
    df["contribution"] = df["weight"] * df["source_value"]
    contributions = df.groupby("source_code")["contribution"].sum().sort_values(ascending=False)
    return contributions


def _cell_source_contributions(
    mapping_matrix: pd.DataFrame,
    source_Z: pd.DataFrame,
    *,
    supplier_code: str,
    buyer_code: str,
) -> pd.Series:
    source_Z = _to_float_frame(source_Z, name="Z")
    source_Z = source_Z.reindex(index=mapping_matrix.columns, columns=mapping_matrix.columns).fillna(0.0)
    supplier_weights = mapping_matrix.loc[supplier_code]
    buyer_weights = mapping_matrix.loc[buyer_code]
    contrib: dict[str, float] = {}
    for source_i in mapping_matrix.columns:
        w_i = float(supplier_weights.get(source_i, 0.0))
        if w_i == 0:
            continue
        value = float((source_Z.loc[source_i] * buyer_weights).sum()) * w_i
        if value != 0:
            contrib[str(source_i)] = value
    return pd.Series(contrib, dtype="float64").sort_values(ascending=False)


def _cell_buyer_contributions(
    mapping_matrix: pd.DataFrame,
    source_Z: pd.DataFrame,
    *,
    supplier_code: str,
    buyer_code: str,
) -> pd.Series:
    source_Z = _to_float_frame(source_Z, name="Z")
    source_Z = source_Z.reindex(index=mapping_matrix.columns, columns=mapping_matrix.columns).fillna(0.0)
    buyer_weights = mapping_matrix.loc[buyer_code]
    contrib: dict[str, float] = {}
    for source_j in mapping_matrix.columns:
        w_j = float(buyer_weights.get(source_j, 0.0))
        if w_j == 0:
            continue
        value = float((mapping_matrix.loc[supplier_code] * source_Z[source_j]).sum()) * w_j
        if value != 0:
            contrib[str(source_j)] = value
    return pd.Series(contrib, dtype="float64").sort_values(ascending=False)


def _classify_mapping_status(
    *,
    source_code: str,
    mapped_targets: int,
    sum_mapping_weight: float,
    source_value: float,
    unknown_target_count: int,
    duplicate_rows: bool,
    tolerance: float = 1e-9,
) -> tuple[str, str]:
    reasons: list[str] = []
    status_flags: list[str] = []
    if duplicate_rows:
        status_flags.append("DUPLICATE_MAPPING")
        reasons.append("duplicate source-target rows present")
    if unknown_target_count > 0:
        status_flags.append("UNKNOWN_TARGET_CODE")
        reasons.append(f"{unknown_target_count} unknown target code(s)")
    if mapped_targets == 0:
        if abs(source_value) > EPS:
            status_flags.append("UNMAPPED_NONZERO")
            reasons.append("source has non-zero value but no mapped targets")
        else:
            status_flags.append("UNMAPPED_ZERO")
            reasons.append("source has zero value and no mapped targets")
    elif sum_mapping_weight > 1.0 + tolerance:
        status_flags.append("OVERMAPPED_WEIGHT_GT_1")
        reasons.append(f"mapping weights sum to {sum_mapping_weight:.3f} > 1")
    elif sum_mapping_weight < 1.0 - tolerance:
        status_flags.append("PARTIALLY_MAPPED")
        reasons.append(f"mapping weights sum to {sum_mapping_weight:.3f} < 1")
    elif mapped_targets > 1:
        status_flags.append("PARTIALLY_MAPPED")
        reasons.append(f"source mapped to {mapped_targets} targets")
    else:
        status_flags.append("OK")
        reasons.append("single-target full mapping")
    return "; ".join(dict.fromkeys(status_flags)), "; ".join(reasons)


def _mapping_coverage_table(
    *,
    mapping_weights: pd.DataFrame,
    exio_source_values: pd.Series,
    exio_row_total: pd.Series,
    exio_col_total: pd.Series,
    exio_y_total: pd.Series,
    exio_labels: Mapping[str, str],
    euro_target_codes: set[str],
) -> pd.DataFrame:
    mapping_weights = mapping_weights.copy()
    duplicate_pairs = mapping_weights.duplicated(subset=["source_code", "target_code"], keep=False)
    duplicate_sources = set(mapping_weights.loc[duplicate_pairs, "source_code"].astype(str))
    rows: list[dict[str, Any]] = []
    all_sources = sorted(set(map(str, exio_source_values.index)).union(set(map(str, mapping_weights["source_code"]))))
    for source_code in all_sources:
        rows_for_source = mapping_weights[mapping_weights["source_code"] == source_code].copy()
        target_codes = rows_for_source["target_code"].astype(str).tolist()
        target_labels = rows_for_source["target_label"].astype(str).tolist()
        unknown_targets = [code for code in target_codes if code not in euro_target_codes]
        mapped_targets = len([code for code in target_codes if code in euro_target_codes])
        mapped_weight = float(rows_for_source.loc[rows_for_source["target_code"].isin(euro_target_codes), "weight"].sum())
        source_value = float(exio_source_values.get(source_code, 0.0))
        status, reason = _classify_mapping_status(
            source_code=source_code,
            mapped_targets=mapped_targets,
            sum_mapping_weight=mapped_weight,
            source_value=source_value,
            unknown_target_count=len(unknown_targets),
            duplicate_rows=source_code in duplicate_sources,
        )
        rows.append(
            {
                "source_code": source_code,
                "source_label": exio_labels.get(source_code, source_code),
                "mapped_targets": mapped_targets,
                "sum_mapping_weight": mapped_weight,
                "status": status,
                "x_value": source_value,
                "y_value": float(exio_y_total.get(source_code, 0.0)),
                "z_row_total": float(exio_row_total.get(source_code, 0.0)),
                "z_col_total": float(exio_col_total.get(source_code, 0.0)),
                "target_codes": " | ".join(target_codes),
                "target_labels": " | ".join(target_labels),
                "unknown_target_codes": " | ".join(unknown_targets),
                "reason": reason,
                "large_value_in_ambiguous_mapping": bool(mapped_targets > 1 and source_value >= DEFAULT_MIN_ABS_MEUR),
            }
        )
    out = pd.DataFrame(rows)
    out["status"] = out["status"].astype(str)
    return out.sort_values(["status", "source_code"], kind="stable").reset_index(drop=True)


def _aggregate_comparison_table(
    *,
    metric: str,
    exio_mapped: pd.Series,
    eurostat: pd.Series,
    exio_contrib_lookup: Mapping[str, pd.Series] | None,
    exio_labels: Mapping[str, str],
    euro_labels: Mapping[str, str],
    rel_threshold: float,
    min_abs_meur: float,
) -> pd.DataFrame:
    exio_mapped = _to_float_series(exio_mapped, name="exio_mapped")
    eurostat = _to_float_series(eurostat, name="eurostat")
    codes = sorted(set(exio_mapped.index).union(set(eurostat.index)))
    rows: list[dict[str, Any]] = []
    for code in codes:
        exio_value = float(exio_mapped.get(code, 0.0))
        euro_value = float(eurostat.get(code, 0.0))
        signed_gap = exio_value - euro_value
        abs_gap = abs(signed_gap)
        rel_gap = _relative_gap(exio_value, euro_value)
        flagged = abs_gap >= min_abs_meur and rel_gap >= rel_threshold
        top_contribs = ""
        if exio_contrib_lookup is not None and code in exio_contrib_lookup:
            top_contribs = _top_contributors_string(exio_contrib_lookup[code], label_map=exio_labels)
        rows.append(
            {
                "target_code": code,
                "target_label": euro_labels.get(code, exio_labels.get(code, code)),
                "metric": metric,
                "exio_mapped_value": exio_value,
                "eurostat_value": euro_value,
                "signed_gap": signed_gap,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                "ratio_exio_to_eurostat": _safe_ratio(exio_value, euro_value),
                "flag": flagged,
                "severity": _severity(rel_gap),
                "top_exio_contributors": top_contribs,
                "issue_type": _metric_issue_type(metric, is_flagged=flagged, low_value_noise=abs_gap < min_abs_meur),
                "interpretation": "" if not flagged else _explain_aggregate_mismatch(metric, exio_value, euro_value, rel_gap, top_contribs),
            }
        )
    return pd.DataFrame(rows).sort_values(["flag", "severity", "abs_gap", "rel_gap"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)


def _explain_aggregate_mismatch(metric: str, exio_value: float, euro_value: float, rel_gap: float, top_contribs: str) -> str:
    if metric == METRIC_X_OUTPUT:
        base = "Mapped EXIO output differs from Eurostat output."
    elif metric == METRIC_Y_FINAL_DEMAND_TOTAL:
        base = "Mapped EXIO final-demand total differs from Eurostat final demand."
    elif metric in {METRIC_Z_ROW_INTERMEDIATE_SALES, METRIC_Z_COL_INTERMEDIATE_PURCHASES}:
        base = "Mapped EXIO intermediate-use total differs from Eurostat."
    elif metric == METRIC_TOTAL_USE:
        base = "Mapped EXIO total use differs from Eurostat total use."
    else:
        base = "Mapped EXIO and Eurostat differ."
    direction = "higher" if exio_value > euro_value else "lower"
    detail = f"Mapped EXIO is {direction} than Eurostat by {rel_gap:.1%}."
    if top_contribs:
        detail = f"{detail} Top EXIO contributors: {top_contribs}."
    return f"{base} {detail}"


def _explain_cell_mismatch(exio_value: float, euro_value: float, source_contribs: pd.Series, label_map: Mapping[str, str]) -> str:
    direction = "higher" if exio_value > euro_value else "lower"
    top = _top_contributors_string(source_contribs, label_map=label_map)
    if top:
        return f"Mapped EXIO cell is {direction} than Eurostat. Top source contributors: {top}."
    return f"Mapped EXIO cell is {direction} than Eurostat."


def _explain_subcomponent_exceedance(*, contribution_value: float, euro_value: float, source_code: str, source_label: str) -> str:
    ratio = _safe_ratio(contribution_value, euro_value)
    if np.isinf(ratio):
        return f"{source_code} {source_label} contributes to a Eurostat-zero cell with a positive mapped value."
    return f"{source_code} {source_label} contributes {ratio:.1f}x the Eurostat cell value."


def _z_cell_table(
    *,
    exio_mapped_Z: pd.DataFrame,
    eurostat_Z: pd.DataFrame,
    mapping_matrix: pd.DataFrame,
    source_Z: pd.DataFrame,
    exio_labels: Mapping[str, str],
    euro_labels: Mapping[str, str],
    rel_threshold: float,
    min_abs_meur: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exio_mapped_Z = _to_float_frame(exio_mapped_Z, name="mapped_Z")
    eurostat_Z = _to_float_frame(eurostat_Z, name="eurostat_Z")
    codes = sorted(set(exio_mapped_Z.index).union(set(eurostat_Z.index)))
    exio_mapped_Z = exio_mapped_Z.reindex(index=codes, columns=codes).fillna(0.0)
    eurostat_Z = eurostat_Z.reindex(index=codes, columns=codes).fillna(0.0)

    rows: list[dict[str, Any]] = []
    drilldown_rows: list[dict[str, Any]] = []
    cell_columns = [
        "supplier_code",
        "supplier_label",
        "buyer_code",
        "buyer_label",
        "exio_mapped_z",
        "eurostat_z",
        "signed_gap",
        "abs_gap",
        "rel_gap",
        "ratio_exio_to_eurostat",
        "flag",
        "severity",
        "top_source_subsectors",
        "top_buyer_subsectors",
        "interpretation",
    ]
    drilldown_columns = [
        "supplier_code",
        "supplier_label",
        "buyer_code",
        "buyer_label",
        "exio_subsector_code",
        "exio_subsector_label",
        "role",
        "contribution_value",
        "share_of_exio_mapped_cell",
        "share_of_eurostat_cell",
        "subsector_exceeds_eurostat_cell",
        "contribution_rank",
        "interpretation",
    ]
    for supplier_code in codes:
        for buyer_code in codes:
            exio_value = float(exio_mapped_Z.loc[supplier_code, buyer_code])
            euro_value = float(eurostat_Z.loc[supplier_code, buyer_code])
            signed_gap = exio_value - euro_value
            abs_gap = abs(signed_gap)
            rel_gap = _relative_gap(exio_value, euro_value)
            flagged = abs_gap >= min_abs_meur and rel_gap >= rel_threshold
            source_contribs = pd.Series(dtype="float64")
            buyer_contribs = pd.Series(dtype="float64")
            if flagged:
                source_contribs = _cell_source_contributions(
                    mapping_matrix,
                    source_Z,
                    supplier_code=supplier_code,
                    buyer_code=buyer_code,
                )
                buyer_contribs = _cell_buyer_contributions(
                    mapping_matrix,
                    source_Z,
                    supplier_code=supplier_code,
                    buyer_code=buyer_code,
                )
            rows.append(
                {
                    "supplier_code": supplier_code,
                    "supplier_label": euro_labels.get(supplier_code, exio_labels.get(supplier_code, supplier_code)),
                    "buyer_code": buyer_code,
                    "buyer_label": euro_labels.get(buyer_code, exio_labels.get(buyer_code, buyer_code)),
                    "exio_mapped_z": exio_value,
                    "eurostat_z": euro_value,
                    "signed_gap": signed_gap,
                    "abs_gap": abs_gap,
                    "rel_gap": rel_gap,
                    "ratio_exio_to_eurostat": _safe_ratio(exio_value, euro_value),
                    "flag": flagged,
                    "severity": _severity(rel_gap),
                    "top_source_subsectors": _top_contributors_string(source_contribs, label_map=exio_labels),
                    "top_buyer_subsectors": _top_contributors_string(buyer_contribs, label_map=exio_labels),
                    "interpretation": "" if not flagged else _explain_cell_mismatch(exio_value, euro_value, source_contribs, exio_labels),
                }
            )
            if flagged:
                for role, contribs in (("source_contributor", source_contribs), ("buyer_contributor", buyer_contribs)):
                    if contribs.empty:
                        continue
                    for rank, (source_code, contribution_value) in enumerate(contribs.items(), start=1):
                        drilldown_rows.append(
                            {
                                "supplier_code": supplier_code,
                                "supplier_label": euro_labels.get(supplier_code, exio_labels.get(supplier_code, supplier_code)),
                                "buyer_code": buyer_code,
                                "buyer_label": euro_labels.get(buyer_code, exio_labels.get(buyer_code, buyer_code)),
                                "exio_subsector_code": source_code,
                                "exio_subsector_label": exio_labels.get(source_code, source_code),
                                "role": role,
                                "contribution_value": float(contribution_value),
                                "share_of_exio_mapped_cell": _safe_ratio(float(contribution_value), exio_value),
                                "share_of_eurostat_cell": _safe_ratio(float(contribution_value), euro_value),
                                "subsector_exceeds_eurostat_cell": bool(float(contribution_value) > (euro_value * DEFAULT_SUBCOMPONENT_RATIO)),
                                "contribution_rank": rank,
                                "interpretation": _explain_subcomponent_exceedance(
                                    contribution_value=float(contribution_value),
                                    euro_value=euro_value,
                                    source_code=source_code,
                                    source_label=exio_labels.get(source_code, source_code),
                                ),
                            }
                        )
    cell_frame = pd.DataFrame(rows, columns=cell_columns)
    drilldown_frame = pd.DataFrame(drilldown_rows, columns=drilldown_columns)
    if not cell_frame.empty:
        cell_frame = cell_frame.sort_values(["flag", "severity", "abs_gap", "rel_gap"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)
    if not drilldown_frame.empty:
        drilldown_frame = drilldown_frame.sort_values(["supplier_code", "buyer_code", "role", "contribution_rank"], kind="stable").reset_index(drop=True)
    return cell_frame, drilldown_frame


def _subcomponent_exceeds_aggregate_table(
    *,
    mapping_weights: pd.DataFrame,
    mapping_matrix: pd.DataFrame,
    source_x: pd.Series,
    source_y_total: pd.Series,
    source_row_total: pd.Series,
    source_Z: pd.DataFrame,
    eurostat_x: pd.Series,
    eurostat_y_total: pd.Series,
    eurostat_row_total: pd.Series,
    eurostat_Z: pd.DataFrame,
    z_cell_drilldown: pd.DataFrame,
    exio_labels: Mapping[str, str],
    euro_labels: Mapping[str, str],
    rel_threshold: float,
    min_abs_meur: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "metric",
        "target_code",
        "target_label",
        "buyer_code",
        "buyer_label",
        "exio_subsector_code",
        "exio_subsector_label",
        "exio_subsector_value",
        "eurostat_aggregate_value",
        "ratio",
        "abs_gap",
        "severity",
        "issue_type",
        "interpretation",
    ]
    for metric, source_values, euro_values in (
        (METRIC_X_OUTPUT, source_x, eurostat_x),
        (METRIC_Y_FINAL_DEMAND_TOTAL, source_y_total, eurostat_y_total),
        (METRIC_Z_ROW_INTERMEDIATE_SALES, source_row_total, eurostat_row_total),
    ):
        for row in mapping_weights.itertuples(index=False):
            if row.target_code not in euro_values.index:
                continue
            source_code = str(row.source_code)
            target_code = str(row.target_code)
            source_value = float(source_values.get(source_code, 0.0)) * float(row.weight)
            euro_value = float(euro_values.get(target_code, 0.0))
            abs_gap = abs(source_value - euro_value)
            rel_gap = _relative_gap(source_value, euro_value)
            if source_value > euro_value * DEFAULT_SUBCOMPONENT_RATIO and abs_gap >= min_abs_meur and rel_gap >= rel_threshold:
                rows.append(
                    {
                        "metric": metric,
                        "target_code": target_code,
                        "target_label": euro_labels.get(target_code, target_code),
                        "buyer_code": "",
                        "buyer_label": "",
                        "exio_subsector_code": source_code,
                        "exio_subsector_label": exio_labels.get(source_code, source_code),
                        "exio_subsector_value": source_value,
                        "eurostat_aggregate_value": euro_value,
                        "ratio": _safe_ratio(source_value, euro_value),
                        "abs_gap": abs_gap,
                        "severity": _severity(rel_gap),
                        "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                        "interpretation": f"{source_code} {exio_labels.get(source_code, source_code)} exceeds the Eurostat aggregate for {target_code}.",
                    }
                )

    exio_mapped_Z = _mapped_matrix(mapping_matrix, source_Z)
    exio_mapped_Z = exio_mapped_Z.reindex(index=eurostat_Z.index, columns=eurostat_Z.columns).fillna(0.0)
    for supplier_code in eurostat_Z.index.astype(str):
        for buyer_code in eurostat_Z.columns.astype(str):
            euro_value = float(eurostat_Z.loc[supplier_code, buyer_code])
            source_contribs = _cell_source_contributions(
                mapping_matrix,
                source_Z,
                supplier_code=supplier_code,
                buyer_code=buyer_code,
            )
            if source_contribs.empty:
                continue
            for source_code, contribution_value in source_contribs.items():
                abs_gap = abs(float(contribution_value) - euro_value)
                rel_gap = _relative_gap(float(contribution_value), euro_value)
                if float(contribution_value) > euro_value * DEFAULT_SUBCOMPONENT_RATIO and abs_gap >= min_abs_meur and rel_gap >= rel_threshold:
                    rows.append(
                        {
                            "metric": METRIC_Z_CELL,
                            "target_code": supplier_code,
                            "target_label": euro_labels.get(supplier_code, supplier_code),
                            "buyer_code": buyer_code,
                            "buyer_label": euro_labels.get(buyer_code, buyer_code),
                            "exio_subsector_code": source_code,
                            "exio_subsector_label": exio_labels.get(source_code, source_code),
                            "exio_subsector_value": float(contribution_value),
                            "eurostat_aggregate_value": euro_value,
                            "ratio": _safe_ratio(float(contribution_value), euro_value),
                            "abs_gap": abs_gap,
                            "severity": _severity(rel_gap),
                            "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                            "interpretation": f"{source_code} {exio_labels.get(source_code, source_code)} exceeds the Eurostat cell {supplier_code}→{buyer_code}.",
                        }
                    )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["metric", "severity", "abs_gap", "ratio"], ascending=[True, False, False, False], kind="stable").reset_index(drop=True)
    return frame


@dataclass(frozen=True)
class ConcordanceAuditResult:
    metadata: dict[str, Any]
    mapping_coverage: pd.DataFrame
    internal_identities: pd.DataFrame
    aggregate_totals: pd.DataFrame
    x_mismatches: pd.DataFrame
    y_mismatches: pd.DataFrame
    z_row_col_mismatches: pd.DataFrame
    z_cell_mismatches: pd.DataFrame
    z_cell_drilldown: pd.DataFrame
    y_drilldown: pd.DataFrame
    subcomponent_exceeds_aggregate: pd.DataFrame
    report_markdown: str


def build_concordance_audit(
    *,
    country: str,
    year: int,
    exio_x: pd.Series,
    exio_Z: pd.DataFrame,
    exio_Y: pd.DataFrame,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    mapping_df: pd.DataFrame,
    exio_labels: Mapping[str, str],
    eurostat_labels: Mapping[str, str],
    eurostat_x: pd.Series | None = None,
    metadata: Mapping[str, Any] | None = None,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    min_abs_meur: float = DEFAULT_MIN_ABS_MEUR,
    identity_mode: str = "diagnostic",
) -> ConcordanceAuditResult:
    exio_x = _to_float_series(exio_x, name="exio_x")
    exio_Z = _to_float_frame(exio_Z, name="exio_Z")
    exio_Y = _to_float_frame(exio_Y, name="exio_Y")
    eurostat_Z = _to_float_frame(eurostat_Z, name="eurostat_Z")
    eurostat_Y = _to_float_frame(eurostat_Y, name="eurostat_Y")
    eurostat_x = _to_float_series(eurostat_x, name="eurostat_x") if eurostat_x is not None else (eurostat_Z.sum(axis=1) + eurostat_Y.sum(axis=1)).astype("float64")

    exio_row_total, exio_col_total, exio_y_total = _source_metrics(exio_Z, exio_Y)
    euro_row_total, euro_col_total, euro_y_total = _source_metrics(eurostat_Z, eurostat_Y)

    mapping_weights = _build_mapping_weights(mapping_df)
    source_codes = list(exio_Z.index.astype(str))
    target_codes = list(eurostat_Z.index.astype(str))
    mapping_matrix = _build_mapping_matrix(mapping_weights, source_codes=source_codes, target_codes=target_codes)

    mapped_x = _mapped_series(mapping_matrix, exio_x)
    mapped_y = _mapped_series(mapping_matrix, exio_y_total)
    flagged_cells = z_cell_drilldown[z_cell_drilldown["subsector_exceeds_eurostat_cell"].fillna(False)].copy()
    if not flagged_cells.empty:
        for row in flagged_cells.itertuples(index=False):
            euro_value = float(eurostat_Z.loc[str(row.supplier_code), str(row.buyer_code)])
            contribution_value = float(row.contribution_value)
            abs_gap = abs(contribution_value - euro_value)
            rel_gap = _relative_gap(contribution_value, euro_value)
            rows.append(
                {
                    "metric": METRIC_Z_CELL,
                    "target_code": str(row.supplier_code),
                    "target_label": euro_labels.get(str(row.supplier_code), str(row.supplier_code)),
                    "buyer_code": str(row.buyer_code),
                    "buyer_label": euro_labels.get(str(row.buyer_code), str(row.buyer_code)),
                    "exio_subsector_code": str(row.exio_subsector_code),
                    "exio_subsector_label": exio_labels.get(str(row.exio_subsector_code), str(row.exio_subsector_code)),
                    "exio_subsector_value": contribution_value,
                    "eurostat_aggregate_value": euro_value,
                    "ratio": _safe_ratio(contribution_value, euro_value),
                    "abs_gap": abs_gap,
                    "severity": _severity(rel_gap),
                    "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                    "interpretation": f"{row.exio_subsector_code} {exio_labels.get(str(row.exio_subsector_code), str(row.exio_subsector_code))} exceeds the Eurostat cell {row.supplier_code}→{row.buyer_code}.",
                }
            )
    row_contrib_lookup = {code: _series_contributions(mapping_weights, exio_row_total, target_code=code) for code in mapped_z.index}
    col_contrib_lookup = {code: _series_contributions(mapping_weights, exio_col_total, target_code=code) for code in mapped_z.columns}

    x_comparison = _aggregate_comparison_table(
        metric=METRIC_X_OUTPUT,
        exio_mapped=mapped_x,
        eurostat=eurostat_x,
        exio_contrib_lookup=x_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    y_comparison = _aggregate_comparison_table(
        metric=METRIC_Y_FINAL_DEMAND_TOTAL,
        exio_mapped=mapped_y,
        eurostat=euro_y_total,
        exio_contrib_lookup=y_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    row_comparison = _aggregate_comparison_table(
        metric=METRIC_Z_ROW_INTERMEDIATE_SALES,
        exio_mapped=mapped_z.sum(axis=1),
        eurostat=euro_row_total,
        exio_contrib_lookup=row_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    col_comparison = _aggregate_comparison_table(
        metric=METRIC_Z_COL_INTERMEDIATE_PURCHASES,
        exio_mapped=mapped_z.sum(axis=0),
        eurostat=euro_col_total,
        exio_contrib_lookup=col_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    total_use_comparison = _aggregate_comparison_table(
        metric=METRIC_TOTAL_USE,
        exio_mapped=mapped_z.sum(axis=1) + mapped_y,
        eurostat=euro_row_total + euro_y_total,
        exio_contrib_lookup=row_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )

    aggregate_totals = pd.concat([x_comparison, y_comparison, row_comparison, col_comparison, total_use_comparison], ignore_index=True)
    z_row_col_mismatches = pd.concat([row_comparison, col_comparison], ignore_index=True)
    x_mismatches = x_comparison[x_comparison["flag"]].copy().reset_index(drop=True)
    y_mismatches = y_comparison[y_comparison["flag"]].copy().reset_index(drop=True)

    z_cell_mismatches, z_cell_drilldown = _z_cell_table(
        exio_mapped_Z=mapped_z,
        eurostat_Z=eurostat_Z,
        mapping_matrix=mapping_matrix,
        source_Z=exio_Z,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )

    y_drilldown_rows: list[dict[str, Any]] = []
    for code, contribs in y_contrib_lookup.items():
        euro_value = float(euro_y_total.get(code, 0.0))
        exio_value = float(mapped_y.get(code, 0.0))
        for rank, (source_code, contribution_value) in enumerate(contribs.items(), start=1):
            y_drilldown_rows.append(
                {
                    "target_code": code,
                    "target_label": eurostat_labels.get(code, exio_labels.get(code, code)),
                    "exio_subsector_code": source_code,
                    "exio_subsector_label": exio_labels.get(source_code, source_code),
                    "contribution_value": float(contribution_value),
                    "share_of_exio_mapped_y": _safe_ratio(float(contribution_value), exio_value),
                    "share_of_eurostat_y": _safe_ratio(float(contribution_value), euro_value),
                    "contribution_rank": rank,
                    "interpretation": _explain_subcomponent_exceedance(
                        contribution_value=float(contribution_value),
                        euro_value=euro_value,
                        source_code=source_code,
                        source_label=exio_labels.get(source_code, source_code),
                    ),
                }
            )
    y_drilldown_columns = [
        "target_code",
        "target_label",
        "exio_subsector_code",
        "exio_subsector_label",
        "contribution_value",
        "share_of_exio_mapped_y",
        "share_of_eurostat_y",
        "contribution_rank",
        "interpretation",
    ]
    y_drilldown = pd.DataFrame(y_drilldown_rows, columns=y_drilldown_columns)
    if not y_drilldown.empty:
        y_drilldown = y_drilldown.sort_values(["target_code", "contribution_rank"], kind="stable").reset_index(drop=True)

    subcomponent_exceeds_aggregate = _subcomponent_exceeds_aggregate_table(
        mapping_weights=mapping_weights,
        mapping_matrix=mapping_matrix,
        source_x=exio_x,
        source_y_total=exio_y_total,
        source_row_total=exio_row_total,
        source_Z=exio_Z,
        eurostat_x=eurostat_x,
        eurostat_y_total=euro_y_total,
        eurostat_row_total=euro_row_total,
        eurostat_Z=eurostat_Z,
        z_cell_drilldown=z_cell_drilldown,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )

    audit_metadata = {
        "country": country,
        "year": year,
        "rel_threshold": rel_threshold,
        "min_abs_meur": min_abs_meur,
        "identity_mode": identity_mode,
        "source_database": metadata.get("source_database") if metadata else None,
        "target_database": metadata.get("target_database") if metadata else None,
        "price_basis": metadata.get("price_basis") if metadata else "unknown",
        "table_basis": metadata.get("table_basis") if metadata else "product-by-product",
        "domestic_import_treatment": metadata.get("domestic_import_treatment") if metadata else "total-use comparison",
        "currency_unit": metadata.get("currency_unit") if metadata else "MIO_EUR",
        "unit_conversion": metadata.get("unit_conversion") if metadata else 1.0,
        "mapping_file": metadata.get("mapping_file") if metadata else None,
        "mapping_file_sha256": metadata.get("mapping_file_sha256") if metadata else None,
        "exio_year": metadata.get("exio_year") if metadata else None,
        "eurostat_dataset": metadata.get("eurostat_dataset") if metadata else None,
        "eurostat_request_url": metadata.get("eurostat_request_url") if metadata else None,
    }

    report_markdown = render_concordance_audit_report(
        metadata=audit_metadata,
        mapping_coverage=mapping_coverage,
        internal_identities=internal_identities,
        aggregate_totals=aggregate_totals,
        z_cell_mismatches=z_cell_mismatches,
        z_cell_drilldown=z_cell_drilldown,
        subcomponent_exceeds_aggregate=subcomponent_exceeds_aggregate,
        y_mismatches=y_mismatches,
        x_mismatches=x_mismatches,
        y_drilldown=y_drilldown,
    )

    return ConcordanceAuditResult(
        metadata=audit_metadata,
        mapping_coverage=mapping_coverage,
        internal_identities=internal_identities,
        aggregate_totals=aggregate_totals,
        x_mismatches=x_mismatches,
        y_mismatches=y_mismatches,
        z_row_col_mismatches=z_row_col_mismatches,
        z_cell_mismatches=z_cell_mismatches,
        z_cell_drilldown=z_cell_drilldown,
        y_drilldown=y_drilldown,
        subcomponent_exceeds_aggregate=subcomponent_exceeds_aggregate,
        report_markdown=report_markdown,
    )


def render_concordance_audit_report(
    *,
    metadata: Mapping[str, Any],
    mapping_coverage: pd.DataFrame,
    internal_identities: pd.DataFrame,
    aggregate_totals: pd.DataFrame,
    z_cell_mismatches: pd.DataFrame,
    z_cell_drilldown: pd.DataFrame,
    subcomponent_exceeds_aggregate: pd.DataFrame,
    y_mismatches: pd.DataFrame,
    x_mismatches: pd.DataFrame,
    y_drilldown: pd.DataFrame,
) -> str:
    lines: list[str] = []
    country = metadata.get("country", "")
    year = metadata.get("year", "")
    lines.append(f"# EXIOBASE–Eurostat Concordance Audit: {country} {year}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Country: {country}")
    lines.append(f"- Year: {year}")
    lines.append(f"- Mapping file SHA256: {metadata.get('mapping_file_sha256', 'unknown')}")
    lines.append(f"- Identity mode: {metadata.get('identity_mode', 'diagnostic')}")
    lines.append(f"- Mapped target sectors: {int(mapping_coverage['mapped_targets'].gt(0).sum())}")
    lines.append(f"- Unmapped non-zero source sectors: {int((mapping_coverage['status'].str.contains('UNMAPPED_NONZERO', na=False)).sum())}")
    lines.append(f"- Aggregate mismatches above threshold: {int(aggregate_totals['flag'].sum())}")
    lines.append(f"- Flagged Z cells: {int(z_cell_mismatches['flag'].sum())}")
    lines.append(f"- Subcomponents exceeding aggregates: {len(subcomponent_exceeds_aggregate)}")
    lines.append("")

    lines.append("## Mapping Coverage")
    lines.append("")
    lines.append(_markdown_table(mapping_coverage.sort_values(["status", "source_code"], kind="stable"), columns=["source_code", "source_label", "mapped_targets", "sum_mapping_weight", "status", "x_value", "y_value", "z_row_total", "z_col_total", "reason"], max_rows=20))
    lines.append("")

    lines.append("## Internal Identities")
    lines.append("")
    lines.append(_markdown_table(internal_identities, columns=["database", "code", "x", "z_row_total", "y_total", "z_plus_y", "abs_gap", "rel_gap", "severity", "status", "identity_mode"], max_rows=20))
    lines.append("")

    lines.append("## Most Serious Aggregate Mismatches")
    lines.append("")
    top_agg = aggregate_totals[aggregate_totals["flag"]].sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable")
    lines.append(_markdown_table(top_agg, columns=["metric", "target_code", "target_label", "exio_mapped_value", "eurostat_value", "ratio_exio_to_eurostat", "abs_gap", "rel_gap", "severity", "issue_type", "top_exio_contributors", "interpretation"], max_rows=12))
    lines.append("")

    lines.append("## Z-Cell Mismatches")
    lines.append("")
    top_cells = z_cell_mismatches[z_cell_mismatches["flag"]].sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable")
    lines.append(_markdown_table(top_cells, columns=["supplier_code", "buyer_code", "exio_mapped_z", "eurostat_z", "ratio_exio_to_eurostat", "abs_gap", "rel_gap", "severity", "top_source_subsectors", "top_buyer_subsectors", "interpretation"], max_rows=12))
    lines.append("")

    lines.append("## Subcomponents Exceeding Aggregates")
    lines.append("")
    lines.append(_markdown_table(subcomponent_exceeds_aggregate.sort_values(["severity", "abs_gap", "ratio"], ascending=[False, False, False], kind="stable"), columns=["metric", "target_code", "buyer_code", "exio_subsector_code", "exio_subsector_value", "eurostat_aggregate_value", "ratio", "severity", "interpretation"], max_rows=12))
    lines.append("")

    if not x_mismatches.empty:
        lines.append("## X-Output Mismatches")
        lines.append("")
        lines.append(_markdown_table(x_mismatches.sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable"), columns=["target_code", "exio_mapped_value", "eurostat_value", "ratio_exio_to_eurostat", "abs_gap", "rel_gap", "top_exio_contributors", "interpretation"], max_rows=8))
        lines.append("")

    if not y_mismatches.empty:
        lines.append("## Y-Final Demand Mismatches")
        lines.append("")
        lines.append(_markdown_table(y_mismatches.sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable"), columns=["target_code", "exio_mapped_value", "eurostat_value", "ratio_exio_to_eurostat", "abs_gap", "rel_gap", "top_exio_contributors", "interpretation"], max_rows=8))
        lines.append("")

    if not y_drilldown.empty:
        lines.append("## Final Demand Drill-Down")
        lines.append("")
        lines.append(_markdown_table(y_drilldown.sort_values(["target_code", "contribution_rank"], kind="stable"), columns=["target_code", "exio_subsector_code", "contribution_value", "share_of_exio_mapped_y", "share_of_eurostat_y", "contribution_rank", "interpretation"], max_rows=12))
        lines.append("")

    biofuel_mask = (
        z_cell_mismatches["supplier_label"].astype(str).str.contains("chemicals and chemical products", case=False, na=False)
        | z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")
        | z_cell_mismatches["top_source_subsectors"].astype(str).str.contains("Biogasoline|Biodiesels|Other Liquid Biofuels", case=False, na=False)
    )
    if biofuel_mask.any():
        lines.append("## CPA_C20 / Biofuels Diagnostic")
        lines.append("")
        bio = z_cell_mismatches[biofuel_mask].sort_values(["severity", "abs_gap"], ascending=[False, False], kind="stable")
        lines.append(_markdown_table(bio, columns=["supplier_code", "buyer_code", "exio_mapped_z", "eurostat_z", "ratio_exio_to_eurostat", "top_source_subsectors", "interpretation"], max_rows=10))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_concordance_audit_outputs(result: ConcordanceAuditResult, output_dir: str | Path) -> dict[str, Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit_metadata.json": outdir / "audit_metadata.json",
        "audit_mapping_coverage.csv": outdir / "audit_mapping_coverage.csv",
        "audit_internal_identities.csv": outdir / "audit_internal_identities.csv",
        "audit_aggregate_totals.csv": outdir / "audit_aggregate_totals.csv",
        "audit_x_mismatches.csv": outdir / "audit_x_mismatches.csv",
        "audit_y_mismatches.csv": outdir / "audit_y_mismatches.csv",
        "audit_z_row_col_mismatches.csv": outdir / "audit_z_row_col_mismatches.csv",
        "audit_z_cell_mismatches.csv": outdir / "audit_z_cell_mismatches.csv",
        "audit_z_cell_drilldown.csv": outdir / "audit_z_cell_drilldown.csv",
        "audit_y_drilldown.csv": outdir / "audit_y_drilldown.csv",
        "audit_subcomponent_exceeds_aggregate.csv": outdir / "audit_subcomponent_exceeds_aggregate.csv",
        "audit_top_flags.md": outdir / "audit_top_flags.md",
        "audit_exio_eurostat_concordance.md": outdir / f"audit_exio_eurostat_concordance_{result.metadata.get('country', '')}_{result.metadata.get('year', '')}.md",
    }
    Path(paths["audit_metadata.json"]).write_text(json.dumps(result.metadata, indent=2, sort_keys=True), encoding="utf-8")
    result.mapping_coverage.to_csv(paths["audit_mapping_coverage.csv"], index=False)
    result.internal_identities.to_csv(paths["audit_internal_identities.csv"], index=False)
    result.aggregate_totals.to_csv(paths["audit_aggregate_totals.csv"], index=False)
    result.x_mismatches.to_csv(paths["audit_x_mismatches.csv"], index=False)
    result.y_mismatches.to_csv(paths["audit_y_mismatches.csv"], index=False)
    result.z_row_col_mismatches.to_csv(paths["audit_z_row_col_mismatches.csv"], index=False)
    result.z_cell_mismatches.to_csv(paths["audit_z_cell_mismatches.csv"], index=False)
    result.z_cell_drilldown.to_csv(paths["audit_z_cell_drilldown.csv"], index=False)
    result.y_drilldown.to_csv(paths["audit_y_drilldown.csv"], index=False)
    result.subcomponent_exceeds_aggregate.to_csv(paths["audit_subcomponent_exceeds_aggregate.csv"], index=False)
    top_flags_lines = [
        f"# Top Concordance Flags: {result.metadata.get('country', '')} {result.metadata.get('year', '')}",
        "",
        "## Aggregate mismatches",
        "",
        _markdown_table(
            result.aggregate_totals[result.aggregate_totals["flag"]].sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable"),
            columns=["metric", "target_code", "exio_mapped_value", "eurostat_value", "ratio_exio_to_eurostat", "abs_gap", "rel_gap", "top_exio_contributors"],
            max_rows=8,
        ),
        "",
        "## Z-cell mismatches",
        "",
        _markdown_table(
            result.z_cell_mismatches[result.z_cell_mismatches["flag"]].sort_values(["severity", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable"),
            columns=["supplier_code", "buyer_code", "exio_mapped_z", "eurostat_z", "ratio_exio_to_eurostat", "top_source_subsectors"],
            max_rows=8,
        ),
        "",
        "## Subcomponents exceeding aggregates",
        "",
        _markdown_table(
            result.subcomponent_exceeds_aggregate.sort_values(["severity", "abs_gap", "ratio"], ascending=[False, False, False], kind="stable"),
            columns=["metric", "target_code", "buyer_code", "exio_subsector_code", "exio_subsector_value", "eurostat_aggregate_value", "ratio"],
            max_rows=8,
        ),
        "",
    ]
    Path(paths["audit_top_flags.md"]).write_text("\n".join(top_flags_lines).rstrip() + "\n", encoding="utf-8")
    Path(paths["audit_exio_eurostat_concordance.md"]).write_text(result.report_markdown, encoding="utf-8")
    return paths


def load_default_label_maps(repo_root: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    root = Path(repo_root)
    exio_labels = _read_label_map(root / "scripts" / "mappings" / "exiobase_products_dictionary.csv")
    euro_labels = _read_label_map(root / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv")
    return exio_labels, euro_labels


def build_audit_metadata(
    *,
    country: str,
    year: int,
    exio_year: int,
    eurostat_dataset: str,
    eurostat_request_url: str,
    mapping_file: str | Path,
    exio_flow: str,
    eurostat_unit: str,
    price_basis: str = "unknown",
    table_basis: str = "product-by-product",
    domestic_import_treatment: str = "total-use comparison",
    unit_conversion: float = 1.0,
) -> dict[str, Any]:
    mapping_file = Path(mapping_file)
    return {
        "country": country,
        "year": year,
        "exio_year": exio_year,
        "eurostat_dataset": eurostat_dataset,
        "eurostat_request_url": eurostat_request_url,
        "mapping_file": str(mapping_file),
        "mapping_file_sha256": _sha256(mapping_file),
        "source_database": {
            "name": "EXIOBASE3",
            "year": exio_year,
            "flow": exio_flow,
            "system": "pxp",
        },
        "target_database": {
            "name": "Eurostat",
            "dataset": eurostat_dataset,
            "unit": eurostat_unit,
            "country": country,
            "year": year,
        },
        "price_basis": price_basis,
        "table_basis": table_basis,
        "domestic_import_treatment": domestic_import_treatment,
        "currency_unit": eurostat_unit,
        "unit_conversion": unit_conversion,
    }
