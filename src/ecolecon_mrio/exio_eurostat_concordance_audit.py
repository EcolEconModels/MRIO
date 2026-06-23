from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.mapping_csv import resolve_weighted_mapping_frame


DEFAULT_REL_THRESHOLD = 0.20
DEFAULT_MIN_ABS_MEUR = 5.0
DEFAULT_SUBCOMPONENT_RATIO = 1.20
EPS = 1e-12

METRIC_X_OUTPUT = "X_OUTPUT"
METRIC_X_OUTPUT_PROXY_TOTAL_USE = "X_OUTPUT_PROXY_TOTAL_USE"
METRIC_Y_FINAL_DEMAND_TOTAL = "Y_FINAL_DEMAND_TOTAL"
METRIC_Z_ROW_INTERMEDIATE_SALES = "Z_ROW_INTERMEDIATE_SALES"
METRIC_Z_COL_INTERMEDIATE_PURCHASES = "Z_COL_INTERMEDIATE_PURCHASES"
METRIC_TOTAL_USE = "TOTAL_USE_Z_PLUS_Y"
METRIC_Z_CELL = "Z_CELL"

METRIC_HUMAN_LABELS = {
    "X_OUTPUT": "Output comparison",
    "X_OUTPUT_PROXY_TOTAL_USE": "Flow-specific total-use proxy comparison",
    "Y_FINAL_DEMAND_TOTAL": "Final demand total",
    "Z_ROW_INTERMEDIATE_SALES": "Intermediate sales by supplying sector",
    "Z_COL_INTERMEDIATE_PURCHASES": "Intermediate purchases by using sector",
    "TOTAL_USE_Z_PLUS_Y": "Total use: intermediate sales + final demand",
    "Z_CELL": "Intermediate flow cell: supplier → buyer",
}

METRIC_EXPLANATIONS = {
    "Z_ROW_INTERMEDIATE_SALES": (
        "For each supplying sector, this sums all intermediate sales from that sector to all buyer sectors."
    ),
    "Z_COL_INTERMEDIATE_PURCHASES": (
        "For each buyer sector, this sums all intermediate inputs purchased by that sector from all supplying sectors."
    ),
    "Z_CELL": (
        "This compares one specific intermediate flow: a supplier sector selling to a buyer sector."
    ),
    "Y_FINAL_DEMAND_TOTAL": (
        "This compares total final demand for the product/sector."
    ),
    "TOTAL_USE_Z_PLUS_Y": (
        "This compares intermediate sales plus final demand."
    ),
    "X_OUTPUT_PROXY_TOTAL_USE": (
        "This is a flow-specific total-use proxy, not independently observed gross output. It uses row total plus final demand when no independent output vector is available."
    ),
}


def fmt_meur(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric_value):
        return ""
    return f"{numeric_value:,.1f}"


def fmt_ratio(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isinf(numeric_value):
        return "∞"
    if pd.isna(numeric_value):
        return ""
    return f"{numeric_value:,.2f}"


def fmt_pct(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric_value):
        return ""
    return f"{100 * numeric_value:,.1f}%"


def _display_code(code: object, labels: Mapping[str, str]) -> str:
    code_text = str(code)
    label_text = labels.get(code_text, code_text)
    return f"{code_text} — {label_text}"


def euro_display(code: object, labels: Mapping[str, str]) -> str:
    return _display_code(code, labels)


def exio_display(code: object, labels: Mapping[str, str]) -> str:
    return _display_code(code, labels)


def human_metric_label(metric: str) -> str:
    return METRIC_HUMAN_LABELS.get(metric, metric)


def human_metric_explanation(metric: str) -> str:
    return METRIC_EXPLANATIONS.get(metric, "")


def _signed_meur(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric_value):
        return ""
    sign = "+" if numeric_value > 0 else ""
    return f"{sign}{numeric_value:,.1f}"


def _display_code_label(code: object, label: object) -> str:
    code_text = str(code)
    label_text = str(label).strip() if label is not None else code_text
    if not label_text or label_text == "nan":
        label_text = code_text
    return f"{code_text} — {label_text}"


def _display_metric(metric: object) -> str:
    return human_metric_label(str(metric))


def _display_contributor_string(raw_value: object) -> str:
    if raw_value is None:
        return ""
    try:
        if pd.isna(raw_value):
            return ""
    except TypeError:
        pass
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return ""
    rendered: list[str] = []
    for segment in text.split(" | "):
        segment = segment.strip()
        if not segment or segment.lower() == "nan":
            continue
        left, separator, right = segment.partition(": ")
        if not separator:
            if segment.lower() != "nan":
                rendered.append(segment)
            continue
        code, _, label = left.partition(" ")
        label = label.strip() or code
        if not code or code.lower() == "nan" or not right or right.lower() == "nan":
            continue
        if "(" in right and right.endswith("%)"):
            value_text, share_text = right.rsplit(" (", 1)
            share_text = share_text.rstrip("%)")
            if not value_text or value_text.lower() == "nan":
                continue
            rendered.append(f"{code} — {label}: {fmt_meur(value_text)} MEUR ({share_text}%)")
        else:
            rendered.append(f"{code} — {label}: {right}")
    return "<br>".join(rendered)


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.replace("\n", "<br>").replace("|", "\\|")


def _direction_label(exio_value: object, euro_value: object) -> str:
    try:
        exio_numeric = float(exio_value)
        euro_numeric = float(euro_value)
    except (TypeError, ValueError):
        return ""
    if exio_numeric > euro_numeric:
        return "EXIOBASE larger"
    if exio_numeric < euro_numeric:
        return "Eurostat larger"
    return "Equal"


def _human_identity_status(status: object) -> str:
    text = str(status).strip().upper()
    if text in {"OK", ""}:
        return "OK"
    if text in {"DIAGNOSTIC_GAP", "INTERNAL_IDENTITY_FAIL"}:
        return "Flagged"
    return text.replace("_", " ").title()


def _human_comparison_prose(*, metric: str, direction: str, exio_value: object, euro_value: object, contributors: str, extra: str = "") -> str:
    metric_text = human_metric_label(metric)
    if not direction:
        return metric_text
    if direction == "Equal":
        summary = f"{metric_text}: EXIOBASE and Eurostat match at {fmt_meur(exio_value)} MEUR."
    else:
        summary = f"{metric_text}: EXIOBASE is {('larger' if direction == 'EXIOBASE larger' else 'smaller')} than Eurostat ({fmt_meur(exio_value)} vs {fmt_meur(euro_value)} MEUR)."
    if contributors:
        summary += f" Top EXIOBASE contributors: {contributors}."
    if extra:
        summary += f" {extra}"
    return summary


def _human_flow_prose(*, supplier_display: str, buyer_display: str, direction: str, exio_value: object, euro_value: object, source_contribs: str, buyer_contribs: str) -> str:
    if direction == "Equal":
        summary = f"{supplier_display} to {buyer_display}: EXIOBASE and Eurostat match at {fmt_meur(exio_value)} MEUR."
    else:
        summary = f"{supplier_display} to {buyer_display}: EXIOBASE is {('larger' if direction == 'EXIOBASE larger' else 'smaller')} than Eurostat ({fmt_meur(exio_value)} vs {fmt_meur(euro_value)} MEUR)."
    pieces: list[str] = []
    if source_contribs:
        pieces.append(f"Top source subsectors: {source_contribs}.")
    if buyer_contribs:
        pieces.append(f"Top buyer subsectors: {buyer_contribs}.")
    if pieces:
        summary += " " + " ".join(pieces)
    return summary


def _human_subcomponent_prose(*, metric: str, aggregate_display: str, buyer_display: str, exio_subsector_display: str, exio_subsector_value: object, euro_value: object, ratio: object) -> str:
    metric_text = human_metric_label(metric)
    if metric == METRIC_Z_CELL:
        comparison = f"{exio_subsector_display} exceeds the Eurostat cell {aggregate_display}"
        if buyer_display:
            comparison = f"{exio_subsector_display} exceeds the Eurostat cell {aggregate_display} → {buyer_display}"
    else:
        comparison = f"{exio_subsector_display} exceeds the Eurostat aggregate for {aggregate_display}"
    return f"{metric_text}: {comparison} ({fmt_meur(exio_subsector_value)} vs {fmt_meur(euro_value)} MEUR; ratio {fmt_ratio(ratio)})."


def _human_y_prose(*, source_display: str, contribution_value: object, share_of_exio: object, share_of_eurostat: object) -> str:
    return f"{source_display} contributes {fmt_meur(contribution_value)} MEUR, or {fmt_pct(share_of_exio)} of mapped EXIOBASE final demand and {fmt_pct(share_of_eurostat)} of Eurostat final demand."


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


def concordance_rel_gap(exio_value: object, eurostat_value: object) -> float:
    try:
        exio_numeric = float(exio_value)
        euro_numeric = float(eurostat_value)
    except (TypeError, ValueError):
        return 0.0
    return abs(exio_numeric - euro_numeric) / max(abs(exio_numeric), abs(euro_numeric), EPS)


def _severity(rel_gap: float, rel_threshold: float = DEFAULT_REL_THRESHOLD) -> str:
    if rel_gap >= 1.0:
        return "CRITICAL"
    if rel_gap >= 0.5:
        return "HIGH"
    if rel_gap >= rel_threshold:
        return "MEDIUM"
    return "OK"


def _severity_rank(severity: str) -> int:
    return {
        "CRITICAL": 3,
        "HIGH": 2,
        "MEDIUM": 1,
        "OK": 0,
    }.get(str(severity).upper(), -1)


def _metric_issue_type(metric: str, *, is_flagged: bool, low_value_noise: bool = False) -> str:
    if low_value_noise:
        return "LOW_VALUE_RELATIVE_NOISE"
    if not is_flagged:
        return "LOW_VALUE_RELATIVE_NOISE"
    if metric == METRIC_Y_FINAL_DEMAND_TOTAL:
        return "FINAL_DEMAND_ALLOCATION_DIFFERENCE"
    if metric in {METRIC_Z_ROW_INTERMEDIATE_SALES, METRIC_Z_COL_INTERMEDIATE_PURCHASES}:
        return "INTERMEDIATE_USE_ALLOCATION_DIFFERENCE"
    if metric in {METRIC_X_OUTPUT, METRIC_X_OUTPUT_PROXY_TOTAL_USE, METRIC_TOTAL_USE}:
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
    headers = [_markdown_cell(col) for col in view.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("")
                else:
                    cells.append(_markdown_cell(f"{value:.3f}"))
            else:
                cells.append(_markdown_cell(value))
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
    rel_threshold: float,
    min_abs_meur: float,
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
        threshold_exceeded = abs_gap >= min_abs_meur and rel_gap >= rel_threshold
        if identity_mode == "diagnostic":
            status = "DIAGNOSTIC_GAP" if threshold_exceeded else "OK"
        else:
            status = "INTERNAL_IDENTITY_FAIL" if threshold_exceeded else "OK"
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
                "severity": _severity(rel_gap, rel_threshold),
                "identity_mode": identity_mode,
                "status": status,
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _build_mapping_weights(
    mapping_df: pd.DataFrame,
    *,
    target_output_by_code: object | None = None,
    target_output_by_target: object | None = None,
) -> pd.DataFrame:
    resolved = resolve_weighted_mapping_frame(
        mapping_df,
        target_output_by_code=target_output_by_code,
        target_output_by_target=target_output_by_target,
    )
    if "notes" not in resolved.columns:
        resolved["notes"] = ""
    return resolved.loc[:, ["source_code", "target_code", "weight", "source_label", "target_label", "notes"]].copy()

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
                "severity": _severity(rel_gap, rel_threshold),
                "top_exio_contributors": top_contribs,
                "issue_type": _metric_issue_type(metric, is_flagged=flagged, low_value_noise=abs_gap < min_abs_meur),
                "interpretation": "" if not flagged else _explain_aggregate_mismatch(metric, exio_value, euro_value, rel_gap, top_contribs),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["severity_rank"] = frame["severity"].map(_severity_rank).fillna(-1).astype(int)
        frame = frame.sort_values(["flag", "severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)
        frame = frame.drop(columns=["severity_rank"])
    return frame


def _explain_aggregate_mismatch(metric: str, exio_value: float, euro_value: float, rel_gap: float, top_contribs: str) -> str:
    if metric == METRIC_X_OUTPUT:
        base = "Mapped EXIO output differs from Eurostat output."
    elif metric == METRIC_X_OUTPUT_PROXY_TOTAL_USE:
        base = "Mapped EXIO output differs from the inferred Eurostat proxy total use."
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
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exio_mapped_Z = _to_float_frame(exio_mapped_Z, name="mapped_Z")
    eurostat_Z = _to_float_frame(eurostat_Z, name="eurostat_Z")
    codes = sorted(set(exio_mapped_Z.index).union(set(exio_mapped_Z.columns), set(eurostat_Z.index), set(eurostat_Z.columns)))
    exio_mapped_Z = exio_mapped_Z.reindex(index=codes, columns=codes).fillna(0.0)
    eurostat_Z = eurostat_Z.reindex(index=codes, columns=codes).fillna(0.0)

    rows: list[dict[str, Any]] = []
    drilldown_rows: list[dict[str, Any]] = []
    total_cells = max(1, len(codes) * len(codes))
    progress_step = max(1, total_cells // 20)
    next_progress_tick = progress_step
    processed_cells = 0
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
            processed_cells += 1
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
                    "severity": _severity(rel_gap, rel_threshold),
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
            if verbose and (processed_cells >= next_progress_tick or processed_cells == total_cells):
                pct_done = 100.0 * processed_cells / total_cells
                pct_remaining = max(0.0, 100.0 - pct_done)
                print(
                    f"  [4.6] Z-cell mismatch table | {pct_done:5.1f}% complete, {pct_remaining:5.1f}% remaining ({processed_cells}/{total_cells} cells)",
                    flush=True,
                )
                next_progress_tick += progress_step
    cell_frame = pd.DataFrame(rows, columns=cell_columns)
    drilldown_frame = pd.DataFrame(drilldown_rows, columns=drilldown_columns)
    if not cell_frame.empty:
        cell_frame["severity_rank"] = cell_frame["severity"].map(_severity_rank).fillna(-1).astype(int)
        cell_frame = cell_frame.sort_values(["flag", "severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False, False], kind="stable").reset_index(drop=True)
        cell_frame = cell_frame.drop(columns=["severity_rank"])
    if not drilldown_frame.empty:
        drilldown_frame = drilldown_frame.sort_values(["supplier_code", "buyer_code", "role", "contribution_rank"], kind="stable").reset_index(drop=True)
    return cell_frame, drilldown_frame


def _subcomponent_exceeds_aggregate_table(
    *,
    x_metric: str,
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
    full_subcomponent_z_scan: bool = False,
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
                        "severity": _severity(rel_gap, rel_threshold),
                        "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                        "interpretation": f"{source_code} {exio_labels.get(source_code, source_code)} exceeds the Eurostat aggregate for {target_code}.",
                    }
                )
    if x_metric != METRIC_X_OUTPUT_PROXY_TOTAL_USE:
        for row in mapping_weights.itertuples(index=False):
            if row.target_code not in eurostat_x.index:
                continue
            source_code = str(row.source_code)
            target_code = str(row.target_code)
            source_value = float(source_x.get(source_code, 0.0)) * float(row.weight)
            euro_value = float(eurostat_x.get(target_code, 0.0))
            abs_gap = abs(source_value - euro_value)
            rel_gap = _relative_gap(source_value, euro_value)
            if source_value > euro_value * DEFAULT_SUBCOMPONENT_RATIO and abs_gap >= min_abs_meur and rel_gap >= rel_threshold:
                rows.append(
                    {
                        "metric": x_metric,
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
                        "severity": _severity(rel_gap, rel_threshold),
                        "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                        "interpretation": f"{source_code} {exio_labels.get(source_code, source_code)} exceeds the Eurostat aggregate for {target_code}.",
                    }
                )

    if full_subcomponent_z_scan:
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
                                "severity": _severity(rel_gap, rel_threshold),
                                "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                                "interpretation": f"{source_code} {exio_labels.get(source_code, source_code)} exceeds the Eurostat cell {supplier_code}→{buyer_code}.",
                            }
                        )
    elif z_cell_drilldown is not None and not z_cell_drilldown.empty:
        flagged_cells = z_cell_drilldown[
            (z_cell_drilldown["role"].astype(str) == "source_contributor")
            & (z_cell_drilldown["subsector_exceeds_eurostat_cell"].fillna(False))
        ].copy()
        for row in flagged_cells.itertuples(index=False):
            supplier_code = str(row.supplier_code)
            buyer_code = str(row.buyer_code)
            contribution_value = float(row.contribution_value)
            euro_value = float(eurostat_Z.loc[supplier_code, buyer_code])
            abs_gap = abs(contribution_value - euro_value)
            rel_gap = _relative_gap(contribution_value, euro_value)
            if abs_gap >= min_abs_meur and rel_gap >= rel_threshold:
                rows.append(
                    {
                        "metric": METRIC_Z_CELL,
                        "target_code": supplier_code,
                        "target_label": euro_labels.get(supplier_code, supplier_code),
                        "buyer_code": buyer_code,
                        "buyer_label": euro_labels.get(buyer_code, buyer_code),
                        "exio_subsector_code": str(row.exio_subsector_code),
                        "exio_subsector_label": exio_labels.get(str(row.exio_subsector_code), str(row.exio_subsector_code)),
                        "exio_subsector_value": contribution_value,
                        "eurostat_aggregate_value": euro_value,
                        "ratio": _safe_ratio(contribution_value, euro_value),
                        "abs_gap": abs_gap,
                        "severity": _severity(rel_gap, rel_threshold),
                        "issue_type": "SUBSECTOR_EXCEEDS_AGGREGATE",
                        "interpretation": f"{row.exio_subsector_code} {exio_labels.get(str(row.exio_subsector_code), str(row.exio_subsector_code))} exceeds the Eurostat cell {supplier_code}→{buyer_code}.",
                    }
                )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["severity_rank"] = frame["severity"].map(_severity_rank).fillna(-1).astype(int)
        frame = frame.sort_values(["metric", "severity_rank", "abs_gap", "ratio"], ascending=[True, False, False, False], kind="stable").reset_index(drop=True)
        frame = frame.drop(columns=["severity_rank"])
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
    mapped_x: pd.Series | None = None
    mapped_y: pd.Series | None = None
    mapped_z: pd.DataFrame | None = None
    eurostat_x: pd.Series | None = None
    eurostat_y_total: pd.Series | None = None
    eurostat_z: pd.DataFrame | None = None


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
    verbose: bool = False,
    full_subcomponent_z_scan: bool = False,
) -> ConcordanceAuditResult:
    metadata = dict(metadata or {})

    def _log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    def _stage_log(stage: str, label: str, started_at: float) -> None:
        if verbose:
            elapsed = time.perf_counter() - started_at
            print(f"  {stage} {label} | elapsed={elapsed:.3f}s", flush=True)

    _log("  [4.1] normalizing inputs")
    stage_started_at = time.perf_counter()
    exio_x = _to_float_series(exio_x, name="exio_x")
    exio_Z = _to_float_frame(exio_Z, name="exio_Z")
    exio_Y = _to_float_frame(exio_Y, name="exio_Y")
    eurostat_Z = _to_float_frame(eurostat_Z, name="eurostat_Z")
    eurostat_Y = _to_float_frame(eurostat_Y, name="eurostat_Y")
    eurostat_x_supplied = eurostat_x is not None
    eurostat_x = _to_float_series(eurostat_x, name="eurostat_x") if eurostat_x is not None else (eurostat_Z.sum(axis=1) + eurostat_Y.sum(axis=1)).astype("float64")
    x_metric = METRIC_X_OUTPUT if eurostat_x_supplied else METRIC_X_OUTPUT_PROXY_TOTAL_USE
    if verbose and not eurostat_x_supplied:
        _log("  [4.3] X comparison uses the Eurostat total-use proxy (no independent Eurostat output vector supplied)")

    _log("  [4.2] building mapping weights/matrix")
    exio_row_total, exio_col_total, exio_y_total = _source_metrics(exio_Z, exio_Y)
    euro_row_total, euro_col_total, euro_y_total = _source_metrics(eurostat_Z, eurostat_Y)

    mapping_weights = _build_mapping_weights(mapping_df, target_output_by_code=eurostat_x)
    source_codes = list(exio_Z.index.astype(str))
    target_codes = sorted(set(eurostat_Z.index.astype(str)).union(set(eurostat_Z.columns.astype(str)), set(eurostat_Y.index.astype(str))))
    mapping_matrix = _build_mapping_matrix(mapping_weights, source_codes=source_codes, target_codes=target_codes)

    _stage_log("[4.2]", "building mapping weights/matrix complete", stage_started_at)

    _log("  [4.3] mapping x/y/Z")
    stage_started_at = time.perf_counter()
    mapped_x = _mapped_series(mapping_matrix, exio_x)
    mapped_y = _mapped_series(mapping_matrix, exio_y_total)
    mapped_z = _mapped_matrix(mapping_matrix, exio_Z)
    _stage_log("[4.3]", "mapping x/y/Z complete", stage_started_at)

    _log("  [4.4] internal identities and mapping coverage")
    internal_mapped = _identity_metric_rows(
        database="EXIOBASE-mapped",
        country=country,
        year=year,
        x=mapped_x,
        z=mapped_z,
        y=pd.DataFrame({"FINAL_DEMAND_TOTAL": mapped_y}),
        identity_mode=identity_mode,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    internal_euro = _identity_metric_rows(
        database="Eurostat",
        country=country,
        year=year,
        x=eurostat_x,
        z=eurostat_Z,
        y=eurostat_Y,
        identity_mode=identity_mode,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    internal_identities = pd.concat([internal_mapped, internal_euro], ignore_index=True)

    euro_target_codes = set(eurostat_Z.index.astype(str)) | set(eurostat_Z.columns.astype(str)) | set(eurostat_Y.index.astype(str))
    mapping_coverage = _mapping_coverage_table(
        mapping_weights=mapping_weights,
        exio_source_values=exio_x,
        exio_row_total=exio_row_total,
        exio_col_total=exio_col_total,
        exio_y_total=exio_y_total,
        exio_labels=exio_labels,
        euro_target_codes=euro_target_codes,
    )
    mapped_exio_source_sectors = int(mapping_coverage["mapped_targets"].gt(0).sum())
    mapped_euro_target_sectors = int(len(set(mapping_weights["target_code"].astype(str)).intersection(euro_target_codes)))

    x_contrib_lookup = {code: _series_contributions(mapping_weights, exio_x, target_code=code) for code in mapped_x.index}
    y_contrib_lookup = {code: _series_contributions(mapping_weights, exio_y_total, target_code=code) for code in mapped_y.index}
    row_contrib_lookup = {code: _series_contributions(mapping_weights, exio_row_total, target_code=code) for code in mapped_z.index}
    col_contrib_lookup = {code: _series_contributions(mapping_weights, exio_col_total, target_code=code) for code in mapped_z.columns}

    _log("  [4.5] aggregate comparisons")
    stage_started_at = time.perf_counter()
    _log("  [4.6] Z-cell mismatch table")
    z_cell_started_at = time.perf_counter()
    z_cell_mismatches, z_cell_drilldown = _z_cell_table(
        exio_mapped_Z=mapped_z,
        eurostat_Z=eurostat_Z,
        mapping_matrix=mapping_matrix,
        source_Z=exio_Z,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
        verbose=verbose,
    )
    _stage_log("[4.6]", "Z-cell mismatch table complete", z_cell_started_at)

    stage_started_at = time.perf_counter()
    x_comparison = _aggregate_comparison_table(
        metric=x_metric,
        exio_mapped=mapped_x,
        eurostat=eurostat_x,
        exio_contrib_lookup=x_contrib_lookup,
        exio_labels=exio_labels,
        euro_labels=eurostat_labels,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
    )
    if bool(metadata.get("exio_x_is_proxy", False)) and bool(metadata.get("eurostat_x_is_proxy", False)) and x_metric == METRIC_X_OUTPUT_PROXY_TOTAL_USE:
        x_comparison = x_comparison.iloc[0:0].copy()

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
    _stage_log("[4.5]", "aggregate comparisons complete", stage_started_at)

    _log("  [4.7] Y drill-down")
    stage_started_at = time.perf_counter()
    y_flagged_codes = set(y_mismatches["target_code"].astype(str))
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
                    "parent_y_flagged": bool(code in y_flagged_codes),
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
        "parent_y_flagged",
        "interpretation",
    ]
    y_drilldown = pd.DataFrame(y_drilldown_rows, columns=y_drilldown_columns)
    if not y_drilldown.empty:
        y_drilldown = y_drilldown.sort_values(["target_code", "contribution_rank"], kind="stable").reset_index(drop=True)
    _stage_log("[4.7]", "Y drill-down complete", stage_started_at)

    _log("  [4.8] subcomponent-exceeds-aggregate")
    stage_started_at = time.perf_counter()
    audit_metadata = {
        "country": country,
        "year": year,
        "rel_threshold": rel_threshold,
        "min_abs_meur": min_abs_meur,
        "identity_mode": identity_mode,
        "comparison_mode": metadata.get("comparison_mode") if metadata else None,
        "source_database": metadata.get("source_database") if metadata else None,
        "target_database": metadata.get("target_database") if metadata else None,
        "exio_flow": metadata.get("exio_flow") if metadata else None,
        "eurostat_flow": metadata.get("eurostat_flow") if metadata else None,
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
        "run_command": metadata.get("run_command") if metadata else None,
        "exio_x_is_proxy": bool(metadata.get("exio_x_is_proxy")) if metadata else False,
        "exio_x_proxy_basis": metadata.get("exio_x_proxy_basis") if metadata else None,
        "eurostat_x_is_proxy": bool(metadata.get("eurostat_x_is_proxy", not eurostat_x_supplied)) if metadata else (not eurostat_x_supplied),
        "eurostat_x_proxy_basis": metadata.get("eurostat_x_proxy_basis") if metadata else None,
        "flow_interpretation": metadata.get("flow_interpretation") if metadata else None,
        "full_subcomponent_z_scan": full_subcomponent_z_scan,
        "z_subcomponent_scan_mode": "full_scan" if full_subcomponent_z_scan else "flagged_cells_only",
        "mapped_exio_source_sectors": mapped_exio_source_sectors,
        "mapped_euro_target_sectors": mapped_euro_target_sectors,
    }
    if metadata and any(key in metadata for key in ("eurostat_final_demand_basis", "eurostat_exports_included", "eurostat_final_demand_columns_available", "eurostat_final_demand_columns_selected", "eurostat_final_demand_columns_excluded")):
        audit_metadata.update(
            {
                "eurostat_final_demand_basis": metadata.get("eurostat_final_demand_basis"),
                "eurostat_exports_included": metadata.get("eurostat_exports_included"),
                "eurostat_final_demand_columns_available": list(metadata.get("eurostat_final_demand_columns_available", [])),
                "eurostat_final_demand_columns_selected": list(metadata.get("eurostat_final_demand_columns_selected", [])),
                "eurostat_final_demand_columns_excluded": list(metadata.get("eurostat_final_demand_columns_excluded", [])),
            }
        )
    else:
        audit_metadata.update(
            {
                "eurostat_final_demand_basis": "caller_supplied_final_demand",
                "eurostat_exports_included": None,
                "eurostat_final_demand_columns_available": [str(col) for col in eurostat_Y.columns],
                "eurostat_final_demand_columns_selected": [str(col) for col in eurostat_Y.columns],
                "eurostat_final_demand_columns_excluded": [],
            }
        )

    subcomponent_exceeds_aggregate = _subcomponent_exceeds_aggregate_table(
        x_metric=x_metric,
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
        full_subcomponent_z_scan=full_subcomponent_z_scan,
    )
    _stage_log("[4.8]", "subcomponent-exceeds-aggregate complete", stage_started_at)

    _log("  [4.9] rendering report")
    stage_started_at = time.perf_counter()
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
    _stage_log("[4.9]", "rendering report complete", stage_started_at)

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
        mapped_x=mapped_x,
        mapped_y=mapped_y,
        mapped_z=mapped_z,
        eurostat_x=eurostat_x,
        eurostat_y_total=euro_y_total,
        eurostat_z=eurostat_Z,
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
    proxy_mode = bool(metadata.get("eurostat_x_is_proxy", False))
    scan_mode = str(metadata.get("z_subcomponent_scan_mode", "flagged_cells_only"))
    comparison_mode = str(metadata.get("comparison_mode", "")).strip()
    exio_flow = str(metadata.get("exio_flow", "")).strip()
    eurostat_flow = str(metadata.get("eurostat_flow", "")).strip()
    flow_interpretation = metadata.get("flow_interpretation", "")

    def _top_flag(frame: pd.DataFrame, metric: str | None = None) -> pd.Series:
        filtered = frame[frame["flag"]].copy()
        if metric is not None and "metric" in filtered.columns:
            filtered = filtered[filtered["metric"].astype(str).eq(metric)]
        if filtered.empty:
            return pd.Series(dtype="float64")
        filtered["severity_rank"] = filtered["severity"].map(_severity_rank).fillna(-1).astype(int)
        return filtered.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"]).iloc[0]

    def _section_heading(title: str) -> None:
        lines.append(title)
        lines.append("")

    lines.append(f"# EXIOBASE–Eurostat Concordance Audit: {country} {year}")
    lines.append("")
    _section_heading("## Flow comparison")
    lines.append("This audit compares:")
    lines.append(f"- EXIOBASE flow: {exio_flow or comparison_mode or 'unknown'}")
    lines.append(f"- Eurostat `stk_flow`: {eurostat_flow or 'unknown'}")
    lines.append("Interpretation:")
    if isinstance(flow_interpretation, dict):
        interpretation_text = str(flow_interpretation.get(comparison_mode, "")).strip()
    else:
        interpretation_text = str(flow_interpretation).strip()
    if interpretation_text:
        lines.append(f"- {interpretation_text}")
    else:
        lines.append("- Flow-specific domestic/import/total comparison based on the selected EXIOBASE and Eurostat flow slices.")
    lines.append("- x-like rows in this report are flow-specific total-use proxy comparisons, not independently observed gross-output comparisons.")
    lines.append("- Producer-side EXIOBASE x_region is not used in this audit runner for the flow comparison rows.")
    lines.append("- Important: Eurostat final demand uses atomic final-demand columns excluding exports. Therefore this audit is destination/use-side: it compares products used by the selected country. TOTAL means domestic + imported products used by the selected country, not a production-side total including exports.")
    final_demand_available = metadata.get("eurostat_final_demand_columns_available", [])
    final_demand_selected = metadata.get("eurostat_final_demand_columns_selected", [])
    final_demand_excluded = metadata.get("eurostat_final_demand_columns_excluded", [])
    if final_demand_available:
        lines.append("- Eurostat final-demand columns available: " + ", ".join(map(str, final_demand_available)))
    if final_demand_selected:
        lines.append("- Eurostat final-demand columns used: " + ", ".join(map(str, final_demand_selected)))
    if final_demand_excluded:
        lines.append("- Eurostat final-demand columns excluded: " + ", ".join(map(str, final_demand_excluded)))
    row_mismatch = metadata.get("eurostat_row_codes_not_in_intermediate_columns", [])
    col_mismatch = metadata.get("eurostat_intermediate_columns_not_in_rows", [])
    if row_mismatch or col_mismatch:
        lines.append("- Warning: some Eurostat row/column codes were not retained in the intermediate-use block.")
        if row_mismatch:
            lines.append("  - Row codes not in intermediate-use columns: " + ", ".join(map(str, row_mismatch)))
        if col_mismatch:
            lines.append("  - Column codes not in row set: " + ", ".join(map(str, col_mismatch)))
    lines.append("")
    _section_heading("## Executive interpretation")
    lines.append("The largest mismatches are concentrated in sectors where EXIOBASE and Eurostat allocate flows differently, and in a few supplier→buyer cells where one EXIOBASE subsector dominates the mapped flow.")
    lines.append("")
    lines.append("1. Sectors where EXIOBASE has no mapped counterpart but Eurostat has large values.")
    lines.append("2. Sectors where Eurostat separates aggregates more finely than EXIOBASE.")
    lines.append("3. Specific supplier→buyer flows where a single EXIOBASE subsector dominates the mapped flow.")
    lines.append("")
    top_agg = aggregate_totals[aggregate_totals["flag"]].copy()
    if not top_agg.empty:
        top_agg["severity_rank"] = top_agg["severity"].map(_severity_rank).fillna(-1).astype(int)
        top_agg = top_agg.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
    top_cell = z_cell_mismatches[z_cell_mismatches["flag"]].copy()
    if not top_cell.empty:
        top_cell["severity_rank"] = top_cell["severity"].map(_severity_rank).fillna(-1).astype(int)
        top_cell = top_cell.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
    top_sub = subcomponent_exceeds_aggregate.copy()
    if not top_sub.empty:
        top_sub["severity_rank"] = top_sub["severity"].map(_severity_rank).fillna(-1).astype(int)
        top_sub = top_sub.sort_values(["severity_rank", "ratio", "abs_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
    c20_focus = z_cell_mismatches[(z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")) & (z_cell_mismatches["buyer_code"].astype(str).eq("CPA_G46"))]
    lines.append(f"- Largest absolute aggregate mismatch: {human_metric_label(str(top_agg.iloc[0]['metric'])) if not top_agg.empty else 'n/a'} on {_display_code_label(top_agg.iloc[0]['target_code'], top_agg.iloc[0]['target_label']) if not top_agg.empty else 'n/a'}.")
    lines.append(f"- Largest Z-cell mismatch: {_display_code_label(top_cell.iloc[0]['supplier_code'], top_cell.iloc[0]['supplier_label']) + ' → ' + _display_code_label(top_cell.iloc[0]['buyer_code'], top_cell.iloc[0]['buyer_label']) if not top_cell.empty else 'n/a'}.")
    lines.append(f"- Largest subcomponent-exceeds-aggregate case: {_display_code_label(top_sub.iloc[0]['exio_subsector_code'], top_sub.iloc[0]['exio_subsector_label']) if not top_sub.empty else 'n/a'}.")
    if not c20_focus.empty:
        focus = c20_focus.iloc[0]
        lines.append(f"- CPA_C20 → CPA_G46 appears and is {'EXIOBASE larger' if float(focus['exio_mapped_z']) > float(focus['eurostat_z']) else 'Eurostat larger'}.")
    else:
        lines.append("- CPA_C20 → CPA_G46 does not appear in the flagged Z-cell summary.")
    lines.append("")

    _section_heading("## How to read this report")
    lines.append("- Intermediate flow cell: supplier → buyer compares one Eurostat supplier sector and one Eurostat buyer sector. Example: CPA_C20 → CPA_G46 means chemicals and chemical products sold as intermediate inputs to wholesale trade.")
    lines.append("- Intermediate sales by supplying sector sums all intermediate sales from one supplying sector across all buyer sectors.")
    lines.append("- Intermediate purchases by using sector sums all intermediate inputs bought by one buyer sector from all supplying sectors.")
    lines.append("- Final demand total compares final demand for that sector/product.")
    lines.append("- Total use means intermediate sales plus final demand.")
    lines.append("- EXIOBASE mapped value is the EXIOBASE value after applying the EXIOBASE→Eurostat concordance.")
    lines.append("- Eurostat value is the corresponding value from the Eurostat IOT.")
    lines.append("- Ratio is EXIOBASE mapped value divided by Eurostat value. Values greater than 1 mean EXIOBASE is larger; values below 1 mean EXIOBASE is smaller.")
    lines.append("- Values are in million euros unless otherwise stated.")
    lines.append("")
    lines.append(f"Important caveat: if the metric is {human_metric_label(METRIC_X_OUTPUT_PROXY_TOTAL_USE)}, the Eurostat side is inferred from row totals and final demand. Do not interpret it as an independently observed Eurostat gross-output vector.")
    if bool((mapping_coverage["mapped_targets"].astype(int) > 1).any()) or bool((mapping_coverage.get("large_value_in_ambiguous_mapping", pd.Series(dtype=bool)).astype(bool).any())):
        lines.append("Warning: multi-target EXIO→Eurostat mappings are resolved to numeric weights before auditing. For mapping files without static weights, the resolver uses the selected Eurostat target-side value reference for the relevant year/flow; discrepancies involving such targets may reflect concordance allocation assumptions rather than database disagreement.")
    lines.append("")

    _section_heading("## Mapping coverage")
    lines.append(_markdown_table(_human_mapping_coverage_table(mapping_coverage), columns=["EXIOBASE source sector", "Mapped targets", "Mapping weight sum", "Status", "EXIOBASE x, MEUR", "EXIOBASE y total, MEUR", "EXIOBASE Z row total, MEUR", "EXIOBASE Z col total, MEUR", "Reason"], max_rows=20))
    if bool((mapping_coverage["mapped_targets"].astype(int) > 1).any()) or bool((mapping_coverage.get("large_value_in_ambiguous_mapping", pd.Series(dtype=bool)).astype(bool).any())):
        lines.append("")
        lines.append("Warning: multi-target EXIO→Eurostat mappings are resolved to numeric weights before auditing. For mapping files without static weights, the resolver uses the selected Eurostat target-side value reference for the relevant year/flow; discrepancies involving such targets may reflect concordance allocation assumptions rather than database disagreement.")

    _section_heading("## Internal identities")
    lines.append(_markdown_table(_human_internal_identities_table(internal_identities), columns=["Database", "Sector", "EXIOBASE mapped x, MEUR", "Z row total, MEUR", "Y total, MEUR", "Z + Y, MEUR", "Absolute gap, MEUR", "Relative gap", "Severity", "Status", "Identity mode"], max_rows=20))
    lines.append("")

    if not x_mismatches.empty:
        _section_heading(f"## {human_metric_label(METRIC_X_OUTPUT_PROXY_TOTAL_USE)} mismatches")
        if proxy_mode:
            lines.append("X comparison uses the inferred Eurostat total-use proxy (x proxy) because no independent Eurostat output vector was supplied.")
            lines.append("")
        x_sorted = x_mismatches.copy()
        x_sorted["severity_rank"] = x_sorted["severity"].map(_severity_rank).fillna(-1).astype(int)
        x_sorted = x_sorted.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        x_human = _human_aggregate_table(x_sorted)
        lines.append(_markdown_table(x_human, columns=["Readable metric", "Eurostat sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Relative gap", "Direction", "Largest EXIOBASE contributors", "Plain-English interpretation"], max_rows=8))
        lines.append("")

    if not y_mismatches.empty:
        _section_heading("## Final-demand mismatches")
        y_sorted = y_mismatches.copy()
        y_sorted["severity_rank"] = y_sorted["severity"].map(_severity_rank).fillna(-1).astype(int)
        y_sorted = y_sorted.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        y_human = _human_aggregate_table(y_sorted)
        lines.append(_markdown_table(y_human, columns=["Readable metric", "Eurostat sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Relative gap", "Direction", "Largest EXIOBASE contributors", "Plain-English interpretation"], max_rows=8))
        lines.append("")

    flagged_z_cells = z_cell_mismatches[z_cell_mismatches["flag"]].copy()
    if not flagged_z_cells.empty:
        _section_heading("## Intermediate supplier→buyer flow mismatches")
        flagged_z_cells["severity_rank"] = flagged_z_cells["severity"].map(_severity_rank).fillna(-1).astype(int)
        flagged_z_cells = flagged_z_cells.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        z_human = _human_z_cell_table(flagged_z_cells)
        lines.append(_markdown_table(z_human, columns=["Eurostat supplier sector", "Eurostat buyer sector", "Meaning of comparison", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Relative gap", "Direction", "Top EXIOBASE source subsectors", "Top EXIOBASE buyer subsectors", "Plain-English interpretation"], max_rows=12))
        lines.append("")
        lines.append("### Detailed drill-down for the top flagged Z-cell mismatches")
        lines.append("")
        for cell in flagged_z_cells.head(10).itertuples(index=False):
            supplier_display = _display_code_label(cell.supplier_code, cell.supplier_label)
            buyer_display = _display_code_label(cell.buyer_code, cell.buyer_label)
            lines.append(f"#### Detailed drill-down: {supplier_display} → {buyer_display}")
            lines.append("")
            lines.append("Eurostat comparison:")
            lines.append(f"- Eurostat supplier sector: {supplier_display}")
            lines.append(f"- Eurostat buyer sector: {buyer_display}")
            lines.append(f"- Eurostat Z value: {fmt_meur(cell.eurostat_z)} MEUR")
            lines.append(f"- EXIOBASE mapped Z value: {fmt_meur(cell.exio_mapped_z)} MEUR")
            lines.append(f"- Ratio: {fmt_ratio(cell.ratio_exio_to_eurostat)}")
            lines.append(f"- Difference: {_signed_meur(cell.signed_gap)} MEUR")
            source_rows = z_cell_drilldown[(z_cell_drilldown["supplier_code"].astype(str).eq(str(cell.supplier_code))) & (z_cell_drilldown["buyer_code"].astype(str).eq(str(cell.buyer_code))) & (z_cell_drilldown["role"].astype(str).eq("source_contributor"))]
            buyer_rows = z_cell_drilldown[(z_cell_drilldown["supplier_code"].astype(str).eq(str(cell.supplier_code))) & (z_cell_drilldown["buyer_code"].astype(str).eq(str(cell.buyer_code))) & (z_cell_drilldown["role"].astype(str).eq("buyer_contributor"))]
            if not source_rows.empty:
                source_human = source_rows.assign(
                    **{
                        "EXIOBASE source subsector": source_rows.apply(lambda r: _display_code_label(r.exio_subsector_code, r.exio_subsector_label), axis=1),
                        "Contribution, MEUR": source_rows["contribution_value"].map(fmt_meur),
                        "Share of EXIO mapped cell": source_rows["share_of_exio_mapped_cell"].map(fmt_pct),
                        "Share of Eurostat cell": source_rows["share_of_eurostat_cell"].map(fmt_pct),
                    }
                )
                lines.append("")
                lines.append(f"Top EXIOBASE source subsectors mapped into {supplier_display}:")
                lines.append(_markdown_table(source_human, columns=["EXIOBASE source subsector", "Contribution, MEUR", "Share of EXIO mapped cell", "Share of Eurostat cell"], max_rows=5))
            if not buyer_rows.empty:
                buyer_human = buyer_rows.assign(
                    **{
                        "EXIOBASE buyer subsector": buyer_rows.apply(lambda r: _display_code_label(r.exio_subsector_code, r.exio_subsector_label), axis=1),
                        "Contribution, MEUR": buyer_rows["contribution_value"].map(fmt_meur),
                        "Share of EXIO mapped cell": buyer_rows["share_of_exio_mapped_cell"].map(fmt_pct),
                    }
                )
                lines.append("")
                lines.append(f"Top EXIOBASE buyer subsectors mapped into {buyer_display}:")
                lines.append(_markdown_table(buyer_human, columns=["EXIOBASE buyer subsector", "Contribution, MEUR", "Share of EXIO mapped cell"], max_rows=5))
            lines.append("")
            lines.append("Summary:")
            lines.append(
                _human_flow_prose(
                    supplier_display=supplier_display,
                    buyer_display=buyer_display,
                    direction=_direction_label(cell.exio_mapped_z, cell.eurostat_z),
                    exio_value=cell.exio_mapped_z,
                    euro_value=cell.eurostat_z,
                    source_contribs=_display_contributor_string(getattr(cell, "top_source_subsectors", "")),
                    buyer_contribs=_display_contributor_string(getattr(cell, "top_buyer_subsectors", "")),
                )
            )
            lines.append("")

    if not subcomponent_exceeds_aggregate.empty:
        _section_heading("## Single EXIOBASE subsectors larger than Eurostat aggregates")
        lines.append("This section flags cases where a single EXIOBASE subsector mapped into a Eurostat aggregate is larger than the whole Eurostat aggregate or cell it maps into. These are especially important because they indicate that the mismatch is not diffuse: it is concentrated in one source subsector.")
        lines.append("")
        sub_sorted = subcomponent_exceeds_aggregate.copy()
        sub_sorted["severity_rank"] = sub_sorted["severity"].map(_severity_rank).fillna(-1).astype(int)
        sub_sorted = sub_sorted.sort_values(["severity_rank", "abs_gap", "ratio"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        sub_human = _human_subcomponent_table(sub_sorted)
        lines.append(_markdown_table(sub_human, columns=["Readable metric", "Eurostat aggregate sector", "Eurostat buyer sector", "EXIOBASE subsector", "EXIOBASE subsector value, MEUR", "Eurostat aggregate/cell value, MEUR", "EXIO / Eurostat ratio", "Reason"], max_rows=12))
        lines.append("")

    if not y_drilldown.empty:
        _section_heading("## Final-demand drill-down")
        if "parent_y_flagged" in y_drilldown.columns:
            y_flagged = y_drilldown[y_drilldown["parent_y_flagged"]].copy()
        else:
            y_flagged = y_drilldown.copy()
        if not y_flagged.empty:
            y_human = _human_y_drilldown_table(y_flagged.sort_values(["target_code", "contribution_rank"], kind="stable"))
            lines.append(_markdown_table(y_human, columns=["Eurostat sector", "EXIOBASE contributing source", "Contribution, MEUR", "Share of EXIO mapped final demand", "Share of Eurostat final demand", "Contribution rank", "Interpretation"], max_rows=12))
        else:
            lines.append("No flagged Y mismatches to drill down.")
        lines.append("")

    biofuel_rows = z_cell_mismatches[z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")].copy()
    biofuel_rows = biofuel_rows[biofuel_rows["flag"]].copy()
    if not biofuel_rows.empty:
        _section_heading("## CPA_C20 / Biofuels Diagnostic")
        lines.append("This section focuses on CPA_C20 because liquid biofuels are mapped statistically into CPA_C20 but may be analytically important for the energy-transition split. The table asks: after mapping EXIOBASE products such as p24.g — Biogasoline and p24.h — Biodiesels into CPA_C20, do the resulting CPA_C20 flows agree with Eurostat?")
        lines.append("")
        c20_focus_all = z_cell_mismatches[(z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")) & (z_cell_mismatches["buyer_code"].astype(str).eq("CPA_G46"))].copy()
        c20_focus = biofuel_rows[biofuel_rows["buyer_code"].astype(str).eq("CPA_G46")].copy()
        if c20_focus.empty:
            c20_focus = c20_focus_all.copy()
        elif not c20_focus_all.empty:
            c20_focus = pd.concat([c20_focus, c20_focus_all[~c20_focus_all.index.isin(c20_focus.index)]], axis=0)
        biofuel_display_rows = biofuel_rows.copy()
        if not c20_focus_all.empty:
            biofuel_display_rows = pd.concat([biofuel_display_rows, c20_focus_all[~c20_focus_all.index.isin(biofuel_display_rows.index)]], axis=0)
        if not biofuel_rows.empty:
            biofuel_rows = biofuel_rows.assign(severity_rank=biofuel_rows["severity"].map(_severity_rank).fillna(-1).astype(int))
            biofuel_rows = biofuel_rows.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable")
        if not c20_focus.empty:
            c20_focus = c20_focus.assign(severity_rank=c20_focus["severity"].map(_severity_rank).fillna(-1).astype(int))
            c20_focus = c20_focus.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        if not c20_focus.empty:
            c20_row = c20_focus.iloc[0]
            lines.append(f"Explicit focus row: {_display_code_label(c20_row.supplier_code, c20_row.supplier_label)} → {_display_code_label(c20_row.buyer_code, c20_row.buyer_label)}.")
            lines.append("")
        bio_human = _human_z_cell_table(biofuel_display_rows.assign(severity_rank=biofuel_display_rows.get("severity", pd.Series(dtype=object)).map(_severity_rank).fillna(-1).astype(int)).sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"], errors="ignore"))
        lines.append(_markdown_table(bio_human, columns=["Eurostat supplier sector", "Eurostat buyer sector", "Meaning of comparison", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Top EXIOBASE source subsectors", "Top EXIOBASE buyer subsectors", "Plain-English interpretation"], max_rows=10))
        lines.append("")
        if not c20_focus.empty:
            lines.append("### Focused CPA_C20 → CPA_G46 check")
            lines.append("")
            focus = c20_focus.iloc[0]
            lines.append(f"- Eurostat supplier sector: {_display_code_label(focus.supplier_code, focus.supplier_label)}")
            lines.append(f"- Eurostat buyer sector: {_display_code_label(focus.buyer_code, focus.buyer_label)}")
            lines.append(f"- EXIOBASE mapped flow: {fmt_meur(focus.exio_mapped_z)} MEUR")
            lines.append(f"- Eurostat flow: {fmt_meur(focus.eurostat_z)} MEUR")
            lines.append(f"- Ratio: {fmt_ratio(focus.ratio_exio_to_eurostat)}")
            lines.append(f"- Difference: {_signed_meur(focus.signed_gap)} MEUR")
            focus_source = z_cell_drilldown[(z_cell_drilldown["supplier_code"].astype(str).eq("CPA_C20")) & (z_cell_drilldown["buyer_code"].astype(str).eq("CPA_G46")) & (z_cell_drilldown["role"].astype(str).eq("source_contributor"))].copy()
            if not focus_source.empty:
                p24g = focus_source[focus_source["exio_subsector_code"].astype(str).eq("p24.g")]
                p24h = focus_source[focus_source["exio_subsector_code"].astype(str).eq("p24.h")]
                if not p24g.empty:
                    p24g_value = float(p24g.iloc[0]["contribution_value"])
                    lines.append(f"- p24.g — Biogasoline alone contributes {fmt_meur(p24g_value)} MEUR and exceeds the whole Eurostat CPA_C20 → CPA_G46 cell ({fmt_meur(focus.eurostat_z)} MEUR).")
                if not p24h.empty:
                    p24h_value = float(p24h.iloc[0]["contribution_value"])
                    lines.append(f"- p24.h — Biodiesels contributes {fmt_meur(p24h_value)} MEUR.")
                subtotal = float(focus_source[focus_source["exio_subsector_code"].astype(str).isin(["p24.g", "p24.h"])]["contribution_value"].sum())
                if subtotal > 0:
                    lines.append(f"- Liquid biofuel subtotal from p24.g + p24.h is {fmt_meur(subtotal)} MEUR.")
            if not focus_source.empty:
                focus_source_human = focus_source.assign(
                    **{
                        "EXIOBASE source subsector": focus_source.apply(lambda r: _display_code_label(r.exio_subsector_code, r.exio_subsector_label), axis=1),
                        "Contribution, MEUR": focus_source["contribution_value"].map(fmt_meur),
                        "Share of EXIO mapped cell": focus_source["share_of_exio_mapped_cell"].map(fmt_pct),
                        "Share of Eurostat cell": focus_source["share_of_eurostat_cell"].map(fmt_pct),
                    }
                )
                lines.append("")
                lines.append("Top EXIOBASE source subsectors for CPA_C20 → CPA_G46:")
                lines.append(_markdown_table(focus_source_human, columns=["EXIOBASE source subsector", "Contribution, MEUR", "Share of EXIO mapped cell", "Share of Eurostat cell"], max_rows=5))
            lines.append("")

    _section_heading("## Glossary")
    lines.append("- Z: intermediate input-use matrix. A Z cell is a supplier-sector → buyer-sector flow.")
    lines.append("- Z row total: total intermediate sales by a supplying sector to all buyer sectors.")
    lines.append("- Z column total: total intermediate purchases by a buyer sector from all supplying sectors.")
    lines.append("- Y: final demand.")
    lines.append("- X: flow-specific total-use proxy if independently available; otherwise, in this audit, a flow-specific total-use proxy is used.")
    lines.append("- EXIOBASE mapped value: EXIOBASE value after applying the EXIOBASE→Eurostat concordance.")
    lines.append("- Eurostat value: corresponding Eurostat IOT value.")
    lines.append("- Subcomponent exceeds aggregate: a single EXIOBASE subsector is larger than the whole Eurostat aggregate or cell it maps into.")
    lines.append("")

    _section_heading("## Technical appendix: raw metric names")
    technical_rows = pd.DataFrame(
        [
            {"Raw metric": "Z_CELL", "Human-readable meaning": "Intermediate supplier→buyer flow"},
            {"Raw metric": "Z_ROW_INTERMEDIATE_SALES", "Human-readable meaning": "Total intermediate sales by supplier"},
            {"Raw metric": "Z_COL_INTERMEDIATE_PURCHASES", "Human-readable meaning": "Total intermediate purchases by buyer"},
            {"Raw metric": "Y_FINAL_DEMAND_TOTAL", "Human-readable meaning": "Final demand total"},
            {"Raw metric": "TOTAL_USE_Z_PLUS_Y", "Human-readable meaning": "Intermediate sales plus final demand"},
            {"Raw metric": "X_OUTPUT_PROXY_TOTAL_USE", "Human-readable meaning": "Flow-specific total-use proxy"},
        ]
    )
    lines.append(_markdown_table(technical_rows, columns=["Raw metric", "Human-readable meaning"], max_rows=20))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_concordance_audit_outputs(result: ConcordanceAuditResult, output_dir: str | Path) -> dict[str, Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_command = str(result.metadata.get("run_command", "")).strip()
    paths = {
        "audit_metadata.json": outdir / "audit_metadata.json",
        "audit_mapping_coverage.csv": outdir / "audit_mapping_coverage.csv",
        "audit_internal_identities.csv": outdir / "audit_internal_identities.csv",
        "audit_aggregate_totals.csv": outdir / "audit_aggregate_totals.csv",
        "audit_x_mismatches.csv": outdir / "audit_x_mismatches.csv",
        "audit_y_mismatches.csv": outdir / "audit_y_mismatches.csv",
        "audit_z_row_col_mismatches.csv": outdir / "audit_z_row_col_mismatches.csv",
        "audit_z_cell_mismatches.csv": outdir / "audit_z_cell_mismatches.csv",
        "audit_z_cell_mismatches_flagged.csv": outdir / "audit_z_cell_mismatches_flagged.csv",
        "audit_z_cell_drilldown.csv": outdir / "audit_z_cell_drilldown.csv",
        "audit_y_drilldown.csv": outdir / "audit_y_drilldown.csv",
        "audit_subcomponent_exceeds_aggregate.csv": outdir / "audit_subcomponent_exceeds_aggregate.csv",
        "audit_top_flags.md": outdir / "audit_top_flags.md",
        "audit_exio_eurostat_concordance.md": outdir / f"audit_exio_eurostat_concordance_{result.metadata.get('country', '')}_{result.metadata.get('year', '')}.md",
        "audit_run.sh": outdir / "audit_run.sh",
    }
    Path(paths["audit_metadata.json"]).write_text(json.dumps(result.metadata, indent=2, sort_keys=True), encoding="utf-8")
    result.mapping_coverage.to_csv(paths["audit_mapping_coverage.csv"], index=False)
    result.internal_identities.to_csv(paths["audit_internal_identities.csv"], index=False)
    result.aggregate_totals.to_csv(paths["audit_aggregate_totals.csv"], index=False)
    result.x_mismatches.to_csv(paths["audit_x_mismatches.csv"], index=False)
    result.y_mismatches.to_csv(paths["audit_y_mismatches.csv"], index=False)
    result.z_row_col_mismatches.to_csv(paths["audit_z_row_col_mismatches.csv"], index=False)
    result.z_cell_mismatches.to_csv(paths["audit_z_cell_mismatches.csv"], index=False)
    result.z_cell_mismatches[result.z_cell_mismatches["flag"]].to_csv(paths["audit_z_cell_mismatches_flagged.csv"], index=False)
    result.z_cell_drilldown.to_csv(paths["audit_z_cell_drilldown.csv"], index=False)
    result.y_drilldown.to_csv(paths["audit_y_drilldown.csv"], index=False)
    result.subcomponent_exceeds_aggregate.to_csv(paths["audit_subcomponent_exceeds_aggregate.csv"], index=False)
    top_agg_human = _human_aggregate_table(
        result.aggregate_totals[result.aggregate_totals["flag"]]
        .assign(severity_rank=lambda frame: frame["severity"].map(_severity_rank).fillna(-1).astype(int))
        .sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable")
        .drop(columns=["severity_rank"])
    ).head(10)
    top_cell_human = _human_z_cell_table(
        result.z_cell_mismatches[result.z_cell_mismatches["flag"]]
        .assign(severity_rank=lambda frame: frame["severity"].map(_severity_rank).fillna(-1).astype(int))
        .sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable")
        .drop(columns=["severity_rank"])
    ).head(10)
    top_sub_human = _human_subcomponent_table(
        result.subcomponent_exceeds_aggregate
        .assign(severity_rank=lambda frame: frame["severity"].map(_severity_rank).fillna(-1).astype(int))
        .sort_values(["severity_rank", "abs_gap", "ratio"], ascending=[False, False, False], kind="stable")
        .drop(columns=["severity_rank"])
    ).head(10)
    c20_flagged = result.z_cell_mismatches[(result.z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")) & (result.z_cell_mismatches["flag"])]
    c20_focus = c20_flagged[c20_flagged["buyer_code"].astype(str).eq("CPA_G46")].copy()
    c20_force = result.z_cell_mismatches[(result.z_cell_mismatches["supplier_code"].astype(str).eq("CPA_C20")) & (result.z_cell_mismatches["buyer_code"].astype(str).eq("CPA_G46"))].copy()
    if c20_focus.empty:
        c20_focus = c20_force.copy()
    elif not c20_force.empty:
        c20_focus = pd.concat([c20_focus, c20_force[~c20_force.index.isin(c20_focus.index)]], axis=0)
    c20_display_rows = c20_flagged.copy()
    if not c20_force.empty:
        c20_display_rows = pd.concat([c20_display_rows, c20_force[~c20_force.index.isin(c20_display_rows.index)]], axis=0)

    def _table_or_message(frame: pd.DataFrame, columns: Sequence[str], empty_message: str) -> str:
        if frame.empty:
            return empty_message
        return _markdown_table(frame, columns=columns, max_rows=10)

    top_flags_lines = [
        f"# Top Concordance Flags: {result.metadata.get('country', '')} {result.metadata.get('year', '')}",
        "",
        "## How to read this short report",
        "",
        "## Flow comparison",
        "",
        f"- EXIOBASE flow: {result.metadata.get('exio_flow', 'unknown')}",
        f"- Eurostat `stk_flow`: {result.metadata.get('eurostat_flow', 'unknown')}",
        f"- x-like rows are {human_metric_label(METRIC_X_OUTPUT_PROXY_TOTAL_USE)} comparisons, not independently observed gross-output comparisons.",
        "",
        "- Intermediate flow cell: supplier → buyer compares one Eurostat supplier sector and one Eurostat buyer sector.",
        "- Intermediate sales by supplying sector sums all intermediate sales from one supplying sector across all buyer sectors.",
        "- Final demand total compares final demand for that sector/product.",
        "- EXIOBASE mapped value is the EXIOBASE value after applying the concordance.",
        "- Eurostat value is the corresponding Eurostat IOT value.",
        "",
        "## Top 10 aggregate mismatches",
        "",
        _table_or_message(top_agg_human, ["Readable metric", "Eurostat sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Direction", "Largest EXIOBASE contributors", "Plain-English interpretation"], "No flagged aggregate mismatches at the current threshold."),
        "",
        "## Top 10 supplier→buyer flow mismatches",
        "",
        _table_or_message(top_cell_human, ["Eurostat supplier sector", "Eurostat buyer sector", "Meaning of comparison", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "EXIO / Eurostat ratio", "Direction", "Top EXIOBASE source subsectors", "Plain-English interpretation"], "No flagged supplier→buyer flow mismatches at the current threshold."),
        "",
        "## Top single-subsector exceedance cases",
        "",
        _table_or_message(top_sub_human, ["Readable metric", "Eurostat aggregate sector", "Eurostat buyer sector", "EXIOBASE subsector", "EXIOBASE subsector value, MEUR", "Eurostat aggregate/cell value, MEUR", "EXIO / Eurostat ratio", "Reason"], "No single-subsector exceedances were flagged at the current threshold."),
        "",
        "## CPA_C20 / liquid biofuels check",
        "",
        "This section focuses on CPA_C20 because liquid biofuels are mapped statistically into CPA_C20 but may be analytically important for the energy-transition split.",
        "",
    ]
    if not c20_focus.empty:
        c20_focus = c20_focus.assign(severity_rank=lambda frame: frame["severity"].map(_severity_rank).fillna(-1).astype(int)).sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        c20_row = c20_focus.iloc[0]
        top_flags_lines.extend([
            f"Explicit focus row: {_display_code_label(c20_row.supplier_code, c20_row.supplier_label)} → {_display_code_label(c20_row.buyer_code, c20_row.buyer_label)}.",
            "",
            _table_or_message(
                _human_z_cell_table(
                    c20_display_rows.assign(severity_rank=lambda frame: frame["severity"].map(_severity_rank).fillna(-1).astype(int)).sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
                ),
                ["Eurostat supplier sector", "Eurostat buyer sector", "Meaning of comparison", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "EXIO / Eurostat ratio", "Direction", "Top EXIOBASE source subsectors", "Top EXIOBASE buyer subsectors", "Plain-English interpretation"],
                "No CPA_C20 → CPA_G46 row was flagged in the current run.",
            ),
            "",
        ])
    else:
        top_flags_lines.append("No CPA_C20 → CPA_G46 row was flagged in the current run.")
        top_flags_lines.append("")
    Path(paths["audit_top_flags.md"]).write_text("\n".join(top_flags_lines).rstrip() + "\n", encoding="utf-8")
    Path(paths["audit_exio_eurostat_concordance.md"]).write_text(result.report_markdown, encoding="utf-8")
    if run_command:
        script_text = "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Generated command for {result.metadata.get('country', '')} {result.metadata.get('year', '')}",
            run_command,
            "",
        ])
    else:
        script_text = "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# No command was recorded for this audit run.",
            "",
        ])
    run_script_path = Path(paths["audit_run.sh"])
    run_script_path.write_text(script_text, encoding="utf-8")
    run_script_path.chmod(0o755)
    return paths


def load_default_label_maps(repo_root: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    root = Path(repo_root)
    exio_labels = _read_label_map(root / "scripts" / "mappings" / "exiobase_products_dictionary.csv")
    euro_labels = _read_label_map(root / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv")
    return exio_labels, euro_labels


def _human_mapping_coverage_table(mapping_coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in mapping_coverage.itertuples(index=False):
        rows.append(
            {
                "EXIOBASE source sector": _display_code_label(row.source_code, row.source_label),
                "Mapped targets": int(row.mapped_targets),
                "Mapping weight sum": fmt_ratio(row.sum_mapping_weight),
                "Status": row.status.replace("_", " "),
                "EXIOBASE x, MEUR": fmt_meur(row.x_value),
                "EXIOBASE y total, MEUR": fmt_meur(row.y_value),
                "EXIOBASE Z row total, MEUR": fmt_meur(row.z_row_total),
                "EXIOBASE Z col total, MEUR": fmt_meur(row.z_col_total),
                "Reason": row.reason,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["EXIOBASE source sector", "Mapped targets", "Mapping weight sum", "Status", "EXIOBASE x, MEUR", "EXIOBASE y total, MEUR", "EXIOBASE Z row total, MEUR", "EXIOBASE Z col total, MEUR", "Reason"],
    )


def _human_internal_identities_table(internal_identities: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in internal_identities.itertuples(index=False):
        rows.append(
            {
                "Database": row.database,
                "Sector": _display_code_label(row.code, row.code),
                "EXIOBASE mapped x, MEUR": fmt_meur(row.x),
                "Z row total, MEUR": fmt_meur(row.z_row_total),
                "Y total, MEUR": fmt_meur(row.y_total),
                "Z + Y, MEUR": fmt_meur(row.z_plus_y),
                "Absolute gap, MEUR": fmt_meur(row.abs_gap),
                "Relative gap": fmt_pct(row.rel_gap),
                "Severity": row.severity,
                "Status": _human_identity_status(row.status),
                "Identity mode": row.identity_mode,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Database", "Sector", "EXIOBASE mapped x, MEUR", "Z row total, MEUR", "Y total, MEUR", "Z + Y, MEUR", "Absolute gap, MEUR", "Relative gap", "Severity", "Status", "Identity mode"],
    )


def _human_aggregate_table(aggregate_totals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in aggregate_totals.itertuples(index=False):
        rows.append(
            {
                "Readable metric": _display_metric(row.metric),
                "Eurostat sector": _display_code_label(row.target_code, row.target_label),
                "EXIOBASE mapped value, MEUR": fmt_meur(row.exio_mapped_value),
                "Eurostat value, MEUR": fmt_meur(row.eurostat_value),
                "EXIO / Eurostat ratio": fmt_ratio(row.ratio_exio_to_eurostat),
                "Difference, MEUR": _signed_meur(row.signed_gap),
                "Relative gap": fmt_pct(row.rel_gap),
                "Direction": _direction_label(row.exio_mapped_value, row.eurostat_value),
                "Largest EXIOBASE contributors": _display_contributor_string(row.top_exio_contributors),
                "Plain-English interpretation": _human_comparison_prose(
                    metric=str(row.metric),
                    direction=_direction_label(row.exio_mapped_value, row.eurostat_value),
                    exio_value=row.exio_mapped_value,
                    euro_value=row.eurostat_value,
                    contributors=_display_contributor_string(row.top_exio_contributors),
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Readable metric", "Eurostat sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Relative gap", "Direction", "Largest EXIOBASE contributors", "Plain-English interpretation"],
    )


def _human_z_cell_table(z_cell_mismatches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in z_cell_mismatches.itertuples(index=False):
        rows.append(
            {
                "Eurostat supplier sector": _display_code_label(row.supplier_code, row.supplier_label),
                "Eurostat buyer sector": _display_code_label(row.buyer_code, row.buyer_label),
                "Meaning of comparison": f"Intermediate flow from {_display_code_label(row.supplier_code, row.supplier_label)} to {_display_code_label(row.buyer_code, row.buyer_label)}",
                "EXIOBASE mapped flow, MEUR": fmt_meur(row.exio_mapped_z),
                "Eurostat flow, MEUR": fmt_meur(row.eurostat_z),
                "EXIO / Eurostat ratio": fmt_ratio(row.ratio_exio_to_eurostat),
                "Difference, MEUR": _signed_meur(row.signed_gap),
                "Relative gap": fmt_pct(row.rel_gap),
                "Direction": _direction_label(row.exio_mapped_z, row.eurostat_z),
                "Top EXIOBASE source subsectors": _display_contributor_string(row.top_source_subsectors),
                "Top EXIOBASE buyer subsectors": _display_contributor_string(row.top_buyer_subsectors),
                "Plain-English interpretation": _human_flow_prose(
                    supplier_display=_display_code_label(row.supplier_code, row.supplier_label),
                    buyer_display=_display_code_label(row.buyer_code, row.buyer_label),
                    direction=_direction_label(row.exio_mapped_z, row.eurostat_z),
                    exio_value=row.exio_mapped_z,
                    euro_value=row.eurostat_z,
                    source_contribs=_display_contributor_string(row.top_source_subsectors),
                    buyer_contribs=_display_contributor_string(row.top_buyer_subsectors),
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Eurostat supplier sector", "Eurostat buyer sector", "Meaning of comparison", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "EXIO / Eurostat ratio", "Difference, MEUR", "Relative gap", "Direction", "Top EXIOBASE source subsectors", "Top EXIOBASE buyer subsectors", "Plain-English interpretation"],
    )


def _human_subcomponent_table(subcomponent_exceeds_aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in subcomponent_exceeds_aggregate.itertuples(index=False):
        if row.metric == METRIC_Z_CELL:
            aggregate_display = _display_code_label(row.target_code, row.target_label)
            buyer_display = _display_code_label(row.buyer_code, row.buyer_label)
        else:
            aggregate_display = _display_code_label(row.target_code, row.target_label)
            buyer_display = ""
        rows.append(
            {
                "Readable metric": _display_metric(row.metric),
                "Eurostat aggregate sector": aggregate_display,
                "Eurostat buyer sector": buyer_display,
                "EXIOBASE subsector": _display_code_label(row.exio_subsector_code, row.exio_subsector_label),
                "EXIOBASE subsector value, MEUR": fmt_meur(row.exio_subsector_value),
                "Eurostat aggregate/cell value, MEUR": fmt_meur(row.eurostat_aggregate_value),
                "EXIO / Eurostat ratio": fmt_ratio(row.ratio),
                "Reason": _human_subcomponent_prose(
                    metric=str(row.metric),
                    aggregate_display=aggregate_display,
                    buyer_display=buyer_display,
                    exio_subsector_display=_display_code_label(row.exio_subsector_code, row.exio_subsector_label),
                    exio_subsector_value=row.exio_subsector_value,
                    euro_value=row.eurostat_aggregate_value,
                    ratio=row.ratio,
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Readable metric", "Eurostat aggregate sector", "Eurostat buyer sector", "EXIOBASE subsector", "EXIOBASE subsector value, MEUR", "Eurostat aggregate/cell value, MEUR", "EXIO / Eurostat ratio", "Reason"],
    )


def _human_y_drilldown_table(y_drilldown: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in y_drilldown.itertuples(index=False):
        rows.append(
            {
                "Eurostat sector": _display_code_label(row.target_code, row.target_label),
                "EXIOBASE contributing source": _display_code_label(row.exio_subsector_code, row.exio_subsector_label),
                "Contribution, MEUR": fmt_meur(row.contribution_value),
                "Share of EXIO mapped final demand": fmt_pct(row.share_of_exio_mapped_y),
                "Share of Eurostat final demand": fmt_pct(row.share_of_eurostat_y),
                "Contribution rank": int(row.contribution_rank),
                "Interpretation": _human_y_prose(
                    source_display=_display_code_label(row.exio_subsector_code, row.exio_subsector_label),
                    contribution_value=row.contribution_value,
                    share_of_exio=row.share_of_exio_mapped_y,
                    share_of_eurostat=row.share_of_eurostat_y,
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Eurostat sector", "EXIOBASE contributing source", "Contribution, MEUR", "Share of EXIO mapped final demand", "Share of Eurostat final demand", "Contribution rank", "Interpretation"],
    )


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
