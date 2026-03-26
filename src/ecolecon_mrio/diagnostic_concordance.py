from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ecolecon_mrio.mapping_csv import validate_mapping_weights


CANONICAL_WEIGHTED_COLUMNS: tuple[str, ...] = (
    "aggregate_sector",
    "exio_sector",
    "weight",
    "enabled",
    "notes",
    "source_name",
    "provenance",
    "transformation_rule",
)

CANONICAL_SPLIT_COLUMNS: tuple[str, ...] = (
    "aggregate_sector",
    "subsector",
    "exio_sector",
    "enabled",
    "notes",
    "source_name",
    "provenance",
    "transformation_rule",
)

CANONICAL_CATEGORY_COLUMNS: tuple[str, ...] = (
    "Category",
    "TargetSector",
    "Share_total",
    "notes",
    "source_name",
    "provenance",
    "transformation_rule",
)

CANONICAL_AGGREGATION_COLUMNS: tuple[str, ...] = (
    "from_sector",
    "to_sector",
    "enabled",
    "notes",
    "source_name",
    "provenance",
    "transformation_rule",
)


def _parse_boolish(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s == "":
        return True
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {x!r}")


def normalize_label(value: object) -> str:
    return " ".join(str(value).strip().casefold().replace("\u00a0", " ").split())


def _ensure_non_empty(df: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    empties = pd.Series(False, index=df.index)
    for column in columns:
        empties = empties | (df[column].astype(str).str.strip() == "")
    if empties.any():
        raise ValueError(f"{label} contains empty required values in rows: {empties[empties].index.tolist()}")


def _combine_notes(values: pd.Series) -> str:
    notes: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str):
        note = value.strip()
        if note and note not in seen:
            notes.append(note)
            seen.add(note)
    return " | ".join(notes)


def _validate_provenance(provenance: str) -> str:
    allowed = {"official", "semi-official", "heuristic", "methodological"}
    prov = str(provenance).strip().lower()
    if prov not in allowed:
        raise ValueError(f"Unsupported provenance {provenance!r}; expected one of {sorted(allowed)}")
    return prov


def normalize_weighted_mapping_frame(
    raw_df: pd.DataFrame,
    *,
    aggregate_col: str,
    exio_col: str,
    weight_col: str | None = None,
    enabled_col: str | None = None,
    notes_col: str | None = None,
    source_name: str,
    provenance: str,
    transformation_rule: str,
) -> pd.DataFrame:
    df = raw_df.copy().fillna("")
    missing = [c for c in (aggregate_col, exio_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Weighted mapping input is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "aggregate_sector": df[aggregate_col].astype(str).str.strip(),
            "exio_sector": df[exio_col].astype(str).str.strip(),
            "weight": 1.0 if weight_col is None else pd.to_numeric(df[weight_col], errors="raise"),
            "enabled": "1"
            if enabled_col is None
            else df[enabled_col].map(_parse_boolish).map(lambda x: "1" if x else "0"),
            "notes": "" if notes_col is None else df[notes_col].astype(str).str.strip(),
            "source_name": str(source_name).strip(),
            "provenance": _validate_provenance(provenance),
            "transformation_rule": str(transformation_rule).strip(),
        }
    )
    _ensure_non_empty(out, ("aggregate_sector", "exio_sector"), label="Weighted mapping")

    out = out.groupby(["aggregate_sector", "exio_sector"], as_index=False).agg(
        {
            "weight": "sum",
            "enabled": "max",
            "notes": _combine_notes,
            "source_name": "first",
            "provenance": "first",
            "transformation_rule": "first",
        }
    )
    validate_weighted_mapping(
        out,
        aggregate_col="aggregate_sector",
        exio_col="exio_sector",
        weight_col="weight",
        enabled_col="enabled",
    )
    return out.loc[:, list(CANONICAL_WEIGHTED_COLUMNS)]


def validate_weighted_mapping(
    mapping_df: pd.DataFrame,
    *,
    aggregate_col: str = "aggregate_sector",
    exio_col: str = "exio_sector",
    weight_col: str = "weight",
    enabled_col: str = "enabled",
) -> None:
    df = mapping_df.copy()
    required = {aggregate_col, exio_col, weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Weighted mapping frame missing required columns: {sorted(missing)}")

    if enabled_col in df.columns:
        df = df[df[enabled_col].map(_parse_boolish)].copy()
    if df.empty:
        raise ValueError("Weighted mapping frame is empty after enabled filtering")

    _ensure_non_empty(
        df.rename(columns={aggregate_col: "aggregate_sector", exio_col: "exio_sector"}),
        ("aggregate_sector", "exio_sector"),
        label="Weighted mapping",
    )
    df[weight_col] = pd.to_numeric(df[weight_col], errors="raise")

    mapping: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        agg = str(row[aggregate_col]).strip()
        exio = str(row[exio_col]).strip()
        weight = float(row[weight_col])
        mapping.setdefault(agg, {})
        mapping[agg][exio] = mapping[agg].get(exio, 0.0) + weight
    validate_mapping_weights(mapping)


def normalize_split_mapping_frame(
    raw_df: pd.DataFrame,
    *,
    aggregate_col: str,
    exio_col: str,
    subsector_col: str | None = None,
    enabled_col: str | None = None,
    notes_col: str | None = None,
    source_name: str,
    provenance: str,
    transformation_rule: str,
) -> pd.DataFrame:
    df = raw_df.copy().fillna("")
    missing = [c for c in (aggregate_col, exio_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Split mapping input is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "aggregate_sector": df[aggregate_col].astype(str).str.strip(),
            "subsector": "" if subsector_col is None else df[subsector_col].astype(str).str.strip(),
            "exio_sector": df[exio_col].astype(str).str.strip(),
            "enabled": "1"
            if enabled_col is None
            else df[enabled_col].map(_parse_boolish).map(lambda x: "1" if x else "0"),
            "notes": "" if notes_col is None else df[notes_col].astype(str).str.strip(),
            "source_name": str(source_name).strip(),
            "provenance": _validate_provenance(provenance),
            "transformation_rule": str(transformation_rule).strip(),
        }
    )
    mask_no_sub = out["subsector"] == ""
    out.loc[mask_no_sub, "subsector"] = out.loc[mask_no_sub, "exio_sector"]
    _ensure_non_empty(out, ("aggregate_sector", "subsector", "exio_sector"), label="Split mapping")
    return out.loc[:, list(CANONICAL_SPLIT_COLUMNS)]


def normalize_category_target_mapping_frame(
    raw_df: pd.DataFrame,
    *,
    category_col: str,
    target_col: str,
    share_col: str | None = None,
    notes_col: str | None = None,
    source_name: str,
    provenance: str,
    transformation_rule: str,
) -> pd.DataFrame:
    df = raw_df.copy().fillna("")
    missing = [c for c in (category_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Category mapping input is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "Category": df[category_col].astype(str).str.strip(),
            "TargetSector": df[target_col].astype(str).str.strip(),
            "Share_total": np.nan if share_col is None else pd.to_numeric(df[share_col], errors="coerce"),
            "notes": "" if notes_col is None else df[notes_col].astype(str).str.strip(),
            "source_name": str(source_name).strip(),
            "provenance": _validate_provenance(provenance),
            "transformation_rule": str(transformation_rule).strip(),
        }
    )
    _ensure_non_empty(out, ("Category", "TargetSector"), label="Category mapping")
    return out.loc[:, list(CANONICAL_CATEGORY_COLUMNS)]


def normalize_aggregation_mapping_frame(
    raw_df: pd.DataFrame,
    *,
    from_col: str,
    to_col: str,
    enabled_col: str | None = None,
    notes_col: str | None = None,
    source_name: str,
    provenance: str,
    transformation_rule: str,
) -> pd.DataFrame:
    df = raw_df.copy().fillna("")
    missing = [c for c in (from_col, to_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Aggregation mapping input is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "from_sector": df[from_col].astype(str).str.strip(),
            "to_sector": df[to_col].astype(str).str.strip(),
            "enabled": "1"
            if enabled_col is None
            else df[enabled_col].map(_parse_boolish).map(lambda x: "1" if x else "0"),
            "notes": "" if notes_col is None else df[notes_col].astype(str).str.strip(),
            "source_name": str(source_name).strip(),
            "provenance": _validate_provenance(provenance),
            "transformation_rule": str(transformation_rule).strip(),
        }
    )
    _ensure_non_empty(out, ("from_sector", "to_sector"), label="Aggregation mapping")
    return out.loc[:, list(CANONICAL_AGGREGATION_COLUMNS)]


def build_mapping_audit(mapping_df: pd.DataFrame, *, mapping_family: str) -> pd.DataFrame:
    df = mapping_df.copy()
    df["mapping_family"] = str(mapping_family)
    if "enabled" in df.columns:
        df["is_enabled"] = df["enabled"].map(_parse_boolish)
    else:
        df["is_enabled"] = True
    if "notes" not in df.columns:
        df["notes"] = ""
    df["has_notes"] = df["notes"].astype(str).str.strip() != ""

    if {"aggregate_sector", "exio_sector", "weight"}.issubset(df.columns):
        enabled_df = df[df["is_enabled"]].copy()
        weight_sums = enabled_df.groupby("exio_sector", dropna=False)["weight"].sum().rename("weight_sum_by_exio")
        df = df.merge(weight_sums, on="exio_sector", how="left")
        df["weight_sum_by_exio"] = df["weight_sum_by_exio"].fillna(0.0)
        df["weight_sum_ok"] = np.isclose(df["weight_sum_by_exio"], 1.0, atol=1e-9)
    return df


@dataclass(frozen=True)
class WeightedAllocationResult:
    detail: pd.DataFrame
    aggregate_totals: pd.Series
    source_reconstructed: pd.Series


def allocate_weighted_values(
    source_values: pd.Series,
    mapping_df: pd.DataFrame,
    *,
    aggregate_col: str = "aggregate_sector",
    exio_col: str = "exio_sector",
    weight_col: str = "weight",
    enabled_col: str = "enabled",
) -> WeightedAllocationResult:
    validate_weighted_mapping(
        mapping_df,
        aggregate_col=aggregate_col,
        exio_col=exio_col,
        weight_col=weight_col,
        enabled_col=enabled_col,
    )
    df = mapping_df.copy()
    if enabled_col in df.columns:
        df = df[df[enabled_col].map(_parse_boolish)].copy()

    values = source_values.copy().astype("float64")
    values.index = values.index.astype(str)
    df["source_value"] = df[exio_col].astype(str).map(values).fillna(0.0)
    df["allocated_value"] = pd.to_numeric(df[weight_col], errors="raise") * df["source_value"]

    aggregate_totals = df.groupby(aggregate_col)["allocated_value"].sum().sort_index()
    source_reconstructed = df.groupby(exio_col)["allocated_value"].sum().sort_index()
    return WeightedAllocationResult(detail=df, aggregate_totals=aggregate_totals, source_reconstructed=source_reconstructed)


def build_weighted_value_comparison(
    source_values: pd.Series,
    reference_values: pd.Series,
    mapping_df: pd.DataFrame,
    *,
    scale_to_reference: bool = True,
) -> pd.DataFrame:
    alloc = allocate_weighted_values(source_values, mapping_df)
    detail = alloc.detail.copy()
    summary = detail.groupby("aggregate_sector", dropna=False).agg(
        mapped_value=("allocated_value", "sum"),
        exio_sector_count=("exio_sector", "nunique"),
        provenance_set=("provenance", lambda s: "|".join(sorted(set(map(str, s))))),
        source_name_set=("source_name", lambda s: "|".join(sorted(set(map(str, s))))),
        exio_sectors=("exio_sector", lambda s: " | ".join(map(str, sorted(set(s))))),
        notes=("notes", _combine_notes),
    )
    ref = reference_values.copy().astype("float64")
    ref.index = ref.index.astype(str)
    summary["reference_value"] = ref.reindex(summary.index).astype("float64")

    scale_factor = 1.0
    mapped_total = float(summary["mapped_value"].sum())
    reference_total = float(summary["reference_value"].fillna(0.0).sum())
    if scale_to_reference and mapped_total > 0 and np.isfinite(mapped_total) and np.isfinite(reference_total):
        scale_factor = reference_total / mapped_total
    summary["scale_factor"] = scale_factor
    summary["mapped_value_scaled"] = summary["mapped_value"] * scale_factor
    summary["abs_gap"] = (summary["mapped_value_scaled"] - summary["reference_value"]).abs()
    summary["pct_gap"] = np.where(
        summary["reference_value"].abs() > 0,
        summary["abs_gap"] / summary["reference_value"].abs(),
        np.nan,
    )
    summary["mismatch_reason"] = np.where(
        summary["provenance_set"].str.contains("heuristic", na=False),
        "heuristic-concordance-or-gap",
        "value-gap",
    )
    return summary.reset_index().rename(columns={"aggregate_sector": "target_sector"})


@dataclass(frozen=True)
class ShareAllocationResult:
    detail: pd.DataFrame
    target_totals: pd.Series
    category_reconstructed: pd.Series


@dataclass(frozen=True)
class ShareMatrixAllocationResult:
    detail: pd.DataFrame
    target_matrix: pd.DataFrame
    category_reconstructed: pd.DataFrame


def allocate_category_values_by_share(
    category_values: pd.Series,
    mapping_df: pd.DataFrame,
    *,
    category_col: str = "Category",
    target_col: str = "TargetSector",
    share_col: str = "Share_used",
) -> ShareAllocationResult:
    df = mapping_df.copy()
    required = {category_col, target_col, share_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Share mapping frame missing required columns: {sorted(missing)}")

    values = category_values.copy().astype("float64")
    values.index = values.index.astype(str)
    df[category_col] = df[category_col].astype(str)
    df[target_col] = df[target_col].astype(str)
    df[share_col] = pd.to_numeric(df[share_col], errors="raise")

    sums = df.groupby(category_col)[share_col].sum()
    bad = sums[~np.isclose(sums, 1.0, atol=1e-9)]
    if not bad.empty:
        sample = ", ".join(f"{idx}={val:.6g}" for idx, val in bad.items())
        raise ValueError(f"Share mapping must sum to 1 by category. Example mismatches: {sample}")

    df["category_value"] = df[category_col].map(values).fillna(0.0)
    df["allocated_value"] = df["category_value"] * df[share_col]
    target_totals = df.groupby(target_col)["allocated_value"].sum().sort_index()
    category_reconstructed = df.groupby(category_col)["allocated_value"].sum().sort_index()
    return ShareAllocationResult(detail=df, target_totals=target_totals, category_reconstructed=category_reconstructed)


def allocate_category_matrix_by_share(
    category_matrix: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    category_col: str = "Category",
    target_col: str = "TargetSector",
    share_col: str = "Share_used",
) -> ShareMatrixAllocationResult:
    df = mapping_df.copy()
    required = {category_col, target_col, share_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Share mapping frame missing required columns: {sorted(missing)}")

    matrix = category_matrix.copy().astype("float64")
    matrix.index = matrix.index.astype(str)
    value_cols = list(matrix.columns)

    df[category_col] = df[category_col].astype(str)
    df[target_col] = df[target_col].astype(str)
    df[share_col] = pd.to_numeric(df[share_col], errors="raise")

    sums = df.groupby(category_col)[share_col].sum()
    bad = sums[~np.isclose(sums, 1.0, atol=1e-9)]
    if not bad.empty:
        sample = ", ".join(f"{idx}={val:.6g}" for idx, val in bad.items())
        raise ValueError(f"Share mapping must sum to 1 by category. Example mismatches: {sample}")

    joined = df.merge(matrix, how="left", left_on=category_col, right_index=True)
    joined[value_cols] = joined[value_cols].fillna(0.0)
    for column in value_cols:
        joined[column] = joined[column] * joined[share_col]

    target_matrix = joined.groupby(target_col)[value_cols].sum().sort_index()
    category_reconstructed = joined.groupby(category_col)[value_cols].sum().sort_index()
    return ShareMatrixAllocationResult(
        detail=joined,
        target_matrix=target_matrix,
        category_reconstructed=category_reconstructed,
    )


def write_canonical_mapping_csv(mapping_df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out, index=False)
    return out


def load_exiobase_nace_concordance(
    path: str | Path,
    *,
    exio_col: str = "EXIOBASE",
    nace_col: str = "NACE",
    source_name: str | None = None,
    provenance: str = "semi-official",
    transformation_rule: str = "vendored-exiobase-nace-concordance",
) -> pd.DataFrame:
    source_path = Path(path)
    df = pd.read_excel(source_path, sheet_name=0).fillna("")
    missing = [c for c in (exio_col, nace_col) if c not in df.columns]
    if missing:
        raise ValueError(f"EXIOBASE concordance workbook missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "exio_sector": df[exio_col].astype(str).str.strip(),
            "nace_code": df[nace_col].astype(str).str.strip(),
            "source_name": source_name or source_path.name,
            "provenance": _validate_provenance(provenance),
            "transformation_rule": str(transformation_rule).strip(),
        }
    )
    out = out[(out["exio_sector"] != "") & (out["nace_code"] != "")].copy()
    out = out.drop_duplicates(subset=["exio_sector", "nace_code"]).reset_index(drop=True)
    return out


def build_exio_nace_audit(concordance_df: pd.DataFrame) -> pd.DataFrame:
    df = concordance_df.copy()
    required = {"exio_sector", "nace_code", "source_name", "provenance", "transformation_rule"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EXIO-NACE concordance frame missing required columns: {sorted(missing)}")

    counts = df.groupby("exio_sector")["nace_code"].nunique().rename("nace_count")
    out = df.merge(counts, on="exio_sector", how="left")
    out["is_one_to_one"] = out["nace_count"] == 1
    return out.sort_values(["nace_count", "exio_sector", "nace_code"], ascending=[False, True, True]).reset_index(drop=True)