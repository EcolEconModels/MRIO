from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecolecon_mrio.canonical_mapping_csv import load_sector_dictionary


MAPPING_CSV_COLUMNS: tuple[str, ...] = (
    "order",
    "kind",
    "source_dataset",
    "source_system",
    "source_version",
    "source_level",
    "target_dataset",
    "target_system",
    "target_version",
    "target_level",
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "enabled",
    "provenance",
    "certainty",
    "allocation_rule",
    "source_name",
    "official_source",
    "notes",
    "source_balance_role",
    "target_sector_role",
    "include_in_direct_energy",
)

SOURCE_BALANCE_ROLES: frozenset[str] = frozenset(
    {
        "final_use",
        "transformation_input",
        "energy_sector_own_use",
        "transformation_output",
        "non_energy_use",
    }
)

TARGET_SECTOR_ROLES: frozenset[str] = frozenset(
    {
        "end_use_sector",
        "transformation_sector",
        "households",
    }
)

SPLIT_SECTOR_MANIFEST_COLUMNS: tuple[str, ...] = (
    "source_code",
    "source_label",
)

SPLIT_OUTPUT_SHARE_RULES: frozenset[str] = frozenset(
    {
        "derived-output-share-split",
        "split-by-output-share",
        "split-by-output-share-year-specific",
    }
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


def _parse_required_boolish(x: object, *, column: str) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError(f"Mapping CSV requires an explicit boolean value in {column!r}.")
    s = str(x).strip()
    if s == "":
        raise ValueError(f"Mapping CSV requires an explicit boolean value in {column!r}.")
    return _parse_boolish(s)


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _dictionary_lookup(dictionary: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row.code): row for row in dictionary.itertuples(index=False)}


def _normalize_role_column(
    frame: pd.DataFrame,
    *,
    column: str,
    allowed: frozenset[str],
) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.lower()
    invalid = sorted({value for value in values.unique().tolist() if value and value not in allowed})
    if invalid:
        raise ValueError(
            f"Mapping CSV contains unsupported values in {column!r}: {invalid!r}."
        )
    return values


def read_mapping_csv(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    required_columns: tuple[str, ...] = MAPPING_CSV_COLUMNS,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if expected_kind in {"split", "aggregate"}:
        required_columns = tuple(column for column in required_columns if column != "order")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Mapping CSV {Path(path)} missing required columns: {missing}")
    if expected_kind is not None:
        kinds = {str(value).strip() for value in frame["kind"].astype(str) if str(value).strip() != ""}
        if kinds != {expected_kind}:
            raise ValueError(
                f"Mapping CSV {Path(path)} expected kind {expected_kind!r}, got {sorted(kinds)!r}"
            )
    return frame


def validate_mapping_weights(
    sector_mapping: dict[str, dict[str, float]],
    *,
    atol: float = 1e-9,
) -> None:
    exio_weight_sum: dict[str, float] = {}
    for aggregate_sector, weights in sector_mapping.items():
        if not isinstance(weights, dict):
            raise ValueError(
                f"Weighted mapping for {aggregate_sector!r} must be a dict of exio->weight."
            )
        for exio_sector, weight in weights.items():
            if weight is None or not np.isfinite(float(weight)):
                raise ValueError(
                    f"Weight for exio {exio_sector!r} in {aggregate_sector!r} is not finite: {weight!r}"
                )
            weight_float = float(weight)
            if weight_float < 0:
                raise ValueError(
                    f"Negative weight for exio {exio_sector!r} in {aggregate_sector!r}: {weight_float}"
                )
            exio_weight_sum[exio_sector] = exio_weight_sum.get(exio_sector, 0.0) + weight_float

    bad = {
        exio_sector: total
        for exio_sector, total in exio_weight_sum.items()
        if abs(total - 1.0) > atol
    }
    if bad:
        sample = ", ".join(f"{key}={value:.6g}" for key, value in list(bad.items())[:8])
        raise ValueError(
            "EXIO weights must sum to 1.0 across mapped target codes. "
            f"Example mismatches: {sample}"
        )


def load_weighted_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="weighted").copy()
    frame["aggregate_sector"] = frame["target_code"].astype(str).str.strip()
    frame["exio_sector"] = frame["source_code"].astype(str).str.strip()
    frame["order"] = pd.to_numeric(frame["order"], errors="raise")
    return frame


def _coerce_weighted_mapping_frame(mapping: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(mapping, pd.DataFrame):
        frame = mapping.copy().fillna("")
        missing = [column for column in ("source_code", "target_code") if column not in frame.columns]
        if missing:
            raise ValueError(f"Weighted mapping frame missing required columns: {missing}")
        if "kind" in frame.columns:
            kinds = {str(value).strip() for value in frame["kind"].astype(str) if str(value).strip() != ""}
            if kinds and kinds != {"weighted"}:
                raise ValueError(f"Weighted mapping frame expected kind 'weighted', got {sorted(kinds)!r}")
        frame["aggregate_sector"] = frame["target_code"].astype(str).str.strip()
        frame["exio_sector"] = frame["source_code"].astype(str).str.strip()
        return frame
    return load_weighted_mapping_frame(mapping)


def _combine_mapping_texts(values: pd.Series) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str):
        text = _normalize_text(value)
        if text and text not in seen:
            parts.append(text)
            seen.add(text)
    return " | ".join(parts)


def _mapping_dict_from_resolved_frame(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        target = str(row.target_code)
        source = str(row.source_code)
        mapping.setdefault(target, {})
        mapping[target][source] = mapping[target].get(source, 0.0) + float(row.weight)
    return mapping


def _finalize_resolved_weight_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("Mapping frame is empty after enabled filtering.")
    rows = rows.copy()
    for column in ("source_label", "target_label", "notes"):
        if column not in rows.columns:
            rows[column] = ""
        rows[column] = rows[column].astype(str).str.strip()
    grouped = rows.groupby(["source_code", "target_code"], as_index=False).agg(
        weight=("weight", "sum"),
        source_label=("source_label", _combine_mapping_texts),
        target_label=("target_label", _combine_mapping_texts),
        notes=("notes", _combine_mapping_texts),
    )
    validate_mapping_weights(_mapping_dict_from_resolved_frame(grouped))
    return grouped.loc[:, ["source_code", "target_code", "weight", "source_label", "target_label", "notes"]].copy()


def resolve_weighted_mapping_frame(
    mapping: str | Path | pd.DataFrame,
    *,
    target_output_by_code: object | None = None,
    target_output_by_target: object | None = None,
) -> pd.DataFrame:
    """Resolve a current-schema weighted mapping to numeric source-target weights.

    Explicit static weights in ``weight`` or ``Share_used`` are preserved. When no
    static weights are present, one-to-one source rows receive weight 1.0 and
    one-to-many source rows are resolved from the supplied target-side output/value
    reference. This is the common runtime resolver used by disaggregation and audit
    code paths.
    """
    frame = _coerce_weighted_mapping_frame(mapping)
    enabled = frame["enabled"].map(_parse_boolish) if "enabled" in frame.columns else pd.Series(True, index=frame.index)
    frame = frame[enabled].copy()
    frame = frame[(frame["aggregate_sector"] != "") & (frame["exio_sector"] != "")]
    if frame.empty:
        raise ValueError("Mapping frame is empty after enabled filtering.")

    for column in ("source_label", "target_label", "notes"):
        if column not in frame.columns:
            frame[column] = ""

    explicit_weight_column: str | None = None
    if "weight" in frame.columns and frame["weight"].astype(str).str.strip().ne("").any():
        explicit_weight_column = "weight"
    elif "Share_used" in frame.columns and frame["Share_used"].astype(str).str.strip().ne("").any():
        explicit_weight_column = "Share_used"

    if explicit_weight_column is not None:
        rows = pd.DataFrame(
            {
                "source_code": frame["exio_sector"].astype(str).str.strip(),
                "target_code": frame["aggregate_sector"].astype(str).str.strip(),
                "weight": pd.to_numeric(frame[explicit_weight_column], errors="coerce").fillna(0.0).astype("float64"),
                "source_label": frame["source_label"].astype(str).str.strip(),
                "target_label": frame["target_label"].astype(str).str.strip(),
                "notes": frame["notes"].astype(str).str.strip(),
            }
        )
        return _finalize_resolved_weight_rows(rows)

    output_by_code = _normalize_numeric_reference(target_output_by_code)
    output_by_target = _normalize_numeric_reference(target_output_by_target)
    resolved_rows: list[dict[str, object]] = []
    for _source, group in frame.groupby("exio_sector", sort=False):
        resolved_weights = _resolve_split_weights_for_source_group(
            group,
            target_output_by_code=output_by_code,
            target_output_by_target=output_by_target,
        )
        for (_, row), weight in zip(group.iterrows(), resolved_weights.tolist(), strict=False):
            resolved_rows.append(
                {
                    "source_code": str(row["exio_sector"]).strip(),
                    "target_code": str(row["aggregate_sector"]).strip(),
                    "weight": float(weight),
                    "source_label": str(row.get("source_label", "")).strip(),
                    "target_label": str(row.get("target_label", "")).strip(),
                    "notes": str(row.get("notes", "")).strip(),
                }
            )
    return _finalize_resolved_weight_rows(pd.DataFrame(resolved_rows))


def _normalize_numeric_reference(reference: object) -> dict[str, float]:
    if reference is None:
        return {}
    if hasattr(reference, "to_dict") and not isinstance(reference, dict):
        reference = reference.to_dict()
    if not isinstance(reference, dict):
        raise TypeError("Output reference must be a mapping or pandas Series-like object.")
    return {
        _normalize_text(key): float(value)
        for key, value in reference.items()
        if _normalize_text(key) != ""
    }


def _resolve_split_weights_for_source_group(
    group: pd.DataFrame,
    *,
    target_output_by_code: dict[str, float],
    target_output_by_target: dict[str, float],
) -> pd.Series:
    if group.empty:
        return pd.Series(dtype="float64")

    if len(group) == 1:
        return pd.Series([1.0], index=group.index, dtype="float64")

    rule_values: set[str] = set()
    for column in ("rule", "allocation_rule", "certainty"):
        if column in group.columns:
            rule_values.update(
                value
                for value in group[column].astype(str).str.strip().tolist()
                if value
            )

    uses_split_rule = any(value in SPLIT_OUTPUT_SHARE_RULES for value in rule_values)
    if target_output_by_code and group["target_code"].astype(str).str.strip().ne("").all():
        keys = group["target_code"].astype(str).tolist()
        reference = target_output_by_code
    elif target_output_by_target:
        keys = group["target_label"].astype(str).tolist()
        reference = target_output_by_target
    else:
        raise ValueError("Split mapping rows require a year-specific output reference when weights are omitted.")

    weights = np.array([max(0.0, float(reference.get(_normalize_text(key), 0.0))) for key in keys], dtype="float64")
    total = float(weights.sum())
    if total <= 0:
        sample = ", ".join(map(str, keys[:8]))
        raise ValueError(
            "Cannot resolve split weights from the supplied output reference. "
            f"Affected targets: {sample}"
        )
    return pd.Series(weights / total, index=group.index, dtype="float64")


def read_weighted_mapping(
    path: str | Path,
    *,
    target_output_by_code: object | None = None,
    target_output_by_target: object | None = None,
) -> dict[str, dict[str, float]]:
    resolved = resolve_weighted_mapping_frame(
        path,
        target_output_by_code=target_output_by_code,
        target_output_by_target=target_output_by_target,
    )
    mapping = _mapping_dict_from_resolved_frame(resolved)
    validate_mapping_weights(mapping)
    return mapping

def load_split_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="split").copy()
    frame["aggregate_sector"] = frame["target_code"].astype(str).str.strip()
    frame["subsector"] = frame["source_code"].astype(str).str.strip()
    frame["exio_sector"] = frame["source_code"].astype(str).str.strip()
    return frame


def load_split_manifest_frame(
    path: str | Path,
    source_dictionary_path: str | Path,
) -> pd.DataFrame:
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype=str).fillna("")
    missing = [column for column in ("source_code",) if column not in frame.columns]
    if missing:
        raise ValueError(f"Split manifest CSV {manifest_path} missing required columns: {missing}")

    source_dictionary = load_sector_dictionary(source_dictionary_path)
    lookup = _dictionary_lookup(source_dictionary)

    frame["source_code"] = frame["source_code"].map(_normalize_text)
    if "source_label" not in frame.columns:
        frame["source_label"] = ""
    frame["source_label"] = frame["source_label"].map(_normalize_text)

    if (frame["source_code"] == "").any():
        raise ValueError(f"Split manifest CSV {manifest_path} contains blank source codes.")

    duplicated = frame[frame.duplicated(subset=["source_code"], keep=False)]["source_code"].tolist()
    if duplicated:
        sample = sorted(dict.fromkeys(duplicated))[:8]
        raise ValueError(f"Split manifest CSV {manifest_path} contains duplicate source codes: {sample!r}")

    validated_labels: list[str] = []
    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        dictionary_row = lookup.get(str(row.source_code))
        if dictionary_row is None:
            raise ValueError(f"{manifest_path} row {row_number}: unknown source code {row.source_code!r}.")
        expected_label = _normalize_text(dictionary_row.label)
        observed_label = _normalize_text(row.source_label)
        if observed_label and observed_label != expected_label:
            raise ValueError(
                f"{manifest_path} row {row_number}: source label mismatch for {row.source_code!r}, "
                f"expected {expected_label!r}, got {observed_label!r}."
            )
        validated_labels.append(expected_label)

    frame["source_label"] = validated_labels
    return frame.loc[:, list(SPLIT_SECTOR_MANIFEST_COLUMNS)].copy()


def read_split_mapping(
    path: str | Path,
    *,
    exio_to_eurostat_mapping_csv: str | Path | None = None,
    source_dictionary_path: str | Path | None = None,
    target_dictionary_path: str | Path | None = None,
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
    raw_frame = pd.read_csv(path, dtype=str).fillna("")
    if "target_code" not in raw_frame.columns:
        if exio_to_eurostat_mapping_csv is None:
            raise ValueError(
                f"Split manifest CSV {Path(path)} requires exio_to_eurostat_mapping_csv to expand subsectors."
            )
        missing = [column for column in SPLIT_SECTOR_MANIFEST_COLUMNS if column not in raw_frame.columns]
        if missing:
            raise ValueError(f"Split manifest CSV {Path(path)} missing required columns: {missing}")

        concordance = load_weighted_mapping_frame(exio_to_eurostat_mapping_csv)
        concordance = concordance[concordance["enabled"].map(_parse_boolish)].copy()
        concordance = concordance[(concordance["aggregate_sector"] != "") & (concordance["exio_sector"] != "")]

        out: dict[str, list[str]] = {}
        for row in raw_frame.itertuples(index=False):
            aggregate_sector = _normalize_text(getattr(row, "source_code", ""))
            if aggregate_sector == "":
                continue
            members = concordance.loc[concordance["aggregate_sector"] == aggregate_sector, "exio_sector"].tolist()
            if not members:
                raise ValueError(
                    f"Split manifest CSV {Path(path)} references aggregate sector {aggregate_sector!r} not present in {Path(exio_to_eurostat_mapping_csv)}."
                )
            out[aggregate_sector] = list(dict.fromkeys(map(str, members)))
        return out

    frame = load_split_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["aggregate_sector"] != "") & (frame["exio_sector"] != "")]

    out_simple: dict[str, list[str]] = {}
    out_grouped: dict[str, dict[str, list[str]]] = {}
    has_subsectors = False
    for aggregate_sector, aggregate_group in frame.groupby("aggregate_sector", sort=False):
        subsector_map: dict[str, list[str]] = {}
        for subsector, subsector_group in aggregate_group.groupby("subsector", sort=False):
            sector_codes = list(map(str, subsector_group["exio_sector"].tolist()))
            subsector_map[str(subsector)] = sector_codes
            if str(subsector) != "" and str(subsector) != str(aggregate_sector):
                has_subsectors = True
        out_simple[str(aggregate_sector)] = list(map(str, aggregate_group["exio_sector"].tolist()))
        out_grouped[str(aggregate_sector)] = subsector_map

    if has_subsectors:
        return out_grouped
    return out_simple


def load_aggregation_mapping_frame(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str).fillna("")
    if "kind" in raw.columns:
        frame = read_mapping_csv(path, expected_kind="aggregate").copy()
    else:
        missing = [column for column in ("source_code", "target_code") if column not in raw.columns]
        if missing:
            raise ValueError(f"Aggregation manifest CSV {Path(path)} missing required columns: {missing}")
        frame = raw.copy()
    frame["from_sector"] = frame["source_code"].astype(str).str.strip()
    frame["to_sector"] = frame["target_code"].astype(str).str.strip()
    return frame


def read_aggregation_mapping(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    frame = load_aggregation_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["from_sector"] != "") & (frame["to_sector"] != "")]

    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        from_sector = str(row.from_sector)
        to_sector = str(row.to_sector)
        if from_sector in mapping and mapping[from_sector] != to_sector:
            raise ValueError(
                f"Aggregation mapping maps {from_sector!r} to multiple targets: "
                f"{mapping[from_sector]!r} and {to_sector!r}"
            )
        mapping[from_sector] = to_sector
    return mapping, frame


def load_category_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="category-target").copy()
    frame["Category"] = frame["source_label"].astype(str).str.strip()
    missing_category = frame["Category"] == ""
    frame.loc[missing_category, "Category"] = frame.loc[missing_category, "source_code"].astype(str).str.strip()
    frame["TargetCode"] = frame["target_code"].astype(str).str.strip()
    frame["TargetSector"] = frame["target_label"].astype(str).str.strip()
    blank_target_sector = frame["TargetSector"] == ""
    frame.loc[blank_target_sector, "TargetSector"] = frame.loc[blank_target_sector, "target_code"].astype(str).str.strip()
    frame["AllocationRule"] = frame["allocation_rule"].astype(str).str.strip()
    frame["SourceBalanceRole"] = _normalize_role_column(
        frame,
        column="source_balance_role",
        allowed=SOURCE_BALANCE_ROLES,
    )
    missing_source_roles = frame["SourceBalanceRole"] == ""
    if missing_source_roles.any():
        raise ValueError(
            "Category-target mapping CSV requires explicit 'source_balance_role' values for all rows."
        )

    frame["TargetSectorRole"] = _normalize_role_column(
        frame,
        column="target_sector_role",
        allowed=TARGET_SECTOR_ROLES,
    )
    missing_target_roles = frame["TargetSectorRole"] == ""
    if missing_target_roles.any():
        raise ValueError(
            "Category-target mapping CSV requires explicit 'target_sector_role' values for all rows."
        )
    frame["IncludeInDirectEnergy"] = frame["include_in_direct_energy"].map(
        lambda value: _parse_required_boolish(value, column="include_in_direct_energy")
    )
    return frame


def read_category_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = load_category_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["Category"] != "") & (frame["TargetSector"] != "")]
    return frame


def load_source_bucket_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="source-bucket").copy()
    frame["SourceBucket"] = frame["source_code"].astype(str).str.strip()
    frame["BucketGroup"] = frame["target_code"].astype(str).str.strip()
    frame["BucketOrder"] = pd.to_numeric(frame["order"], errors="raise")
    return frame


def read_source_bucket_mapping(path: str | Path) -> list[str]:
    frame = load_source_bucket_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[frame["SourceBucket"] != ""]
    frame = frame.sort_values(["BucketOrder", "SourceBucket"], kind="stable")
    return list(map(str, frame["SourceBucket"].tolist()))


def read_source_bucket_groups(path: str | Path) -> dict[str, str]:
    frame = load_source_bucket_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["SourceBucket"] != "") & (frame["BucketGroup"] != "")]
    return dict(zip(frame["SourceBucket"].astype(str), frame["BucketGroup"].astype(str), strict=False))


def load_product_source_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="product-source").copy()
    frame["ProductCode"] = frame["source_code"].astype(str).str.strip()
    frame["SourceBucket"] = frame["target_code"].astype(str).str.strip()
    return frame


def read_product_source_mapping(path: str | Path) -> dict[str, tuple[str, ...]]:
    frame = load_product_source_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["ProductCode"] != "") & (frame["SourceBucket"] != "")]

    out: dict[str, tuple[str, ...]] = {}
    for source_bucket, group in frame.groupby("SourceBucket", sort=False):
        out[str(source_bucket)] = tuple(map(str, group["ProductCode"].tolist()))
    return out


def flatten_exio_sectors_from_mapping(
    sector_mapping: dict[str, list[str]] | dict[str, dict[str, list[str]]] | dict[str, dict[str, float]],
) -> list[str]:
    out: list[str] = []
    for members in sector_mapping.values():
        if isinstance(members, dict):
            if members and any(isinstance(value, list) for value in members.values()):
                for value in members.values():
                    if isinstance(value, list):
                        out.extend(list(map(str, value)))
            else:
                out.extend(list(map(str, members.keys())))
        else:
            out.extend(list(map(str, members)))
    return list(dict.fromkeys(out))
