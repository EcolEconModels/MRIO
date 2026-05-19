from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecolecon_mrio.canonical_mapping_csv import load_sector_dictionary


MAPPING_CSV_COLUMNS: tuple[str, ...] = (
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
    "subtarget_code",
    "order",
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
    frame = load_weighted_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["aggregate_sector"] != "") & (frame["exio_sector"] != "")]

    output_by_code = _normalize_numeric_reference(target_output_by_code)
    output_by_target = _normalize_numeric_reference(target_output_by_target)

    mapping: dict[str, dict[str, float]] = {}
    for _, group in frame.groupby("exio_sector", sort=False):
        resolved_weights = _resolve_split_weights_for_source_group(
            group,
            target_output_by_code=output_by_code,
            target_output_by_target=output_by_target,
        )
        for row, weight in zip(group.itertuples(index=False), resolved_weights.tolist(), strict=False):
            mapping.setdefault(str(row.aggregate_sector), {})
            mapping[str(row.aggregate_sector)][str(row.exio_sector)] = (
                mapping[str(row.aggregate_sector)].get(str(row.exio_sector), 0.0) + float(weight)
            )
    validate_mapping_weights(mapping)
    return mapping


def load_split_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="split").copy()
    frame["aggregate_sector"] = frame["target_code"].astype(str).str.strip()
    frame["subsector"] = frame["subtarget_code"].astype(str).str.strip()
    frame["exio_sector"] = frame["source_code"].astype(str).str.strip()
    mask = frame["subsector"] == ""
    frame.loc[mask, "subsector"] = frame.loc[mask, "exio_sector"]
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
    frame = read_mapping_csv(path, expected_kind="aggregate").copy()
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