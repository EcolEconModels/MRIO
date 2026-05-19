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
    "source_key",
    "source_code",
    "source_label",
    "target_key",
    "target_code",
    "target_label",
    "subtarget_key",
    "weight",
    "share",
    "enabled",
    "provenance",
    "certainty",
    "transformation_rule",
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


def read_mapping_csv(path: str | Path, *, expected_kind: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in MAPPING_CSV_COLUMNS if column not in frame.columns]
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
    frame["aggregate_sector"] = frame["target_key"].astype(str).str.strip()
    frame["exio_sector"] = frame["source_key"].astype(str).str.strip()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    return frame


def read_weighted_mapping(path: str | Path) -> dict[str, dict[str, float]]:
    frame = load_weighted_mapping_frame(path)
    frame = frame[frame["enabled"].map(_parse_boolish)].copy()
    frame = frame[(frame["aggregate_sector"] != "") & (frame["exio_sector"] != "")]

    mapping: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        mapping.setdefault(str(row.aggregate_sector), {})
        mapping[str(row.aggregate_sector)][str(row.exio_sector)] = (
            mapping[str(row.aggregate_sector)].get(str(row.exio_sector), 0.0) + float(row.weight)
        )
    validate_mapping_weights(mapping)
    return mapping


def load_split_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="split").copy()
    frame["aggregate_sector"] = frame["target_key"].astype(str).str.strip()
    frame["subsector"] = frame["subtarget_key"].astype(str).str.strip()
    frame["exio_sector"] = frame["source_key"].astype(str).str.strip()
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
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)

    if {"target_key", "source_key"}.issubset(columns):
        frame = load_split_mapping_frame(path)
        frame = frame[frame["enabled"].map(_parse_boolish)].copy()
        frame = frame[(frame["aggregate_sector"] != "") & (frame["exio_sector"] != "")]

        if "subtarget_key" not in frame.columns:
            out_simple: dict[str, list[str]] = {}
            for aggregate_sector, group in frame.groupby("aggregate_sector", sort=False):
                out_simple[str(aggregate_sector)] = list(map(str, group["exio_sector"].tolist()))
            return out_simple

        out_grouped: dict[str, dict[str, list[str]]] = {}
        for aggregate_sector, aggregate_group in frame.groupby("aggregate_sector", sort=False):
            sub_dict: dict[str, list[str]] = {}
            for subsector, subsector_group in aggregate_group.groupby("subsector", sort=False):
                sub_dict[str(subsector)] = list(map(str, subsector_group["exio_sector"].tolist()))
            out_grouped[str(aggregate_sector)] = sub_dict
        return out_grouped

    if "source_code" not in columns:
        raise ValueError(f"Split mapping CSV {path} must include either 'source_key' or 'source_code' columns.")
    if exio_to_eurostat_mapping_csv is None:
        raise ValueError(
            f"Split mapping CSV {path} contains only sector codes; provide exio_to_eurostat_mapping_csv to expand it."
        )

    source_dictionary = source_dictionary_path or (path.parent / "eurostat_cpa65_products_dictionary.csv")
    target_dictionary = target_dictionary_path or source_dictionary
    manifest = load_split_manifest_frame(path, source_dictionary)
    exio_dictionary = load_sector_dictionary(path.parent / "exiobase_products_dictionary.csv")
    exio_lookup = _dictionary_lookup(exio_dictionary)
    target_lookup = _dictionary_lookup(load_sector_dictionary(target_dictionary))

    weighted_frame = pd.read_csv(exio_to_eurostat_mapping_csv, dtype=str).fillna("")
    required = {"source_code", "source_label", "target_code", "target_label"}
    missing = required - set(weighted_frame.columns)
    if missing:
        raise ValueError(
            f"EXIO-to-Eurostat mapping CSV {exio_to_eurostat_mapping_csv} missing required columns: {sorted(missing)!r}"
        )

    weighted_frame["source_code"] = weighted_frame["source_code"].map(_normalize_text)
    weighted_frame["source_label"] = weighted_frame["source_label"].map(_normalize_text)
    weighted_frame["target_code"] = weighted_frame["target_code"].map(_normalize_text)
    weighted_frame["target_label"] = weighted_frame["target_label"].map(_normalize_text)
    if "enabled" in weighted_frame.columns:
        weighted_frame = weighted_frame[weighted_frame["enabled"].map(_parse_boolish)].copy()

    requested_targets = set(manifest["source_code"].tolist())
    selected = weighted_frame[weighted_frame["target_code"].isin(requested_targets)].copy()
    if selected.empty:
        raise ValueError(
            f"Split manifest {path} refers to sectors missing from {exio_to_eurostat_mapping_csv}: {sorted(requested_targets)!r}"
        )

    selected_missing = sorted(requested_targets - set(selected["target_code"].tolist()))
    if selected_missing:
        raise ValueError(
            f"Split manifest {path} refers to sectors missing from {exio_to_eurostat_mapping_csv}: {selected_missing[:8]!r}"
        )

    out: dict[str, list[str]] = {}
    for target_code, group in selected.groupby("target_code", sort=False):
        target_row = target_lookup.get(str(target_code))
        if target_row is None:
            raise ValueError(
                f"{exio_to_eurostat_mapping_csv}: unknown target code {target_code!r} for split expansion."
            )
        expected_target_label = _normalize_text(target_row.label)
        observed_target_label = _normalize_text(group.iloc[0]["target_label"])
        if observed_target_label != expected_target_label:
            raise ValueError(
                f"{exio_to_eurostat_mapping_csv}: target label mismatch for {target_code!r}, "
                f"expected {expected_target_label!r}, got {observed_target_label!r}."
            )

        members: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(group.itertuples(index=False), start=2):
            source_label = _normalize_text(row.source_label)
            source_code = _normalize_text(row.source_code)
            if source_code:
                source_row = exio_lookup.get(source_code)
                if source_row is None:
                    raise ValueError(
                        f"{exio_to_eurostat_mapping_csv} row {row_number}: unknown source code {source_code!r}."
                    )
                expected_source_label = _normalize_text(source_row.label)
                if source_label and source_label != expected_source_label:
                    raise ValueError(
                        f"{exio_to_eurostat_mapping_csv} row {row_number}: source label mismatch for {source_code!r}, "
                        f"expected {expected_source_label!r}, got {source_label!r}."
                    )
                sector_label = expected_source_label
            else:
                source_row = exio_lookup.get(source_label)
                if source_row is None:
                    raise ValueError(
                        f"{exio_to_eurostat_mapping_csv} row {row_number}: unknown EXIO sector label {source_label!r}."
                    )
                sector_label = _normalize_text(source_row.label)

            if sector_label not in seen:
                seen.add(sector_label)
                members.append(sector_label)

        out[str(target_code)] = members

    return out


def load_aggregation_mapping_frame(path: str | Path) -> pd.DataFrame:
    frame = read_mapping_csv(path, expected_kind="aggregate").copy()
    frame["from_sector"] = frame["source_key"].astype(str).str.strip()
    frame["to_sector"] = frame["target_key"].astype(str).str.strip()
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
    frame["Category"] = frame["source_key"].astype(str).str.strip()
    frame["TargetCode"] = frame["target_code"].astype(str).str.strip()
    frame["TargetSector"] = frame["target_label"].astype(str).str.strip()
    blank_target_sector = frame["TargetSector"] == ""
    frame.loc[blank_target_sector, "TargetSector"] = frame.loc[blank_target_sector, "target_key"].astype(str).str.strip()
    declared_share = frame["share"].astype(str).str.strip()
    frame["DeclaredShare"] = pd.to_numeric(declared_share.mask(declared_share == ""), errors="raise")
    frame["AllocationRule"] = frame["transformation_rule"].astype(str).str.strip()
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
    frame["SourceBucket"] = frame["source_key"].astype(str).str.strip()
    frame["BucketGroup"] = frame["target_key"].astype(str).str.strip()
    frame["BucketOrder"] = pd.to_numeric(frame["weight"], errors="raise")
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
    frame["ProductCode"] = frame["source_key"].astype(str).str.strip()
    frame["SourceBucket"] = frame["target_key"].astype(str).str.strip()
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