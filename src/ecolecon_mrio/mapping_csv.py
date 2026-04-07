from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


def read_split_mapping(
    path: str | Path,
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
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