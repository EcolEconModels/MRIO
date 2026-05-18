from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import re


CANONICAL_DICTIONARY_COLUMNS: tuple[str, ...] = (
    "code",
    "label",
    "parent_code",
    "can_be_source",
    "can_be_target",
    "can_be_split",
    "active",
)

CANONICAL_MAPPING_COLUMNS: tuple[str, ...] = (
    "mapping_set",
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "subtarget_code",
    "subtarget_label",
    "weight",
    "enabled",
    "notes",
)


def _parse_bool(value: object, *, column: str, required: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if required:
            raise ValueError(f"CSV requires an explicit boolean value in {column!r}.")
        return False
    text = str(value).strip().lower()
    if text == "":
        if required:
            raise ValueError(f"CSV requires an explicit boolean value in {column!r}.")
        return False
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value for {column!r}: {value!r}")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, path: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Canonical mapping CSV {path} missing required columns: {missing}")


def _read_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _label_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalized_text(value).casefold())


def _split_code_prefixed_label(value: object) -> tuple[str, str]:
    text = _normalized_text(value)
    if not text:
        return "", ""
    if "::" in text:
        prefix, suffix = text.split("::", 1)
        return _normalized_text(prefix), _normalized_text(suffix)
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        return _normalized_text(prefix), _normalized_text(suffix)
    return "", text


def load_sector_dictionary(path: str | Path) -> pd.DataFrame:
    dictionary_path = Path(path)
    frame = _read_frame(dictionary_path)
    _require_columns(frame, CANONICAL_DICTIONARY_COLUMNS, path=dictionary_path)

    for column in ("code", "label"):
        frame[column] = frame[column].map(_normalized_text)
    frame["parent_code"] = frame["parent_code"].map(_normalized_text)
    frame["can_be_source"] = frame["can_be_source"].map(lambda value: _parse_bool(value, column="can_be_source", required=True))
    frame["can_be_target"] = frame["can_be_target"].map(lambda value: _parse_bool(value, column="can_be_target", required=True))
    frame["can_be_split"] = frame["can_be_split"].map(lambda value: _parse_bool(value, column="can_be_split", required=True))
    frame["active"] = frame["active"].map(lambda value: _parse_bool(value, column="active", required=True))

    if (frame["code"] == "").any():
        raise ValueError(f"Canonical dictionary {dictionary_path} contains blank codes.")
    if (frame["label"] == "").any():
        raise ValueError(f"Canonical dictionary {dictionary_path} contains blank labels.")

    duplicated_codes = frame[frame.duplicated(subset=["code"], keep=False)]["code"].tolist()
    if duplicated_codes:
        sample = sorted(dict.fromkeys(duplicated_codes))[:8]
        raise ValueError(f"Canonical dictionary {dictionary_path} contains duplicate codes: {sample!r}")

    if not (frame[["can_be_source", "can_be_target", "can_be_split"]].any(axis=1)).all():
        raise ValueError(
            f"Canonical dictionary {dictionary_path} contains rows that are disabled for every mapping role."
        )

    return frame.loc[:, CANONICAL_DICTIONARY_COLUMNS].copy()


def _dictionary_lookup(dictionary: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row.code): row for row in dictionary.itertuples(index=False)}


def _validate_code_and_label(
    *,
    dictionary_row: pd.Series,
    code: str,
    label: str,
    path: Path,
    row_number: int,
    field: str,
) -> None:
    expected_code = _normalized_text(dictionary_row.code)
    expected_label = _normalized_text(dictionary_row.label)
    observed_code = _normalized_text(code)
    observed_label = _normalized_text(label)
    if observed_code != expected_code:
        raise ValueError(
            f"{path} row {row_number}: {field} code mismatch, expected {expected_code!r}, got {observed_code!r}."
        )
    if observed_label == expected_label:
        return
    prefix, suffix = _split_code_prefixed_label(observed_label)
    if prefix == expected_code and suffix == expected_label and "::" not in expected_code:
        return
    if prefix and suffix == expected_label:
        raise ValueError(
            f"{path} row {row_number}: {field} label mismatch for {observed_code!r}, expected {expected_label!r}, got {observed_label!r}."
        )
    if observed_label != expected_label:
        raise ValueError(
            f"{path} row {row_number}: {field} label mismatch for {observed_code!r}, expected {expected_label!r}, got {observed_label!r}."
        )


def load_canonical_mapping_frame(
    path: str | Path,
    dictionary_path: str | Path,
    *,
    require_weight: bool = True,
) -> pd.DataFrame:
    mapping_path = Path(path)
    frame = _read_frame(mapping_path)
    _require_columns(frame, CANONICAL_MAPPING_COLUMNS, path=mapping_path)

    dictionary = load_sector_dictionary(dictionary_path)
    lookup = _dictionary_lookup(dictionary)
    frame["mapping_set"] = frame["mapping_set"].map(_normalized_text)
    frame["source_code"] = frame["source_code"].map(_normalized_text)
    frame["source_label"] = frame["source_label"].map(_normalized_text)
    frame["target_code"] = frame["target_code"].map(_normalized_text)
    frame["target_label"] = frame["target_label"].map(_normalized_text)
    frame["subtarget_code"] = frame["subtarget_code"].map(_normalized_text)
    frame["subtarget_label"] = frame["subtarget_label"].map(_normalized_text)
    frame["notes"] = frame["notes"].map(_normalized_text)
    frame["enabled"] = frame["enabled"].map(lambda value: _parse_bool(value, column="enabled", required=True))

    if require_weight:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    elif (frame["weight"].astype(str).str.strip() != "").any():
        frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")

    if (frame["mapping_set"] == "").any():
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains blank mapping_set values.")
    if (frame["source_code"] == "").any() or (frame["target_code"] == "").any():
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains blank source or target codes.")

    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        source_row = lookup.get(str(row.source_code))
        target_row = lookup.get(str(row.target_code))
        if source_row is None:
            raise ValueError(f"{mapping_path} row {row_number}: unknown source code {row.source_code!r}.")
        if target_row is None:
            raise ValueError(f"{mapping_path} row {row_number}: unknown target code {row.target_code!r}.")
        if not bool(source_row.active):
            raise ValueError(f"{mapping_path} row {row_number}: inactive source code {row.source_code!r}.")
        if not bool(target_row.active):
            raise ValueError(f"{mapping_path} row {row_number}: inactive target code {row.target_code!r}.")
        if not bool(source_row.can_be_source):
            raise ValueError(f"{mapping_path} row {row_number}: source code {row.source_code!r} is not allowed as a source.")
        if not bool(target_row.can_be_target):
            raise ValueError(f"{mapping_path} row {row_number}: target code {row.target_code!r} is not allowed as a target.")
        _validate_code_and_label(
            dictionary_row=source_row,
            code=row.source_code,
            label=row.source_label,
            path=mapping_path,
            row_number=row_number,
            field="source",
        )
        _validate_code_and_label(
            dictionary_row=target_row,
            code=row.target_code,
            label=row.target_label,
            path=mapping_path,
            row_number=row_number,
            field="target",
        )
        if row.subtarget_code:
            subtarget_row = lookup.get(str(row.subtarget_code))
            if subtarget_row is None:
                raise ValueError(f"{mapping_path} row {row_number}: unknown subtarget code {row.subtarget_code!r}.")
            if not bool(subtarget_row.active):
                raise ValueError(f"{mapping_path} row {row_number}: inactive subtarget code {row.subtarget_code!r}.")
            if not bool(subtarget_row.can_be_split):
                raise ValueError(
                    f"{mapping_path} row {row_number}: subtarget code {row.subtarget_code!r} is not allowed as a split code."
                )
            _validate_code_and_label(
                dictionary_row=subtarget_row,
                code=row.subtarget_code,
                label=row.subtarget_label,
                path=mapping_path,
                row_number=row_number,
                field="subtarget",
            )

    duplicates = frame[frame.duplicated(subset=["mapping_set", "source_code", "target_code", "subtarget_code"], keep=False)]
    if not duplicates.empty:
        sample = duplicates.loc[:, ["mapping_set", "source_code", "target_code", "subtarget_code"]].head(8).to_dict(orient="records")
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains duplicate edges: {sample!r}")

    if require_weight:
        active = frame[frame["enabled"]].copy()
        grouped = active.groupby(["mapping_set", "source_code"], sort=False)["weight"].sum()
        bad = grouped[(grouped - 1.0).abs() > 1e-9]
        if not bad.empty:
            sample = ", ".join(f"{mapping_set}/{source_code}={float(total):.6g}" for (mapping_set, source_code), total in bad.head(8).items())
            raise ValueError(f"Canonical mapping CSV {mapping_path} has weight-sum drift: {sample}")

    return frame.loc[:, [column for column in CANONICAL_MAPPING_COLUMNS if column in frame.columns]].copy()


def validate_canonical_chain(
    split_mapping_path: str | Path,
    reaggregation_mapping_path: str | Path,
    dictionary_path: str | Path,
) -> None:
    split_frame = load_canonical_mapping_frame(split_mapping_path, dictionary_path)
    reaggregation_frame = load_canonical_mapping_frame(reaggregation_mapping_path, dictionary_path)

    split_active = split_frame[split_frame["enabled"]].copy()
    reaggregation_active = reaggregation_frame[reaggregation_frame["enabled"]].copy()

    split_codes = set(code for code in split_active["subtarget_code"].tolist() if str(code).strip())
    if not split_codes:
        split_codes = set(split_active["target_code"].tolist())
    reaggregation_sources = set(reaggregation_active["source_code"].tolist())
    missing = sorted(split_codes - reaggregation_sources)
    if missing:
        raise ValueError(
            "Canonical mapping chain is incomplete: split subsectors missing from reaggregation: "
            f"{missing[:12]!r}"
        )


def read_canonical_split_mapping(
    path: str | Path,
    dictionary_path: str | Path,
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
    frame = load_canonical_mapping_frame(path, dictionary_path)
    active = frame[frame["enabled"]].copy()
    active = active[(active["target_code"] != "") & (active["source_code"] != "")]
    dictionary = load_sector_dictionary(dictionary_path)
    lookup = _dictionary_lookup(dictionary)

    if active["subtarget_code"].astype(str).str.strip().eq("").all():
        out_simple: dict[str, list[str]] = {}
        for target_code, group in active.groupby("target_code", sort=False):
            out_simple[str(target_code)] = [
                str(lookup[str(source_code)].label) for source_code in group["source_code"].tolist()
            ]
        return out_simple

    out_grouped: dict[str, dict[str, list[str]]] = {}
    for target_code, target_group in active.groupby("target_code", sort=False):
        sub_dict: dict[str, list[str]] = {}
        for subsector_code, subsector_group in target_group.groupby("subtarget_code", sort=False):
            key = str(lookup[str(subsector_code)].label).strip() if str(subsector_code).strip() else str(lookup[str(target_code)].label)
            sub_dict[key] = [str(lookup[str(source_code)].label) for source_code in subsector_group["source_code"].tolist()]
        out_grouped[str(target_code)] = sub_dict
    return out_grouped


def read_canonical_aggregation_mapping(
    path: str | Path,
    dictionary_path: str | Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    frame = load_canonical_mapping_frame(path, dictionary_path)
    active = frame[frame["enabled"]].copy()
    active = active[(active["source_code"] != "") & (active["target_code"] != "")]
    dictionary = load_sector_dictionary(dictionary_path)
    lookup = _dictionary_lookup(dictionary)

    mapping: dict[str, str] = {}
    for row in active.itertuples(index=False):
        source_code = str(lookup[str(row.source_code)].label)
        target_code = str(lookup[str(row.target_code)].label)
        if source_code in mapping and mapping[source_code] != target_code:
            raise ValueError(
                f"Canonical aggregation mapping maps {source_code!r} to multiple targets: "
                f"{mapping[source_code]!r} and {target_code!r}"
            )
        mapping[source_code] = target_code

    returned = active.copy()
    returned["from_sector"] = returned["source_code"].map(lambda code: str(lookup[str(code)].label))
    returned["to_sector"] = returned["target_code"].map(lambda code: str(lookup[str(code)].label))
    return mapping, returned