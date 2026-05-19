from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_DICTIONARY_COLUMNS: tuple[str, ...] = (
    "code",
    "label",
    "description",
)

CANONICAL_MAPPING_COLUMNS: tuple[str, ...] = (
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "order",
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

    for column in ("code", "label", "description"):
        frame[column] = frame[column].map(_normalized_text)

    if (frame["code"] == "").any():
        raise ValueError(f"Canonical dictionary {dictionary_path} contains blank codes.")
    if (frame["label"] == "").any():
        raise ValueError(f"Canonical dictionary {dictionary_path} contains blank labels.")

    duplicated_codes = frame[frame.duplicated(subset=["code"], keep=False)]["code"].tolist()
    if duplicated_codes:
        sample = sorted(dict.fromkeys(duplicated_codes))[:8]
        raise ValueError(f"Canonical dictionary {dictionary_path} contains duplicate codes: {sample!r}")

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
    source_dictionary_path: str | Path,
    *,
    target_dictionary_path: str | Path | None = None,
    validate_target_codes: bool = True,
    require_order: bool = True,
) -> pd.DataFrame:
    mapping_path = Path(path)
    frame = _read_frame(mapping_path)
    _require_columns(frame, tuple(column for column in CANONICAL_MAPPING_COLUMNS if column != "order"), path=mapping_path)

    source_dictionary = load_sector_dictionary(source_dictionary_path)
    target_dictionary = load_sector_dictionary(target_dictionary_path or source_dictionary_path)
    source_lookup = _dictionary_lookup(source_dictionary)
    target_lookup = _dictionary_lookup(target_dictionary)

    frame["source_code"] = frame["source_code"].map(_normalized_text)
    frame["source_label"] = frame["source_label"].map(_normalized_text)
    frame["target_code"] = frame["target_code"].map(_normalized_text)
    frame["target_label"] = frame["target_label"].map(_normalized_text)
    frame["notes"] = frame["notes"].map(_normalized_text)
    frame["enabled"] = frame["enabled"].map(lambda value: _parse_bool(value, column="enabled", required=True))

    if require_order:
        frame["order"] = pd.to_numeric(frame["order"], errors="raise")
    elif "order" in frame.columns:
        frame["order"] = pd.to_numeric(frame["order"], errors="raise")

    if (frame["source_code"] == "").any() or (frame["target_code"] == "").any():
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains blank source or target codes.")

    duplicated_sources = frame[frame.duplicated(subset=["source_code"], keep=False)]["source_code"].tolist()
    if duplicated_sources:
        sample = sorted(dict.fromkeys(duplicated_sources))[:8]
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains duplicate source codes: {sample!r}")

    for row_number, row in enumerate(frame.itertuples(index=False), start=2):
        source_row = source_lookup.get(str(row.source_code))
        if source_row is None:
            raise ValueError(f"{mapping_path} row {row_number}: unknown source code {row.source_code!r}.")
        _validate_code_and_label(
            dictionary_row=source_row,
            code=row.source_code,
            label=row.source_label,
            path=mapping_path,
            row_number=row_number,
            field="source",
        )

        if validate_target_codes:
            target_row = target_lookup.get(str(row.target_code))
            if target_row is None:
                raise ValueError(f"{mapping_path} row {row_number}: unknown target code {row.target_code!r}.")
            _validate_code_and_label(
                dictionary_row=target_row,
                code=row.target_code,
                label=row.target_label,
                path=mapping_path,
                row_number=row_number,
                field="target",
            )

    duplicates = frame[frame.duplicated(subset=["source_code", "target_code"], keep=False)]
    if not duplicates.empty:
        sample = duplicates.loc[:, ["source_code", "target_code"]].head(8).to_dict(orient="records")
        raise ValueError(f"Canonical mapping CSV {mapping_path} contains duplicate edges: {sample!r}")

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
    source_dictionary_path: str | Path,
    target_dictionary_path: str | Path | None = None,
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
    frame = load_canonical_mapping_frame(
        path,
        source_dictionary_path,
        target_dictionary_path=target_dictionary_path,
        validate_target_codes=True,
    )
    active = frame[frame["enabled"]].copy()
    active = active[(active["target_code"] != "") & (active["source_code"] != "")]
    source_dictionary = load_sector_dictionary(source_dictionary_path)
    source_lookup = _dictionary_lookup(source_dictionary)

    out_simple: dict[str, list[str]] = {}
    for target_code, group in active.groupby("target_code", sort=False):
        target_name = _normalized_text(group.iloc[0]["target_label"])
        out_simple[target_name] = [
            str(source_lookup[str(source_code)].label) for source_code in group["source_code"].tolist()
        ]
    return out_simple


def read_canonical_aggregation_mapping(
    path: str | Path,
    source_dictionary_path: str | Path,
    target_dictionary_path: str | Path | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    frame = load_canonical_mapping_frame(
        path,
        source_dictionary_path,
        target_dictionary_path=target_dictionary_path,
        validate_target_codes=False,
    )
    active = frame[frame["enabled"]].copy()
    active = active[(active["source_code"] != "") & (active["target_code"] != "")]
    dictionary = load_sector_dictionary(dictionary_path)
    lookup = _dictionary_lookup(dictionary)

    mapping: dict[str, str] = {}
    for row in active.itertuples(index=False):
        source_code = str(lookup[str(row.source_code)].label)
        target_code = _normalized_text(row.target_label)
        if source_code in mapping and mapping[source_code] != target_code:
            raise ValueError(
                f"Canonical aggregation mapping maps {source_code!r} to multiple targets: "
                f"{mapping[source_code]!r} and {target_code!r}"
            )
        mapping[source_code] = target_code

    returned = active.copy()
    returned["from_sector"] = returned["source_code"].map(lambda code: str(lookup[str(code)].label))
    returned["to_sector"] = returned["target_label"].map(_normalized_text)
    return mapping, returned