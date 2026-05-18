#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ecolecon_mrio.canonical_mapping_csv import CANONICAL_DICTIONARY_COLUMNS


DEFAULT_INPUTS = (
    Path("scripts/mappings/exio_to_eurostat_cpa.csv"),
    Path("scripts/mappings/exio_energy_split_cpa_with_electricity_source.csv"),
    Path("scripts/mappings/exio_energy_split_cpa_no_electricity_source.csv"),
    Path("scripts/mappings/energy_green_brown_2sector_from_split_subsectors.csv"),
    Path("scripts/mappings/energy_balance_to_cpa65_products_mapping.csv"),
    Path("outputs/AT_2021_fuels_generated_mappings/exio_fuels_split_from_workbook.csv"),
    Path("outputs/AT_2021_fuels_generated_mappings/fuels_identity_from_workbook.csv"),
    Path("outputs/AT_2021_fuels_generated_mappings/fuels_reaggregation_from_workbook.csv"),
    Path("outputs/AT_2021_fuels_more_generated_mappings/exio_fuels_more_split_from_workbook.csv"),
    Path("outputs/AT_2021_fuels_more_generated_mappings/fuels_more_identity_from_workbook.csv"),
    Path("outputs/AT_2021_fuels_more_generated_mappings/fuels_more_reaggregation_from_workbook.csv"),
)


def _normalized(value: object) -> str:
    return " ".join(str(value).strip().split())


def _label_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalized(value).casefold())


def _collect_rows(frame: pd.DataFrame, *, role_source: bool, role_target: bool, role_split: bool) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}

    def _store(code_value: object, label_value: object, *, parent_code: object = "") -> None:
        code = _normalized(code_value)
        label = _normalized(label_value)
        if not code or not label:
            return
        entry = rows.setdefault(
            code,
            {
                "code": code,
                "label": label,
                "parent_code": _normalized(parent_code),
                "can_be_source": False,
                "can_be_target": False,
                "can_be_split": False,
                "active": True,
            },
        )
        if _label_key(entry["label"]) != _label_key(label):
            raise ValueError(f"Conflicting labels for code {code!r}: {entry['label']!r} vs {label!r}")
        if len(label) > len(str(entry["label"])):
            entry["label"] = label
        if _normalized(parent_code) and not entry["parent_code"]:
            entry["parent_code"] = _normalized(parent_code)
        entry["can_be_source"] = bool(entry["can_be_source"] or role_source)
        entry["can_be_target"] = bool(entry["can_be_target"] or role_target)
        entry["can_be_split"] = bool(entry["can_be_split"] or role_split)

    for _, row in frame.iterrows():
        source_code = row.get("source_code", "") or row.get("source_key", "")
        source_label = row.get("source_label", "") or row.get("source_key", "")
        target_code = row.get("target_code", "") or row.get("target_key", "")
        target_label = row.get("target_label", "") or row.get("target_key", "")
        subtarget_code = row.get("subtarget_code", "") or row.get("subtarget_key", "")
        subtarget_label = row.get("subtarget_label", "") or row.get("subtarget_key", "")

        _store(source_code, source_label)
        _store(target_code, target_label)
        if _normalized(subtarget_code):
            _store(subtarget_code, subtarget_label)

    return rows


def build_dictionary(inputs: list[Path]) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for path in inputs:
        frame = pd.read_csv(path, dtype=str).fillna("")
        collected = _collect_rows(
            frame,
            role_source=True,
            role_target=True,
            role_split=path.name != "energy_balance_to_cpa65_products_mapping.csv",
        )
        for code, entry in collected.items():
            existing = rows.setdefault(code, entry.copy())
            if existing["label"] != entry["label"]:
                raise ValueError(f"Conflicting labels for code {code!r}: {existing['label']!r} vs {entry['label']!r}")
            existing["can_be_source"] = bool(existing["can_be_source"] or entry["can_be_source"])
            existing["can_be_target"] = bool(existing["can_be_target"] or entry["can_be_target"])
            existing["can_be_split"] = bool(existing["can_be_split"] or entry["can_be_split"])

    frame = pd.DataFrame(sorted(rows.values(), key=lambda item: item["code"]))
    if frame.empty:
        raise ValueError("No codes were discovered while building the canonical sector dictionary.")
    frame["parent_code"] = frame["parent_code"].fillna("").astype(str)
    frame["active"] = True
    return frame.loc[:, CANONICAL_DICTIONARY_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical sector dictionary from existing mapping CSVs.")
    parser.add_argument("--output", default="scripts/mappings/sector_dictionary.csv")
    parser.add_argument("inputs", nargs="*", type=Path, help="Optional explicit CSV inputs.")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    input_paths = args.inputs or [root / path for path in DEFAULT_INPUTS]
    frame = build_dictionary(input_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())