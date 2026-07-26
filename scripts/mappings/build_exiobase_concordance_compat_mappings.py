#!/usr/bin/env python3
"""Convert EXIOBASE official-concordance output to CPA_64/CPA_65 mapping CSV schema.

This script does not rebuild the EXIOBASE concordance itself. It consumes the
collapsed output produced by ``build_exiobase3_to_eurostat_iot_a64_mapping.py``
and writes current-schema mapping CSVs usable by the existing MRIO mapping APIs.
Generated labels are read from the committed CPA_64/CPA_65 dictionaries.

Two projections are supported:

* CPA_64: canonical 64-sector target with one ``CPA_L68`` real-estate sector.
* CPA_65: compatibility target that splits only ``CPA_L68`` into ``CPA_L68A``
  and ``CPA_L68B``. No static L68 weights are stored; runtime code should use
  the selected Eurostat target-side value reference for split weights.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DERIVED_DIR = REPO_ROOT / "data" / "exiobase_concordance_sources" / "derived" / "exiobase_eurostat_iot_a64"
DEFAULT_COLLAPSED = DEFAULT_DERIVED_DIR / "exiobase3_products_to_eurostat_iot_a64_collapsed.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "scripts" / "mappings" / "generated"

CPA64_OUTPUT_NAME = "exio_to_eurostat_cpa64_products_from_exiobase_concordances.csv"
CPA65_OUTPUT_NAME = "exio_to_eurostat_cpa65_products_from_exiobase_concordances.csv"
MANIFEST_NAME = "exiobase_concordance_compat_manifest.json"
DEFAULT_CPA64_DICTIONARY = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_products_dictionary.csv"
DEFAULT_CPA65_DICTIONARY = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv"

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
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

REQUIRED_COLLAPSED_COLUMNS: Final[set[str]] = {
    "exio3_product_code",
    "exio3_product_name",
    "iot_a64_code",
    "iot_a64_label",
    "source_nace2_codes",
    "source_nace2_divisions",
    "source_exio_codes",
    "bridge_key_kinds",
    "n_iot_a64_targets_for_exio3_product",
    "is_multi_target",
}

A64_CODE_TO_CPA64: Final[dict[str, str]] = {
    "A01": "CPA_A01",
    "A02": "CPA_A02",
    "A03": "CPA_A03",
    "B": "CPA_B",
    "C10-C12": "CPA_C10-12",
    "C13-C15": "CPA_C13-15",
    "C16": "CPA_C16",
    "C17": "CPA_C17",
    "C18": "CPA_C18",
    "C19": "CPA_C19",
    "C20": "CPA_C20",
    "C21": "CPA_C21",
    "C22": "CPA_C22",
    "C23": "CPA_C23",
    "C24": "CPA_C24",
    "C25": "CPA_C25",
    "C26": "CPA_C26",
    "C27": "CPA_C27",
    "C28": "CPA_C28",
    "C29": "CPA_C29",
    "C30": "CPA_C30",
    "C31-C32": "CPA_C31_32",
    "C33": "CPA_C33",
    "D35": "CPA_D",
    "E36": "CPA_E36",
    "E37-E39": "CPA_E37-39",
    "F": "CPA_F",
    "G45": "CPA_G45",
    "G46": "CPA_G46",
    "G47": "CPA_G47",
    "H49": "CPA_H49",
    "H50": "CPA_H50",
    "H51": "CPA_H51",
    "H52": "CPA_H52",
    "H53": "CPA_H53",
    "I": "CPA_I",
    "J58": "CPA_J58",
    "J59-J60": "CPA_J59_60",
    "J61": "CPA_J61",
    "J62-J63": "CPA_J62_63",
    "K64": "CPA_K64",
    "K65": "CPA_K65",
    "K66": "CPA_K66",
    "L68": "CPA_L68",
    "M69-M70": "CPA_M69_70",
    "M71": "CPA_M71",
    "M72": "CPA_M72",
    "M73": "CPA_M73",
    "M74-M75": "CPA_M74_75",
    "N77": "CPA_N77",
    "N78": "CPA_N78",
    "N79": "CPA_N79",
    "N80-N82": "CPA_N80-82",
    "O84": "CPA_O",
    "P85": "CPA_P",
    "Q86": "CPA_Q86",
    "Q87-Q88": "CPA_Q87_88",
    "R90-R92": "CPA_R90-92",
    "R93": "CPA_R93",
    "S94": "CPA_S94",
    "S95": "CPA_S95",
    "S96": "CPA_S96",
    "T": "CPA_T",
    "U": "CPA_U",
}

CPA65_REAL_ESTATE_TARGETS: Final[tuple[str, ...]] = ("CPA_L68A", "CPA_L68B")

OFFICIAL_SOURCE = (
    "EXIOBASE/XIO concordances: NACE2_EXIOBASE20p_list.csv | "
    "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx | "
    "Eurostat IOT A*64 division rule"
)

def read_target_label_map(path: Path) -> dict[str, str]:
    """Read committed target-code labels from a CPA dictionary CSV."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"code", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Target dictionary {path} is missing required columns: {missing}")
    codes = pd.Series(frame["code"], index=frame.index, dtype="object").astype(str).str.strip()
    labels = pd.Series(frame["label"], index=frame.index, dtype="object").astype(str).str.strip()
    blank_codes = sorted(codes[codes.eq("")].index.tolist())
    blank_labels = sorted(codes[labels.eq("")].tolist())
    if blank_codes:
        raise ValueError(f"Target dictionary {path} has blank code rows: {blank_codes[:20]}")
    if blank_labels:
        raise ValueError(f"Target dictionary {path} has blank labels for: {blank_labels[:20]}")
    duplicates = sorted(codes[codes.duplicated()].unique().tolist())
    if duplicates:
        raise ValueError(f"Target dictionary {path} has duplicate codes: {duplicates[:20]}")
    return dict(zip(codes.tolist(), labels.tolist(), strict=False))


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _validate_collapsed(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLLAPSED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Collapsed mapping is missing required columns: {missing}")
    blank_source = frame["exio3_product_code"].astype(str).str.strip().eq("")
    blank_target = frame["iot_a64_code"].astype(str).str.strip().eq("")
    if bool(blank_source.any()) or bool(blank_target.any()):
        raise ValueError("Collapsed mapping contains blank EXIOBASE source or A64 target codes.")
    unknown_a64 = sorted(set(frame["iot_a64_code"].astype(str).str.strip()) - set(A64_CODE_TO_CPA64))
    if unknown_a64:
        raise ValueError(f"Collapsed mapping contains unknown A64 target codes: {unknown_a64}")


def _source_target_count(frame: pd.DataFrame) -> pd.Series:
    source_codes = pd.Series(frame["exio3_product_code"], index=frame.index, dtype="object").astype(str).str.strip()
    target_codes = pd.Series(frame["iot_a64_code"], index=frame.index, dtype="object")
    transformed = target_codes.groupby(source_codes).transform("nunique")
    return pd.Series(transformed, index=frame.index, dtype="int64")


def _base_note(row: object, *, target_code: str, target_version: str) -> str:
    return (
        "Derived from the EXIOBASE official concordance chain "
        "NACE Rev.2 class -> EXIOBASE2 product -> EXIOBASE3 product -> Eurostat IOT A*64. "
        f"Original A64 target: {_clean_text(getattr(row, 'iot_a64_code'))}. "
        f"Runtime target: {target_code}. Target version: {target_version}. "
        f"Source NACE2 codes: {_clean_text(getattr(row, 'source_nace2_codes'))}. "
        f"Source NACE2 divisions: {_clean_text(getattr(row, 'source_nace2_divisions'))}. "
        f"Source EXIOBASE concordance codes: {_clean_text(getattr(row, 'source_exio_codes'))}. "
        f"Bridge key kinds: {_clean_text(getattr(row, 'bridge_key_kinds'))}."
    )


def _label_for(labels: Mapping[str, str], code: str, *, dictionary_name: str) -> str:
    try:
        return labels[code]
    except KeyError as exc:
        raise ValueError(f"{dictionary_name} is missing required target code: {code}") from exc


def _target_rows_for_a64(
    row: object,
    *,
    projection: str,
    cpa64_labels: Mapping[str, str],
    cpa65_labels: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    a64_code = _clean_text(getattr(row, "iot_a64_code"))
    cpa64_code = A64_CODE_TO_CPA64[a64_code]
    if projection == "cpa64":
        return [(cpa64_code, _label_for(cpa64_labels, cpa64_code, dictionary_name="CPA_64 dictionary"), "CPA_64")]
    if projection == "cpa65":
        if cpa64_code == "CPA_L68":
            return [
                (code, _label_for(cpa65_labels, code, dictionary_name="CPA_65 dictionary"), "CPA_65")
                for code in CPA65_REAL_ESTATE_TARGETS
            ]
        return [(cpa64_code, _label_for(cpa65_labels, cpa64_code, dictionary_name="CPA_65 dictionary"), "CPA_65")]
    raise ValueError(f"Unknown projection: {projection!r}")


def convert_collapsed_to_canonical(
    frame: pd.DataFrame,
    *,
    projection: str,
    cpa64_labels: Mapping[str, str] | None = None,
    cpa65_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return current-schema rows for the requested target projection."""
    _validate_collapsed(frame)
    cpa64_label_map = cpa64_labels if cpa64_labels is not None else read_target_label_map(DEFAULT_CPA64_DICTIONARY)
    cpa65_label_map = cpa65_labels if cpa65_labels is not None else read_target_label_map(DEFAULT_CPA65_DICTIONARY)
    work = frame.copy().fillna("")
    work["_source_target_count"] = _source_target_count(work)

    rows: list[dict[str, object]] = []
    for _source_code, source_group in work.groupby(work["exio3_product_code"].astype(str).str.strip(), sort=True):
        expanded_targets: list[tuple[object, str, str, str]] = []
        for row in source_group.itertuples(index=False):
            for target_code, target_label, target_version in _target_rows_for_a64(
                row,
                projection=projection,
                cpa64_labels=cpa64_label_map,
                cpa65_labels=cpa65_label_map,
            ):
                expanded_targets.append((row, target_code, target_label, target_version))
        n_runtime_targets = len({target_code for _row, target_code, _label, _version in expanded_targets})
        for row, target_code, target_label, target_version in expanded_targets:
            is_split = n_runtime_targets > 1
            allocation_rule = "split-by-output-share" if is_split else "direct-single-target"
            certainty = "official-chain-multi-target" if is_split else "official-chain-single-target"
            note = _base_note(row, target_code=target_code, target_version=target_version)
            if is_split:
                note += (
                    " No static weight is stored. Existing mapping APIs should resolve this row with "
                    "the selected Eurostat target-side value reference for the relevant year/flow."
                )
            if projection == "cpa65" and target_code in {"CPA_L68A", "CPA_L68B"}:
                note += (
                    " This row is a CPA_65 compatibility projection of the canonical CPA_64 CPA_L68 "
                    "real-estate target."
                )
            rows.append(
                {
                    "order": len(rows) + 1,
                    "kind": "weighted",
                    "source_dataset": "EXIOBASE3",
                    "source_system": "pxp",
                    "source_version": "3",
                    "source_level": "product",
                    "target_dataset": "Eurostat_IO",
                    "target_system": "pxp",
                    "target_version": target_version,
                    "target_level": "product",
                    "source_code": _clean_text(getattr(row, "exio3_product_code")),
                    "source_label": _clean_text(getattr(row, "exio3_product_name")),
                    "target_code": target_code,
                    "target_label": target_label,
                    "enabled": "1",
                    "provenance": "official-exiobase-concordance-derived",
                    "certainty": certainty,
                    "allocation_rule": allocation_rule,
                    "source_name": "exiobase3_products_to_eurostat_iot_a64_collapsed.csv",
                    "official_source": OFFICIAL_SOURCE,
                    "notes": note,
                    "source_balance_role": "",
                    "target_sector_role": "",
                    "include_in_direct_energy": "",
                }
            )

    out = pd.DataFrame(rows).loc[:, list(CANONICAL_COLUMNS)].copy()
    duplicated = out.duplicated(subset=["source_code", "target_code"], keep=False)
    if bool(duplicated.any()):
        duplicate_pairs = out.loc[duplicated, ["source_code", "target_code"]].drop_duplicates().head(20).to_dict(orient="records")
        raise ValueError(f"Converted mapping contains duplicate source/target pairs: {duplicate_pairs}")
    source_count = out["source_code"].nunique()
    if source_count != 200:
        raise ValueError(f"Expected 200 distinct EXIOBASE3 source products; found {source_count}.")
    return out


def coverage_summary(mapping: pd.DataFrame) -> dict[str, object]:
    active = mapping[mapping["enabled"].map(_as_bool)].copy()
    source_codes = pd.Series(active["source_code"], index=active.index, dtype="object")
    target_codes = pd.Series(active["target_code"], index=active.index, dtype="object")
    target_versions = pd.Series(active["target_version"], index=active.index, dtype="object")
    by_source = target_codes.groupby(source_codes).nunique()
    return {
        "rows": int(len(active)),
        "distinct_sources": int(source_codes.nunique()),
        "distinct_targets": int(target_codes.nunique()),
        "multi_target_sources": int((by_source > 1).sum()),
        "target_version_values": sorted(map(str, target_versions.unique().tolist())),
        "has_static_weight_column": "weight" in active.columns,
    }


def write_outputs(
    collapsed_path: Path,
    out_dir: Path,
    *,
    cpa64_dictionary: Path = DEFAULT_CPA64_DICTIONARY,
    cpa65_dictionary: Path = DEFAULT_CPA65_DICTIONARY,
) -> dict[str, object]:
    collapsed = pd.read_csv(collapsed_path, dtype=str).fillna("")
    out_dir.mkdir(parents=True, exist_ok=True)
    cpa64_labels = read_target_label_map(cpa64_dictionary)
    cpa65_labels = read_target_label_map(cpa65_dictionary)

    cpa64 = convert_collapsed_to_canonical(collapsed, projection="cpa64", cpa64_labels=cpa64_labels, cpa65_labels=cpa65_labels)
    cpa65 = convert_collapsed_to_canonical(collapsed, projection="cpa65", cpa64_labels=cpa64_labels, cpa65_labels=cpa65_labels)

    cpa64_path = out_dir / CPA64_OUTPUT_NAME
    cpa65_path = out_dir / CPA65_OUTPUT_NAME
    cpa64.to_csv(cpa64_path, index=False)
    cpa65.to_csv(cpa65_path, index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_collapsed_csv": str(collapsed_path),
        "target_dictionaries": {
            "cpa64": str(cpa64_dictionary),
            "cpa65": str(cpa65_dictionary),
        },
        "outputs": {
            "cpa64": str(cpa64_path),
            "cpa65_compat": str(cpa65_path),
        },
        "coverage": {
            "cpa64": coverage_summary(cpa64),
            "cpa65_compat": coverage_summary(cpa65),
        },
        "method_notes": [
            "Generated from EXIOBASE official-concordance-derived A64 collapsed mapping.",
            "Rows use the existing MRIO canonical weighted mapping schema for script compatibility.",
            "No static weight column is written. Multi-target sources use allocation_rule=split-by-output-share.",
            "The CPA_64 output keeps one CPA_L68 real-estate target.",
            "The CPA_65 compatibility output splits only CPA_L68 into CPA_L68A and CPA_L68B for older workflows.",
        ],
        "licensing_notice": "Generated files derive from EXIOBASE/XIO concordances; check upstream terms before redistribution.",
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collapsed-csv", type=Path, default=DEFAULT_COLLAPSED, help="Collapsed A64 CSV generated from EXIOBASE concordance sources.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for generated current-schema mapping CSVs.")
    parser.add_argument("--cpa64-dictionary", type=Path, default=DEFAULT_CPA64_DICTIONARY, help="Committed CPA_64 target dictionary used for generated CPA_64 labels.")
    parser.add_argument("--cpa65-dictionary", type=Path, default=DEFAULT_CPA65_DICTIONARY, help="Committed CPA_65 target dictionary used for generated CPA_65 labels.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.collapsed_csv.exists():
        raise SystemExit(f"Collapsed input not found: {args.collapsed_csv}")
    manifest = write_outputs(args.collapsed_csv, args.out_dir, cpa64_dictionary=args.cpa64_dictionary, cpa65_dictionary=args.cpa65_dictionary)
    print(json.dumps(manifest["coverage"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
