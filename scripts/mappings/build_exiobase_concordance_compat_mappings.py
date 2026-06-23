#!/usr/bin/env python3
"""Convert EXIOBASE official-concordance output to CPA_64/CPA_65 mapping CSV schema.

This script does not rebuild the EXIOBASE concordance itself. It consumes the
collapsed output produced by ``build_exiobase3_to_eurostat_iot_a64_mapping.py``
and writes current-schema mapping CSVs usable by the existing MRIO mapping APIs.

Two projections are supported:

* CPA_64: canonical 64-sector target with one ``CPA_L68`` real-estate sector.
* CPA_65: compatibility target that splits only ``CPA_L68`` into ``CPA_L68A``
  and ``CPA_L68B``. No static L68 weights are stored; runtime code should use
  the selected Eurostat target-side value reference for split weights.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
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

CPA65_REAL_ESTATE_TARGETS: Final[tuple[tuple[str, str], ...]] = (
    ("CPA_L68A", "Imputed rents of owner-occupied dwellings"),
    ("CPA_L68B", "Real estate services excluding imputed rents"),
)

OFFICIAL_SOURCE = (
    "EXIOBASE/XIO concordances: NACE2_EXIOBASE20p_list.csv | "
    "EXIOBASE2 products comparison with EXIOBASE3 products.xlsx | "
    "Eurostat IOT A*64 division rule"
)

CPA64_DICTIONARY_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("CPA_A01", "Products of crop and animal production, hunting and related services", "A*64 A01; NACE/CPA division 01."),
    ("CPA_A02", "Products of forestry and logging", "A*64 A02; NACE/CPA division 02."),
    ("CPA_A03", "Fish and other fishing products; aquaculture products; support services to fishing", "A*64 A03; NACE/CPA division 03."),
    ("CPA_B", "Mining and quarrying products", "A*64 B; NACE/CPA divisions 05-09."),
    ("CPA_C10-12", "Food products, beverages and tobacco products", "A*64 C10-C12; NACE/CPA divisions 10-12."),
    ("CPA_C13-15", "Textiles, wearing apparel, leather and related products", "A*64 C13-C15; NACE/CPA divisions 13-15."),
    ("CPA_C16", "Wood and products of wood and cork, except furniture; articles of straw and plaiting materials", "A*64 C16; NACE/CPA division 16."),
    ("CPA_C17", "Paper and paper products", "A*64 C17; NACE/CPA division 17."),
    ("CPA_C18", "Printing and reproduction services of recorded media", "A*64 C18; NACE/CPA division 18."),
    ("CPA_C19", "Coke and refined petroleum products", "A*64 C19; NACE/CPA division 19."),
    ("CPA_C20", "Chemicals and chemical products", "A*64 C20; NACE/CPA division 20."),
    ("CPA_C21", "Basic pharmaceutical products and pharmaceutical preparations", "A*64 C21; NACE/CPA division 21."),
    ("CPA_C22", "Rubber and plastic products", "A*64 C22; NACE/CPA division 22."),
    ("CPA_C23", "Other non-metallic mineral products", "A*64 C23; NACE/CPA division 23."),
    ("CPA_C24", "Basic metals", "A*64 C24; NACE/CPA division 24."),
    ("CPA_C25", "Fabricated metal products, except machinery and equipment", "A*64 C25; NACE/CPA division 25."),
    ("CPA_C26", "Computer, electronic and optical products", "A*64 C26; NACE/CPA division 26."),
    ("CPA_C27", "Electrical equipment", "A*64 C27; NACE/CPA division 27."),
    ("CPA_C28", "Machinery and equipment n.e.c.", "A*64 C28; NACE/CPA division 28."),
    ("CPA_C29", "Motor vehicles, trailers and semi-trailers", "A*64 C29; NACE/CPA division 29."),
    ("CPA_C30", "Other transport equipment", "A*64 C30; NACE/CPA division 30."),
    ("CPA_C31_32", "Furniture; other manufactured goods", "A*64 C31-C32; NACE/CPA divisions 31-32."),
    ("CPA_C33", "Repair and installation services of machinery and equipment", "A*64 C33; NACE/CPA division 33."),
    ("CPA_D", "Electricity, gas, steam and air-conditioning", "A*64 D35; NACE/CPA division 35."),
    ("CPA_E36", "Natural water; water treatment and supply services", "A*64 E36; NACE/CPA division 36."),
    ("CPA_E37-39", "Sewerage; waste collection, treatment and disposal; remediation and other waste management services", "A*64 E37-E39; NACE/CPA divisions 37-39."),
    ("CPA_F", "Constructions and construction works", "A*64 F; NACE/CPA divisions 41-43."),
    ("CPA_G45", "Wholesale and retail trade and repair services of motor vehicles and motorcycles", "A*64 G45; NACE/CPA division 45."),
    ("CPA_G46", "Wholesale trade services, except of motor vehicles and motorcycles", "A*64 G46; NACE/CPA division 46."),
    ("CPA_G47", "Retail trade services, except of motor vehicles and motorcycles", "A*64 G47; NACE/CPA division 47."),
    ("CPA_H49", "Land transport services and transport services via pipelines", "A*64 H49; NACE/CPA division 49."),
    ("CPA_H50", "Water transport services", "A*64 H50; NACE/CPA division 50."),
    ("CPA_H51", "Air transport services", "A*64 H51; NACE/CPA division 51."),
    ("CPA_H52", "Warehousing and support services for transportation", "A*64 H52; NACE/CPA division 52."),
    ("CPA_H53", "Postal and courier services", "A*64 H53; NACE/CPA division 53."),
    ("CPA_I", "Accommodation and food services", "A*64 I; NACE/CPA divisions 55-56."),
    ("CPA_J58", "Publishing services", "A*64 J58; NACE/CPA division 58."),
    ("CPA_J59_60", "Motion picture, video, television programme production, sound recording and music publishing; programming and broadcasting services", "A*64 J59-J60; NACE/CPA divisions 59-60."),
    ("CPA_J61", "Telecommunications services", "A*64 J61; NACE/CPA division 61."),
    ("CPA_J62_63", "Computer programming, consultancy and related services; information services", "A*64 J62-J63; NACE/CPA divisions 62-63."),
    ("CPA_K64", "Financial services, except insurance and pension funding", "A*64 K64; NACE/CPA division 64."),
    ("CPA_K65", "Insurance, reinsurance and pension funding services, except compulsory social security", "A*64 K65; NACE/CPA division 65."),
    ("CPA_K66", "Services auxiliary to financial services and insurance services", "A*64 K66; NACE/CPA division 66."),
    ("CPA_L68", "Real estate services", "A*64 L68; NACE/CPA division 68. This 64-sector target does not split imputed rents."),
    ("CPA_M69_70", "Legal and accounting services; services of head offices; management consulting services", "A*64 M69-M70; NACE/CPA divisions 69-70."),
    ("CPA_M71", "Architectural and engineering services; technical testing and analysis services", "A*64 M71; NACE/CPA division 71."),
    ("CPA_M72", "Scientific research and development services", "A*64 M72; NACE/CPA division 72."),
    ("CPA_M73", "Advertising and market research services", "A*64 M73; NACE/CPA division 73."),
    ("CPA_M74_75", "Other professional, scientific and technical services; veterinary services", "A*64 M74-M75; NACE/CPA divisions 74-75."),
    ("CPA_N77", "Rental and leasing services", "A*64 N77; NACE/CPA division 77."),
    ("CPA_N78", "Employment services", "A*64 N78; NACE/CPA division 78."),
    ("CPA_N79", "Travel agency, tour operator and other reservation services and related services", "A*64 N79; NACE/CPA division 79."),
    ("CPA_N80-82", "Security and investigation; services to buildings and landscape; office administrative and support services", "A*64 N80-N82; NACE/CPA divisions 80-82."),
    ("CPA_O", "Public administration and defence services; compulsory social security services", "A*64 O84; NACE/CPA division 84."),
    ("CPA_P", "Education services", "A*64 P85; NACE/CPA division 85."),
    ("CPA_Q86", "Human health services", "A*64 Q86; NACE/CPA division 86."),
    ("CPA_Q87_88", "Residential care services; social work services without accommodation", "A*64 Q87-Q88; NACE/CPA divisions 87-88."),
    ("CPA_R90-92", "Creative, arts and entertainment; libraries, archives, museums and other cultural; gambling and betting services", "A*64 R90-R92; NACE/CPA divisions 90-92."),
    ("CPA_R93", "Sporting services and amusement and recreation services", "A*64 R93; NACE/CPA division 93."),
    ("CPA_S94", "Services furnished by membership organisations", "A*64 S94; NACE/CPA division 94."),
    ("CPA_S95", "Repair services of computers and personal and household goods", "A*64 S95; NACE/CPA division 95."),
    ("CPA_S96", "Other personal services", "A*64 S96; NACE/CPA division 96."),
    ("CPA_T", "Services of households as employers; undifferentiated goods and services produced by households for own use", "A*64 T; NACE/CPA divisions 97-98."),
    ("CPA_U", "Services provided by extraterritorial organisations and bodies", "A*64 U; NACE/CPA division 99."),
)

CPA64_LABEL_BY_CODE: Final[dict[str, str]] = {code: label for code, label, _description in CPA64_DICTIONARY_ROWS}


def write_cpa64_dictionary(path: Path) -> None:
    frame = pd.DataFrame(CPA64_DICTIONARY_ROWS, columns=["code", "label", "description"])
    if frame["code"].nunique() != 64:
        raise AssertionError("CPA_64 dictionary must contain 64 unique target codes.")
    if "CPA_L68" not in set(frame["code"]):
        raise AssertionError("CPA_64 dictionary must contain CPA_L68.")
    if {"CPA_L68A", "CPA_L68B"} & set(frame["code"]):
        raise AssertionError("CPA_64 dictionary must not contain CPA_L68A/CPA_L68B.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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


def _target_rows_for_a64(row: object, *, projection: str) -> list[tuple[str, str, str]]:
    a64_code = _clean_text(getattr(row, "iot_a64_code"))
    cpa64_code = A64_CODE_TO_CPA64[a64_code]
    if projection == "cpa64":
        return [(cpa64_code, CPA64_LABEL_BY_CODE[cpa64_code], "CPA_64")]
    if projection == "cpa65":
        if cpa64_code == "CPA_L68":
            return [(code, label, "CPA_65") for code, label in CPA65_REAL_ESTATE_TARGETS]
        return [(cpa64_code, CPA64_LABEL_BY_CODE[cpa64_code], "CPA_65")]
    raise ValueError(f"Unknown projection: {projection!r}")


def convert_collapsed_to_canonical(frame: pd.DataFrame, *, projection: str) -> pd.DataFrame:
    """Return current-schema rows for the requested target projection."""
    _validate_collapsed(frame)
    work = frame.copy().fillna("")
    work["_source_target_count"] = _source_target_count(work)

    rows: list[dict[str, object]] = []
    for _source_code, source_group in work.groupby(work["exio3_product_code"].astype(str).str.strip(), sort=True):
        expanded_targets: list[tuple[object, str, str, str]] = []
        for row in source_group.itertuples(index=False):
            for target_code, target_label, target_version in _target_rows_for_a64(row, projection=projection):
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


def write_outputs(collapsed_path: Path, out_dir: Path) -> dict[str, object]:
    collapsed = pd.read_csv(collapsed_path, dtype=str).fillna("")
    out_dir.mkdir(parents=True, exist_ok=True)

    cpa64 = convert_collapsed_to_canonical(collapsed, projection="cpa64")
    cpa65 = convert_collapsed_to_canonical(collapsed, projection="cpa65")

    cpa64_path = out_dir / CPA64_OUTPUT_NAME
    cpa65_path = out_dir / CPA65_OUTPUT_NAME
    cpa64.to_csv(cpa64_path, index=False)
    cpa65.to_csv(cpa65_path, index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_collapsed_csv": str(collapsed_path),
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
    parser.add_argument("--write-cpa64-dictionary", type=Path, default=REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_products_dictionary.csv", help="Write/update the CPA_64 64-sector target dictionary at this path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_cpa64_dictionary(args.write_cpa64_dictionary)
    if not args.collapsed_csv.exists():
        raise SystemExit(f"Collapsed input not found: {args.collapsed_csv}")
    manifest = write_outputs(args.collapsed_csv, args.out_dir)
    print(json.dumps(manifest["coverage"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
