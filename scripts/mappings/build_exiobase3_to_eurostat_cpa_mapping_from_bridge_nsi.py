#!/usr/bin/env python3
"""Build EXIOBASE3 -> Eurostat CPA mappings from Bridge_NSI_AT.xlsx.

The bridge workbook is an Austria-specific EXIOBASE/XIO national-statistics
bridge.  This generator converts the product bridge sheet into the current MRIO
canonical weighted mapping schema.  Runtime code should still resolve omitted
many-target weights from a selected Eurostat target-side value reference.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CONCORDANCES_ZIP: Final[Path] = REPO_ROOT / "data" / "exiobase_concordance_sources" / "exiobase_concordances.zip"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated"
DEFAULT_CPA64_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa64_products_dictionary.csv"
DEFAULT_CPA65_DICTIONARY: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "eurostat_cpa65_products_dictionary.csv"
DEFAULT_BRIDGE_MEMBER_SUFFIX: Final[str] = "Bridge_NSI_AT.xlsx"
DEFAULT_BRIDGE_SHEET: Final[str] = "AT_r_sup"

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

# Bridge_NSI_AT.xlsx uses older NSI product rows.  This table is deliberately
# transparent and conservative: broad old rows become many-target rows so the
# existing weighted-mapping resolver can apply selected-country/year/flow
# target-side value weights instead of static shares.
NSI_PRODUCT_TO_CPA64: Final[dict[str, tuple[str, ...]]] = {
    "p01": ("CPA_A01",),
    "p02": ("CPA_A02",),
    "p03": ("CPA_A03",),
    "p04": ("CPA_B",),
    "p05": ("CPA_B",),
    "p06": ("CPA_B",),
    "p07": ("CPA_B",),
    "p08": ("CPA_B",),
    "p09": ("CPA_C10-12",),
    "p10": ("CPA_C10-12",),
    "p11": ("CPA_C13-15",),
    "p12": ("CPA_C13-15",),
    "p13": ("CPA_C13-15",),
    "p14": ("CPA_C16",),
    "p15": ("CPA_C17",),
    "p16": ("CPA_C18", "CPA_J58", "CPA_J59_60"),
    "p17": ("CPA_C19",),
    "p18": ("CPA_C20", "CPA_C21"),
    "p19": ("CPA_C22",),
    "p20": ("CPA_C23",),
    "p21": ("CPA_C24",),
    "p22": ("CPA_C25", "CPA_C33"),
    "p23": ("CPA_C28",),
    "p24": ("CPA_C26",),
    "p25": ("CPA_C27",),
    "p26": ("CPA_C26",),
    "p27": ("CPA_C26",),
    "p28": ("CPA_C29",),
    "p29": ("CPA_C30",),
    "p30": ("CPA_C31_32",),
    "p31": ("CPA_E37-39",),
    "p32": ("CPA_D",),
    "p33": ("CPA_E36",),
    "p34": ("CPA_F",),
    "p35": ("CPA_G45", "CPA_G47"),
    "p36": ("CPA_G46",),
    "p37": ("CPA_G47", "CPA_S95"),
    "p38": ("CPA_I",),
    "p39": ("CPA_H49",),
    "p40": ("CPA_H50",),
    "p41": ("CPA_H51",),
    "p42": ("CPA_H52", "CPA_N79"),
    "p43": ("CPA_H53", "CPA_J61"),
    "p44": ("CPA_K64",),
    "p45": ("CPA_K65",),
    "p46": ("CPA_K66",),
    "p47": ("CPA_L68",),
    "p48": ("CPA_N77",),
    "p49": ("CPA_J62_63",),
    "p50": ("CPA_M72",),
    "p51": ("CPA_M69_70", "CPA_M71", "CPA_M73", "CPA_M74_75", "CPA_N78", "CPA_N80-82"),
    "p52": ("CPA_O",),
    "p53": ("CPA_P",),
    "p54": ("CPA_Q86", "CPA_Q87_88"),
    "p55": ("CPA_E37-39",),
    "p56": ("CPA_S94",),
    "p57": ("CPA_J59_60", "CPA_R90-92", "CPA_R93"),
    "p58": ("CPA_S96",),
    "p59": ("CPA_T",),
}

EXIO_CODE_ALIASES: Final[dict[str, str]] = {
    "i23.3": "p23.3",
    "p26.w.1": "p26.a.w",
}


@dataclass(frozen=True)
class BridgeBuildResult:
    mapping: pd.DataFrame
    edges: pd.DataFrame
    diagnostics: pd.DataFrame
    coverage: dict[str, object]


def clean_text(value: object) -> str:
    """Normalize a value into a stable one-line string key."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a byte payload."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_member_by_suffix(zf: zipfile.ZipFile, suffix: str) -> str:
    """Find exactly one ZIP member whose normalized path ends with suffix."""

    matches = [name for name in zf.namelist() if name.replace("\\", "/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No zip member ending with {suffix!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple zip members ending with {suffix!r}: {matches}")
    return matches[0]


def load_bridge_workbook_bytes(concordances_zip: Path, member_suffix: str) -> tuple[bytes, str]:
    """Read Bridge_NSI_AT.xlsx bytes from the concordance ZIP."""

    if not concordances_zip.exists():
        raise FileNotFoundError(f"Missing concordance archive: {concordances_zip}")
    with zipfile.ZipFile(concordances_zip) as zf:
        member = find_member_by_suffix(zf, member_suffix)
        return zf.read(member), member


def load_target_labels(dictionary_path: Path) -> dict[str, str]:
    """Load target CPA code labels from the committed dictionary."""

    if not dictionary_path.exists():
        raise FileNotFoundError(f"Missing target dictionary: {dictionary_path}")
    frame = pd.read_csv(dictionary_path, dtype=str).fillna("")
    missing = sorted({"code", "label"} - set(frame.columns))
    if missing:
        raise ValueError(f"Dictionary {dictionary_path} missing columns: {missing}")
    return dict(zip(frame["code"].astype(str).str.strip(), frame["label"].astype(str).str.strip(), strict=False))


def target_codes_for_version(nsi_code: str, target_version: str) -> tuple[str, ...]:
    """Return CPA64/CPA65 target codes for an old Bridge_NSI_AT product row."""

    base = NSI_PRODUCT_TO_CPA64.get(nsi_code, tuple())
    if target_version.casefold() != "cpa65":
        return base
    out: list[str] = []
    for code in base:
        if code == "CPA_L68":
            out.extend(["CPA_L68A", "CPA_L68B"])
        else:
            out.append(code)
    return tuple(out)


def extract_bridge_product_edges(workbook_bytes: bytes, *, sheet_name: str) -> pd.DataFrame:
    """Extract non-zero EXIO product -> NSI product row edges from a bridge sheet.

    The workbook layout has four header rows.  Row 0 stores EXIO labels, row 2
    stores EXIO product codes, and rows from 4 onward store NSI row labels/codes.
    """

    raw = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name=sheet_name, header=None)
    exio_columns: list[tuple[int, str, str, str]] = []
    for col in range(2, raw.shape[1]):
        raw_code = clean_text(raw.iat[2, col])
        if not re.match(r"^[pi]\d", raw_code):
            continue
        exio_code = EXIO_CODE_ALIASES.get(raw_code, raw_code)
        exio_columns.append((col, raw_code, exio_code, clean_text(raw.iat[0, col])))

    records: list[dict[str, object]] = []
    for row_idx in range(4, raw.shape[0]):
        nsi_code = clean_text(raw.iat[row_idx, 1])
        if not re.match(r"^p\d{2}$", nsi_code):
            continue
        nsi_label = clean_text(raw.iat[row_idx, 0])
        for col, raw_exio_code, exio_code, exio_label in exio_columns:
            value = pd.to_numeric(pd.Series([raw.iat[row_idx, col]]), errors="coerce").fillna(0.0).iloc[0]
            if float(value) == 0.0:
                continue
            records.append(
                {
                    "bridge_sheet": sheet_name,
                    "bridge_value": float(value),
                    "bridge_nsi_code": nsi_code,
                    "bridge_nsi_label": nsi_label,
                    "raw_exio_code": raw_exio_code,
                    "exio3_product_code": exio_code,
                    "exio3_product_name": exio_label,
                    "code_normalized": raw_exio_code != exio_code,
                }
            )
    return pd.DataFrame.from_records(records)


def build_mapping(
    *,
    workbook_bytes: bytes,
    workbook_sha256: str,
    bridge_member: str,
    sheet_name: str,
    target_version: str,
    target_dictionary: Path,
) -> BridgeBuildResult:
    """Build a canonical EXIOBASE3 -> CPA weighted mapping from Bridge_NSI_AT."""

    labels = load_target_labels(target_dictionary)
    bridge_edges = extract_bridge_product_edges(workbook_bytes, sheet_name=sheet_name)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for edge in bridge_edges.itertuples(index=False):
        nsi_code = str(edge.bridge_nsi_code)
        target_codes = target_codes_for_version(nsi_code, target_version)
        if not target_codes:
            diagnostics.append(
                {
                    "diagnostic_kind": "unmapped_nsi_product_row",
                    "bridge_nsi_code": nsi_code,
                    "bridge_nsi_label": str(edge.bridge_nsi_label),
                    "exio3_product_code": str(edge.exio3_product_code),
                    "message": "No NSI-product-row to CPA target rule is declared.",
                }
            )
            continue
        for target_code in target_codes:
            if target_code not in labels:
                diagnostics.append(
                    {
                        "diagnostic_kind": "target_missing_from_dictionary",
                        "bridge_nsi_code": nsi_code,
                        "bridge_nsi_label": str(edge.bridge_nsi_label),
                        "exio3_product_code": str(edge.exio3_product_code),
                        "message": f"Target {target_code!r} is not present in {target_dictionary}.",
                    }
                )
                continue
            rows.append(
                {
                    "source_code": str(edge.exio3_product_code),
                    "source_label": str(edge.exio3_product_name),
                    "target_code": target_code,
                    "target_label": labels[target_code],
                    "bridge_nsi_code": nsi_code,
                    "bridge_nsi_label": str(edge.bridge_nsi_label),
                    "raw_exio_code": str(edge.raw_exio_code),
                    "bridge_value": float(edge.bridge_value),
                    "code_normalized": bool(edge.code_normalized),
                }
            )

    edge_frame = pd.DataFrame.from_records(rows)
    if edge_frame.empty:
        mapping = pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        grouped = edge_frame.groupby(["source_code", "source_label", "target_code", "target_label"], as_index=False).agg(
            bridge_nsi_codes=("bridge_nsi_code", lambda values: ";".join(sorted(set(map(str, values))))),
            bridge_nsi_labels=("bridge_nsi_label", lambda values: " | ".join(sorted(set(map(str, values))))),
            raw_exio_codes=("raw_exio_code", lambda values: ";".join(sorted(set(map(str, values))))),
            bridge_values=("bridge_value", "sum"),
            any_code_normalized=("code_normalized", "any"),
        )
        target_counts = grouped.groupby("source_code")["target_code"].transform("nunique").astype(int)
        grouped["n_targets_for_exio3_product"] = target_counts
        mapping_rows: list[dict[str, object]] = []
        for order, row in enumerate(grouped.sort_values(["source_code", "target_code"]).itertuples(index=False), start=1):
            allocation_rule = "split-by-output-share" if int(row.n_targets_for_exio3_product) > 1 else "direct-single-target"
            mapping_rows.append(
                {
                    "order": order,
                    "kind": "weighted",
                    "source_dataset": "EXIOBASE",
                    "source_system": "pxp",
                    "source_version": "3.10.2-compatible-code-normalized",
                    "source_level": "product",
                    "target_dataset": "Eurostat_IO",
                    "target_system": "pxp",
                    "target_version": target_version.upper(),
                    "target_level": "CPA_product",
                    "source_code": row.source_code,
                    "source_label": row.source_label,
                    "target_code": row.target_code,
                    "target_label": row.target_label,
                    "enabled": "1",
                    "provenance": "Bridge_NSI_AT",
                    "certainty": "at-nsi-bridge-relational",
                    "allocation_rule": allocation_rule,
                    "source_name": "Bridge_NSI_AT.xlsx",
                    "official_source": bridge_member,
                    "notes": (
                        f"AT NSI bridge row(s) {row.bridge_nsi_codes}: {row.bridge_nsi_labels}. "
                        f"Bridge value sum={float(row.bridge_values):.12g}; raw EXIO code(s)={row.raw_exio_codes}. "
                        "Weights are not static; many-target rows should be resolved from Eurostat target-side values."
                    ),
                    "source_balance_role": "",
                    "target_sector_role": "",
                    "include_in_direct_energy": "1",
                }
            )
        mapping = pd.DataFrame(mapping_rows, columns=CANONICAL_COLUMNS)

    diagnostics_df = pd.DataFrame.from_records(
        diagnostics,
        columns=["diagnostic_kind", "bridge_nsi_code", "bridge_nsi_label", "exio3_product_code", "message"],
    ).fillna("")
    coverage = {
        "source_archive_or_workbook_sha256": workbook_sha256,
        "bridge_member": bridge_member,
        "bridge_sheet": sheet_name,
        "target_version": target_version,
        "raw_bridge_edges": int(len(bridge_edges)),
        "mapped_edge_rows": int(len(edge_frame)),
        "mapping_rows": int(len(mapping)),
        "distinct_exio3_products": int(mapping["source_code"].nunique()) if not mapping.empty else 0,
        "distinct_target_codes": int(mapping["target_code"].nunique()) if not mapping.empty else 0,
        "diagnostic_rows": int(len(diagnostics_df)),
        "normalized_exio_codes": sorted(set(bridge_edges.loc[bridge_edges.get("code_normalized", False), "raw_exio_code"].astype(str))) if not bridge_edges.empty else [],
    }
    return BridgeBuildResult(mapping=mapping, edges=edge_frame, diagnostics=diagnostics_df, coverage=coverage)


def write_outputs(result: BridgeBuildResult, *, out_dir: Path, target_version: str) -> dict[str, object]:
    """Write mapping, edge audit, diagnostics, and manifest outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"exio_to_eurostat_{target_version.lower()}_products_from_bridge_nsi_at"
    mapping_path = out_dir / f"{stem}.csv"
    edges_path = out_dir / f"{stem}_edge_audit.csv"
    diagnostics_path = out_dir / f"{stem}_diagnostics.csv"
    manifest_path = out_dir / f"{stem}_manifest.json"
    result.mapping.to_csv(mapping_path, index=False)
    result.edges.to_csv(edges_path, index=False)
    result.diagnostics.to_csv(diagnostics_path, index=False)
    manifest = {
        "mapping_name": stem,
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/mappings/build_exiobase3_to_eurostat_cpa_mapping_from_bridge_nsi.py",
        "method_notes": [
            "Reads Bridge_NSI_AT.xlsx product bridge sheet and emits current-schema weighted EXIOBASE3->CPA mapping rows.",
            "Old Bridge_NSI_AT product rows are mapped to current CPA targets by explicit transparent row rules.",
            "Many-target rows omit static weights and use allocation_rule=split-by-output-share for runtime Eurostat target-side value weighting.",
        ],
        "coverage": result.coverage,
        "output_files": {
            "mapping": mapping_path.name,
            "edge_audit": edges_path.name,
            "diagnostics": diagnostics_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concordances-zip", type=Path, default=DEFAULT_CONCORDANCES_ZIP)
    parser.add_argument("--bridge-xlsx", type=Path, default=None, help="Optional direct Bridge_NSI_AT.xlsx path; overrides --concordances-zip.")
    parser.add_argument("--bridge-member-suffix", default=DEFAULT_BRIDGE_MEMBER_SUFFIX)
    parser.add_argument("--bridge-sheet", default=DEFAULT_BRIDGE_SHEET)
    parser.add_argument("--target-version", choices=["cpa64", "cpa65"], default="cpa65")
    parser.add_argument("--target-dictionary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge-based mapping generator."""

    args = parse_args(argv)
    dictionary = args.target_dictionary
    if dictionary is None:
        dictionary = DEFAULT_CPA65_DICTIONARY if args.target_version == "cpa65" else DEFAULT_CPA64_DICTIONARY
    if args.bridge_xlsx is not None:
        workbook_bytes = args.bridge_xlsx.read_bytes()
        bridge_member = str(args.bridge_xlsx)
        workbook_sha = sha256_file(args.bridge_xlsx)
    else:
        workbook_bytes, bridge_member = load_bridge_workbook_bytes(args.concordances_zip, args.bridge_member_suffix)
        workbook_sha = sha256_bytes(workbook_bytes)
    result = build_mapping(
        workbook_bytes=workbook_bytes,
        workbook_sha256=workbook_sha,
        bridge_member=bridge_member,
        sheet_name=args.bridge_sheet,
        target_version=args.target_version,
        target_dictionary=dictionary,
    )
    if not args.check_only:
        write_outputs(result, out_dir=args.out_dir, target_version=args.target_version)
    print(json.dumps({"coverage": result.coverage, "out_dir": str(args.out_dir), "wrote_outputs": not args.check_only}, indent=2, sort_keys=True))
    if result.coverage["diagnostic_rows"]:
        raise SystemExit("Bridge NSI mapping has diagnostics; inspect the diagnostics CSV before runtime use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
