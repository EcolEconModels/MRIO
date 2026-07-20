#!/usr/bin/env python3
"""Build canonical EXIOBASE3 -> CPA64 and CPA65 maps from the official direct concordance."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/exiobase_concordance_sources/cpasut64_exiobase3p.csv"
DEFAULT_EXIO_DICTIONARY = REPO_ROOT / "scripts/mappings/exiobase_products_dictionary.csv"
DEFAULT_CPA64_DICTIONARY = REPO_ROOT / "scripts/mappings/eurostat_cpa64_products_dictionary.csv"
DEFAULT_CPA65_DICTIONARY = REPO_ROOT / "scripts/mappings/eurostat_cpa65_products_dictionary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "scripts/mappings/generated/official_direct"
OFFICIAL_SOURCE_URL = (
    "https://github.com/EXIOBASE/00-concordances-public/blob/main/"
    "concordances/international/cpasut64_exiobase3p.csv"
)
CANONICAL_COLUMNS = (
    "order", "kind", "source_dataset", "source_system", "source_version",
    "source_level", "target_dataset", "target_system", "target_version",
    "target_level", "source_code", "source_label", "target_code",
    "target_label", "enabled", "provenance", "certainty",
    "allocation_rule", "source_name", "official_source", "notes",
    "source_balance_role", "target_sector_role", "include_in_direct_energy",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def read_dictionary(path: Path) -> tuple[list[str], dict[str, str]]:
    rows = read_rows(path)
    if not rows or not {"code", "label"}.issubset(rows[0]):
        raise ValueError(f"Dictionary must contain code,label columns: {path}")
    codes = [row["code"] for row in rows]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError(f"Dictionary contains blank or duplicate codes: {path}")
    return codes, {row["code"]: row["label"] for row in rows}


def normalize_cpa64(code: str) -> str:
    # The upstream direct file calls the A*64 real-estate row CPA_L; the repo dictionary uses CPA_L68.
    return "CPA_L68" if code == "CPA_L" else code


def target_codes(cpa64_code: str, target_version: str) -> tuple[str, ...]:
    if target_version == "CPA_65" and cpa64_code == "CPA_L68":
        return ("CPA_L68A", "CPA_L68B")
    return (cpa64_code,)


def validate_source(rows: list[dict[str, str]], exio_codes: set[str], cpa64_codes: set[str]) -> list[tuple[str, str]]:
    required = {"source_code", "target_code", "weight"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Official concordance must contain columns {sorted(required)}")
    edges: list[tuple[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        cpa = normalize_cpa64(row["source_code"])
        exio = row["target_code"]
        if not cpa or not exio:
            raise ValueError(f"Blank source/target code at official row {row_number}")
        try:
            weight = float(row["weight"])
        except ValueError as exc:
            raise ValueError(f"Invalid incidence weight at official row {row_number}: {row['weight']!r}") from exc
        if weight != 1.0:
            raise ValueError(f"Expected binary incidence weight 1.0 at official row {row_number}, got {weight}")
        if cpa not in cpa64_codes:
            raise ValueError(f"Unknown CPA64 code after normalization at official row {row_number}: {cpa}")
        if exio not in exio_codes:
            raise ValueError(f"Unknown EXIOBASE3 code at official row {row_number}: {exio}")
        edges.append((exio, cpa))
    if len(edges) != len(set(edges)):
        duplicates = [edge for edge, count in Counter(edges).items() if count > 1]
        raise ValueError(f"Duplicate official concordance edges: {duplicates[:8]}")
    covered_exio = {exio for exio, _ in edges}
    covered_cpa = {cpa for _, cpa in edges}
    if covered_exio != exio_codes:
        raise ValueError(f"EXIOBASE3 coverage mismatch; missing={sorted(exio_codes-covered_exio)}, extra={sorted(covered_exio-exio_codes)}")
    if covered_cpa != cpa64_codes:
        raise ValueError(f"CPA64 coverage mismatch; missing={sorted(cpa64_codes-covered_cpa)}, extra={sorted(covered_cpa-cpa64_codes)}")
    return edges


def build_mapping(
    edges: list[tuple[str, str]],
    *,
    target_version: str,
    exio_order: list[str],
    exio_labels: dict[str, str],
    target_order: list[str],
    target_labels: dict[str, str],
) -> list[dict[str, str]]:
    expanded: set[tuple[str, str]] = set()
    for exio, cpa64 in edges:
        expanded.update((exio, target) for target in target_codes(cpa64, target_version))
    expected_targets = set(target_order)
    actual_targets = {target for _, target in expanded}
    if actual_targets != expected_targets:
        raise ValueError(f"{target_version} target coverage mismatch; missing={sorted(expected_targets-actual_targets)}, extra={sorted(actual_targets-expected_targets)}")
    target_rank = {code: index for index, code in enumerate(target_order)}
    exio_rank = {code: index for index, code in enumerate(exio_order)}
    target_count: Counter[str] = Counter(exio for exio, _ in expanded)
    rows: list[dict[str, str]] = []
    for order, (exio, target) in enumerate(
        sorted(expanded, key=lambda edge: (exio_rank[edge[0]], target_rank[edge[1]])), start=1
    ):
        multiple = target_count[exio] > 1
        rows.append({
            "order": str(order),
            "kind": "weighted",
            "source_dataset": "EXIOBASE3",
            "source_system": "pxp",
            "source_version": "3",
            "source_level": "product",
            "target_dataset": "Eurostat_IO",
            "target_system": "pxp",
            "target_version": target_version,
            "target_level": "CPA_product",
            "source_code": exio,
            "source_label": exio_labels[exio],
            "target_code": target,
            "target_label": target_labels[target],
            "enabled": "1",
            "provenance": "official-direct-cpasut64-exiobase3p",
            "certainty": "official-direct-topology",
            "allocation_rule": "split-by-output-share" if multiple else "direct-single-target",
            "source_name": "cpasut64_exiobase3p.csv",
            "official_source": OFFICIAL_SOURCE_URL,
            "notes": (
                "Official binary CPA64-EXIOBASE3 incidence edge, reversed for runtime use. "
                + ("Runtime target-side output/use values resolve this multi-target EXIOBASE3 product."
                   if multiple else "This EXIOBASE3 product has one target in this target classification.")
            ),
            "source_balance_role": "",
            "target_sector_role": "",
            "include_in_direct_energy": "1",
        })
    if {row["source_code"] for row in rows} != set(exio_order):
        raise ValueError(f"{target_version} output does not cover the EXIOBASE3 dictionary")
    return rows


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = list(CANONICAL_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exio-dictionary", type=Path, default=DEFAULT_EXIO_DICTIONARY)
    parser.add_argument("--cpa64-dictionary", type=Path, default=DEFAULT_CPA64_DICTIONARY)
    parser.add_argument("--cpa65-dictionary", type=Path, default=DEFAULT_CPA65_DICTIONARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    exio_order, exio_labels = read_dictionary(args.exio_dictionary)
    cpa64_order, cpa64_labels = read_dictionary(args.cpa64_dictionary)
    cpa65_order, cpa65_labels = read_dictionary(args.cpa65_dictionary)
    official_rows = read_rows(args.source)
    edges = validate_source(official_rows, set(exio_order), set(cpa64_order))

    outputs = (
        ("CPA_64", cpa64_order, cpa64_labels, "exio_to_eurostat_cpa64_products_from_official_direct.csv"),
        ("CPA_65", cpa65_order, cpa65_labels, "exio_to_eurostat_cpa65_products_from_official_direct.csv"),
    )
    for target_version, target_order, target_labels, filename in outputs:
        rows = build_mapping(
            edges,
            target_version=target_version,
            exio_order=exio_order,
            exio_labels=exio_labels,
            target_order=target_order,
            target_labels=target_labels,
        )
        output = args.out_dir / filename
        write_mapping(output, rows)
        n_multi = len({row["source_code"] for row in rows if row["allocation_rule"] == "split-by-output-share"})
        print(f"Wrote {output}: rows={len(rows)}, EXIOBASE3=200, targets={len(target_order)}, multi-target EXIOBASE3={n_multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
