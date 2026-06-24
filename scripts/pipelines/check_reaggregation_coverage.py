#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _parse_boolish(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    if text == "":
        return True
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _active_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if _parse_boolish(row.get("enabled", "true")):
            out.append(row)
    return out


def _required_columns(rows: list[dict[str, str]], path: Path, columns: set[str]) -> None:
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    fieldnames = set(rows[0].keys())
    missing = sorted(columns - fieldnames)
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")


def split_parent_codes(path: Path) -> list[str]:
    rows = _read_csv(path)
    _required_columns(rows, path, {"source_code"})
    parents: list[str] = []
    seen: set[str] = set()
    for row in _active_rows(rows):
        code = str(row.get("source_code", "")).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        parents.append(code)
    return parents


def expand_selected_children(split_csv: Path, concordance_csv: Path) -> list[dict[str, str]]:
    parents = split_parent_codes(split_csv)
    parent_set = set(parents)
    concordance_rows = _active_rows(_read_csv(concordance_csv))
    _required_columns(concordance_rows, concordance_csv, {"source_code", "target_code"})
    targets_by_source: dict[str, set[str]] = {}
    for row in concordance_rows:
        source_code = str(row.get("source_code", "")).strip()
        target_code = str(row.get("target_code", "")).strip()
        if source_code and target_code:
            targets_by_source.setdefault(source_code, set()).add(target_code)

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in concordance_rows:
        target_code = str(row.get("target_code", "")).strip()
        source_code = str(row.get("source_code", "")).strip()
        if not target_code or not source_code or target_code not in parent_set:
            continue
        child_code = f"{source_code}@{target_code}" if len(targets_by_source.get(source_code, set())) > 1 else source_code
        key = (target_code, child_code)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "parent_code": target_code,
                "child_code": child_code,
                "source_code": source_code,
                "source_label": str(row.get("source_label", "")).strip(),
                "target_code": target_code,
                "target_label": str(row.get("target_label", "")).strip(),
            }
        )

    found_parents = {row["parent_code"] for row in selected}
    missing_parents = sorted(parent_set - found_parents)
    if missing_parents:
        raise SystemExit(
            "Split manifest references parent sectors not present as targets in the concordance: "
            + ", ".join(missing_parents)
        )
    return selected


def reaggregation_sources(path: Path) -> set[str]:
    rows = _read_csv(path)
    _required_columns(rows, path, {"source_code", "target_code"})
    sources: set[str] = set()
    for row in _active_rows(rows):
        source = str(row.get("source_code", "")).strip()
        target = str(row.get("target_code", "")).strip()
        if source and target:
            sources.add(source)
    return sources


def write_report(path: Path, rows: list[dict[str, str]], covered: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parent_code",
        "child_code",
        "source_code",
        "source_label",
        "target_code",
        "target_label",
        "covered_by_reaggregation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "covered_by_reaggregation": str(row["child_code"] in covered)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight-check split-manifest children against a reaggregation manifest."
    )
    parser.add_argument("--split-mapping-csv", required=True)
    parser.add_argument("--reagg-mapping-csv", required=True)
    parser.add_argument("--concordance-csv", required=True)
    parser.add_argument("--report-csv", default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    args = parser.parse_args(argv)

    split_csv = Path(args.split_mapping_csv)
    reagg_csv = Path(args.reagg_mapping_csv)
    concordance_csv = Path(args.concordance_csv)
    selected = expand_selected_children(split_csv, concordance_csv)
    covered_sources = reaggregation_sources(reagg_csv)
    missing_rows = [row for row in selected if row["child_code"] not in covered_sources]

    if args.report_csv:
        write_report(Path(args.report_csv), selected, covered_sources)

    summary = {
        "split_mapping_csv": str(split_csv),
        "reagg_mapping_csv": str(reagg_csv),
        "concordance_csv": str(concordance_csv),
        "n_selected_children": len(selected),
        "n_covered_children": len(selected) - len(missing_rows),
        "n_missing_children": len(missing_rows),
        "missing_children": [
            {
                "parent_code": row["parent_code"],
                "child_code": row["child_code"],
                "source_label": row["source_label"],
            }
            for row in missing_rows
        ],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Reaggregation coverage preflight: "
            f"selected={summary['n_selected_children']} "
            f"covered={summary['n_covered_children']} "
            f"missing={summary['n_missing_children']}"
        )
        if missing_rows:
            print("Missing selected split children:")
            for row in missing_rows[:50]:
                label = f" — {row['source_label']}" if row["source_label"] else ""
                print(f"  {row['parent_code']}: {row['child_code']}{label}")
            if len(missing_rows) > 50:
                print(f"  ... and {len(missing_rows) - 50} more")

    if missing_rows:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
