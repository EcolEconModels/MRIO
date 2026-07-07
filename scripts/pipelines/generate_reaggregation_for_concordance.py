#!/usr/bin/env python3
"""Generate a fuels reaggregation draft for a route-specific EXIO->CPA map.

The fuels split pipeline selects EXIOBASE children by expanding selected CPA
parents through a supplied EXIO->Eurostat concordance.  If an EXIO source maps to
multiple CPA targets, the split child is represented as ``source@target``.  This
helper prepares a reaggregation CSV for such route-specific child codes without
silently classifying genuinely new children.

Rules:

* existing base reaggregation rows are preserved;
* missing edge children like ``p40.11.a@CPA_D`` are copied from an existing
  unsuffixed base row ``p40.11.a`` when available;
* children with no exact row and no unsuffixed source row are written to a review
  CSV and are not added automatically.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

TRUE_VALUES: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES: Final[set[str]] = {"0", "false", "f", "no", "n", "off"}
REAGG_COLUMNS: Final[list[str]] = ["source_code", "source_label", "target_code", "target_label", "enabled", "notes"]
REVIEW_COLUMNS: Final[list[str]] = [
    "parent_code",
    "child_code",
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "reason",
]


@dataclass(frozen=True)
class SelectedChild:
    parent_code: str
    child_code: str
    source_code: str
    source_label: str
    target_code: str
    target_label: str


def parse_boolish(value: object) -> bool:
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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def require_columns(path: Path, fieldnames: Iterable[str], required: set[str]) -> None:
    present = set(fieldnames)
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")


def active_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if parse_boolish(row.get("enabled", "true"))]


def split_parent_codes(path: Path) -> list[str]:
    fieldnames, rows = read_csv(path)
    require_columns(path, fieldnames, {"source_code"})
    parents: list[str] = []
    seen: set[str] = set()
    for row in active_rows(rows):
        code = row.get("source_code", "").strip()
        if code and code not in seen:
            parents.append(code)
            seen.add(code)
    if not parents:
        raise SystemExit(f"{path} has no active split parent rows")
    return parents


def selected_children(split_csv: Path, concordance_csv: Path) -> list[SelectedChild]:
    parent_set = set(split_parent_codes(split_csv))
    fieldnames, concordance_rows = read_csv(concordance_csv)
    require_columns(concordance_csv, fieldnames, {"source_code", "target_code"})
    active_concordance_rows = active_rows(concordance_rows)
    targets_by_source: dict[str, set[str]] = {}
    for row in active_concordance_rows:
        source_code = row.get("source_code", "").strip()
        target_code = row.get("target_code", "").strip()
        if source_code and target_code:
            targets_by_source.setdefault(source_code, set()).add(target_code)

    selected: list[SelectedChild] = []
    seen: set[tuple[str, str]] = set()
    for row in active_concordance_rows:
        source_code = row.get("source_code", "").strip()
        target_code = row.get("target_code", "").strip()
        if not source_code or not target_code or target_code not in parent_set:
            continue
        child_code = f"{source_code}@{target_code}" if len(targets_by_source.get(source_code, set())) > 1 else source_code
        key = (target_code, child_code)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            SelectedChild(
                parent_code=target_code,
                child_code=child_code,
                source_code=source_code,
                source_label=row.get("source_label", "").strip(),
                target_code=target_code,
                target_label=row.get("target_label", "").strip(),
            )
        )
    missing_parents = sorted(parent_set - {child.parent_code for child in selected})
    if missing_parents:
        raise SystemExit(
            "Split manifest references parent sectors not present as targets in the concordance: "
            + ", ".join(missing_parents)
        )
    return selected


def normalize_base_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, rows = read_csv(path)
    require_columns(path, fieldnames, set(REAGG_COLUMNS))
    out: list[dict[str, str]] = []
    for row in rows:
        normalized = {column: row.get(column, "") for column in fieldnames}
        out.append(normalized)
    return fieldnames, out


def append_note(existing: str, addition: str) -> str:
    existing_text = existing.strip()
    if not existing_text:
        return addition
    if addition in existing_text:
        return existing_text
    return f"{existing_text} {addition}"


def generate_reaggregation(
    *,
    split_csv: Path,
    base_reagg_csv: Path,
    concordance_csv: Path,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    fieldnames, base_rows = normalize_base_rows(base_reagg_csv)
    children = selected_children(split_csv, concordance_csv)
    active_base = active_rows(base_rows)
    active_by_source: dict[str, list[dict[str, str]]] = {}
    for row in active_base:
        source_code = row.get("source_code", "").strip()
        if source_code:
            active_by_source.setdefault(source_code, []).append(row)

    covered = set(active_by_source)
    output_rows = list(base_rows)
    review_rows: list[dict[str, str]] = []
    added_aliases = 0

    for child in children:
        if child.child_code in covered:
            continue
        source_rows = active_by_source.get(child.source_code, [])
        if source_rows:
            for source_row in source_rows:
                new_row = dict(source_row)
                new_row["source_code"] = child.child_code
                label = child.source_label or source_row.get("source_label", "")
                new_row["source_label"] = f"{label} allocated to {child.target_code}" if "@" in child.child_code else label
                new_row["enabled"] = "True"
                new_row["notes"] = append_note(
                    source_row.get("notes", ""),
                    (
                        f"Generated by generate_reaggregation_for_concordance.py: route-specific child "
                        f"{child.child_code} selected under {child.parent_code} copies base reaggregation "
                        f"from {child.source_code}."
                    ),
                )
                output_rows.append(new_row)
                added_aliases += 1
            covered.add(child.child_code)
            continue
        review_rows.append(
            {
                "parent_code": child.parent_code,
                "child_code": child.child_code,
                "source_code": child.source_code,
                "source_label": child.source_label,
                "target_code": child.target_code,
                "target_label": child.target_label,
                "reason": "No exact child row and no unsuffixed base reaggregation source row; manual economic classification required.",
            }
        )

    summary = {
        "selected_children": len(children),
        "base_rows": len(base_rows),
        "added_alias_rows": added_aliases,
        "review_required_rows": len(review_rows),
        "output_rows": len(output_rows),
    }
    return fieldnames, output_rows, review_rows, summary


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-mapping-csv", required=True, type=Path)
    parser.add_argument("--base-reagg-csv", required=True, type=Path)
    parser.add_argument("--concordance-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help="Exit 0 even when review-required rows remain. The review rows are still not added to the output CSV.",
    )
    args = parser.parse_args(argv)

    fieldnames, output_rows, review_rows, summary = generate_reaggregation(
        split_csv=args.split_mapping_csv,
        base_reagg_csv=args.base_reagg_csv,
        concordance_csv=args.concordance_csv,
    )
    write_rows(args.out_csv, fieldnames, output_rows)
    write_rows(args.review_csv, REVIEW_COLUMNS, review_rows)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by_script": "scripts/pipelines/generate_reaggregation_for_concordance.py",
        "split_mapping_csv": str(args.split_mapping_csv),
        "base_reagg_csv": str(args.base_reagg_csv),
        "concordance_csv": str(args.concordance_csv),
        "out_csv": str(args.out_csv),
        "review_csv": str(args.review_csv),
        **summary,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if review_rows and not args.allow_review_required:
        raise SystemExit(
            f"Generated {args.out_csv} with {len(review_rows)} review-required rows in {args.review_csv}; "
            "review/classify them or rerun with --allow-review-required to accept a partial draft."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
