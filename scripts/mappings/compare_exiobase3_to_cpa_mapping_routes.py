#!/usr/bin/env python3
"""Compare old A64/compat and new official-concordance EXIOBASE3->CPA maps."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_COMPARE_OUT_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated" / "compare_routes" / "exiobase3_to_cpa"
ANCHOR_SOURCE_MODES: Final[tuple[str, ...]] = ("direct-only", "direct-plus-curated-gaps")
PAIR_COLUMNS: Final[tuple[str, ...]] = ("source_code", "source_label", "target_code", "target_label")


def clean_text(value: object) -> str:
    """Normalize cells for stable comparisons and CSV output."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def canonical_target_version(value: str) -> str:
    """Return the lowercase target token used in generated filenames."""

    token = clean_text(value).lower().replace("_", "")
    if token not in {"cpa64", "cpa65"}:
        raise ValueError(f"Unsupported target version {value!r}; expected cpa64 or cpa65.")
    return token


def default_old_csv(target_version: str) -> Path:
    """Default old-route A64/compat generated mapping path."""

    target = canonical_target_version(target_version)
    return (
        REPO_ROOT
        / "scripts"
        / "mappings"
        / "generated"
        / "exiobase_concordance_compat"
        / f"exio_to_eurostat_{target}_products_from_exiobase_concordances.csv"
    )


def default_new_csv(target_version: str, anchor_source_mode: str) -> Path:
    """Default new-route official-concordance generated mapping path."""

    target = canonical_target_version(target_version)
    suffix = "" if anchor_source_mode == "direct-only" else f"_{anchor_source_mode.replace('-', '_')}"
    return (
        REPO_ROOT
        / "scripts"
        / "mappings"
        / "generated"
        / "official_concordances"
        / f"exio_to_eurostat_{target}_products_from_official_concordances{suffix}.csv"
    )


def read_pairs(path: Path, *, route_name: str) -> dict[tuple[str, str], dict[str, str]]:
    """Read a canonical mapping CSV into one record per source-target pair."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {route_name} mapping CSV: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = sorted(set(PAIR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if "enabled" in frame.columns:
        enabled = frame["enabled"].map(clean_text).str.casefold()
        frame = frame.loc[~enabled.isin({"0", "false", "no", "off"})].copy()
    records = cast(list[dict[str, Any]], frame.loc[:, list(PAIR_COLUMNS)].to_dict(orient="records"))
    out: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        source_code = clean_text(record.get("source_code", ""))
        target_code = clean_text(record.get("target_code", ""))
        if not source_code or not target_code:
            continue
        pair = (source_code, target_code)
        out.setdefault(
            pair,
            {
                "source_code": source_code,
                "target_code": target_code,
                f"{route_name}_source_label": clean_text(record.get("source_label", "")),
                f"{route_name}_target_label": clean_text(record.get("target_label", "")),
            },
        )
    return out


def pair_rows(
    pairs: set[tuple[str, str]],
    old_pairs: Mapping[tuple[str, str], Mapping[str, str]],
    new_pairs: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, str]]:
    """Render pair records with old/new labels where available."""

    rows: list[dict[str, str]] = []
    for source_code, target_code in sorted(pairs):
        old = old_pairs.get((source_code, target_code), {})
        new = new_pairs.get((source_code, target_code), {})
        rows.append(
            {
                "source_code": source_code,
                "target_code": target_code,
                "old_source_label": clean_text(old.get("old_source_label", "")),
                "old_target_label": clean_text(old.get("old_target_label", "")),
                "new_source_label": clean_text(new.get("new_source_label", "")),
                "new_target_label": clean_text(new.get("new_target_label", "")),
            }
        )
    return rows


def targets_by_source(pairs: set[tuple[str, str]]) -> dict[str, set[str]]:
    """Group target codes by source product."""

    grouped: dict[str, set[str]] = {}
    for source_code, target_code in pairs:
        grouped.setdefault(source_code, set()).add(target_code)
    return grouped


def target_set_difference_rows(old_pair_set: set[tuple[str, str]], new_pair_set: set[tuple[str, str]]) -> list[dict[str, str]]:
    """Return one row per source with a changed old/new target set."""

    old_by_source = targets_by_source(old_pair_set)
    new_by_source = targets_by_source(new_pair_set)
    rows: list[dict[str, str]] = []
    for source_code in sorted(set(old_by_source) | set(new_by_source)):
        old_targets = old_by_source.get(source_code, set())
        new_targets = new_by_source.get(source_code, set())
        if old_targets == new_targets:
            continue
        rows.append(
            {
                "source_code": source_code,
                "old_targets": ";".join(sorted(old_targets)),
                "new_targets": ";".join(sorted(new_targets)),
                "old_target_count": str(len(old_targets)),
                "new_target_count": str(len(new_targets)),
                "only_old_targets": ";".join(sorted(old_targets - new_targets)),
                "only_new_targets": ";".join(sorted(new_targets - old_targets)),
            }
        )
    return rows


def multi_target_rows(pairs: set[tuple[str, str]], *, route_name: str) -> list[dict[str, str]]:
    """Return sources with more than one target in a route."""

    rows: list[dict[str, str]] = []
    for source_code, targets in sorted(targets_by_source(pairs).items()):
        if len(targets) <= 1:
            continue
        rows.append(
            {
                "route": route_name,
                "source_code": source_code,
                "target_count": str(len(targets)),
                "targets": ";".join(sorted(targets)),
            }
        )
    return rows


def write_records(path: Path, rows: list[dict[str, str]], columns: Sequence[str]) -> None:
    """Write records with stable columns, including empty outputs."""

    frame = pd.DataFrame.from_records(rows, columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def compare_routes(
    *,
    old_csv: Path,
    new_csv: Path,
    out_dir: Path,
    target_version: str,
    output_prefix: str | None = None,
) -> dict[str, object]:
    """Compare source-target pairs between old and new generated maps."""

    target = canonical_target_version(target_version)
    prefix = clean_text(output_prefix) or target
    old_pairs = read_pairs(old_csv, route_name="old")
    new_pairs = read_pairs(new_csv, route_name="new")
    old_pair_set = set(old_pairs)
    new_pair_set = set(new_pairs)
    both = old_pair_set & new_pair_set
    only_old = old_pair_set - new_pair_set
    only_new = new_pair_set - old_pair_set
    changed_sources = target_set_difference_rows(old_pair_set, new_pair_set)
    old_multi = multi_target_rows(old_pair_set, route_name="old")
    new_multi = multi_target_rows(new_pair_set, route_name="new")

    summary: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "target_version": target,
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
        "old_rows": len(old_pair_set),
        "new_rows": len(new_pair_set),
        "same_source_target_pairs": len(both),
        "only_old_pairs": len(only_old),
        "only_new_pairs": len(only_new),
        "changed_exio_products": len(changed_sources),
        "old_multi_target_products": len(old_multi),
        "new_multi_target_products": len(new_multi),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_columns = ["source_code", "target_code", "old_source_label", "old_target_label", "new_source_label", "new_target_label"]
    write_records(out_dir / f"{prefix}_source_target_pairs_both.csv", pair_rows(both, old_pairs, new_pairs), pair_columns)
    write_records(out_dir / f"{prefix}_source_target_pairs_only_old.csv", pair_rows(only_old, old_pairs, new_pairs), pair_columns)
    write_records(out_dir / f"{prefix}_source_target_pairs_only_new.csv", pair_rows(only_new, old_pairs, new_pairs), pair_columns)
    write_records(
        out_dir / f"{prefix}_per_source_target_set_differences.csv",
        changed_sources,
        ["source_code", "old_targets", "new_targets", "old_target_count", "new_target_count", "only_old_targets", "only_new_targets"],
    )
    write_records(out_dir / f"{prefix}_multi_target_sources_old.csv", old_multi, ["route", "source_code", "target_count", "targets"])
    write_records(out_dir / f"{prefix}_multi_target_sources_new.csv", new_multi, ["route", "source_code", "target_count", "targets"])
    summary_path = out_dir / f"{prefix}_route_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", choices=["cpa64", "cpa65"], default="cpa65")
    parser.add_argument("--new-anchor-source-mode", choices=list(ANCHOR_SOURCE_MODES), default="direct-only")
    parser.add_argument("--old-csv", type=Path, default=None)
    parser.add_argument("--new-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_COMPARE_OUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run route comparison from the command line."""

    args = parse_args(argv)
    old_csv = args.old_csv if args.old_csv is not None else default_old_csv(args.target_version)
    new_csv = args.new_csv if args.new_csv is not None else default_new_csv(args.target_version, args.new_anchor_source_mode)
    summary = compare_routes(
        old_csv=old_csv,
        new_csv=new_csv,
        out_dir=args.out_dir,
        target_version=args.target_version,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
