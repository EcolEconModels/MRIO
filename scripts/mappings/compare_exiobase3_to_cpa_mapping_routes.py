#!/usr/bin/env python3
"""Compare old A64/compat and new official-concordance EXIOBASE3->CPA maps.

The old route is:

    NACE Rev.2 -> EXIOBASE2 product -> EXIOBASE3 product -> Eurostat IOT A*64
    -> CPA64/CPA65 compatibility projection

The new route is:

    EXIOBASE2 direct CPA2002 anchors -> CPA2008 -> CPA2.1 -> CPA64/CPA65

This script compares active source-target pairs only.  It does not decide which
route is economically preferable; it writes explicit diffs for audit and pipeline
experiments.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_OLD_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated"
DEFAULT_NEW_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated" / "official_concordances"
DEFAULT_OUT_DIR: Final[Path] = REPO_ROOT / "scripts" / "mappings" / "generated" / "compare_routes" / "exiobase3_to_cpa"
DIRECT_ONLY_ANCHOR_SOURCE_MODE: Final[str] = "direct-only"
DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE: Final[str] = "direct-plus-curated-gaps"
ANCHOR_SOURCE_MODES: Final[set[str]] = {DIRECT_ONLY_ANCHOR_SOURCE_MODE, DIRECT_PLUS_CURATED_GAPS_ANCHOR_SOURCE_MODE}
PAIR_COLUMNS: Final[tuple[str, ...]] = ("source_code", "source_label", "target_code", "target_label")
TRUTHY: Final[set[str]] = {"1", "true", "t", "yes", "y", "on"}


def clean_text(value: object) -> str:
    """Normalize text for stable comparisons and CSV output."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def _enabled(value: object) -> bool:
    return clean_text(value).lower() in TRUTHY


def default_old_csv(target_version: str, old_dir: Path = DEFAULT_OLD_DIR) -> Path:
    """Return the default old A64/compat route output path for a target version."""

    target = target_version.lower()
    return old_dir / f"exio_to_eurostat_{target}_products_from_exiobase_concordances.csv"


def default_new_csv(
    target_version: str,
    *,
    anchor_source_mode: str = DIRECT_ONLY_ANCHOR_SOURCE_MODE,
    new_dir: Path = DEFAULT_NEW_DIR,
) -> Path:
    """Return the default new official-concordance route output path."""

    target = target_version.lower()
    stem = f"exio_to_eurostat_{target}_products_from_official_concordances"
    if anchor_source_mode != DIRECT_ONLY_ANCHOR_SOURCE_MODE:
        stem = f"{stem}_{anchor_source_mode.replace('-', '_')}"
    return new_dir / f"{stem}.csv"


def require_columns(frame: pd.DataFrame, *, path: Path, columns: Sequence[str]) -> None:
    """Require columns in a CSV input."""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def read_active_pairs(path: Path, *, route_name: str, include_disabled: bool = False) -> pd.DataFrame:
    """Read one mapping CSV and return de-duplicated active source-target pairs."""

    if not path.exists():
        raise FileNotFoundError(f"Missing {route_name} mapping CSV: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    require_columns(frame, path=path, columns=["source_code", "target_code"])
    work = frame.copy()
    if not include_disabled and "enabled" in work.columns:
        enabled_mask = cast(pd.Series, work["enabled"]).map(_enabled)
        work = cast(pd.DataFrame, work.loc[enabled_mask].copy())
    for col in ("source_label", "target_label"):
        if col not in work.columns:
            work[col] = ""

    grouped: dict[tuple[str, str], dict[str, set[str]]] = {}
    records = cast(list[dict[str, Any]], work.loc[:, list(PAIR_COLUMNS)].to_dict(orient="records"))
    for record in records:
        source_code = clean_text(record.get("source_code", ""))
        target_code = clean_text(record.get("target_code", ""))
        if not source_code or not target_code:
            continue
        slot = grouped.setdefault((source_code, target_code), {"source_label": set(), "target_label": set()})
        source_label = clean_text(record.get("source_label", ""))
        target_label = clean_text(record.get("target_label", ""))
        if source_label:
            slot["source_label"].add(source_label)
        if target_label:
            slot["target_label"].add(target_label)

    rows = [
        {
            "source_code": source_code,
            "source_label": " | ".join(sorted(labels["source_label"])),
            "target_code": target_code,
            "target_label": " | ".join(sorted(labels["target_label"])),
        }
        for (source_code, target_code), labels in sorted(grouped.items())
    ]
    return pd.DataFrame.from_records(rows, columns=list(PAIR_COLUMNS))


def pair_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return source-target key pairs from a pair frame."""

    records = cast(list[dict[str, Any]], frame.loc[:, ["source_code", "target_code"]].to_dict(orient="records"))
    return {(clean_text(record.get("source_code", "")), clean_text(record.get("target_code", ""))) for record in records}


def frame_for_keys(frame: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    """Return rows whose source-target pair is in keys, sorted for stable output."""

    rows: list[dict[str, str]] = []
    records = cast(list[dict[str, Any]], frame.loc[:, list(PAIR_COLUMNS)].to_dict(orient="records"))
    for record in records:
        key = (clean_text(record.get("source_code", "")), clean_text(record.get("target_code", "")))
        if key in keys:
            rows.append({col: clean_text(record.get(col, "")) for col in PAIR_COLUMNS})
    rows.sort(key=lambda row: (row["source_code"], row["target_code"]))
    return pd.DataFrame.from_records(rows, columns=list(PAIR_COLUMNS))


def targets_by_source(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Return target-code sets by source code."""

    out: dict[str, set[str]] = {}
    records = cast(list[dict[str, Any]], frame.loc[:, ["source_code", "target_code"]].to_dict(orient="records"))
    for record in records:
        source = clean_text(record.get("source_code", ""))
        target = clean_text(record.get("target_code", ""))
        if source and target:
            out.setdefault(source, set()).add(target)
    return out


def labels_by_source(frame: pd.DataFrame) -> dict[str, str]:
    """Return a compact source-label map."""

    labels: dict[str, set[str]] = {}
    records = cast(list[dict[str, Any]], frame.loc[:, ["source_code", "source_label"]].to_dict(orient="records"))
    for record in records:
        source = clean_text(record.get("source_code", ""))
        label = clean_text(record.get("source_label", ""))
        if source and label:
            labels.setdefault(source, set()).add(label)
    return {source: " | ".join(sorted(values)) for source, values in labels.items()}


def per_source_differences(old_pairs: pd.DataFrame, new_pairs: pd.DataFrame) -> pd.DataFrame:
    """Return sources whose old/new target sets differ."""

    old_targets = targets_by_source(old_pairs)
    new_targets = targets_by_source(new_pairs)
    old_labels = labels_by_source(old_pairs)
    new_labels = labels_by_source(new_pairs)
    rows: list[dict[str, str]] = []
    for source in sorted(set(old_targets) | set(new_targets)):
        old_set = old_targets.get(source, set())
        new_set = new_targets.get(source, set())
        if old_set == new_set:
            continue
        rows.append(
            {
                "source_code": source,
                "old_source_label": old_labels.get(source, ""),
                "new_source_label": new_labels.get(source, ""),
                "old_targets": ";".join(sorted(old_set)),
                "new_targets": ";".join(sorted(new_set)),
                "old_target_count": str(len(old_set)),
                "new_target_count": str(len(new_set)),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=["source_code", "old_source_label", "new_source_label", "old_targets", "new_targets", "old_target_count", "new_target_count"],
    )


def multi_target_sources(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sources that map to more than one target."""

    target_sets = targets_by_source(frame)
    labels = labels_by_source(frame)
    rows = [
        {
            "source_code": source,
            "source_label": labels.get(source, ""),
            "target_count": str(len(targets)),
            "targets": ";".join(sorted(targets)),
        }
        for source, targets in sorted(target_sets.items())
        if len(targets) > 1
    ]
    return pd.DataFrame.from_records(rows, columns=["source_code", "source_label", "target_count", "targets"])


def write_comparison_outputs(
    *,
    old_pairs: pd.DataFrame,
    new_pairs: pd.DataFrame,
    old_csv: Path,
    new_csv: Path,
    out_dir: Path,
    target_version: str,
    anchor_source_mode: str,
) -> dict[str, object]:
    """Write route-comparison CSVs and return summary metadata."""

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = target_version.lower()
    old_keys = pair_keys(old_pairs)
    new_keys = pair_keys(new_pairs)
    same_keys = old_keys & new_keys
    only_old_keys = old_keys - new_keys
    only_new_keys = new_keys - old_keys

    only_old = frame_for_keys(old_pairs, only_old_keys)
    only_new = frame_for_keys(new_pairs, only_new_keys)
    both = frame_for_keys(new_pairs, same_keys)
    per_source = per_source_differences(old_pairs, new_pairs)
    old_multi = multi_target_sources(old_pairs)
    new_multi = multi_target_sources(new_pairs)

    outputs = {
        "summary": out_dir / f"{prefix}_route_comparison_summary.json",
        "per_source_differences": out_dir / f"{prefix}_per_source_target_set_differences.csv",
        "only_old_pairs": out_dir / f"{prefix}_source_target_pairs_only_old.csv",
        "only_new_pairs": out_dir / f"{prefix}_source_target_pairs_only_new.csv",
        "both_pairs": out_dir / f"{prefix}_source_target_pairs_both.csv",
        "old_multi_target_sources": out_dir / f"{prefix}_multi_target_sources_old.csv",
        "new_multi_target_sources": out_dir / f"{prefix}_multi_target_sources_new.csv",
    }

    per_source.to_csv(outputs["per_source_differences"], index=False)
    only_old.to_csv(outputs["only_old_pairs"], index=False)
    only_new.to_csv(outputs["only_new_pairs"], index=False)
    both.to_csv(outputs["both_pairs"], index=False)
    old_multi.to_csv(outputs["old_multi_target_sources"], index=False)
    new_multi.to_csv(outputs["new_multi_target_sources"], index=False)

    old_target_sets = targets_by_source(old_pairs)
    new_target_sets = targets_by_source(new_pairs)
    summary: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "target_version": target_version.lower(),
        "new_anchor_source_mode": anchor_source_mode,
        "old_route": "NACE2_EXIOBASE20p_list -> EXIOBASE2/3 bridge -> hard-coded Eurostat IOT A64 -> CPA64/CPA65 compatibility projection",
        "new_route": "EXIOBASE direct CPA2002 anchors -> level-aware CPA2002->CPA2008->CPA2.1 official correspondences -> CPA64/CPA65 division map",
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
        "old_rows": int(len(old_pairs)),
        "new_rows": int(len(new_pairs)),
        "same_source_target_pairs": int(len(same_keys)),
        "only_old_pairs": int(len(only_old_keys)),
        "only_new_pairs": int(len(only_new_keys)),
        "changed_exio_products": int(len(per_source)),
        "old_distinct_sources": int(len(old_target_sets)),
        "new_distinct_sources": int(len(new_target_sets)),
        "old_multi_target_products": int(len(old_multi)),
        "new_multi_target_products": int(len(new_multi)),
        "output_files": {name: str(path) for name, path in outputs.items()},
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", choices=["cpa64", "cpa65"], default="cpa65")
    parser.add_argument(
        "--new-anchor-source-mode",
        choices=sorted(ANCHOR_SOURCE_MODES),
        default=DIRECT_ONLY_ANCHOR_SOURCE_MODE,
        help="Anchor mode used when selecting the default new-route CSV path.",
    )
    parser.add_argument("--old-csv", type=Path, default=None, help="Old A64/compat generated mapping CSV. Defaults to scripts/mappings/generated/...")
    parser.add_argument("--new-csv", type=Path, default=None, help="New official-concordance generated mapping CSV. Defaults to scripts/mappings/generated/official_concordances/...")
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR, help="Directory for default old-route CSV lookup.")
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR, help="Directory for default new-route CSV lookup.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-disabled", action="store_true", help="Include rows whose enabled column is falsey.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the route comparison."""

    args = parse_args(argv)
    target = clean_text(args.target_version).lower()
    mode = clean_text(args.new_anchor_source_mode) or DIRECT_ONLY_ANCHOR_SOURCE_MODE
    old_csv = args.old_csv if args.old_csv is not None else default_old_csv(target, old_dir=args.old_dir)
    new_csv = args.new_csv if args.new_csv is not None else default_new_csv(target, anchor_source_mode=mode, new_dir=args.new_dir)
    old_pairs = read_active_pairs(old_csv, route_name="old A64/compat route", include_disabled=bool(args.include_disabled))
    new_pairs = read_active_pairs(new_csv, route_name="new official-concordance route", include_disabled=bool(args.include_disabled))
    summary = write_comparison_outputs(
        old_pairs=old_pairs,
        new_pairs=new_pairs,
        old_csv=old_csv,
        new_csv=new_csv,
        out_dir=args.out_dir,
        target_version=target,
        anchor_source_mode=mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
