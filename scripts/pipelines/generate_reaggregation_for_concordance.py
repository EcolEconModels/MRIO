#!/usr/bin/env python3
"""Generate edge-consistent fuels reaggregation mappings for a concordance route.

The split pipeline resolves EXIO->Eurostat many-to-many weights upstream and emits
materialized edge children such as ``p36@CPA_C20`` when a raw EXIO product maps to
several CPA targets.  This helper therefore does not write weights.  It writes the
structural child->output-bucket mapping used by the later many-to-one aggregation
step, preserving the selected edge target unless an explicit fuel-expanded bucket
rule says otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REAGG_COLUMNS: tuple[str, ...] = (
    "source_code",
    "source_label",
    "target_code",
    "target_label",
    "enabled",
    "notes",
)


def clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_bool(value: object) -> bool:
    text = clean(value).casefold()
    if text == "":
        return True
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def active_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "enabled" not in frame.columns:
        return frame.copy()
    mask = frame["enabled"].map(parse_bool)
    return frame.loc[mask].copy()


def row_dict_by_source(path: Path) -> dict[str, dict[str, str]]:
    frame = active_rows(read_csv_frame(path))
    missing = [column for column in ("source_code", "target_code") if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        source = clean(row.get("source_code", ""))
        target = clean(row.get("target_code", ""))
        if not source or not target:
            continue
        if source in out and clean(out[source].get("target_code", "")) != target:
            raise ValueError(f"{path} maps source_code {source!r} to multiple targets.")
        out[source] = {column: clean(row.get(column, "")) for column in REAGG_COLUMNS}
        out[source]["enabled"] = "True"
    return out


def source_targets_from_concordance(concordance_csv: Path) -> tuple[dict[str, set[str]], dict[tuple[str, str], dict[str, str]]]:
    frame = active_rows(read_csv_frame(concordance_csv))
    required = {"source_code", "target_code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{concordance_csv} missing required columns: {missing}")
    targets_by_source: dict[str, set[str]] = {}
    edge_info: dict[tuple[str, str], dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        source = clean(row.get("source_code", ""))
        target = clean(row.get("target_code", ""))
        if not source or not target:
            continue
        targets_by_source.setdefault(source, set()).add(target)
        edge_info.setdefault(
            (source, target),
            {
                "source_code": source,
                "source_label": clean(row.get("source_label", "")),
                "target_code": target,
                "target_label": clean(row.get("target_label", "")) or target,
            },
        )
    return targets_by_source, edge_info


def split_parent_codes(split_mapping_csv: Path) -> list[str]:
    frame = active_rows(read_csv_frame(split_mapping_csv))
    if "target_code" in frame.columns:
        values = frame["target_code"].astype(str).tolist()
    elif "source_code" in frame.columns:
        values = frame["source_code"].astype(str).tolist()
    else:
        raise ValueError(f"{split_mapping_csv} must contain source_code or target_code")
    return list(dict.fromkeys(clean(value) for value in values if clean(value)))


def selected_children(
    *,
    split_mapping_csv: Path,
    concordance_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    parents = split_parent_codes(split_mapping_csv)
    targets_by_source, edge_info = source_targets_from_concordance(concordance_csv)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent in parents:
        for (source, target), info in sorted(edge_info.items()):
            if target != parent:
                continue
            source_n_targets = len(targets_by_source.get(source, set()))
            child_code = f"{source}@{target}" if source_n_targets > 1 else source
            if child_code in seen:
                continue
            seen.add(child_code)
            rows.append(
                {
                    "child_code": child_code,
                    "source_code": source,
                    "source_label": info.get("source_label", ""),
                    "edge_target_code": target,
                    "edge_target_label": info.get("target_label", "") or target,
                    "source_n_targets": source_n_targets,
                    "is_edge_child": "@" in child_code,
                }
            )
    return rows, targets_by_source


def target_compatible(route_target: str, selected_target: str) -> bool:
    route = clean(route_target)
    selected = clean(selected_target)
    if not route or not selected:
        return False
    return route == selected or route.startswith(f"{selected}_")


def render_row(
    *,
    child: dict[str, Any],
    target_code: str,
    target_label: str,
    rule: str,
    note: str,
) -> dict[str, str]:
    child_code = clean(child["child_code"])
    source_label = clean(child.get("source_label", ""))
    label = source_label if child_code == clean(child.get("source_code", "")) else f"{source_label} allocated through {child.get('edge_target_code', '')}".strip()
    return {
        "source_code": child_code,
        "source_label": label,
        "target_code": clean(target_code),
        "target_label": clean(target_label) or clean(target_code),
        "enabled": "True",
        "notes": f"closure_rule={rule}. {clean(note)}".strip(),
    }


def route_child(
    child: dict[str, Any],
    *,
    base_by_source: dict[str, dict[str, str]],
    rules_by_source: dict[str, dict[str, str]],
    targets_by_source: dict[str, set[str]],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    child_code = clean(child["child_code"])
    source = clean(child["source_code"])
    edge_target = clean(child["edge_target_code"])
    edge_target_label = clean(child.get("edge_target_label", "")) or edge_target
    edge_key = f"{source}@{edge_target}" if source and edge_target else ""

    for key, rule_name in ((child_code, "explicit_route_rule"), (edge_key, "explicit_edge_route_rule"), (source, "explicit_source_route_rule")):
        if key and key in rules_by_source:
            rule = rules_by_source[key]
            return render_row(
                child=child,
                target_code=rule["target_code"],
                target_label=rule.get("target_label", "") or rule["target_code"],
                rule=rule_name,
                note=rule.get("notes", ""),
            ), None

    if child_code in base_by_source:
        base = base_by_source[child_code]
        return render_row(
            child=child,
            target_code=base["target_code"],
            target_label=base.get("target_label", "") or base["target_code"],
            rule="existing_exact_reaggregation_row",
            note=base.get("notes", ""),
        ), None

    if bool(child.get("is_edge_child")):
        base = base_by_source.get(source)
        if base is not None and target_compatible(base.get("target_code", ""), edge_target):
            return render_row(
                child=child,
                target_code=base["target_code"],
                target_label=base.get("target_label", "") or base["target_code"],
                rule="existing_unsuffixed_reaggregation_row_same_edge_target",
                note=base.get("notes", ""),
            ), None
        return render_row(
            child=child,
            target_code=edge_target,
            target_label=edge_target_label,
            rule="edge_target_residual_bucket",
            note="Materialized source-target edge child closes to its selected concordance target; weights were already resolved upstream by the split pipeline.",
        ), None

    base = base_by_source.get(source)
    if base is not None:
        return render_row(
            child=child,
            target_code=base["target_code"],
            target_label=base.get("target_label", "") or base["target_code"],
            rule="existing_unsuffixed_reaggregation_row_raw_child",
            note=base.get("notes", ""),
        ), None

    targets = sorted(targets_by_source.get(source, set()))
    if len(targets) == 1:
        return render_row(
            child=child,
            target_code=targets[0],
            target_label=edge_target_label if targets[0] == edge_target else targets[0],
            rule="raw_single_target_concordance_home",
            note="Raw child has exactly one active EXIO-to-CPA target in the concordance.",
        ), None

    return None, {
        "source_code": child_code,
        "source_label": clean(child.get("source_label", "")),
        "edge_target_code": edge_target,
        "edge_target_label": edge_target_label,
        "candidate_targets": " | ".join(targets),
        "reason": "raw selected child has multiple active concordance targets and was not materialized as source@target",
    }


def build_closed_reaggregation(
    *,
    split_mapping_csv: Path,
    base_reagg_csv: Path,
    concordance_csv: Path,
    rule_csvs: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    base_by_source = row_dict_by_source(base_reagg_csv)
    rules_by_source: dict[str, dict[str, str]] = {}
    for rule_csv in rule_csvs:
        if not rule_csv.exists():
            raise FileNotFoundError(rule_csv)
        rules_by_source.update(row_dict_by_source(rule_csv))
    children, targets_by_source = selected_children(split_mapping_csv=split_mapping_csv, concordance_csv=concordance_csv)
    out_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    rule_counts: Counter[str] = Counter()
    for child in children:
        row, review = route_child(
            child,
            base_by_source=base_by_source,
            rules_by_source=rules_by_source,
            targets_by_source=targets_by_source,
        )
        if row is not None:
            rule = row["notes"].split(".", 1)[0].replace("closure_rule=", "")
            rule_counts[rule] += 1
            out_rows.append(row)
        elif review is not None:
            rule_counts["review_required"] += 1
            review_rows.append(review)
    out_rows = sorted(out_rows, key=lambda row: row["source_code"])
    review_rows = sorted(review_rows, key=lambda row: row["source_code"])
    manifest: dict[str, Any] = {
        "created_by_script": "scripts/pipelines/generate_reaggregation_for_concordance.py",
        "created_utc": datetime.now(UTC).isoformat(),
        "split_mapping_csv": str(split_mapping_csv),
        "base_reagg_csv": str(base_reagg_csv),
        "concordance_csv": str(concordance_csv),
        "rule_csvs": [str(path) for path in rule_csvs],
        "selected_children": len(children),
        "base_rows": len(base_by_source),
        "rule_rows": len(rules_by_source),
        "output_rows": len(out_rows),
        "review_required_rows": len(review_rows),
        "closure_rule_counts": dict(sorted(rule_counts.items())),
    }
    return out_rows, review_rows, manifest


def write_rows(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-mapping-csv", type=Path, required=True)
    parser.add_argument("--base-reagg-csv", type=Path, required=True)
    parser.add_argument("--concordance-csv", type=Path, required=True)
    parser.add_argument("--rule-csv", dest="rule_csvs", type=Path, action="append", default=[])
    parser.add_argument("--override-csv", dest="rule_csvs", type=Path, action="append")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--allow-review-required", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rule_csvs = [path for path in (args.rule_csvs or []) if path is not None]
    out_rows, review_rows, manifest = build_closed_reaggregation(
        split_mapping_csv=args.split_mapping_csv,
        base_reagg_csv=args.base_reagg_csv,
        concordance_csv=args.concordance_csv,
        rule_csvs=rule_csvs,
    )
    manifest.update(
        {
            "out_csv": str(args.out_csv),
            "review_csv": str(args.review_csv),
            "manifest_json": str(args.manifest_json),
        }
    )
    write_rows(args.out_csv, out_rows, REAGG_COLUMNS)
    write_rows(args.review_csv, review_rows, ("source_code", "source_label", "edge_target_code", "edge_target_label", "candidate_targets", "reason"))
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if review_rows and not args.allow_review_required:
        raise SystemExit(f"{len(review_rows)} selected split child row(s) still require manual closure. See {args.review_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
