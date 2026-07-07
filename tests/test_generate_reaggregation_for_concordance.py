from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "pipelines" / "generate_reaggregation_for_concordance.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("generate_reaggregation_for_concordance", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_aliases_edge_child_from_unsuffixed_base_row(tmp_path: Path) -> None:
    helper = load_helper()
    split = tmp_path / "split.csv"
    base = tmp_path / "base_reagg.csv"
    concordance = tmp_path / "concordance.csv"
    out = tmp_path / "generated_reagg.csv"
    review = tmp_path / "review.csv"
    manifest = tmp_path / "manifest.json"

    write_csv(split, [{"source_code": "CPA_D", "source_label": "Electricity"}])
    write_csv(
        base,
        [
            {
                "source_code": "p40.11.a",
                "source_label": "Electricity by coal",
                "target_code": "CPA_D_Electricity_by_coal",
                "target_label": "CPA_D: Electricity by coal",
                "enabled": "True",
                "notes": "base row",
            }
        ],
    )
    write_csv(
        concordance,
        [
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_D", "target_label": "Electricity", "enabled": "1"},
            {"source_code": "p40.11.a", "source_label": "Electricity by coal", "target_code": "CPA_E37-39", "target_label": "Waste", "enabled": "1"},
        ],
    )

    assert helper.main(
        [
            "--split-mapping-csv", str(split),
            "--base-reagg-csv", str(base),
            "--concordance-csv", str(concordance),
            "--out-csv", str(out),
            "--review-csv", str(review),
            "--manifest-json", str(manifest),
        ]
    ) == 0

    rows = read_csv(out)
    assert {row["source_code"] for row in rows} == {"p40.11.a", "p40.11.a@CPA_D"}
    alias = [row for row in rows if row["source_code"] == "p40.11.a@CPA_D"][0]
    assert alias["target_code"] == "CPA_D_Electricity_by_coal"
    assert "copies base reaggregation from p40.11.a" in alias["notes"]
    assert read_csv(review) == []
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    assert summary["added_alias_rows"] == 1
    assert summary["review_required_rows"] == 0


def test_review_required_rows_are_written_but_not_added(tmp_path: Path) -> None:
    helper = load_helper()
    split = tmp_path / "split.csv"
    base = tmp_path / "base_reagg.csv"
    concordance = tmp_path / "concordance.csv"
    out = tmp_path / "generated_reagg.csv"
    review = tmp_path / "review.csv"
    manifest = tmp_path / "manifest.json"

    write_csv(split, [{"source_code": "CPA_C19", "source_label": "Coke and refined petroleum"}])
    write_csv(
        base,
        [
            {
                "source_code": "p23.20.g",
                "source_label": "Heavy Fuel Oil",
                "target_code": "CPA_C19_OIL",
                "target_label": "CPA_C19: Oil",
                "enabled": "True",
                "notes": "base row",
            }
        ],
    )
    write_csv(
        concordance,
        [
            {"source_code": "p10.f", "source_label": "Lignite/Brown Coal", "target_code": "CPA_C19", "target_label": "Coke and refined petroleum", "enabled": "1"}
        ],
    )

    try:
        helper.main(
            [
                "--split-mapping-csv", str(split),
                "--base-reagg-csv", str(base),
                "--concordance-csv", str(concordance),
                "--out-csv", str(out),
                "--review-csv", str(review),
                "--manifest-json", str(manifest),
            ]
        )
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("Expected review-required rows to exit nonzero by default")

    assert {row["source_code"] for row in read_csv(out)} == {"p23.20.g"}
    review_rows = read_csv(review)
    assert len(review_rows) == 1
    assert review_rows[0]["child_code"] == "p10.f"
    assert "manual economic classification required" in review_rows[0]["reason"]
