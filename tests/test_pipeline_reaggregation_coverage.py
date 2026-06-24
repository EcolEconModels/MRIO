from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "pipelines" / "check_reaggregation_coverage.py"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_checker(tmp_path: Path, *, include_p10g_reaggregation: bool) -> subprocess.CompletedProcess[str]:
    split_csv = tmp_path / "split.csv"
    concordance_csv = tmp_path / "concordance.csv"
    reagg_csv = tmp_path / "reagg.csv"
    report_csv = tmp_path / "report.csv"

    _write_csv(
        split_csv,
        [
            {"source_code": "CPA_C19", "source_label": "Coke and refined petroleum products"},
        ],
    )
    _write_csv(
        concordance_csv,
        [
            {
                "source_code": "p23.1.a",
                "source_label": "Coke Oven Coke",
                "target_code": "CPA_C19",
                "target_label": "Coke and refined petroleum products",
                "enabled": "True",
            },
            {
                "source_code": "p10.g",
                "source_label": "BKB/Peat Briquettes",
                "target_code": "CPA_C19",
                "target_label": "Coke and refined petroleum products",
                "enabled": "True",
            },
        ],
    )
    reagg_rows = [
        {
            "source_code": "p23.1.a",
            "source_label": "Coke Oven Coke",
            "target_code": "CPA_C19_COAL",
            "target_label": "CPA_C19: Coal",
            "enabled": "True",
        },
    ]
    if include_p10g_reaggregation:
        reagg_rows.append(
            {
                "source_code": "p10.g",
                "source_label": "BKB/Peat Briquettes",
                "target_code": "CPA_C19_COAL",
                "target_label": "CPA_C19: Coal",
                "enabled": "True",
            }
        )
    _write_csv(reagg_csv, reagg_rows)

    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--split-mapping-csv",
            str(split_csv),
            "--reagg-mapping-csv",
            str(reagg_csv),
            "--concordance-csv",
            str(concordance_csv),
            "--report-csv",
            str(report_csv),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_reaggregation_coverage_preflight_fails_for_missing_selected_child(tmp_path: Path) -> None:
    result = _run_checker(tmp_path, include_p10g_reaggregation=False)
    assert result.returncode == 1
    assert "p10.g" in result.stdout
    report = pd.read_csv(tmp_path / "report.csv")
    covered = report["covered_by_reaggregation"].astype(str).str.lower().eq("true")
    missing = report.loc[~covered, "child_code"].tolist()
    assert missing == ["p10.g"]


def test_reaggregation_coverage_preflight_passes_when_selected_child_is_covered(tmp_path: Path) -> None:
    result = _run_checker(tmp_path, include_p10g_reaggregation=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing=0" in result.stdout
    report = pd.read_csv(tmp_path / "report.csv")
    assert set(report["child_code"]) == {"p23.1.a", "p10.g"}
    assert report["covered_by_reaggregation"].astype(str).str.lower().eq("true").all()
