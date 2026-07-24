from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from scripts.audit_eurostat_iot_accounting import audit_variants, write_audit_outputs


def _variant(*, z: float, p3_detail: tuple[float, float, float], p6: float) -> ParsedEurostatIOT:
    index = pd.Index(["a"])
    p3 = sum(p3_detail)
    return ParsedEurostatIOT(
        Z=pd.DataFrame([[z]], index=index, columns=index),
        Y=pd.DataFrame(
            {
                "P3": [p3],
                "P3_S13": [p3_detail[0]],
                "P3_S14": [p3_detail[1]],
                "P3_S15": [p3_detail[2]],
                "P6": [p6],
            },
            index=index,
        ),
        V=pd.DataFrame([[0.0]], index=pd.Index(["P1"]), columns=index),
        M=pd.DataFrame(
            [[0.0] * 5],
            index=pd.Index(["P1"]),
            columns=pd.Index(["P3", "P3_S13", "P3_S14", "P3_S15", "P6"]),
        ),
        meta={"source_rounding_quantum": 0.01},
    )


def _valid_variants() -> dict[str, ParsedEurostatIOT]:
    domestic = _variant(z=3.0, p3_detail=(1.0, 1.0, 1.0), p6=1.0)
    imported = _variant(z=2.0, p3_detail=(0.5, 0.5, 1.0), p6=0.5)
    total = _variant(z=5.0, p3_detail=(1.5, 1.5, 2.0), p6=1.5)
    return {"DOM": domestic, "IMP": imported, "TOTAL": total}


def test_read_only_audit_accepts_balanced_source_triplet() -> None:
    report = audit_variants(_valid_variants(), requested_flows=["DOM", "IMP", "TOTAL"])
    assert report.ok
    assert not [check for check in report.checks if check.status == "fail"]
    assert {"source.cross_flow.Z", "source.cross_flow.Y"}.issubset(
        {check.check_id for check in report.checks}
    )


def test_read_only_audit_reports_source_rounding_failure() -> None:
    variants = _valid_variants()
    variants["TOTAL"].Z.loc["a", "a"] = 5.1
    report = audit_variants(variants, requested_flows=["DOM", "IMP", "TOTAL"])
    failed = {check.check_id: check for check in report.checks if check.status == "fail"}
    assert failed["source.cross_flow.Z"].severity == "warning"
    assert failed["source.cross_flow.Z"].affected_labels == ("a|a",)


def test_audit_writes_required_csv_outputs(tmp_path: Path) -> None:
    report = write_audit_outputs(
        variants=_valid_variants(),
        requested_flows=["DOM", "IMP", "TOTAL"],
        outdir=tmp_path,
        dataset="naio_10_cp1700",
        geo="AT",
        year=2021,
        unit="MIO_EUR",
    )
    assert report.ok
    expected = {
        "iot_invariant_checks.csv",
        "iot_invariant_failures.csv",
        "iot_domain_summary.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    domain = pd.read_csv(tmp_path / "iot_domain_summary.csv")
    assert set(domain["flow_role"]) == {"DOM", "IMP", "TOTAL"}
    assert set(domain["valuation"]) == {"not_resolved_from_current_parser_metadata"}
    assert domain["parser_missingness_contract"].str.contains("filled with zero").all()
