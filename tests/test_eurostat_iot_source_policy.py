from __future__ import annotations

import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_iot_source_policy import (
    AT_2021_NAIO_10_CP1700_PROFILE,
    EurostatIOTSourceProfile,
    validate_source_profile_schema,
)


def _profile() -> EurostatIOTSourceProfile:
    base = AT_2021_NAIO_10_CP1700_PROFILE
    return EurostatIOTSourceProfile(
        profile_id="toy",
        dataset=base.dataset,
        geo=base.geo,
        year=base.year,
        unit=base.unit,
        row_dim=base.row_dim,
        col_dim=base.col_dim,
        flow_roles=("DOM", "IMP", "TOTAL"),
        product_count=1,
        final_demand_columns=("P3", "P6"),
        atomic_final_demand_columns=("P3", "P6"),
        aggregate_partitions={},
        aggregate_authoritative_columns=("P3", "P6"),
        expected_supplementary_rows={
            "DOM": ("P1",),
            "IMP": ("CPA_TOTAL",),
            "TOTAL": ("P1", "P7", "TS_BP"),
        },
        role_evidence=base.role_evidence,
        unresolved_questions=base.unresolved_questions,
    )


def _variant(rows: tuple[str, ...]) -> ParsedEurostatIOT:
    products = pd.Index(["a"])
    final = pd.Index(["P3", "P6"])
    return ParsedEurostatIOT(
        Z=pd.DataFrame([[1.0]], index=products, columns=products),
        Y=pd.DataFrame([[2.0, 3.0]], index=products, columns=final),
        V=pd.DataFrame([[0.0] for _ in rows], index=pd.Index(rows), columns=products),
        M=pd.DataFrame([[0.0, 0.0] for _ in rows], index=pd.Index(rows), columns=final),
        meta={},
    )


def _variants() -> dict[str, ParsedEurostatIOT]:
    return {
        "DOM": _variant(("P1",)),
        "IMP": _variant(("CPA_TOTAL",)),
        "TOTAL": _variant(("P1", "P7", "TS_BP")),
    }


def test_reviewed_profile_records_only_authorized_equations() -> None:
    profile = AT_2021_NAIO_10_CP1700_PROFILE
    authorized = {item.code for item in profile.role_evidence if item.equation_authorized}
    assert authorized == {"P1", "P3", "P5", "P7", "TS_BP"}
    excluded = {
        code
        for question in profile.unresolved_questions
        for code in question.excluded_codes
    }
    assert {"P2_ADJ", "D21X31", "D29X39", "IMP"}.issubset(excluded)


def test_profile_schema_accepts_exact_reviewed_domains() -> None:
    report = validate_source_profile_schema(_variants(), profile=_profile())
    assert report.ok
    assert len(report.checks) == 12


def test_profile_schema_rejects_supplementary_row_drift() -> None:
    variants = _variants()
    variants["TOTAL"] = _variant(("P1", "TS_BP", "P7"))
    report = validate_source_profile_schema(variants, profile=_profile())
    failed = {check.check_id for check in report.checks if check.status == "fail"}
    assert failed == {"profile.supplementary_v.total", "profile.supplementary_m.total"}


def test_profile_schema_rejects_final_demand_drift() -> None:
    variants = _variants()
    total = variants["TOTAL"]
    variants["TOTAL"] = ParsedEurostatIOT(
        Z=total.Z,
        Y=total.Y.loc[:, ["P6", "P3"]],
        V=total.V,
        M=total.M.loc[:, ["P6", "P3"]] if total.M is not None else None,
        meta=total.meta,
    )
    report = validate_source_profile_schema(variants, profile=_profile())
    failed = {check.check_id for check in report.checks if check.status == "fail"}
    assert "profile.final_demand.total" in failed
