from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, cast

import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.iot_validation import FlowRole, InvariantCheck, InvariantReport

EvidenceAuthority = Literal["source_code_and_label", "numerically_confirmed_source_identity"]
ResolutionStatus = Literal["resolved_for_profile", "explicitly_unresolved"]


@dataclass(frozen=True)
class SourceRoleEvidence:
    code: str
    role: str
    blocks: tuple[str, ...]
    flows: tuple[str, ...]
    authority: EvidenceAuthority
    equation_authorized: bool


@dataclass(frozen=True)
class UnresolvedSourceQuestion:
    question_id: str
    excluded_codes: tuple[str, ...]
    reason: str
    status: ResolutionStatus = "explicitly_unresolved"


@dataclass(frozen=True)
class EurostatIOTSourceProfile:
    profile_id: str
    dataset: str
    geo: str
    year: int
    unit: str
    row_dim: str
    col_dim: str
    flow_roles: tuple[str, ...]
    product_count: int
    final_demand_columns: tuple[str, ...]
    atomic_final_demand_columns: tuple[str, ...]
    aggregate_partitions: Mapping[str, tuple[str, ...]]
    aggregate_authoritative_columns: tuple[str, ...]
    expected_supplementary_rows: Mapping[str, tuple[str, ...]]
    role_evidence: tuple[SourceRoleEvidence, ...]
    unresolved_questions: tuple[UnresolvedSourceQuestion, ...]


AT_2021_NAIO_10_CP1700_PROFILE = EurostatIOTSourceProfile(
    profile_id="AT_2021_naio_10_cp1700_MIO_EUR",
    dataset="naio_10_cp1700",
    geo="AT",
    year=2021,
    unit="MIO_EUR",
    row_dim="prd_ava",
    col_dim="prd_use",
    flow_roles=("DOM", "IMP", "TOTAL"),
    product_count=65,
    final_demand_columns=(
        "P3",
        "P3_S13",
        "P3_S14",
        "P3_S15",
        "P5",
        "P51G",
        "P52",
        "P53",
        "P5M",
        "P6",
        "P6_B0",
        "P6_D0",
        "P6_U2",
        "P6_U3",
        "TFU",
        "TOTAL",
        "TU",
    ),
    atomic_final_demand_columns=(
        "P3_S13",
        "P3_S14",
        "P3_S15",
        "P51G",
        "P52",
        "P53",
        "P6",
    ),
    aggregate_partitions={
        "P3": ("P3_S13", "P3_S14", "P3_S15"),
        "P5": ("P51G", "P52", "P53"),
    },
    aggregate_authoritative_columns=("P6",),
    expected_supplementary_rows={
        "DOM": (
            "B1G",
            "B2A3G",
            "B2A3N",
            "CPA_TOTAL",
            "D1",
            "D11",
            "D21X31",
            "D29X39",
            "IMP",
            "P1",
            "P2_ADJ",
            "P51C",
        ),
        "IMP": ("CPA_TOTAL",),
        "TOTAL": (
            "B1G",
            "B2A3G",
            "B2A3N",
            "CPA_TOTAL",
            "D1",
            "D11",
            "D21X31",
            "D29X39",
            "P1",
            "P2_ADJ",
            "P51C",
            "P7",
            "P7_B0",
            "P7_D0",
            "P7_U2",
            "P7_U3",
            "TS_BP",
        ),
    },
    role_evidence=(
        SourceRoleEvidence(
            code="P1",
            role="output_control",
            blocks=("V", "M"),
            flows=("DOM", "TOTAL"),
            authority="source_code_and_label",
            equation_authorized=True,
        ),
        SourceRoleEvidence(
            code="B1G",
            role="gross_value_added_control",
            blocks=("V", "M"),
            flows=("DOM", "TOTAL"),
            authority="source_code_and_label",
            equation_authorized=False,
        ),
        SourceRoleEvidence(
            code="P7",
            role="imports_control",
            blocks=("V", "M"),
            flows=("TOTAL",),
            authority="source_code_and_label",
            equation_authorized=True,
        ),
        SourceRoleEvidence(
            code="TS_BP",
            role="total_supply_at_basic_prices_control",
            blocks=("V", "M"),
            flows=("TOTAL",),
            authority="source_code_and_label",
            equation_authorized=True,
        ),
        SourceRoleEvidence(
            code="P3",
            role="rounded_aggregate_control",
            blocks=("Y",),
            flows=("DOM", "IMP", "TOTAL"),
            authority="numerically_confirmed_source_identity",
            equation_authorized=True,
        ),
        SourceRoleEvidence(
            code="P5",
            role="rounded_aggregate_control",
            blocks=("Y",),
            flows=("DOM", "IMP", "TOTAL"),
            authority="numerically_confirmed_source_identity",
            equation_authorized=True,
        ),
    ),
    unresolved_questions=(
        UnresolvedSourceQuestion(
            question_id="p2_adj_valuation",
            excluded_codes=("P2_ADJ",),
            reason="Source evidence does not establish its valuation basis or adjustment content.",
        ),
        UnresolvedSourceQuestion(
            question_id="supply_use_valuation",
            excluded_codes=("TS_BP", "TU", "D21X31"),
            reason="A common valuation and required bridge components are not established.",
        ),
        UnresolvedSourceQuestion(
            question_id="p6_regional_exhaustiveness",
            excluded_codes=("P6_B0", "P6_D0", "P6_U2", "P6_U3"),
            reason="Regional export columns overlap and exhaustiveness is not established.",
        ),
        UnresolvedSourceQuestion(
            question_id="value_added_decomposition",
            excluded_codes=("B2A3G", "B2A3N", "D1", "D11", "D29X39", "P51C"),
            reason="The observed rows are not established as one exhaustive non-overlapping B1G partition.",
        ),
        UnresolvedSourceQuestion(
            question_id="tax_margin_bridge",
            excluded_codes=("D21X31", "D29X39"),
            reason="Tax/subsidy rows are observed but no complete purchaser/basic-price bridge is established.",
        ),
        UnresolvedSourceQuestion(
            question_id="imported_use_bridge",
            excluded_codes=("IMP", "CPA_TOTAL"),
            reason="The relationship between DOM supplementary IMP controls and the IMP-flow table is not yet validated cellwise.",
        ),
        UnresolvedSourceQuestion(
            question_id="source_missingness",
            excluded_codes=(),
            reason="The parser fills sparse SDMX observations with zero, preventing applicability inference.",
        ),
    ),
)


def _check(
    *,
    check_id: str,
    flow_role: str,
    block: str,
    ok: bool,
    affected_labels: Sequence[str] = (),
    explanation: str,
) -> InvariantCheck:
    return InvariantCheck(
        check_id=check_id,
        category="source_profile",
        status="pass" if ok else "fail",
        severity="error",
        flow_role=cast(FlowRole, flow_role),
        block=block,
        max_abs_gap=None,
        tolerance=None,
        affected_labels=tuple(map(str, affected_labels))[:50],
        explanation=explanation,
    )


def validate_source_profile_schema(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    profile: EurostatIOTSourceProfile = AT_2021_NAIO_10_CP1700_PROFILE,
) -> InvariantReport:
    """Validate only the reviewed source schema, not unresolved economic equations."""
    checks: list[InvariantCheck] = []
    reference_products: pd.Index | None = None
    for flow in profile.flow_roles:
        parsed = variants.get(flow)
        if parsed is None:
            checks.append(
                _check(
                    check_id=f"profile.flow.{flow.lower()}",
                    flow_role=flow,
                    block="Z|Y|V|M",
                    ok=False,
                    explanation=f"Reviewed profile requires flow {flow!r}.",
                )
            )
            continue

        products = parsed.Z.index.astype(str)
        product_ok = (
            len(products) == profile.product_count
            and products.equals(parsed.Z.columns.astype(str))
            and products.equals(parsed.Y.index.astype(str))
        )
        if reference_products is None:
            reference_products = products
        else:
            product_ok = product_ok and products.equals(reference_products)
        checks.append(
            _check(
                check_id=f"profile.products.{flow.lower()}",
                flow_role=flow,
                block="Z|Y",
                ok=product_ok,
                explanation="The reviewed Austria profile requires one identical ordered 65-product domain.",
            )
        )

        actual_y = tuple(parsed.Y.columns.astype(str))
        checks.append(
            _check(
                check_id=f"profile.final_demand.{flow.lower()}",
                flow_role=flow,
                block="Y",
                ok=actual_y == profile.final_demand_columns,
                affected_labels=(actual_y if actual_y != profile.final_demand_columns else ()),
                explanation="The reviewed Austria profile requires the observed ordered 17-column final-demand domain.",
            )
        )

        expected_rows = profile.expected_supplementary_rows[flow]
        actual_v = () if parsed.V is None else tuple(parsed.V.index.astype(str))
        actual_m = () if parsed.M is None else tuple(parsed.M.index.astype(str))
        checks.append(
            _check(
                check_id=f"profile.supplementary_v.{flow.lower()}",
                flow_role=flow,
                block="V",
                ok=actual_v == expected_rows,
                affected_labels=(actual_v if actual_v != expected_rows else ()),
                explanation="The reviewed profile preserves the exact positional V row domain observed for Austria 2021.",
            )
        )
        checks.append(
            _check(
                check_id=f"profile.supplementary_m.{flow.lower()}",
                flow_role=flow,
                block="M",
                ok=actual_m == expected_rows,
                affected_labels=(actual_m if actual_m != expected_rows else ()),
                explanation="The reviewed profile preserves the exact positional M row domain observed for Austria 2021.",
            )
        )
    return InvariantReport(checks=tuple(checks))
