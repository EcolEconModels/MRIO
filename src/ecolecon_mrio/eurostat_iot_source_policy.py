from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, cast

import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.iot_validation import (
    AdditivePartition,
    FlowRole,
    InvariantCheck,
    InvariantReport,
)

EvidenceAuthority = Literal["source_code_and_label", "numerically_confirmed_source_identity"]
ResolutionStatus = Literal["resolved_for_profile", "explicitly_unresolved"]
IdentityState = Literal[
    "recognized_role",
    "candidate_identity",
    "numerically_confirmed_source_identity",
    "authorized_source_identity",
    "strict_generated_identity",
]


@dataclass(frozen=True)
class EurostatIOTAccountingPolicy:
    """Dataset-level accounting contract for one Eurostat IOT table type.

    This policy names source codes and equations defined by the published table
    layout.  It deliberately contains no country, year, product count or
    product-order assumption; those facts belong to source evidence records.
    """

    policy_id: str
    dataset: str
    table_type: str
    valuation: str
    atomic_final_demand_columns: tuple[str, ...]
    required_final_demand_columns: tuple[str, ...]
    required_rows_by_flow: Mapping[str, tuple[str, ...]]
    row_roles: Mapping[str, str]
    final_demand_partitions: tuple[AdditivePartition, ...]
    import_partitions: tuple[AdditivePartition, ...]
    production_account_rows: tuple[str, ...]
    vm_total_rows_by_flow: Mapping[str, tuple[str, ...]]
    nonnegative_y_columns_by_flow: Mapping[str, tuple[str, ...]]
    nonnegative_v_rows_by_flow: Mapping[str, tuple[str, ...]]
    total_control_column: str
    identity_states: Mapping[str, IdentityState]


@dataclass(frozen=True)
class EurostatIOTSourceEvidence:
    """Country/year source observations used to review, not define, a policy."""

    evidence_id: str
    dataset: str
    geo: str
    year: int
    unit: str
    product_count: int
    observed_final_demand_columns: tuple[str, ...]
    observed_supplementary_rows: Mapping[str, tuple[str, ...]]
    source_rounding_quantum: float
    numerically_confirmed_identities: tuple[str, ...]


NAIO_10_CP1700_ACCOUNTING_POLICY = EurostatIOTAccountingPolicy(
    policy_id="naio_10_cp1700_product_by_product_v1",
    dataset="naio_10_cp1700",
    table_type="product_by_product_symmetric_iot",
    valuation="published basic-price IOT with row-specific purchaser-price bridges",
    atomic_final_demand_columns=(
        "P3_S13",
        "P3_S14",
        "P3_S15",
        "P51G",
        "P52",
        "P53",
        "P6",
    ),
    required_final_demand_columns=(
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
    required_rows_by_flow={
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
    row_roles={
        "CPA_TOTAL": "published_column_total_control",
        "P2_ADJ": "intermediate_consumption_at_declared_use_valuation",
        "P1": "domestic_output_control",
        "B1G": "gross_value_added_control",
        "D1": "compensation_of_employees",
        "D11": "wages_and_salaries",
        "D21X31": "taxes_less_subsidies_on_products",
        "D29X39": "other_taxes_less_other_subsidies_on_production",
        "P51C": "consumption_of_fixed_capital",
        "B2A3G": "gross_operating_surplus_and_mixed_income",
        "B2A3N": "net_operating_surplus_and_mixed_income",
        "IMP": "imported_use_control",
        "P7": "imports_supply_control",
        "P7_B0": "imports_supply_from_eu",
        "P7_D0": "imports_supply_from_non_eu",
        "P7_U2": "imports_supply_from_euro_area",
        "P7_U3": "imports_supply_from_eu_outside_euro_area",
        "TS_BP": "total_supply_at_basic_prices",
        "P6_D0": "exports_to_non_eu",
        "P6_U2": "exports_to_euro_area",
        "P6_U3": "exports_to_eu_outside_euro_area",
    },
    final_demand_partitions=(
        AdditivePartition("full.y1.p3", "P3", ("P3_S13", "P3_S14", "P3_S15")),
        AdditivePartition("full.y2.p5m", "P5M", ("P52", "P53")),
        AdditivePartition("full.y3.p5_nested", "P5", ("P51G", "P5M")),
        AdditivePartition("full.y4.p5_flat", "P5", ("P51G", "P52", "P53")),
        AdditivePartition("full.y5.p6_world", "P6", ("P6_B0", "P6_D0")),
        AdditivePartition("full.y6.p6_eu", "P6_B0", ("P6_U2", "P6_U3")),
    ),
    import_partitions=(
        AdditivePartition("full.i5.p7_geography", "P7", ("P7_B0", "P7_D0")),
        AdditivePartition("full.i6.p7_eu", "P7_B0", ("P7_U2", "P7_U3")),
    ),
    production_account_rows=(
        "P1",
        "B1G",
        "D1",
        "D11",
        "D21X31",
        "D29X39",
        "P51C",
        "B2A3G",
        "B2A3N",
    ),
    vm_total_rows_by_flow={
        "DOM": (
            "CPA_TOTAL",
            "P1",
            "P2_ADJ",
            "B1G",
            "D1",
            "D11",
            "D21X31",
            "D29X39",
            "P51C",
            "B2A3G",
            "B2A3N",
            "IMP",
        ),
        "IMP": ("CPA_TOTAL",),
        "TOTAL": (
            "CPA_TOTAL",
            "P1",
            "P2_ADJ",
            "B1G",
            "D1",
            "D11",
            "D21X31",
            "D29X39",
            "P51C",
            "B2A3G",
            "B2A3N",
            "P7",
            "P7_B0",
            "P7_D0",
            "P7_U2",
            "P7_U3",
            "TS_BP",
        ),
    },
    # Inventory change (P52), valuables (P53), taxes/subsidies and surplus
    # remain signed. These named transaction/use and supply controls do not.
    nonnegative_y_columns_by_flow={
        flow: ("P3_S13", "P3_S14", "P3_S15", "P51G", "P6", "TU")
        for flow in ("DOM", "IMP", "TOTAL")
    },
    nonnegative_v_rows_by_flow={
        "DOM": ("CPA_TOTAL", "D1", "D11", "P1", "P2_ADJ", "P51C"),
        "IMP": ("CPA_TOTAL",),
        "TOTAL": (
            "CPA_TOTAL", "D1", "D11", "P1", "P2_ADJ", "P51C",
            "P7", "P7_B0", "P7_D0", "P7_U2", "P7_U3", "TS_BP",
        ),
    },
    total_control_column="TOTAL",
    identity_states={
        "roles": "recognized_role",
        "final_demand_partitions": "numerically_confirmed_source_identity",
        "source_controls": "authorized_source_identity",
        "generated_identities": "strict_generated_identity",
        "gdp": "candidate_identity",
    },
)


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


AT_2021_NAIO_10_CP1700_EVIDENCE = EurostatIOTSourceEvidence(
    evidence_id="AT_2021_naio_10_cp1700_MIO_EUR_reviewed_source",
    dataset="naio_10_cp1700",
    geo="AT",
    year=2021,
    unit="MIO_EUR",
    product_count=65,
    observed_final_demand_columns=AT_2021_NAIO_10_CP1700_PROFILE.final_demand_columns,
    observed_supplementary_rows=AT_2021_NAIO_10_CP1700_PROFILE.expected_supplementary_rows,
    source_rounding_quantum=0.01,
    numerically_confirmed_identities=(
        "full.y1.p3",
        "full.y2.p5m",
        "full.y3.p5_nested",
        "full.y4.p5_flat",
        "full.y5.p6_world",
        "full.y6.p6_eu",
        "full.r4.total_equals_dom_plus_imp_z",
        "full.r5.total_equals_dom_plus_imp_atomic_y",
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
