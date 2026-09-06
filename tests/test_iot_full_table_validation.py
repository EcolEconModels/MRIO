from __future__ import annotations

import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_iot_source_policy import NAIO_10_CP1700_ACCOUNTING_POLICY
from ecolecon_mrio.iot_full_table_validation import validate_eurostat_iot_full_table
from ecolecon_mrio.iot_validation import InvariantCheck, InvariantReport


_Y_COLUMNS = (
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
)


def _y(*, z: float, scale: float, total_control: float) -> pd.DataFrame:
    values = {
        "P3_S13": 0.5 * scale,
        "P3_S14": 1.0 * scale,
        "P3_S15": 1.5 * scale,
        "P51G": 2.0 * scale,
        "P52": 0.5 * scale,
        "P53": 0.5 * scale,
        "P6": 4.0 * scale,
        "P6_B0": 3.0 * scale,
        "P6_D0": 1.0 * scale,
        "P6_U2": 2.0 * scale,
        "P6_U3": 1.0 * scale,
    }
    values["P3"] = values["P3_S13"] + values["P3_S14"] + values["P3_S15"]
    values["P5M"] = values["P52"] + values["P53"]
    values["P5"] = values["P51G"] + values["P5M"]
    values["TFU"] = sum(
        values[column]
        for column in NAIO_10_CP1700_ACCOUNTING_POLICY.atomic_final_demand_columns
    )
    values["TOTAL"] = total_control
    values["TU"] = z + values["TFU"]
    return pd.DataFrame({column: [values[column]] for column in _Y_COLUMNS}, index=pd.Index(["a"]))


def _supplementary(
    *,
    flow: str,
    y: pd.DataFrame,
    z: float,
    p1: float,
    p7: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = NAIO_10_CP1700_ACCOUNTING_POLICY.required_rows_by_flow[flow]
    rows: dict[str, float] = {"CPA_TOTAL": z}
    if flow != "IMP":
        rows.update(
            {
                "P2_ADJ": 15.0,
                "P1": p1,
                "B1G": 5.0,
                "D1": 1.0,
                "D11": 1.0,
                "D21X31": 1.0,
                "D29X39": 1.0,
                "P51C": 1.0,
                "B2A3G": 3.0,
                "B2A3N": 2.0,
            }
        )
    if flow == "DOM":
        rows["IMP"] = 4.0
    if flow == "TOTAL":
        assert p7 is not None
        rows.update(
            {
                "P7": p7,
                "P7_B0": 7.0,
                "P7_D0": 4.0,
                "P7_U2": 4.0,
                "P7_U3": 3.0,
                "TS_BP": p1 + p7,
            }
        )
    # Keep the rich fixture independent of the generic dataset-level mandatory
    # row set so optional-detail identities remain exercised after policy generalization.
    present_rows = list(required)
    for row in rows:
        if row not in present_rows:
            present_rows.append(row)
    V = pd.DataFrame(
        [[rows[row]] for row in present_rows],
        index=pd.Index(present_rows),
        columns=pd.Index(["a"]),
    )
    m_rows: dict[str, list[float]] = {}
    for row in present_rows:
        values = [0.0] * len(_Y_COLUMNS)
        if row == "CPA_TOTAL":
            values = [float(y.loc["a", column]) for column in _Y_COLUMNS]
        elif row == "P2_ADJ":
            values = [float(y.loc["a", column]) for column in _Y_COLUMNS]
            values[_Y_COLUMNS.index("TOTAL")] += 1.0
        elif row == "IMP" and flow == "DOM":
            values = [float(y.loc["a", column]) for column in _Y_COLUMNS]
            # The DOM lower row records the complete IMP displayed-use controls.
            values = [value * (7.0 / 10.0) for value in values]
        if row in rows:
            values[_Y_COLUMNS.index("TOTAL")] = rows[row]
        m_rows[row] = values
    M = pd.DataFrame.from_dict(m_rows, orient="index", columns=pd.Index(_Y_COLUMNS))
    return V, M


def _triplet() -> dict[str, ParsedEurostatIOT]:
    dom_y = _y(z=10.0, scale=1.0, total_control=10.0)
    imp_y = _y(z=4.0, scale=0.7, total_control=4.0)
    # The final-use display TOTAL is deliberately not used in any additive identity.
    total_y = dom_y + imp_y
    total_y.loc["a", "TOTAL"] = 14.0
    total_y.loc["a", "TU"] = 31.0
    dom_v, dom_m = _supplementary(flow="DOM", y=dom_y, z=10.0, p1=20.0)
    imp_v, imp_m = _supplementary(flow="IMP", y=imp_y, z=4.0, p1=0.0)
    total_v, total_m = _supplementary(flow="TOTAL", y=total_y, z=14.0, p1=20.0, p7=11.0)
    # DOM M IMP must be the imported displayed-use column sums, not a DOM share.
    dom_m.loc["IMP", :] = imp_y.loc["a", :]
    meta = {"source_rounding_quantum": 0.1}
    return {
        "DOM": ParsedEurostatIOT(pd.DataFrame([[10.0]], index=["a"], columns=["a"]), dom_y, meta, dom_v, dom_m),
        "IMP": ParsedEurostatIOT(pd.DataFrame([[4.0]], index=["a"], columns=["a"]), imp_y, meta, imp_v, imp_m),
        "TOTAL": ParsedEurostatIOT(pd.DataFrame([[14.0]], index=["a"], columns=["a"]), total_y, meta, total_v, total_m),
    }


def _checks(report: InvariantReport) -> dict[str, InvariantCheck]:
    return {check.check_id: check for check in report.checks}


def test_complete_generic_triplet_passes_without_austria_shape() -> None:
    variants = _triplet()
    for flow in ("DOM", "TOTAL"):
        parsed = variants[flow]
        optional = ["D11", "P51C", "B2A3N"]
        if flow == "TOTAL":
            optional += ["P7_B0", "P7_U2", "P7_U3"]
        variants[flow] = ParsedEurostatIOT(
            parsed.Z, parsed.Y, parsed.meta,
            parsed.V.drop(index=optional, errors="ignore") if parsed.V is not None else None,
            parsed.M.drop(index=optional, errors="ignore") if parsed.M is not None else None,
        )
    assert variants["TOTAL"].M is not None and variants["DOM"].M is not None
    for column in ("P6_B0", "P6_U2", "P6_U3"):
        variants["TOTAL"].M.loc[["P2_ADJ", "D21X31"], column] = float("nan")
        variants["DOM"].M.loc["IMP", column] = float("nan")
    variants["TOTAL"].M.loc["B1G", "TOTAL"] += 50.0

    report = validate_eurostat_iot_full_table(
        variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="source"
    )
    checks = _checks(report)
    assert report.ok
    assert checks["full.s7.total"].status == "pass"
    assert checks["full.schema.rows.total"].status == "pass"
    assert checks["full.p2_2.total_m"].status == "pass"
    assert checks["full.i3.dom_m_imp"].status == "pass"
    assert checks["full.v2.b1g.total"].status == "pass"
    assert checks["full.vm.total.b1g"].status == "not_tested"


def test_missing_atomic_column_is_structural_diagnostic_not_key_error() -> None:
    variants = _triplet()
    total = variants["TOTAL"]
    variants["TOTAL"] = ParsedEurostatIOT(
        total.Z,
        total.Y.drop(columns=["P53"]),
        total.meta,
        total.V,
        total.M.drop(columns=["P53"]) if total.M is not None else None,
    )
    checks = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY
        )
    )
    assert checks["full.atomic.total"].status == "fail"
    assert "P53" in checks["full.atomic.total"].affected_labels


def test_complete_nested_p5_triplet_uses_p51g_plus_p5m_without_flat_controls() -> None:
    """A valid nested investment family is not rejected as missing P52/P53."""
    variants = _triplet()
    for flow, parsed in tuple(variants.items()):
        variants[flow] = ParsedEurostatIOT(
            parsed.Z,
            parsed.Y.drop(columns=["P52", "P53"]),
            parsed.meta,
            parsed.V,
            parsed.M.drop(columns=["P52", "P53"]) if parsed.M is not None else None,
        )
    report = validate_eurostat_iot_full_table(
        variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY
    )
    checks = _checks(report)
    assert report.ok
    assert checks["full.y3.p5_nested.total"].status == "pass"
    assert checks["full.y2.p5m.total"].status == "not_tested"
    assert checks["full.y4.p5_flat.total"].status == "not_tested"


def test_source_rounding_gap_passes_but_generated_gap_fails() -> None:
    variants = _triplet()
    variants["TOTAL"].Y.loc["a", "P3"] += 0.1
    source = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="source"
        )
    )
    generated = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )
    assert source["full.y1.p3.total"].status == "pass"
    assert generated["full.y1.p3.total"].status == "fail"
    assert generated["full.y1.p3.total"].affected_labels == ("a",)


def test_dom_ts_bp_uses_domestic_output_not_import_supply() -> None:
    variants = _triplet()
    dom = variants["DOM"]
    assert dom.V is not None
    dom.V.loc["P7"] = 11.0
    dom.V.loc["TS_BP"] = dom.V.loc["P1"]

    source = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="source"
        )
    )
    generated = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )
    assert source["full.sp0.dom_ts_output"].status == "pass"
    assert generated["full.sp0.dom_ts_output"].status == "pass"

    dom.V.loc["TS_BP"] = dom.V.loc["P1"] + dom.V.loc["P7"]
    failed = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )
    assert failed["full.sp0.dom_ts_output"].status == "fail"
    assert failed["full.sp0.dom_ts_output"].affected_labels == ("a",)


def test_generated_p2_tfu_uses_canonical_y_and_d21_controls() -> None:
    variants = _triplet()
    total = variants["TOTAL"]
    assert total.M is not None
    total.M.loc["P2_ADJ", "TFU"] = (
        total.Y["TFU"].sum() + total.M.loc["D21X31", "TFU"]
    )
    checks = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )
    assert checks["full.p2_2.total_m"].status == "pass"


def test_mutations_are_localised_to_the_named_accounting_control() -> None:
    variants = _triplet()
    variants["TOTAL"].V.loc["P2_ADJ", "a"] += 1.0
    variants["TOTAL"].M.loc["D21X31", "P3_S13"] += 1.0
    checks = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY
        )
    )
    assert checks["full.p2_1.total_v"].affected_labels == ("a",)
    assert checks["full.p2_2.total_m"].affected_labels == ("P3_S13",)
    assert checks["full.v4.d1.total"].status == "pass"
    assert checks["full.p2_dom_not_derived"].status == "not_applicable"


def test_negative_transactions_and_supply_controls_fail_domain_checks() -> None:
    """Balanced equalities cannot legitimise negative intermediate or import supply."""
    variants = _triplet()
    variants["IMP"].Z.loc["a", "a"] = -1.0
    variants["TOTAL"].V.loc["P7", "a"] = -1.0

    checks = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )

    assert checks["full.domain.z.imp"].status == "fail"
    assert checks["full.domain.v.total.p7"].status == "fail"


def test_total_v_and_m_are_not_blanket_additive_contracts() -> None:
    checks = _checks(
        validate_eurostat_iot_full_table(
            _triplet(), policy=NAIO_10_CP1700_ACCOUNTING_POLICY
        )
    )
    assert checks["full.v7.no_blanket_v_additivity"].status == "not_applicable"
    assert checks["full.vm.no_blanket_m_additivity"].status == "not_applicable"
    assert checks["full.gdp1.production"].status == "not_tested"

def test_validator_does_not_equate_p7_supply_with_signed_imp_row_use() -> None:
    variants = _triplet()
    imp = variants["IMP"]
    dom = variants["DOM"]
    total = variants["TOTAL"]
    # Make the IMP transaction row use negative through signed inventories while
    # preserving DOM+IMP=TOTAL transactions. P7 remains its independent supply control.
    delta = -30.0
    affected = ["P52", "P5M", "P5", "TFU", "TU"]
    for parsed in (imp, total):
        parsed.Y.loc["a", "P52"] += delta
        parsed.Y.loc["a", "P5M"] += delta
        parsed.Y.loc["a", "P5"] += delta
        parsed.Y.loc["a", "TFU"] += delta
        parsed.Y.loc["a", "TU"] += delta
        assert parsed.M is not None
        parsed.M.loc["CPA_TOTAL", affected] = parsed.Y.loc["a", affected]
    assert total.M is not None
    total.M.loc["P2_ADJ", affected] = total.Y.loc["a", affected]
    # Keep the DOM imported-use supplementary final-use row aligned to IMP Y.
    assert dom.M is not None
    dom.M.loc["IMP", :] = imp.Y.loc["a", :]
    checks = _checks(
        validate_eurostat_iot_full_table(
            variants, policy=NAIO_10_CP1700_ACCOUNTING_POLICY, mode="generated"
        )
    )
    assert "full.domain.y.imp.tu" not in checks
    assert checks["full.r2.imports_supply"].status == "not_tested"
    assert checks["full.i2.dom_v_imp"].status == "pass"
    assert checks["full.i3.dom_m_imp"].status == "pass"
    assert checks["full.i8.import_orientation"].status == "pass"
