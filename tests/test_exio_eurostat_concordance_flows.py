from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio.concordance_audit_flows import (
    FlowAuditRun,
    build_flow_reconciliation,
    build_flow_total_use_proxy,
    resolve_comparison_flows,
    write_flow_consolidated_outputs,
)
from ecolecon_mrio.exio_eurostat_concordance_audit import ConcordanceAuditResult, build_concordance_audit, write_concordance_audit_outputs


def _make_matrix(label: str, values: list[list[float]]) -> pd.DataFrame:
    codes = ["CPA_A", "CPA_B"]
    return pd.DataFrame(values, index=codes, columns=codes if label == "z" else ["FD"])


def _make_result(
    *,
    comparison_mode: str,
    exio_flow: str,
    eurostat_flow: str,
    mapped_z: pd.DataFrame,
    mapped_y: pd.DataFrame,
    mapped_x: pd.Series,
    eurostat_z: pd.DataFrame,
    eurostat_y_total: pd.Series,
    eurostat_x: pd.Series,
    flag_value: float,
) -> ConcordanceAuditResult:
    aggregate_totals = pd.DataFrame(
        [
            {
                "metric": "X_OUTPUT_PROXY_TOTAL_USE",
                "target_code": "CPA_A",
                "target_label": "Sector A",
                "exio_mapped_value": float(mapped_x.get("CPA_A", 0.0)),
                "eurostat_value": float(eurostat_x.get("CPA_A", 0.0)),
                "signed_gap": float(mapped_x.get("CPA_A", 0.0) - eurostat_x.get("CPA_A", 0.0)),
                "abs_gap": abs(float(mapped_x.get("CPA_A", 0.0) - eurostat_x.get("CPA_A", 0.0))),
                "rel_gap": 0.5,
                "ratio_exio_to_eurostat": 1.0,
                "flag": bool(flag_value),
                "severity": "HIGH",
                "top_exio_contributors": "p1 Sector A: 10.0 (100.0%)",
            },
            {
                "metric": "Y_FINAL_DEMAND_TOTAL",
                "target_code": "CPA_B",
                "target_label": "Sector B",
                "exio_mapped_value": float(mapped_y.loc["CPA_B", "FD"]),
                "eurostat_value": float(eurostat_y_total.get("CPA_B", 0.0)),
                "signed_gap": float(mapped_y.loc["CPA_B", "FD"] - eurostat_y_total.get("CPA_B", 0.0)),
                "abs_gap": abs(float(mapped_y.loc["CPA_B", "FD"] - eurostat_y_total.get("CPA_B", 0.0))),
                "rel_gap": 0.5,
                "ratio_exio_to_eurostat": 1.0,
                "flag": bool(flag_value),
                "severity": "HIGH",
                "top_exio_contributors": "p2 Sector B: 10.0 (100.0%)",
            },
        ]
    )
    z_cell_mismatches = pd.DataFrame(
        [
            {
                "metric": "Z_CELL",
                "supplier_code": "CPA_A",
                "supplier_label": "Sector A",
                "buyer_code": "CPA_B",
                "buyer_label": "Sector B",
                "exio_mapped_z": float(mapped_z.loc["CPA_A", "CPA_B"]),
                "eurostat_z": float(eurostat_z.loc["CPA_A", "CPA_B"]),
                "signed_gap": float(mapped_z.loc["CPA_A", "CPA_B"] - eurostat_z.loc["CPA_A", "CPA_B"]),
                "abs_gap": abs(float(mapped_z.loc["CPA_A", "CPA_B"] - eurostat_z.loc["CPA_A", "CPA_B"])),
                "rel_gap": 0.5 if flag_value else 0.0,
                "ratio_exio_to_eurostat": 1.0,
                "flag": bool(flag_value),
                "severity": "HIGH" if flag_value else "OK",
                "top_source_subsectors": "p1 Sector A: 10.0 (100.0%)",
                "top_buyer_subsectors": "p2 Sector B: 10.0 (100.0%)",
            }
        ]
    )
    empty = pd.DataFrame()
    return ConcordanceAuditResult(
        metadata={
            "country": "AT",
            "year": 2020,
            "comparison_mode": comparison_mode,
            "exio_flow": exio_flow,
            "eurostat_flow": eurostat_flow,
            "exio_x_is_proxy": True,
            "exio_x_proxy_basis": "Z row total + Y row total for selected EXIO flow slice",
            "eurostat_x_is_proxy": True,
            "eurostat_x_proxy_basis": "Z row total + Y row total for selected Eurostat stk_flow",
            "flow_interpretation": "Test flow interpretation.",
        },
        mapping_coverage=empty,
        internal_identities=empty,
        aggregate_totals=aggregate_totals,
        x_mismatches=aggregate_totals[aggregate_totals["metric"].astype(str).eq("X_OUTPUT_PROXY_TOTAL_USE")].copy(),
        y_mismatches=aggregate_totals[aggregate_totals["metric"].astype(str).eq("Y_FINAL_DEMAND_TOTAL")].copy(),
        z_row_col_mismatches=empty,
        z_cell_mismatches=z_cell_mismatches,
        z_cell_drilldown=empty,
        y_drilldown=empty,
        subcomponent_exceeds_aggregate=empty,
        report_markdown="",
        mapped_x=mapped_x,
        mapped_y=mapped_y,
        mapped_z=mapped_z,
        eurostat_x=eurostat_x,
        eurostat_y_total=eurostat_y_total,
        eurostat_z=eurostat_z,
    )


def _make_flow_run(mode: str, *, delta: float = 0.0) -> FlowAuditRun:
    codes = ["CPA_A", "CPA_B"]
    mapped_z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=codes, columns=codes)
    mapped_y = pd.DataFrame({"FD": [10.0, 20.0]}, index=codes)
    mapped_x = build_flow_total_use_proxy(mapped_z, mapped_y)
    eurostat_z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=codes, columns=codes)
    eurostat_y = pd.Series({"CPA_A": 10.0, "CPA_B": 20.0})
    eurostat_x = build_flow_total_use_proxy(eurostat_z, mapped_y)
    if delta:
        eurostat_z = eurostat_z.copy()
        eurostat_z.loc["CPA_A", "CPA_B"] += delta
        eurostat_x = build_flow_total_use_proxy(eurostat_z, mapped_y)
    result = _make_result(
        comparison_mode=mode,
        exio_flow=mode,
        eurostat_flow={"domestic": "DOM", "imports": "IMP", "total": "TOTAL"}[mode],
        mapped_z=mapped_z,
        mapped_y=mapped_y,
        mapped_x=mapped_x,
        eurostat_z=eurostat_z,
        eurostat_y_total=eurostat_y,
        eurostat_x=eurostat_x,
        flag_value=1.0,
    )
    return FlowAuditRun(mode=mode, exio_flow=mode, eurostat_flow={"domestic": "DOM", "imports": "IMP", "total": "TOTAL"}[mode], result=result)


def test_resolve_comparison_flows_modes() -> None:
    assert resolve_comparison_flows(comparison_mode="domestic", exio_flow=None, eurostat_flow=None) == ("domestic", "domestic", "DOM")
    assert resolve_comparison_flows(comparison_mode="imports", exio_flow=None, eurostat_flow=None) == ("imports", "imports", "IMP")
    assert resolve_comparison_flows(comparison_mode="total", exio_flow=None, eurostat_flow=None) == ("total", "total", "TOTAL")


def test_resolve_comparison_flows_default_is_domestic() -> None:
    assert resolve_comparison_flows(comparison_mode=None, exio_flow=None, eurostat_flow=None) == ("domestic", "domestic", "DOM")


def test_invalid_flow_pair_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_comparison_flows(comparison_mode=None, exio_flow="imports", eurostat_flow="DOM", allow_flow_mismatch=False)


def test_invalid_flow_pair_allowed_for_debug() -> None:
    mode, exio_flow, euro_flow = resolve_comparison_flows(comparison_mode=None, exio_flow="imports", eurostat_flow="DOM", allow_flow_mismatch=True)
    assert exio_flow == "imports"
    assert euro_flow == "DOM"
    assert mode.startswith("exio_imports__eurostat_DOM")


def test_build_flow_total_use_proxy() -> None:
    Z = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["a", "b"])
    Y = pd.DataFrame([[10, 20], [30, 40]], index=["a", "b"], columns=["fd1", "fd2"])
    proxy = build_flow_total_use_proxy(Z, Y)
    assert proxy["a"] == 1 + 2 + 10 + 20
    assert proxy["b"] == 3 + 4 + 30 + 40


def test_reconciliation_no_flags_when_domestic_plus_imports_equals_total() -> None:
    codes = ["CPA_A", "CPA_B"]
    dom_z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=codes, columns=codes)
    imp_z = pd.DataFrame([[2.0, 1.0], [1.0, 2.0]], index=codes, columns=codes)
    tot_z = dom_z + imp_z
    dom_y = pd.Series({"CPA_A": 10.0, "CPA_B": 20.0})
    imp_y = pd.Series({"CPA_A": 5.0, "CPA_B": 7.0})
    tot_y = dom_y + imp_y
    dom_x = build_flow_total_use_proxy(dom_z, pd.DataFrame({"FD": [10.0, 20.0]}, index=codes))
    imp_x = build_flow_total_use_proxy(imp_z, pd.DataFrame({"FD": [5.0, 7.0]}, index=codes))
    tot_x = build_flow_total_use_proxy(tot_z, pd.DataFrame({"FD": [15.0, 27.0]}, index=codes))
    domestic = FlowAuditRun("domestic", "domestic", "DOM", _make_result(comparison_mode="domestic", exio_flow="domestic", eurostat_flow="DOM", mapped_z=dom_z, mapped_y=pd.DataFrame({"FD": [10.0, 20.0]}, index=codes), mapped_x=dom_x, eurostat_z=dom_z, eurostat_y_total=dom_y, eurostat_x=dom_x, flag_value=0.0))
    imports = FlowAuditRun("imports", "imports", "IMP", _make_result(comparison_mode="imports", exio_flow="imports", eurostat_flow="IMP", mapped_z=imp_z, mapped_y=pd.DataFrame({"FD": [5.0, 7.0]}, index=codes), mapped_x=imp_x, eurostat_z=imp_z, eurostat_y_total=imp_y, eurostat_x=imp_x, flag_value=0.0))
    total = FlowAuditRun("total", "total", "TOTAL", _make_result(comparison_mode="total", exio_flow="total", eurostat_flow="TOTAL", mapped_z=tot_z, mapped_y=pd.DataFrame({"FD": [15.0, 27.0]}, index=codes), mapped_x=tot_x, eurostat_z=tot_z, eurostat_y_total=tot_y, eurostat_x=tot_x, flag_value=0.0))
    reconciliation = build_flow_reconciliation(
        domestic=domestic,
        imports=imports,
        total=total,
        rel_threshold=0.20,
        min_abs_meur=1.0,
        eurostat_labels={"CPA_A": "Sector A", "CPA_B": "Sector B"},
    )
    assert not reconciliation[reconciliation["flag"]].any().any()


def test_reconciliation_flags_mismatch() -> None:
    codes = ["CPA_A", "CPA_B"]
    dom_z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=codes, columns=codes)
    imp_z = pd.DataFrame([[2.0, 1.0], [1.0, 2.0]], index=codes, columns=codes)
    tot_z = dom_z + imp_z
    tot_z.loc["CPA_A", "CPA_B"] += 100.0
    dom_y = pd.Series({"CPA_A": 10.0, "CPA_B": 20.0})
    imp_y = pd.Series({"CPA_A": 5.0, "CPA_B": 7.0})
    tot_y = dom_y + imp_y
    dom_x = build_flow_total_use_proxy(dom_z, pd.DataFrame({"FD": [10.0, 20.0]}, index=codes))
    imp_x = build_flow_total_use_proxy(imp_z, pd.DataFrame({"FD": [5.0, 7.0]}, index=codes))
    tot_x = build_flow_total_use_proxy(tot_z, pd.DataFrame({"FD": [15.0, 27.0]}, index=codes))
    domestic = FlowAuditRun("domestic", "domestic", "DOM", _make_result(comparison_mode="domestic", exio_flow="domestic", eurostat_flow="DOM", mapped_z=dom_z, mapped_y=pd.DataFrame({"FD": [10.0, 20.0]}, index=codes), mapped_x=dom_x, eurostat_z=dom_z, eurostat_y_total=dom_y, eurostat_x=dom_x, flag_value=1.0))
    imports = FlowAuditRun("imports", "imports", "IMP", _make_result(comparison_mode="imports", exio_flow="imports", eurostat_flow="IMP", mapped_z=imp_z, mapped_y=pd.DataFrame({"FD": [5.0, 7.0]}, index=codes), mapped_x=imp_x, eurostat_z=imp_z, eurostat_y_total=imp_y, eurostat_x=imp_x, flag_value=1.0))
    total = FlowAuditRun("total", "total", "TOTAL", _make_result(comparison_mode="total", exio_flow="total", eurostat_flow="TOTAL", mapped_z=tot_z, mapped_y=pd.DataFrame({"FD": [15.0, 27.0]}, index=codes), mapped_x=tot_x, eurostat_z=tot_z, eurostat_y_total=tot_y, eurostat_x=tot_x, flag_value=1.0))
    reconciliation = build_flow_reconciliation(
        domestic=domestic,
        imports=imports,
        total=total,
        rel_threshold=0.20,
        min_abs_meur=1.0,
        eurostat_labels={"CPA_A": "Sector A", "CPA_B": "Sector B"},
    )
    flagged = reconciliation[reconciliation["flag"]]
    assert not flagged.empty
    assert "Z_CELL" in set(flagged["metric"])


def test_metadata_and_report_text_contain_flow_section(tmp_path: Path) -> None:
    exio_x = pd.Series({"CPA_A": 15.0, "CPA_B": 25.0})
    exio_Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    exio_Y = pd.DataFrame({"FD": [10.0, 20.0]}, index=["CPA_A", "CPA_B"])
    euro_Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"FD": [10.0, 20.0]}, index=["CPA_A", "CPA_B"])
    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        mapping_df=pd.DataFrame(
            [
                {"source_code": "CPA_A", "target_code": "CPA_A", "enabled": "1", "source_label": "Sector A", "target_label": "Sector A", "notes": ""},
                {"source_code": "CPA_B", "target_code": "CPA_B", "enabled": "1", "source_label": "Sector B", "target_label": "Sector B", "notes": ""},
            ]
        ),
        exio_labels={"CPA_A": "Sector A", "CPA_B": "Sector B"},
        eurostat_labels={"CPA_A": "Sector A", "CPA_B": "Sector B"},
        metadata={
            "comparison_mode": "imports",
            "exio_flow": "imports",
            "eurostat_flow": "IMP",
            "exio_x_is_proxy": True,
            "exio_x_proxy_basis": "Z row total + Y row total for selected EXIO flow slice",
            "eurostat_x_is_proxy": True,
            "eurostat_x_proxy_basis": "Z row total + Y row total for selected Eurostat stk_flow",
            "flow_interpretation": "EXIOBASE imports mean origin != selected country and destination/use == selected country, summed across all foreign origins; Eurostat IMP is the import-flow table.",
        },
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )
    assert result.metadata["comparison_mode"] == "imports"
    assert result.metadata["exio_x_is_proxy"] is True
    assert result.metadata["eurostat_x_is_proxy"] is True

    paths = write_concordance_audit_outputs(result, tmp_path)
    report_text = Path(paths["audit_exio_eurostat_concordance.md"]).read_text(encoding="utf-8")
    assert "Flow comparison" in report_text
    assert "EXIOBASE flow" in report_text
    assert "Eurostat `stk_flow`" in report_text
    assert "Flow-specific total-use proxy comparison" in report_text
    assert "foreign origins" in report_text
    assert "summed across all foreign origins" in report_text or "summed" in report_text


def test_all_flows_output_structure(tmp_path: Path) -> None:
    domestic = _make_flow_run("domestic")
    imports = _make_flow_run("imports")
    total = _make_flow_run("total")
    consolidated = tmp_path / "consolidated"
    paths = write_flow_consolidated_outputs(
        domestic=domestic,
        imports=imports,
        total=total,
        output_dir=consolidated,
        eurostat_labels={"CPA_A": "Sector A", "CPA_B": "Sector B"},
    )
    for expected in [
        consolidated / "audit_consolidated_summary.md",
        consolidated / "audit_flow_reconciliation.csv",
        consolidated / "audit_flow_reconciliation.md",
        consolidated / "consolidated_aggregate_flags.csv",
        consolidated / "consolidated_z_cell_flags.csv",
        consolidated / "consolidated_sector_flow_classification.csv",
        consolidated / "consolidated_reconciliation_flags.csv",
    ]:
        assert expected.exists(), expected
    assert paths["audit_consolidated_summary.md"].exists()
