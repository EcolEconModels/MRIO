from __future__ import annotations

from pathlib import Path

import pandas as pd


def _build_mapping(source_codes: list[str], target_codes: list[str], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for source_code, target_code in zip(source_codes, target_codes, strict=False):
        rows.append(
            {
                "source_code": source_code,
                "target_code": target_code,
                "enabled": "1",
                "source_label": labels.get(source_code, source_code),
                "target_label": labels.get(target_code, target_code),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _build_base_frames() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    source_codes = ["p24.g", "p46"]
    target_codes = ["CPA_C20", "CPA_G46"]
    labels = {
        "p24.g": "Biogasoline",
        "p46": "Wholesale trade except motor vehicles",
        "CPA_C20": "Chemicals and chemical products",
        "CPA_G46": "Wholesale trade except motor vehicles",
    }
    mapping = _build_mapping(source_codes, target_codes, labels)

    exio_Z = pd.DataFrame(
        [[5.0, 30.0], [2.0, 4.0]],
        index=source_codes,
        columns=source_codes,
    )
    exio_Y = pd.DataFrame(
        {"FD": [10.0, 8.0]},
        index=source_codes,
    )
    exio_x = exio_Z.sum(axis=1) + exio_Y.sum(axis=1)

    euro_Z = pd.DataFrame(
        [[5.0, 30.0], [2.0, 4.0]],
        index=target_codes,
        columns=target_codes,
    )
    euro_Y = pd.DataFrame(
        {"FD": [10.0, 8.0]},
        index=target_codes,
    )
    euro_x = euro_Z.sum(axis=1) + euro_Y.sum(axis=1)
    return exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels


def test_concordance_audit_perfect_agreement_has_no_flags():
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit

    exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels = _build_base_frames()
    eurostat_labels = {"CPA_C20": "Chemicals and chemical products", "CPA_G46": "Wholesale trade except motor vehicles"}

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels=eurostat_labels,
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )

    assert result.mapping_coverage["status"].eq("OK").all()
    assert result.aggregate_totals["flag"].sum() == 0
    assert result.z_cell_mismatches["flag"].sum() == 0
    assert result.x_mismatches.empty
    assert result.y_mismatches.empty


def test_concordance_audit_flags_aggregate_x_mismatch():
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit

    exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels = _build_base_frames()
    euro_x = euro_x.copy()
    euro_x.loc["CPA_C20"] = 100.0

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels={"CPA_C20": "Chemicals and chemical products", "CPA_G46": "Wholesale trade except motor vehicles"},
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )

    flagged = result.x_mismatches[result.x_mismatches["target_code"] == "CPA_C20"]
    assert not flagged.empty
    assert flagged.iloc[0]["flag"]
    assert flagged.iloc[0]["ratio_exio_to_eurostat"] < 1.0


def test_concordance_audit_flags_z_cell_mismatch():
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit

    exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels = _build_base_frames()
    euro_Z = euro_Z.copy()
    euro_Z.loc["CPA_C20", "CPA_G46"] = 10.0

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels={"CPA_C20": "Chemicals and chemical products", "CPA_G46": "Wholesale trade except motor vehicles"},
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )

    cell = result.z_cell_mismatches[(result.z_cell_mismatches["supplier_code"] == "CPA_C20") & (result.z_cell_mismatches["buyer_code"] == "CPA_G46")]
    assert not cell.empty
    assert cell.iloc[0]["flag"]
    assert cell.iloc[0]["exio_mapped_z"] == 30.0
    assert cell.iloc[0]["eurostat_z"] == 10.0


def test_concordance_audit_flags_subcomponent_exceedance():
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit

    exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels = _build_base_frames()
    euro_Z = euro_Z.copy()
    euro_Z.loc["CPA_C20", "CPA_G46"] = 20.0

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels={"CPA_C20": "Chemicals and chemical products", "CPA_G46": "Wholesale trade except motor vehicles"},
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )

    exceed = result.subcomponent_exceeds_aggregate[result.subcomponent_exceeds_aggregate["metric"] == "Z_CELL"]
    assert not exceed.empty
    assert (exceed["exio_subsector_label"] == "Biogasoline").any()
    assert exceed.iloc[0]["ratio"] >= 1.2


def test_concordance_audit_suppresses_tiny_values_with_min_abs_threshold():
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit

    source_codes = ["a", "b"]
    target_codes = ["A", "B"]
    labels = {"a": "A", "b": "B", "A": "A", "B": "B"}
    mapping = _build_mapping(source_codes, target_codes, labels)

    exio_Z = pd.DataFrame([[0.001, 0.0], [0.0, 0.001]], index=source_codes, columns=source_codes)
    exio_Y = pd.DataFrame({"FD": [0.001, 0.001]}, index=source_codes)
    exio_x = exio_Z.sum(axis=1) + exio_Y.sum(axis=1)
    euro_Z = pd.DataFrame([[0.002, 0.0], [0.0, 0.001]], index=target_codes, columns=target_codes)
    euro_Y = pd.DataFrame({"FD": [0.001, 0.001]}, index=target_codes)
    euro_x = euro_Z.sum(axis=1) + euro_Y.sum(axis=1)

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels={"A": "A", "B": "B"},
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )

    assert result.aggregate_totals["flag"].sum() == 0
    assert result.z_cell_mismatches["flag"].sum() == 0


def test_at_2020_audit_report_contains_motivating_biofuels_case():
    report_path = Path("/home/agilra/EcolEconModels/MRIO/outputs/audits/exio_eurostat_concordance/AT/2020/audit_exio_eurostat_concordance_AT_2020.md")
    z_cell_path = Path("/home/agilra/EcolEconModels/MRIO/outputs/audits/exio_eurostat_concordance/AT/2020/audit_z_cell_mismatches.csv")
    drilldown_path = Path("/home/agilra/EcolEconModels/MRIO/outputs/audits/exio_eurostat_concordance/AT/2020/audit_z_cell_drilldown.csv")
    subcomponent_path = Path("/home/agilra/EcolEconModels/MRIO/outputs/audits/exio_eurostat_concordance/AT/2020/audit_subcomponent_exceeds_aggregate.csv")
    if not report_path.exists() or not z_cell_path.exists() or not drilldown_path.exists() or not subcomponent_path.exists():
        import pytest

        pytest.skip("AT 2020 concordance audit outputs are not available")

    report_text = report_path.read_text(encoding="utf-8")
    z_cell_frame = pd.read_csv(z_cell_path)
    drilldown_frame = pd.read_csv(drilldown_path)
    subcomponent_frame = pd.read_csv(subcomponent_path)

    assert "CPA_C20 / Biofuels Diagnostic" in report_text
    assert ((z_cell_frame["supplier_code"] == "CPA_C20") & (z_cell_frame["buyer_code"] == "CPA_G46")).any()
    assert (z_cell_frame["supplier_code"] == "CPA_C20").any()
    assert ((drilldown_frame["supplier_code"] == "CPA_C20") & (drilldown_frame["buyer_code"] == "CPA_G46") & (drilldown_frame["exio_subsector_code"] == "p24.g")).any() or ((subcomponent_frame["metric"] == "Z_CELL") & (subcomponent_frame["exio_subsector_code"] == "p24.g")).any()


def test_write_concordance_audit_outputs_emits_run_script(tmp_path):
    from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit, write_concordance_audit_outputs

    exio_x, exio_Z, exio_Y, euro_x, euro_Z, euro_Y, mapping, labels = _build_base_frames()
    eurostat_labels = {"CPA_C20": "Chemicals and chemical products", "CPA_G46": "Wholesale trade except motor vehicles"}

    result = build_concordance_audit(
        country="AT",
        year=2020,
        exio_x=exio_x,
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_x=euro_x,
        mapping_df=mapping,
        exio_labels=labels,
        eurostat_labels=eurostat_labels,
        rel_threshold=0.20,
        min_abs_meur=1.0,
    )
    result.metadata["run_command"] = "PYTHONPATH=src .venv/bin/python scripts/run_exio_eurostat_concordance_audit.py --country AT --year 2020"

    paths = write_concordance_audit_outputs(result, tmp_path)
    run_script = paths["audit_run.sh"]

    assert run_script.exists()
    script_text = run_script.read_text(encoding="utf-8")
    assert "--country AT --year 2020" in script_text
    assert script_text.startswith("#!/usr/bin/env bash")
