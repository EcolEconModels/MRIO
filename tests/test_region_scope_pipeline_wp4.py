from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.exiobase3 import read_exiobase3_scope_export_vector
from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions
from ecolecon_mrio.region_aggregation import RegionScope
from scripts.split_eurostat_energy_by_exiobase import _resolve_pipeline_scope


def _write_table(zf: zipfile.ZipFile, name: str, regions: list[str], labels: list[str], rows: list[tuple[str,str,list[float]]]) -> None:
    text = "\t\t" + "\t".join(regions) + "\n"
    text += "region\tsector\t" + "\t".join(labels) + "\n"
    text += "unit\tunit\t" + "\t".join(["M.EUR"] * len(labels)) + "\n"
    for region, sector, values in rows:
        text += region + "\t" + sector + "\t" + "\t".join(map(str, values)) + "\n"
    zf.writestr(name, text)


def test_scope_export_excludes_intra_scope_trade(tmp_path: Path) -> None:
    zp = tmp_path / "iot.zip"
    # one sector; destination columns AT, DE, US. AT+DE is the scope.
    rows = [
        ("AT", "S", [1.0, 2.0, 3.0]),
        ("DE", "S", [4.0, 5.0, 6.0]),
        ("US", "S", [7.0, 8.0, 9.0]),
    ]
    with zipfile.ZipFile(zp, "w") as zf:
        _write_table(zf, "Z.txt", ["AT", "DE", "US"], ["S", "S", "S"], rows)
        _write_table(zf, "Y.txt", ["AT", "DE", "US"], ["HH", "HH", "HH"], rows)
    scope = RegionScope(code="EU_TEST", members=("AT", "DE"))
    exports = read_exiobase3_scope_export_vector(source_zip=zp, scope=scope)
    # AT->US (3) + DE->US (6), in both Z and Y = 18. Intra AT<->DE is excluded.
    assert exports.loc["S"] == pytest.approx(18.0)


def test_singleton_scope_needs_no_mapping() -> None:
    scope = _resolve_pipeline_scope(scope_code="AT", mapping_csv=None, source_regions=None)
    assert scope == RegionScope(code="AT", members=("AT",))


def test_mapping_backed_scope_uses_file_members(tmp_path: Path) -> None:
    mapping = tmp_path / "scope.csv"
    pd.DataFrame(
        {
            "scope_code": ["EU_TEST", "EU_TEST"],
            "scope_label": ["EU test", "EU test"],
            "member_region": ["AT", "DE"],
            "member_order": [1, 2],
            "enabled": [1, 1],
            "notes": ["", ""],
        }
    ).to_csv(mapping, index=False)
    scope = _resolve_pipeline_scope(
        scope_code="EU_TEST", mapping_csv=mapping, source_regions=("AT", "DE", "US")
    )
    assert scope.members == ("AT", "DE")


def test_boundary_aware_environmental_reaggregation_uses_flow_denominator() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["a", "b"],
            "boundary": ["external_imports_direct", "external_imports_direct"],
            "denominator_role": ["target_import_supply", "target_import_supply"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 6.0],
            "monetary_output": [10.0, 30.0],
            "intensity": [0.2, 0.2],
        }
    )
    out = aggregate_environmental_extensions(
        frame,
        {"a": "A", "b": "A"},
        output=pd.Series({"a": 10.0, "b": 30.0}),
    )
    assert out is not None
    row = out.iloc[0]
    assert row["boundary"] == "external_imports_direct"
    assert row["value"] == pytest.approx(8.0)
    assert row["monetary_output"] == pytest.approx(40.0)
    assert row["intensity"] == pytest.approx(0.2)


def test_boundary_reaggregation_rejects_domestic_x_as_import_denominator() -> None:
    frame = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [2.0],
            "monetary_output": [10.0],
            "intensity": [0.2],
        }
    )
    with pytest.raises(ValueError, match="denominator disagrees"):
        aggregate_environmental_extensions(
            frame, {"a": "A"}, output=pd.Series({"a": 99.0})
        )


def test_scope_loader_forwards_resolved_group_scope(tmp_path: Path, monkeypatch) -> None:
    """A mapping-backed donor scope must reach the provider unchanged, not as members[0]."""
    from ecolecon_mrio.region_aggregation import MrioScopeSlice
    from scripts import split_eurostat_energy_by_exiobase as split_mod

    captured: dict[str, object] = {}

    def fake_loader(dataset, year, *, scope, flow, **kwargs):
        captured.update(dataset=dataset, year=year, scope=scope, flow=flow)
        sectors = pd.Index(["S"], name="sector")
        return MrioScopeSlice(
            scope=scope,
            Z=pd.DataFrame([[1.0]], index=sectors, columns=sectors),
            Y=pd.DataFrame([[2.0]], index=sectors, columns=pd.Index(["HH"])),
            x_scope=pd.Series([3.0], index=sectors, name="x_scope"),
            flow=flow,
            metadata={"source_zip": str(tmp_path / "iot.zip")},
        )

    monkeypatch.setattr(split_mod, "load_mrio_iot_scope_slice", fake_loader)
    scope = RegionScope(code="EU_TEST", members=("AT", "DE"))
    result = split_mod._load_exio_scope_slice(
        year=2021,
        scope=scope,
        flow="domestic",
        data_root=tmp_path,
        doi="10.5281/zenodo.20051562",
        force_download=False,
        verbose=False,
    )

    assert captured["scope"] == scope
    assert captured["flow"] == "internal"
    assert result.scope == scope


def test_group_scope_export_callback_cannot_fallback_to_first_member(tmp_path: Path) -> None:
    """A legacy region-only callback must fail for a grouped scope instead of using member 0."""
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import _project_export_donor_for_split
    from ecolecon_mrio.region_aggregation import MrioScopeSlice

    sectors = pd.Index(["S"], name="sector")
    exio_slice = MrioScopeSlice(
        scope=RegionScope(code="EU_TEST", members=("AT", "DE")),
        Z=pd.DataFrame([[1.0]], index=sectors, columns=sectors),
        Y=pd.DataFrame([[1.0]], index=sectors, columns=pd.Index(["HH"])),
        x_scope=pd.Series([2.0], index=sectors, name="x_scope"),
        flow="internal",
        metadata={"source_zip": str(tmp_path / "iot.zip"), "system": "pxp"},
    )
    share = pd.Series([1.0], index=sectors, dtype="float64")
    profiles = {
        "CPA_X": SplitProfiles(
            aggregate_sector="CPA_X",
            subsectors=("S",),
            output_share=share.copy(),
            intermediate_sales_share=share.copy(),
            intermediate_input_share=share.copy(),
            final_demand_share=share.copy(),
            subsector_to_exio_sectors={"S": ["S"]},
        )
    }

    def region_only_loader(*, source_zip, region):
        return pd.Series([1.0], index=sectors, dtype="float64")

    with pytest.raises(TypeError, match="scope"):
        _project_export_donor_for_split(
            exio_slice=exio_slice,
            split_profiles=profiles,
            eurostat_sector_mapping={"CPA_X": {"S": 1.0}},
            emitted_labels=["S"],
            export_vector_loader=region_only_loader,
        )


def test_environmental_reaggregation_requires_explicit_boundary_metadata() -> None:
    """A stale sidecar without boundary identity must fail rather than become DOM."""
    frame = pd.DataFrame(
        {
            "sector": ["a"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [2.0],
            "monetary_output": [10.0],
            "intensity": [0.2],
        }
    )
    with pytest.raises(ValueError, match="boundary|denominator_role"):
        aggregate_environmental_extensions(
            frame, {"a": "A"}, output=pd.Series({"a": 10.0})
        )


def test_split_cli_initializes_global_scope_context_once() -> None:
    """WP4 orchestration must not duplicate expensive global-x or scope-resolution setup."""
    source = Path("scripts/split_eurostat_energy_by_exiobase.py").read_text(
        encoding="utf-8"
    )
    assert source.count("environmental_source_output: pd.Series | None = None") == 1
    assert source.count(
        "Archive-driven environmental selection requires the already-loaded "
    ) == 1
    assert source.count("exio_slice=exio_slice_single") >= 1
    assert "Calling: load_exiobase3_region_slice" not in source
    assert "Returned: load_exiobase3_region_slice" not in source


def test_split_provenance_hashes_exact_bytes(tmp_path: Path) -> None:
    """Byte-level provenance must change whenever the exact local source bytes change."""
    import hashlib
    from scripts.split_eurostat_energy_by_exiobase import _sha256_file

    path = tmp_path / "input.bin"
    path.write_bytes(b"abc\x00def")
    assert _sha256_file(path) == hashlib.sha256(b"abc\x00def").hexdigest()
    first = _sha256_file(path)
    path.write_bytes(b"abc\x00deg")
    assert _sha256_file(path) != first


def test_validator_uses_total_supply_not_domestic_x_for_total_environmental_rows(
    tmp_path: Path,
) -> None:
    """TOTAL environmental validation uses Z+Y supply even when shared domestic x differs."""
    from scripts.validate_many_many_pipeline_outputs import validate_pipeline_outputs

    split_dir = tmp_path / "split" / "stk_flow=TOTAL"
    agg_dir = tmp_path / "agg" / "stk_flow=TOTAL"
    split_dir.mkdir(parents=True)
    agg_dir.mkdir(parents=True)
    sectors = pd.Index(["S"])
    Z = pd.DataFrame([[20.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame([[10.0]], index=sectors, columns=pd.Index(["P3"]))
    x = pd.Series([100.0], index=sectors, name="x")
    Z.to_csv(split_dir / "Z_split.csv")
    Y.to_csv(split_dir / "Y_split.csv")
    x.to_csv(split_dir / "x_split.csv")
    Z.to_csv(agg_dir / "Z_agg.csv")
    Y.to_csv(agg_dir / "Y_agg.csv")
    x.to_csv(agg_dir / "x_agg.csv")

    target = pd.DataFrame(
        {
            "sector": ["S"],
            "boundary": ["total_use_direct"],
            "denominator_role": ["target_total_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [6.0],
            "monetary_output": [30.0],
            "intensity": [0.2],
        }
    )
    target.to_csv(split_dir / "environmental_extensions.csv", index=False)
    completed = target.copy()
    completed.insert(0, "sector_position", pd.Series([1], dtype="int64"))
    completed.to_csv(agg_dir / "environmental_extensions.csv", index=False)

    # This copied raw F provenance deliberately looks like DOM source evidence;
    # target TOTAL boundary identity must still come from the target sidecar.
    raw = pd.DataFrame(
        {
            "source_label": ["S"],
            "extension": ["air"],
            "kind": ["F"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [4.0],
            "monetary_denominator": [20.0],
            "intensity": [0.2],
        }
    )
    raw.to_csv(split_dir / "exio_environmental_extensions.csv", index=False)
    raw.to_csv(agg_dir / "exio_environmental_extensions.csv", index=False)

    mapping_csv = tmp_path / "identity.csv"
    pd.DataFrame(
        {
            "source_code": ["S"],
            "target_code": ["S"],
            "enabled": [True],
            "notes": ["identity"],
        }
    ).to_csv(mapping_csv, index=False)

    report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=mapping_csv,
    )
    assert report["ok"], report
def test_final_validator_requires_complete_whole_iot_reconciliation_evidence(
    tmp_path: Path,
) -> None:
    from scripts import validate_many_many_pipeline_outputs as validation_mod

    split_dir = tmp_path / "split" / "stk_flow=TOTAL"
    split_dir.mkdir(parents=True)
    sectors = pd.Index(["S"])
    pd.DataFrame([[1.0]], index=sectors, columns=sectors).to_csv(
        split_dir / "Z_split.csv"
    )
    pd.DataFrame({"P3": [1.0]}, index=sectors).to_csv(split_dir / "Y_split.csv")
    pd.Series([2.0], index=sectors, name="x").to_csv(split_dir / "x_split.csv")
    (split_dir / "split_info.json").write_text(
        '{"producer_column_reconciliation":"joint_dom_imp_kl_production_account_feasible","balancing_applied":true,"balancing_method":"KL_DYKSTRA_PARTITION_CONSTRAINTS"}\n',
        encoding="utf-8",
    )
    rows = [
        {
            "validation_class": "transformation_coherence",
            "flow": flow,
            "check_id": check_id,
            "status": "pass",
            "max_abs_gap": 0.0,
            "rtol": 1e-12,
            "atol": 1e-10,
        }
        for flow, check_id in sorted(
            validation_mod._REQUIRED_TRANSFORMATION_COHERENCE_CHECKS
        )
    ]
    report_path = split_dir.parent / "transformation_coherence_validation.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    pd.DataFrame(
        [{"check_id": "synthetic.full", "status": "pass", "severity": "error"}]
    ).to_csv(split_dir.parent / "split_full_table_validation.csv", index=False)

    accepted = validation_mod.validate_pipeline_outputs(split_dir=split_dir)
    assert accepted["ok"], accepted
    check = next(
        row
        for row in accepted["checks"]
        if row["name"] == "producer_technology_reconciliation_coherence"
    )
    assert check["ok"]

    pd.DataFrame(rows[:-1]).to_csv(report_path, index=False)
    rejected = validation_mod.validate_pipeline_outputs(split_dir=split_dir)
    assert not rejected["ok"]
    failed = next(
        row
        for row in rejected["checks"]
        if row["name"] == "producer_technology_reconciliation_coherence"
    )
    assert not failed["ok"]
    assert failed["missing_checks"]


def test_environmental_target_retarget_preserves_signed_import_use() -> None:
    """Signed target import use is preserved rather than clipped as supply noise."""
    from ecolecon_mrio.region_aggregation import _retarget_direct_environmental_account

    account = pd.DataFrame(
        {
            "axis_position": [1],
            "axis_label": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["scope_external_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [2.0],
            "monetary_denominator": [10.0],
            "intensity": [0.2],
        }
    )
    for target_value in (-5e-7, -2e-6):
        retargeted = _retarget_direct_environmental_account(
            account,
            target_denominator=pd.Series([target_value], index=pd.Index(["a"])),
            boundary="external_imports_direct",
            denominator_role="target_import_use",
        )
        assert float(retargeted.loc[0, "monetary_denominator"]) == pytest.approx(target_value)
        assert float(retargeted.loc[0, "value"]) == pytest.approx(0.2 * target_value)
        assert float(retargeted.loc[0, "intensity"]) == pytest.approx(0.2)


def test_total_sidecar_combiner_normalizes_components_before_total() -> None:
    """TOTAL must be formed from normalized DOM/IMP controls, not noisy raw components."""
    from ecolecon_mrio.region_aggregation import _combine_target_supply_extension_sidecars_direct

    dom = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["scope_production"],
            "denominator_role": ["target_domestic_output"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [0.0],
            "monetary_output": [0.0],
            "intensity": [np.nan],
        }
    )
    imp = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [10.0],
            "monetary_output": [5.0],
            "intensity": [2.0],
        }
    )
    total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp,
        target_domestic_output=pd.Series([-5e-7], index=pd.Index(["a"])),
        target_import_use=pd.Series([5.0], index=pd.Index(["a"])),
    )
    assert float(total.loc[0, "monetary_output"]) == 5.0
    assert float(total.loc[0, "value"]) == 10.0
    assert float(total.loc[0, "intensity"]) == 2.0



def test_environmental_target_retarget_preserves_unavailable_positive_mass() -> None:
    """Undefined donor coefficient over positive target mass is NA evidence, not zero."""
    from ecolecon_mrio.region_aggregation import _retarget_direct_environmental_account

    donor = pd.DataFrame(
        {
            "axis_position": [1],
            "axis_label": ["CPA_E36"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["scope_external_import_use"],
            "extension": ["water"],
            "stressor": ["Synthetic water stressor"],
            "unit": ["kg"],
            "value": [0.0],
            "monetary_denominator": [0.0],
            "intensity": [np.nan],
        }
    )
    target = _retarget_direct_environmental_account(
        donor,
        target_denominator=pd.Series([5.0], index=pd.Index(["CPA_E36"])),
        boundary="external_imports_direct",
        denominator_role="target_import_use",
    )
    assert float(target.loc[0, "monetary_denominator"]) == 5.0
    assert pd.isna(target.loc[0, "value"])
    assert pd.isna(target.loc[0, "intensity"])


def test_total_sidecar_propagates_unavailable_positive_component_only() -> None:
    """Positive unavailable IMP poisons TOTAL; unavailable zero-mass IMP does not."""
    from ecolecon_mrio.region_aggregation import (
        _combine_target_supply_extension_sidecars_direct,
    )

    dom = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["scope_production"],
            "denominator_role": ["target_domestic_output"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [20.0],
            "monetary_output": [10.0],
            "intensity": [2.0],
        }
    )
    imp_missing = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [np.nan],
            "monetary_output": [5.0],
            "intensity": [np.nan],
        }
    )
    unavailable_total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp_missing,
        target_domestic_output=pd.Series([10.0], index=["a"]),
        target_import_use=pd.Series([5.0], index=["a"]),
    )
    assert float(unavailable_total.loc[0, "monetary_output"]) == 15.0
    assert pd.isna(unavailable_total.loc[0, "value"])
    assert pd.isna(unavailable_total.loc[0, "intensity"])

    imp_zero_mass = imp_missing.copy()
    imp_zero_mass["monetary_output"] = 0.0
    supported_total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp_zero_mass,
        target_domestic_output=pd.Series([10.0], index=["a"]),
        target_import_use=pd.Series([0.0], index=["a"]),
    )
    assert float(supported_total.loc[0, "monetary_output"]) == 10.0
    assert float(supported_total.loc[0, "value"]) == 20.0
    assert float(supported_total.loc[0, "intensity"]) == 2.0


def test_total_sidecar_rejects_infinite_intensity_at_zero_denominator() -> None:
    """Zero-denominator evidence must use NA intensity; +/-Inf is malformed."""
    from ecolecon_mrio.region_aggregation import (
        _combine_target_supply_extension_sidecars_direct,
    )

    zero = pd.DataFrame(
        {
            "sector": ["a"],
            "boundary": ["scope_production"],
            "denominator_role": ["target_domestic_output"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [0.0],
            "monetary_output": [0.0],
            "intensity": [np.inf],
        }
    )
    imp = zero.copy()
    imp["boundary"] = "external_imports_direct"
    imp["denominator_role"] = "target_import_use"
    with pytest.raises(ValueError, match="intensity must be missing at zero denominator"):
        _combine_target_supply_extension_sidecars_direct(
            zero,
            imp,
            target_domestic_output=pd.Series([0.0], index=["a"]),
            target_import_use=pd.Series([0.0], index=["a"]),
        )


def test_validator_accepts_imp_target_environmental_grid_without_raw_sidecar(
    tmp_path: Path,
) -> None:
    """IMP target evidence is authoritative even though IMP has no raw F/F_Y sidecar."""
    from scripts.validate_many_many_pipeline_outputs import validate_pipeline_outputs

    split_dir = tmp_path / "split" / "stk_flow=IMP"
    agg_dir = tmp_path / "agg" / "stk_flow=IMP"
    split_dir.mkdir(parents=True)
    agg_dir.mkdir(parents=True)
    sectors = pd.Index(["S"])
    Z = pd.DataFrame([[0.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame({"P3": [5.0]}, index=sectors)
    x = pd.Series([10.0], index=sectors, name="x")
    Z.to_csv(split_dir / "Z_split.csv")
    Y.to_csv(split_dir / "Y_split.csv")
    x.to_csv(split_dir / "x_split.csv")
    Z.to_csv(agg_dir / "Z_agg.csv")
    Y.to_csv(agg_dir / "Y_agg.csv")
    x.to_csv(agg_dir / "x_agg.csv")

    target = pd.DataFrame(
        {
            "sector": ["S"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["water"],
            "stressor": ["Synthetic water stressor"],
            "unit": ["kg"],
            "value": [np.nan],
            "monetary_output": [5.0],
            "intensity": [np.nan],
        }
    )
    target.to_csv(split_dir / "environmental_extensions.csv", index=False)
    completed = target.copy()
    completed.insert(0, "sector_position", pd.Series([1], dtype="int64"))
    completed.to_csv(agg_dir / "environmental_extensions.csv", index=False)

    mapping_csv = tmp_path / "identity.csv"
    pd.DataFrame(
        {
            "source_code": ["S"],
            "target_code": ["S"],
            "enabled": [True],
            "notes": ["identity"],
        }
    ).to_csv(mapping_csv, index=False)

    report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=mapping_csv,
    )
    assert report["ok"], report


def test_total_environmental_sidecar_carries_dom_final_demand_rows(tmp_path: Path) -> None:
    """TOTAL combines product rows but copies destination-scope F_Y from DOM."""
    from scripts.derive_total_iot_flow import _derive_total_environmental_extensions

    dirs = {flow: tmp_path / flow for flow in ("DOM", "IMP", "TOTAL")}
    for directory in dirs.values():
        directory.mkdir()

    dom = pd.DataFrame(
        {
            "sector": ["S", "HH"],
            "boundary": ["scope_production", "scope_final_demand_direct"],
            "denominator_role": ["target_domestic_output", "scope_final_demand"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 4.0],
            "monetary_output": [10.0, 100.0],
            "intensity": [0.2, 0.04],
        }
    )
    imp = pd.DataFrame(
        {
            "sector": ["S"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [1.0],
            "monetary_output": [5.0],
            "intensity": [0.2],
        }
    )
    dom.to_csv(dirs["DOM"] / "environmental_extensions.csv", index=False)
    imp.to_csv(dirs["IMP"] / "environmental_extensions.csv", index=False)
    pd.Series([10.0], index=["S"], name="use_total").to_csv(
        dirs["DOM"] / "use_total_split.csv"
    )
    pd.Series([5.0], index=["S"], name="use_total").to_csv(
        dirs["IMP"] / "use_total_split.csv"
    )

    _derive_total_environmental_extensions(dirs)
    total = pd.read_csv(dirs["TOTAL"] / "environmental_extensions.csv")
    assert total["boundary"].tolist() == ["total_use_direct", "scope_final_demand_direct"]
    product = total.iloc[0]
    assert float(product["value"]) == pytest.approx(3.0)
    assert float(product["monetary_output"]) == pytest.approx(15.0)
    fy = total.iloc[1]
    assert fy["sector"] == "HH"
    assert fy["denominator_role"] == "scope_final_demand"
    assert float(fy["value"]) == pytest.approx(4.0)
    assert float(fy["monetary_output"]) == pytest.approx(100.0)


def test_environmental_grid_validates_required_fy_passthrough(tmp_path: Path) -> None:
    """DOM/TOTAL grid validation requires exact F_Y passthrough from raw provenance."""
    from scripts.validate_many_many_pipeline_outputs import _validate_environmental_grid

    raw = pd.DataFrame(
        {
            "source_label": ["S", "HH"],
            "extension": ["air", "air"],
            "kind": ["F", "F_Y"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 4.0],
            "monetary_denominator": [10.0, 100.0],
            "intensity": [0.2, 0.04],
        }
    )
    observed = pd.DataFrame(
        {
            "sector_position": [1, 1],
            "sector": ["S", "HH"],
            "boundary": ["scope_production", "scope_final_demand_direct"],
            "denominator_role": ["target_domestic_output", "scope_final_demand"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 4.0],
            "monetary_output": [10.0, 100.0],
            "intensity": [0.2, 0.04],
        }
    )
    raw_path = tmp_path / "raw.csv"
    grid_path = tmp_path / "environmental.csv"
    raw.to_csv(raw_path, index=False)
    observed.to_csv(grid_path, index=False)

    report = _validate_environmental_grid(
        environmental_path=grid_path,
        raw_path=raw_path,
        output=pd.Series([10.0], index=["S"], dtype="float64"),
        rtol=1e-9,
        atol=1e-6,
    )
    assert report["ok"], report
    assert report["n_final_demand_rows"] == 1

    observed.iloc[[0]].to_csv(grid_path, index=False)
    missing_report = _validate_environmental_grid(
        environmental_path=grid_path,
        raw_path=raw_path,
        output=pd.Series([10.0], index=["S"], dtype="float64"),
        rtol=1e-9,
        atol=1e-6,
    )
    assert not missing_report["ok"]
    assert any("scope_final_demand_direct" in error for error in missing_report["errors"])
