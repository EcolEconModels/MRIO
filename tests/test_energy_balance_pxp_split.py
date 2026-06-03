from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.energy_balance_pxp import (
    build_energy_balance_component_values,
    build_price_resolution,
    build_split_profiles_from_component_values,
    build_energy_balance_pxp_quantity_lookup,
    compile_energy_balance_pxp_split_plan,
    load_energy_balance_dictionary_frame,
    run_energy_balance_pxp_split,
    validate_energy_balance_pxp_inputs,
    validate_split_against_energy_balance_dictionary,
)


def _split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": 1,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "Chemicals and chemical products",
                "child_code": "EB_C20_ADVANCED_BIODIESEL",
                "child_label": "Advanced biodiesel",
                "component_kind": "advanced_external_quantity",
                "quantity_rule": "external_quantity_cap",
                "subsplit_group": "C20_BIODIESEL",
                "eb_selector_system": "Eurostat_Energy_Balances:SIEC",
                "eb_product_selector": "R5220P|R5220B",
                "eb_flow_selector": "nrg_bal_c:NRG_PR_E",
                "quantity_unit": "GWh",
                "external_quantity_mtoe": "0.01",
                "external_quantity_source": "test",
                "price_code": "advanced_biodiesel",
                "residual_rule": "",
                "share_roles": "all_roles",
                "allow_zero": False,
                "notes": "",
            },
            {
                "order": 2,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "Chemicals and chemical products",
                "child_code": "EB_C20_CONVENTIONAL_BIODIESEL",
                "child_label": "Conventional biodiesel",
                "component_kind": "selector_residual",
                "quantity_rule": "selector_total_minus_group_external",
                "subsplit_group": "C20_BIODIESEL",
                "eb_selector_system": "Eurostat_Energy_Balances:SIEC",
                "eb_product_selector": "R5220P|R5220B",
                "eb_flow_selector": "nrg_bal_c:NRG_PR_E",
                "quantity_unit": "GWh",
                "external_quantity_mtoe": "",
                "external_quantity_source": "",
                "price_code": "conventional_biodiesel",
                "residual_rule": "selector_total_minus_enabled_external_quantities",
                "share_roles": "all_roles",
                "allow_zero": False,
                "notes": "",
            },
            {
                "order": 3,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "Chemicals and chemical products",
                "child_code": "EB_C20_OTHERS",
                "child_label": "C20 other chemicals",
                "component_kind": "parent_residual",
                "quantity_rule": "parent_output_minus_declared_components",
                "subsplit_group": "C20_PARENT",
                "eb_selector_system": "",
                "eb_product_selector": "",
                "eb_flow_selector": "",
                "quantity_unit": "MIO_EUR",
                "external_quantity_mtoe": "",
                "external_quantity_source": "",
                "price_code": "",
                "residual_rule": "parent_output_minus_declared_components",
                "share_roles": "all_roles",
                "allow_zero": True,
                "notes": "",
            },
        ]
    )


def _reagg_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_code": "EB_C20_ADVANCED_BIODIESEL", "source_label": "Advanced biodiesel", "target_code": "CPA_C20_ADVANCED_BIODIESEL", "target_label": "Advanced biodiesel", "enabled": True, "notes": ""},
            {"source_code": "EB_C20_CONVENTIONAL_BIODIESEL", "source_label": "Conventional biodiesel", "target_code": "CPA_C20_CONVENTIONAL_BIODIESEL", "target_label": "Conventional biodiesel", "enabled": True, "notes": ""},
            {"source_code": "EB_C20_OTHERS", "source_label": "C20 others", "target_code": "CPA_C20_others", "target_label": "C20 others", "enabled": True, "notes": ""},
        ]
    )


def _tech_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": 1,
                "enabled": True,
                "parent_code": "CPA_C20",
                "parent_label": "Chemicals and chemical products",
                "buyer_split_code": "EB_C20_ADVANCED_BIODIESEL|EB_C20_CONVENTIONAL_BIODIESEL",
                "buyer_split_label": "biofuels",
                "supplier_selector_system": "target_or_sector_code",
                "supplier_selector": "CPA_C20|CPA_C20",
                "allocation_weight": 1.0,
                "apply_to": "Z_columns",
                "rule": "biofuel_feedstock_column_allocation",
                "fallback_method": "output_share_for_unmatched_suppliers",
                "notes": "",
            }
        ]
    )


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"price_code": "advanced_biodiesel", "price_label": "Advanced biodiesel", "price_value": 100.0, "price_unit": "EUR/MWh_LHV", "toe_per_t": np.nan, "eur_per_mwh_lhv": 100.0, "status": "test", "applies_to": "", "source_invariant": False, "notes": "", "conversion_note": ""},
            {"price_code": "conventional_biodiesel", "price_label": "Conventional biodiesel", "price_value": 50.0, "price_unit": "EUR/MWh_LHV", "toe_per_t": np.nan, "eur_per_mwh_lhv": 50.0, "status": "placeholder_to_replace", "applies_to": "", "source_invariant": False, "notes": "", "conversion_note": ""},
        ]
    )


def _dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "nrg_bal_c", "dimension": "nrg_bal", "code": "NRG_PR_E", "label": "Production", "description": "Production", "provenance_source": "Eurostat SDMX metadata:nrg_bal_c:nrg_bal"},
            {"dataset": "nrg_bal_c", "dimension": "siec", "code": "R5220P", "label": "Pure biodiesels", "description": "Pure biodiesels", "provenance_source": "Eurostat SDMX metadata:nrg_bal_c:siec"},
            {"dataset": "nrg_bal_c", "dimension": "siec", "code": "R5220B", "label": "Blended biodiesels", "description": "Blended biodiesels", "provenance_source": "Eurostat SDMX metadata:nrg_bal_c:siec"},
        ]
    )


def _quantity_table(total_gwh: float = 200.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"child_code": "EB_C20_ADVANCED_BIODIESEL", "parent_code": "CPA_C20", "subsplit_group": "C20_BIODIESEL", "eb_product_selector": "R5220P|R5220B", "eb_flow_selector": "nrg_bal_c:NRG_PR_E", "quantity_GWh": total_gwh, "selector_total_GWh": total_gwh, "quantity_source": "test", "notes": ""},
            {"child_code": "EB_C20_CONVENTIONAL_BIODIESEL", "parent_code": "CPA_C20", "subsplit_group": "C20_BIODIESEL", "eb_product_selector": "R5220P|R5220B", "eb_flow_selector": "nrg_bal_c:NRG_PR_E", "quantity_GWh": total_gwh, "selector_total_GWh": total_gwh, "quantity_source": "test", "notes": ""},
        ]
    )


def test_dictionary_validation_rejects_non_official_siec_code() -> None:
    split = _split_frame()
    split.loc[0, "eb_product_selector"] = "NOT_A_REAL_SIEC"
    with pytest.raises(ValueError, match="non-official SIEC"):
        validate_split_against_energy_balance_dictionary(split, _dictionary_frame())


def test_dictionary_validation_rejects_non_official_flow_code() -> None:
    split = _split_frame()
    split.loc[0, "eb_flow_selector"] = "nrg_bal_c:NOT_A_REAL_FLOW"
    with pytest.raises(ValueError, match="non-official nrg_bal"):
        validate_split_against_energy_balance_dictionary(split, _dictionary_frame())


def test_explicit_flow_selector_rejects_vague_legacy_value() -> None:
    split = _split_frame()
    split.loc[0, "eb_flow_selector"] = "production_or_available_energy"
    with pytest.raises(ValueError, match="Unsupported eb_flow_selector"):
        validate_energy_balance_pxp_inputs(
            split_frame=split,
            reaggregation_frame=_reagg_frame(),
            technology_frame=_tech_frame(),
            prices_frame=_prices_frame(),
            energy_balance_dictionary_frame=None,
        )


def test_rejects_cpa_c10_12_parent() -> None:
    split = _split_frame()
    split.loc[0, "parent_code"] = "CPA_C10-12"
    with pytest.raises(ValueError, match="CPA_C10-12"):
        validate_energy_balance_pxp_inputs(
            split_frame=split,
            reaggregation_frame=_reagg_frame(),
            technology_frame=_tech_frame(),
            prices_frame=_prices_frame(),
        )


def test_component_values_include_selector_and_parent_residuals() -> None:
    resolved = build_price_resolution(_prices_frame())
    components, shares, selector_residuals, parent_residuals, _missing = build_energy_balance_component_values(
        _split_frame(),
        resolved_prices=resolved,
        quantity_lookup=_quantity_table(200.0),
        parent_output_by_code={"CPA_C20": 1000.0},
    )
    assert set(components["child_code"]) == {
        "EB_C20_ADVANCED_BIODIESEL",
        "EB_C20_CONVENTIONAL_BIODIESEL",
        "EB_C20_OTHERS",
    }
    assert not selector_residuals.empty
    assert not parent_residuals.empty
    assert np.isclose(float(shares["child_share"].sum()), 1.0)
    advanced_q = float(components.loc[components["child_code"].eq("EB_C20_ADVANCED_BIODIESEL"), "quantity_GWh"].iloc[0])
    # 0.01 Mtoe = 116.3 GWh, capped by selector total 200 GWh.
    assert np.isclose(advanced_q, 116.3)


def test_compile_and_run_split_uses_existing_disaggregation_api() -> None:
    Z = pd.DataFrame([[100.0]], index=["CPA_C20"], columns=["CPA_C20"])
    Y = pd.DataFrame([[900.0]], index=["CPA_C20"], columns=["FD"])
    plan = compile_energy_balance_pxp_split_plan(
        Z=Z,
        Y=Y,
        split_frame=_split_frame(),
        reaggregation_frame=_reagg_frame(),
        technology_frame=_tech_frame(),
        prices_frame=_prices_frame(),
        energy_balance_dictionary_frame=_dictionary_frame(),
        geo="AT",
        year=2021,
        quantity_table_override=_quantity_table(200.0),
    )
    assert "CPA_C20" in plan.split_profiles
    assert "EB_C20_OTHERS" in plan.split_profiles["CPA_C20"].subsectors
    result = run_energy_balance_pxp_split(Z, Y, plan, balancing_scope="local", create_io_system=False)
    assert "EB_C20_OTHERS" in result.Z.index
    assert np.isclose(float(result.x.sum()), 1000.0)


def test_no_legacy_split_iot_wrapper() -> None:
    import ecolecon_mrio.energy_balance_pxp as mod

    assert not hasattr(mod, "split_iot_by_energy_balances")


def test_dictionary_loader_requires_provenance(tmp_path: Path) -> None:
    bad = pd.DataFrame(
        [
            {"dataset": "nrg_bal_c", "dimension": "siec", "code": "R5220P", "label": "Pure biodiesels", "description": "Pure biodiesels", "provenance_source": ""}
        ]
    )
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="blank required values"):
        load_energy_balance_dictionary_frame(path)


def test_build_quantity_lookup_uses_explicit_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(**kwargs):
        assert kwargs["dataset"] == "nrg_bal_c"
        assert kwargs["nrg_bal"] == ["NRG_PR_E"]
        return pd.DataFrame([[10.0, 20.0]], index=["NRG_PR_E"], columns=["R5220P", "R5220B"]), {"dimensions": {}}

    monkeypatch.setattr("ecolecon_mrio.energy_balance_pxp.load_eurostat_energy_balance_slice", fake_loader)
    q = build_energy_balance_pxp_quantity_lookup(split_frame=_split_frame(), geo="AT", year=2021)
    assert set(q["child_code"]) == {"EB_C20_ADVANCED_BIODIESEL", "EB_C20_CONVENTIONAL_BIODIESEL"}
    assert np.isclose(float(q["quantity_GWh"].iloc[0]), 30.0)
