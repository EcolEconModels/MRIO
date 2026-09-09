from __future__ import annotations

import ast
import inspect
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio import (
    aggregate_external_import_extension_direct,
    aggregate_scope_final_demand_extension,
    aggregate_scope_production_extension,
    combine_total_supply_extension_direct,
)
from ecolecon_mrio.exiobase3 import read_exiobase3_scope_external_import_use_by_origin
from ecolecon_mrio.region_aggregation import RegionScope


ACCOUNT_COLUMNS = [
    "axis_position",
    "axis_label",
    "boundary",
    "denominator_role",
    "extension",
    "stressor",
    "unit",
    "value",
    "monetary_denominator",
    "intensity",
]


def _production_vectors() -> tuple[pd.Series, pd.Series]:
    basis = pd.MultiIndex.from_product(
        [("AT", "DE", "US", "CN"), ("k1", "k2")],
        names=["region", "sector"],
    )
    F = pd.Series(
        [2.0, 8.0, 12.0, 8.0, 10.0, 20.0, 30.0, 10.0],
        index=basis,
        name="F",
        dtype="float64",
    )
    x = pd.Series(
        [20.0, 40.0, 60.0, 80.0, 100.0, 100.0, 100.0, 200.0],
        index=basis,
        name="x",
        dtype="float64",
    )
    return F, x


def _projection() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 0.5], [0.0, 0.5]],
        index=pd.Index(["a", "b"]),
        columns=pd.Index(["k1", "k2"]),
        dtype="float64",
    )


def test_scope_production_sums_extensive_accounts_before_projection_and_division() -> None:
    F, x = _production_vectors()
    out = aggregate_scope_production_extension(
        F,
        x,
        scope=RegionScope(code="EU", members=("AT", "DE")),
        extension="emissions",
        stressor="CO2",
        unit="kg",
        product_projection=_projection(),
    )
    assert list(out.columns) == ACCOUNT_COLUMNS
    assert out["boundary"].unique().tolist() == ["scope_production"]
    # Member sums: F=(14,16), x=(80,120); P then gives (22,8)/(140,60).
    np.testing.assert_allclose(out["value"], [22.0, 8.0])
    np.testing.assert_allclose(out["monetary_denominator"], [140.0, 60.0])
    np.testing.assert_allclose(out["intensity"], [22.0 / 140.0, 8.0 / 60.0])


def test_zero_output_production_donor_survives_retarget_and_reaggregation() -> None:
    """Zero-output donor evidence stays unavailable unless a coefficient exists."""
    from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions
    from ecolecon_mrio.region_aggregation import (
        _environmental_account_to_split_sidecar,
        _retarget_direct_environmental_account,
    )

    basis = pd.MultiIndex.from_product(
        [("AT",), ("k1", "k2")], names=["region", "sector"]
    )
    F = pd.Series([2.0, 8.0], index=basis, dtype="float64")
    x = pd.Series([0.0, 10.0], index=basis, dtype="float64")
    projection = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index(["a", "b"]),
        columns=pd.Index(["k1", "k2"]),
        dtype="float64",
    )

    donor = aggregate_scope_production_extension(
        F,
        x,
        scope=RegionScope(code="AT", members=("AT",)),
        extension="air",
        stressor="CH4",
        unit="kg",
        product_projection=projection,
    ).set_index("axis_label")
    assert float(donor.loc["a", "value"]) == 2.0
    assert float(donor.loc["a", "monetary_denominator"]) == 0.0
    assert pd.isna(donor.loc["a", "intensity"])
    assert float(donor.loc["b", "intensity"]) == pytest.approx(0.8)

    target = _retarget_direct_environmental_account(
        donor.reset_index().loc[:, ACCOUNT_COLUMNS],
        target_denominator=pd.Series([0.0, 20.0], index=["a", "b"]),
        boundary="scope_production",
        denominator_role="target_domestic_output",
    )
    target_by_sector = target.set_index("axis_label")
    assert pd.isna(target_by_sector.loc["a", "value"])
    assert pd.isna(target_by_sector.loc["a", "intensity"])
    assert float(target_by_sector.loc["a", "monetary_denominator"]) == 0.0
    assert float(target_by_sector.loc["b", "value"]) == pytest.approx(16.0)
    assert float(target_by_sector.loc["b", "intensity"]) == pytest.approx(0.8)

    zero_target = _retarget_direct_environmental_account(
        donor.reset_index().loc[:, ACCOUNT_COLUMNS],
        target_denominator=pd.Series([0.0, 0.0], index=["a", "b"]),
        boundary="scope_production",
        denominator_role="target_domestic_output",
    ).set_index("axis_label")
    assert pd.isna(zero_target.loc["a", "value"])
    assert float(zero_target.loc["b", "value"]) == 0.0
    assert pd.isna(zero_target.loc["b", "intensity"])

    with pytest.raises(
        ValueError, match="Positive target domestic output requires a defined mapped production donor coefficient"
    ):
        _retarget_direct_environmental_account(
            donor.reset_index().loc[:, ACCOUNT_COLUMNS],
            target_denominator=pd.Series([5.0, 20.0], index=["a", "b"]),
            boundary="scope_production",
            denominator_role="target_domestic_output",
        )

    sidecar = _environmental_account_to_split_sidecar(target)
    aggregated = aggregate_environmental_extensions(
        sidecar,
        {"a": "parent", "b": "parent"},
        output=pd.Series([0.0, 20.0], index=["a", "b"], dtype="float64"),
    )
    assert aggregated is not None
    parent = aggregated.set_index("sector").loc["parent"]
    assert float(parent["value"]) == pytest.approx(16.0)
    assert float(parent["monetary_output"]) == pytest.approx(20.0)
    assert float(parent["intensity"]) == pytest.approx(0.8)



def test_many_to_many_zero_output_production_donor_state_machine() -> None:
    from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions
    from ecolecon_mrio.region_aggregation import (
        _environmental_account_to_split_sidecar, _retarget_direct_environmental_account,
    )
    basis=pd.MultiIndex.from_tuples([("AT","k1"),("AT","k2")],names=["region","sector"])
    F=pd.Series([2.0,8.0],index=basis,dtype="float64"); x=pd.Series([0.0,10.0],index=basis,dtype="float64")
    projection=pd.DataFrame([[0.6,0.0],[0.4,1.0]],index=pd.Index(["a","b"]),columns=pd.Index(["k1","k2"]),dtype="float64")
    donor=aggregate_scope_production_extension(
        F,x,scope=RegionScope(code="AT",members=("AT",)),extension="air",stressor="CH4",unit="kg",
        product_projection=projection,
    ).set_index("axis_label")
    assert float(donor.loc["a","value"]) == pytest.approx(1.2)
    assert float(donor.loc["a","monetary_denominator"]) == 0.0 and pd.isna(donor.loc["a","intensity"])
    assert float(donor.loc["b","value"]) == pytest.approx(8.8)
    assert float(donor.loc["b","intensity"]) == pytest.approx(0.88)
    target=_retarget_direct_environmental_account(
        donor.reset_index().loc[:,ACCOUNT_COLUMNS],target_denominator=pd.Series([0.0,20.0],index=["a","b"]),
        boundary="scope_production",denominator_role="target_domestic_output",
    )
    tb=target.set_index("axis_label")
    assert pd.isna(tb.loc["a","value"]) and pd.isna(tb.loc["a","intensity"])
    assert float(tb.loc["b","value"]) == pytest.approx(17.6)
    agg=aggregate_environmental_extensions(
        _environmental_account_to_split_sidecar(target),{"a":"parent","b":"parent"},
        output=pd.Series([0.0,20.0],index=["a","b"],dtype="float64"),
    )
    assert agg is not None
    row=agg.set_index("sector").loc["parent"]
    assert float(row["value"]) == pytest.approx(17.6) and float(row["intensity"]) == pytest.approx(0.88)
    with pytest.raises(ValueError,match="Positive target domestic output requires a defined mapped production donor coefficient"):
        _retarget_direct_environmental_account(
            donor.reset_index().loc[:,ACCOUNT_COLUMNS],target_denominator=pd.Series([1.0,20.0],index=["a","b"]),
            boundary="scope_production",denominator_role="target_domestic_output",
        )


def test_projection_must_be_complete_nonnegative_partition() -> None:
    F, x = _production_vectors()
    bad = _projection()
    bad.loc["b", "k2"] = 0.4
    with pytest.raises(ValueError, match="weights summing to 1.0"):
        aggregate_scope_production_extension(
            F,
            x,
            scope=RegionScope(code="EU", members=("AT", "DE")),
            extension="emissions",
            stressor="CO2",
            unit="kg",
            product_projection=bad,
        )


def test_external_import_extension_retains_origin_through_many_many_projection() -> None:
    F, x = _production_vectors()
    import_index = pd.MultiIndex.from_product(
        [("US", "CN"), ("k1", "k2")], names=["region", "sector"]
    )
    # Signed finite use is preserved algebraically; no component is silently clamped.
    use = pd.Series([50.0, 10.0, 20.0, -5.0], index=import_index, dtype="float64")
    out = aggregate_external_import_extension_direct(
        F,
        x,
        use,
        scope=RegionScope(code="EU", members=("AT", "DE")),
        product_projection=_projection(),
        extension="emissions",
        stressor="CO2",
        unit="kg",
    )

    P = _projection().to_numpy(dtype="float64")
    expected_value = np.zeros(2)
    expected_use = np.zeros(2)
    for origin in ("US", "CN"):
        f_o = F.xs(origin, level="region").to_numpy(dtype="float64")
        x_o = x.xs(origin, level="region").to_numpy(dtype="float64")
        u_o = use.xs(origin, level="region").to_numpy(dtype="float64")
        f_bar = P @ f_o
        x_bar = P @ x_o
        u_bar = P @ u_o
        expected_value += (f_bar / x_bar) * u_bar
        expected_use += u_bar
    np.testing.assert_allclose(out["value"], expected_value)
    np.testing.assert_allclose(out["monetary_denominator"], expected_use)

    # Falsification: collapsing origins before F/x is a different calculation here.
    f_collapsed = P @ (
        F.xs("US", level="region").to_numpy()
        + F.xs("CN", level="region").to_numpy()
    )
    x_collapsed = P @ (
        x.xs("US", level="region").to_numpy()
        + x.xs("CN", level="region").to_numpy()
    )
    u_collapsed = P @ (
        use.xs("US", level="region").to_numpy()
        + use.xs("CN", level="region").to_numpy()
    )
    wrong = (f_collapsed / x_collapsed) * u_collapsed
    assert not np.allclose(out["value"].to_numpy(), wrong)


def test_external_import_extension_fails_when_used_mapped_output_is_undefined() -> None:
    F, x = _production_vectors()
    x_bad = x.copy()
    x_bad.loc[("US", slice(None))] = 0.0
    import_index = pd.MultiIndex.from_product(
        [("US", "CN"), ("k1", "k2")], names=["region", "sector"]
    )
    use = pd.Series([1.0, 0.0, 0.0, 0.0], index=import_index, dtype="float64")
    with pytest.raises(ValueError, match="nonzero mapped import use.*non-positive mapped production output"):
        aggregate_external_import_extension_direct(
            F,
            x_bad,
            use,
            scope=RegionScope(code="EU", members=("AT", "DE")),
            product_projection=_projection(),
            extension="emissions",
            stressor="CO2",
            unit="kg",
        )


def test_scope_final_demand_stays_on_category_axis_and_rejects_zero_denominator() -> None:
    index = pd.MultiIndex.from_product(
        [("AT", "DE", "US"), ("HH", "GOV")],
        names=["region", "final_demand_category"],
    )
    F_Y = pd.Series([1.0, 2.0, 3.0, 3.0, 9.0, 9.0], index=index, dtype="float64")
    denominator = pd.Series([100.0, 50.0], index=pd.Index(["HH", "GOV"]), dtype="float64")
    out = aggregate_scope_final_demand_extension(
        F_Y,
        denominator,
        scope=RegionScope(code="EU", members=("AT", "DE")),
        extension="emissions",
        stressor="direct CO2",
        unit="kg",
    )
    np.testing.assert_allclose(out["value"], [4.0, 5.0])
    np.testing.assert_allclose(out["intensity"], [0.04, 0.1])
    assert "product_projection" not in inspect.signature(
        aggregate_scope_final_demand_extension
    ).parameters

    with pytest.raises(ValueError, match="nonzero F_Y with zero monetary final demand"):
        aggregate_scope_final_demand_extension(
            F_Y,
            pd.Series([100.0, 0.0], index=pd.Index(["HH", "GOV"])),
            scope=RegionScope(code="EU", members=("AT", "DE")),
            extension="emissions",
            stressor="direct CO2",
            unit="kg",
        )


def test_total_supply_combines_coefficients_on_authoritative_target_controls() -> None:
    F, x = _production_vectors()
    scope = RegionScope(code="EU", members=("AT", "DE"))
    production = aggregate_scope_production_extension(
        F,
        x,
        scope=scope,
        extension="emissions",
        stressor="CO2",
        unit="kg",
        product_projection=_projection(),
    )
    import_index = pd.MultiIndex.from_product(
        [("US", "CN"), ("k1", "k2")], names=["region", "sector"]
    )
    imports = aggregate_external_import_extension_direct(
        F,
        x,
        pd.Series([50.0, 10.0, 20.0, 5.0], index=import_index, dtype="float64"),
        scope=scope,
        product_projection=_projection(),
        extension="emissions",
        stressor="CO2",
        unit="kg",
    )
    target_dom = pd.Series([1000.0, 500.0], index=pd.Index(["a", "b"]), dtype="float64")
    target_imp = pd.Series([200.0, 100.0], index=pd.Index(["a", "b"]), dtype="float64")
    target_total = target_dom + target_imp
    out = combine_total_supply_extension_direct(
        production_account=production,
        import_account=imports,
        target_domestic_output=target_dom,
        target_import_supply=target_imp,
        target_total_supply=target_total,
    )
    prod_e = production["intensity"].to_numpy(dtype="float64")
    imp_e = imports["intensity"].to_numpy(dtype="float64")
    expected = prod_e * target_dom.to_numpy() + imp_e * target_imp.to_numpy()
    np.testing.assert_allclose(out["value"], expected)
    np.testing.assert_allclose(out["monetary_denominator"], target_total.to_numpy())

    with pytest.raises(ValueError, match="Target total-use identity failed"):
        combine_total_supply_extension_direct(
            production_account=production,
            import_account=imports,
            target_domestic_output=target_dom,
            target_import_supply=target_imp,
            target_total_supply=target_total + 1.0,
        )


def _write_import_use_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\tDE\tDE\tUS\tUS\n"
            "\t\tk1\tk2\tk1\tk2\tk1\tk2\n"
            "\t\tu\tu\tu\tu\tu\tu\n"
            "AT\tk1\t0\t0\t0\t0\t0\t0\n"
            "AT\tk2\t0\t0\t0\t0\t0\t0\n"
            "DE\tk1\t0\t0\t0\t0\t0\t0\n"
            "DE\tk2\t0\t0\t0\t0\t0\t0\n"
            "US\tk1\t1\t2\t3\t4\t5\t6\n"
            "US\tk2\t10\t20\t30\t40\t50\t60\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tDE\tUS\n"
            "\t\tHH\tHH\tHH\n"
            "\t\tu\tu\tu\n"
            "AT\tk1\t0\t0\t0\n"
            "AT\tk2\t0\t0\t0\n"
            "DE\tk1\t0\t0\t0\n"
            "DE\tk2\t0\t0\t0\n"
            "US\tk1\t7\t8\t9\n"
            "US\tk2\t70\t80\t90\n",
        )


def test_exio_provider_retains_external_origin_product_until_wp3_weighting(tmp_path: Path) -> None:
    source = tmp_path / "tiny.zip"
    _write_import_use_zip(source)
    scope = RegionScope(code="EU", members=("AT", "DE"))
    use = read_exiobase3_scope_external_import_use_by_origin(
        source_zip=source, scope=scope
    )
    expected_index = pd.MultiIndex.from_tuples(
        [("US", "k1"), ("US", "k2")], names=["region", "sector"]
    )
    assert use.index.equals(expected_index)
    # Z into AT+DE: 1+2+3+4=10 and 10+20+30+40=100;
    # Y into AT+DE adds 15 and 150.
    np.testing.assert_allclose(use.to_numpy(), [25.0, 250.0])

    all_regions = read_exiobase3_scope_external_import_use_by_origin(
        source_zip=source,
        scope=RegionScope(code="ALL", members=("AT", "DE", "US")),
    )
    assert all_regions.empty
    assert isinstance(all_regions.index, pd.MultiIndex)
    assert list(all_regions.index.names) == ["region", "sector"]


def test_wp3_public_functions_are_unique_and_final_demand_has_no_product_projection() -> None:
    source = Path("src/ecolecon_mrio/region_aggregation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    expected = {
        "aggregate_scope_production_extension",
        "aggregate_external_import_extension_direct",
        "aggregate_scope_final_demand_extension",
        "combine_total_supply_extension_direct",
    }
    counts = {
        name: sum(
            isinstance(node, ast.FunctionDef) and node.name == name
            for node in tree.body
        )
        for name in expected
    }
    assert counts == {name: 1 for name in expected}

def test_target_supply_sidecar_combiner_preserves_unavailable_zero_mass() -> None:
    from ecolecon_mrio.region_aggregation import _combine_target_supply_extension_sidecars_direct

    dom = pd.DataFrame(
        {
            "sector": ["a", "b"],
            "boundary": ["scope_production"] * 2,
            "denominator_role": ["target_domestic_output"] * 2,
            "extension": ["air"] * 2,
            "stressor": ["CO2"] * 2,
            "unit": ["kg"] * 2,
            "value": [20.0, np.nan],
            "monetary_output": [10.0, 0.0],
            "intensity": [2.0, np.nan],
        }
    )
    imp = pd.DataFrame(
        {
            "sector": ["a", "b"],
            "boundary": ["external_imports_direct"] * 2,
            "denominator_role": ["target_import_use"] * 2,
            "extension": ["air"] * 2,
            "stressor": ["CO2"] * 2,
            "unit": ["kg"] * 2,
            "value": [15.0, np.nan],
            "monetary_output": [5.0, 0.0],
            "intensity": [3.0, np.nan],
        }
    )
    total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp,
        target_domestic_output=pd.Series([10.0, 0.0], index=["a", "b"]),
        target_import_use=pd.Series([5.0, 0.0], index=["a", "b"]),
    )
    row_a = total.set_index("sector").loc["a"]
    assert float(row_a["value"]) == pytest.approx(35.0)
    assert float(row_a["monetary_output"]) == pytest.approx(15.0)
    row_b = total.set_index("sector").loc["b"]
    assert pd.isna(row_b["value"])
    assert pd.isna(row_b["intensity"])
    assert float(row_b["monetary_output"]) == 0.0


def test_target_supply_sidecar_combiner_propagates_missing_positive_evidence() -> None:
    """Positive-mass unavailable donor evidence remains unavailable in TOTAL."""
    from ecolecon_mrio.region_aggregation import _combine_target_supply_extension_sidecars_direct

    dom = pd.DataFrame(
        {
            "sector": ["a"], "boundary": ["scope_production"],
            "denominator_role": ["target_domestic_output"], "extension": ["air"],
            "stressor": ["CO2"], "unit": ["kg"], "value": [np.nan],
            "monetary_output": [1.0], "intensity": [np.nan],
        }
    )
    imp = pd.DataFrame(
        {
            "sector": ["a"], "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"], "extension": ["air"],
            "stressor": ["CO2"], "unit": ["kg"], "value": [0.0],
            "monetary_output": [0.0], "intensity": [np.nan],
        }
    )
    total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp,
        target_domestic_output=pd.Series([1.0], index=["a"]),
        target_import_use=pd.Series([0.0], index=["a"]),
    )
    row = total.set_index("sector").loc["a"]
    assert float(row["monetary_output"]) == 1.0
    assert pd.isna(row["value"])
    assert pd.isna(row["intensity"])



def test_public_total_supply_propagates_unavailable_positive_import_evidence() -> None:
    """The public API must match target-sidecar NaN semantics for missing coefficients."""
    production = pd.DataFrame(
        {
            "axis_position": [1, 2],
            "axis_label": ["a", "b"],
            "boundary": ["scope_production", "scope_production"],
            "denominator_role": ["scope_production_output", "scope_production_output"],
            "extension": ["air", "air"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [10.0, 20.0],
            "monetary_denominator": [10.0, 10.0],
            "intensity": [1.0, 2.0],
        }
    )
    imports = production.copy()
    imports["boundary"] = "external_imports_direct"
    imports["denominator_role"] = "scope_external_import_use"
    imports.loc[0, ["value", "monetary_denominator", "intensity"]] = [0.0, 0.0, np.nan]
    imports.loc[1, ["value", "monetary_denominator", "intensity"]] = [5.0, 10.0, 0.5]

    out = combine_total_supply_extension_direct(
        production_account=production,
        import_account=imports,
        target_domestic_output=pd.Series([100.0, 100.0], index=["a", "b"]),
        target_import_supply=pd.Series([20.0, 20.0], index=["a", "b"]),
        target_total_supply=pd.Series([120.0, 120.0], index=["a", "b"]),
    )
    row_a = out.set_index("axis_label").loc["a"]
    row_b = out.set_index("axis_label").loc["b"]
    assert float(row_a["monetary_denominator"]) == 120.0
    assert pd.isna(row_a["value"])
    assert pd.isna(row_a["intensity"])
    assert float(row_b["value"]) == 210.0
    assert float(row_b["intensity"]) == 1.75


def test_public_environmental_account_rejects_infinite_intensity_but_preserves_zero_output_donor() -> None:
    production = pd.DataFrame(
        {
            "axis_position": [1],
            "axis_label": ["a"],
            "boundary": ["scope_production"],
            "denominator_role": ["scope_production_output"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [0.0],
            "monetary_denominator": [0.0],
            "intensity": [np.inf],
        }
    )
    imports = production.copy()
    imports["boundary"] = "external_imports_direct"
    imports["denominator_role"] = "scope_external_import_use"
    imports["intensity"] = np.nan
    with pytest.raises(ValueError, match="intensity must be missing at zero denominator"):
        combine_total_supply_extension_direct(
            production_account=production,
            import_account=imports,
            target_domestic_output=pd.Series([0.0], index=["a"]),
            target_import_supply=pd.Series([0.0], index=["a"]),
            target_total_supply=pd.Series([0.0], index=["a"]),
        )

    production["intensity"] = np.nan
    production["value"] = 1.0
    unavailable = combine_total_supply_extension_direct(
        production_account=production,
        import_account=imports,
        target_domestic_output=pd.Series([0.0], index=["a"]),
        target_import_supply=pd.Series([0.0], index=["a"]),
        target_total_supply=pd.Series([0.0], index=["a"]),
    ).iloc[0]
    assert float(unavailable["monetary_denominator"]) == 0.0
    assert pd.isna(unavailable["value"])
    assert pd.isna(unavailable["intensity"])

    imports.loc[0, ["value", "monetary_denominator", "intensity"]] = [5.0, 10.0, 0.5]
    import_only = combine_total_supply_extension_direct(
        production_account=production,
        import_account=imports,
        target_domestic_output=pd.Series([0.0], index=["a"]),
        target_import_supply=pd.Series([20.0], index=["a"]),
        target_total_supply=pd.Series([20.0], index=["a"]),
    ).iloc[0]
    assert float(import_only["value"]) == pytest.approx(10.0)
    assert float(import_only["intensity"]) == pytest.approx(0.5)

def test_retarget_direct_environmental_account_accepts_signed_import_use() -> None:
    from ecolecon_mrio.region_aggregation import _retarget_direct_environmental_account

    donor = pd.DataFrame(
        {
            "axis_position": [1], "axis_label": ["a"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["scope_external_import_use"],
            "extension": ["air"], "stressor": ["CO2"], "unit": ["kg"],
            "value": [5.0], "monetary_denominator": [10.0], "intensity": [0.5],
        }
    )
    out = _retarget_direct_environmental_account(
        donor, target_denominator=pd.Series([-20.0], index=["a"]),
        boundary="external_imports_direct", denominator_role="target_import_use",
        allow_signed_denominator=True,
    )
    row = out.iloc[0]
    assert float(row["value"]) == pytest.approx(-10.0)
    assert float(row["monetary_denominator"]) == pytest.approx(-20.0)
    assert float(row["intensity"]) == pytest.approx(0.5)

    donor.loc[0, ["value", "monetary_denominator", "intensity"]] = [5.0, 0.0, np.nan]
    unavailable = _retarget_direct_environmental_account(
        donor, target_denominator=pd.Series([20.0], index=["a"]),
        boundary="external_imports_direct", denominator_role="target_import_use",
        allow_signed_denominator=True,
    ).iloc[0]
    assert float(unavailable["monetary_denominator"]) == 20.0
    assert pd.isna(unavailable["value"]) and pd.isna(unavailable["intensity"])

    unavailable_zero_mass = _retarget_direct_environmental_account(
        donor, target_denominator=pd.Series([0.0], index=["a"]),
        boundary="external_imports_direct", denominator_role="target_import_use",
        allow_signed_denominator=True,
    ).iloc[0]
    assert float(unavailable_zero_mass["monetary_denominator"]) == 0.0
    assert pd.isna(unavailable_zero_mass["value"])
    assert pd.isna(unavailable_zero_mass["intensity"])

def test_public_total_combiner_accepts_signed_target_import_use() -> None:
    production = pd.DataFrame(
        {
            "axis_position": [1], "axis_label": ["a"], "boundary": ["scope_production"],
            "denominator_role": ["scope_production_output"], "extension": ["air"],
            "stressor": ["CO2"], "unit": ["kg"], "value": [20.0],
            "monetary_denominator": [10.0], "intensity": [2.0],
        }
    )
    imports = production.copy()
    imports["boundary"] = "external_imports_direct"
    imports["denominator_role"] = "scope_external_import_use"
    imports[["value", "monetary_denominator", "intensity"]] = [5.0, 10.0, 0.5]
    out = combine_total_supply_extension_direct(
        production_account=production, import_account=imports,
        target_domestic_output=pd.Series([100.0], index=["a"]),
        target_import_supply=pd.Series([-20.0], index=["a"]),
        target_total_supply=pd.Series([80.0], index=["a"]),
    ).iloc[0]
    assert out["boundary"] == "total_use_direct"
    assert float(out["value"]) == pytest.approx(190.0)

    imports.loc[0, ["value", "monetary_denominator", "intensity"]] = [5.0, 0.0, np.nan]
    unavailable = combine_total_supply_extension_direct(
        production_account=production, import_account=imports,
        target_domestic_output=pd.Series([100.0], index=["a"]),
        target_import_supply=pd.Series([20.0], index=["a"]),
        target_total_supply=pd.Series([120.0], index=["a"]),
    ).iloc[0]
    assert pd.isna(unavailable["value"]) and pd.isna(unavailable["intensity"])


def test_target_environmental_sidecar_includes_scope_final_demand_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DOM target artifact exposes already-aggregated F_Y beside production F."""
    from ecolecon_mrio.exiobase3_extensions import ExtensionRowResult
    from ecolecon_mrio import exiobase3 as exio_mod
    from ecolecon_mrio import exiobase3_extensions as ext_mod
    from ecolecon_mrio import region_aggregation as region_mod

    basis = pd.MultiIndex.from_tuples([("AT", "k1")], names=["region", "sector"])
    source_output = pd.Series([10.0], index=basis, dtype="float64", name="x")
    projection = pd.DataFrame(
        [[1.0]], index=pd.Index(["CPA_X"]), columns=pd.Index(["k1"]), dtype="float64"
    )

    def fake_rows(
        container,
        *,
        extension,
        kind="F",
        keywords=(),
        region=None,
        system="pxp",
        basis_index=None,
        require_unit=True,
    ):
        del container, keywords, region, system, basis_index, require_unit
        if kind == "F":
            values = pd.Series([2.0], index=basis, dtype="float64")
        else:
            values = pd.Series(
                [3.0, 4.0],
                index=pd.MultiIndex.from_tuples(
                    [("AT", "HH"), ("AT", "GOV")],
                    names=["region", "final_demand_category"],
                ),
                dtype="float64",
            )
        return (
            ExtensionRowResult(
                year=None,
                values=values,
                stressor="CO2",
                unit="kg",
                unit_source="test",
                extension=str(extension),
                kind=str(kind),
                metadata={},
            ),
        )

    monkeypatch.setattr(ext_mod, "load_exiobase3_extension_rows_with_units", fake_rows)
    monkeypatch.setattr(
        exio_mod,
        "read_exiobase3_scope_final_demand_totals",
        lambda **kwargs: pd.Series(
            [100.0, 50.0],
            index=pd.Index(["HH", "GOV"], name="final_demand_category"),
            dtype="float64",
        ),
    )

    target, raw = region_mod.build_scope_target_environmental_extensions(
        source_zip="unused.zip",
        scope=RegionScope(code="AT", members=("AT",)),
        flow="internal",
        source_output=source_output,
        source_basis=basis,
        product_projection=projection,
        target_denominator=pd.Series([20.0], index=pd.Index(["CPA_X"]), dtype="float64"),
        extension_requests={"air": ("CO2",)},
    )

    assert target["boundary"].tolist() == [
        "scope_production",
        "scope_final_demand_direct",
        "scope_final_demand_direct",
    ]
    fy = target.loc[target["boundary"].eq("scope_final_demand_direct")].reset_index(drop=True)
    assert fy["sector"].tolist() == ["HH", "GOV"]
    assert fy["denominator_role"].unique().tolist() == ["scope_final_demand"]
    np.testing.assert_allclose(fy["value"], [3.0, 4.0])
    np.testing.assert_allclose(fy["monetary_output"], [100.0, 50.0])
    assert raw is not None
    assert raw["kind"].tolist() == ["F", "F_Y", "F_Y"]
