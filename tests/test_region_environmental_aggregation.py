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

    with pytest.raises(ValueError, match="Target total-supply identity failed"):
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
            "denominator_role": ["target_import_supply"] * 2,
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
        target_import_supply=pd.Series([5.0, 0.0], index=["a", "b"]),
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
            "denominator_role": ["target_import_supply"], "extension": ["air"],
            "stressor": ["CO2"], "unit": ["kg"], "value": [0.0],
            "monetary_output": [0.0], "intensity": [np.nan],
        }
    )
    total = _combine_target_supply_extension_sidecars_direct(
        dom,
        imp,
        target_domestic_output=pd.Series([1.0], index=["a"]),
        target_import_supply=pd.Series([0.0], index=["a"]),
    )
    row = total.set_index("sector").loc["a"]
    assert float(row["monetary_output"]) == 1.0
    assert pd.isna(row["value"])
    assert pd.isna(row["intensity"])
