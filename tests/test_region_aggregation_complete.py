from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pymrio
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio.data import derive_A_from_Z_and_x
from ecolecon_mrio.region_aggregation import (
    RegionAggregationPlan,
    aggregate_mrio_regions,
    compile_region_aggregation,
)


def _synthetic_bundle() -> tuple[
    dict[str, csr_matrix | pd.Series | pd.DataFrame], pd.MultiIndex
]:
    regions = ("AT", "DE", "US")
    sectors = ("A", "B")
    categories = ("HH", "GOV")
    basis = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )
    y_columns = pd.MultiIndex.from_product(
        [regions, categories], names=["region", "category"]
    )
    Z = csr_matrix(np.arange(36, dtype="float64").reshape(6, 6))
    x = pd.Series(
        [10.0, 0.0, 30.0, 0.0, 50.0, 60.0],
        index=basis,
        name="x",
        dtype="float64",
    )
    # Signed values ensure geography aggregation does not silently clamp inventory-like
    # final-demand entries or otherwise reinterpret extensive source accounts.
    Y = pd.DataFrame(
        np.arange(36, dtype="float64").reshape(6, 6) - 8.0,
        index=basis,
        columns=y_columns,
        dtype="float64",
    )
    return {"Z": Z, "x": x, "Y": Y}, basis


def _eu_row_plan() -> RegionAggregationPlan:
    mapping = pd.DataFrame(
        [
            ["EU", "EU test", "AT", 1, True, ""],
            ["EU", "EU test", "DE", 2, True, ""],
        ],
        columns=[
            "scope_code",
            "scope_label",
            "member_region",
            "member_order",
            "enabled",
            "notes",
        ],
    )
    return compile_region_aggregation(
        source_regions=("AT", "DE", "US"),
        mapping=mapping,
        scopes=("EU",),
        include_complement=True,
        complement_code="RoW",
    )


def test_complete_region_aggregation_matches_explicit_extensive_operators() -> None:
    matrices, basis = _synthetic_bundle()
    source_Z = matrices["Z"]
    source_x = matrices["x"]
    source_Y = matrices["Y"]
    assert isinstance(source_Z, csr_matrix)
    assert isinstance(source_x, pd.Series)
    assert isinstance(source_Y, pd.DataFrame)
    Z_before = source_Z.copy()
    x_before = source_x.copy()
    Y_before = source_Y.copy()

    out, out_basis, metadata = aggregate_mrio_regions(
        matrices,
        basis_index=basis,
        plan=_eu_row_plan(),
        metadata={"dataset": "synthetic", "year": 2020},
    )

    expected_basis = pd.MultiIndex.from_product(
        [("EU", "RoW"), ("A", "B")], names=["region", "sector"]
    )
    expected_Y_columns = pd.MultiIndex.from_product(
        [("EU", "RoW"), ("HH", "GOV")], names=["region", "category"]
    )
    assert out_basis.equals(expected_basis)
    out_Z = out["Z"]
    out_x = out["x"]
    out_Y = out["Y"]
    assert isinstance(out_Z, csr_matrix)
    assert isinstance(out_x, pd.Series)
    assert isinstance(out_Y, pd.DataFrame)
    assert out_Y.columns.equals(expected_Y_columns)
    assert set(out) == {"Z", "x", "Y"}

    assignment = {"AT": "EU", "DE": "EU", "US": "RoW"}
    target_lookup = {key: pos for pos, key in enumerate(expected_basis)}
    R = np.zeros((len(expected_basis), len(basis)), dtype="float64")
    for source_pos, (region, sector) in enumerate(basis):
        R[target_lookup[(assignment[str(region)], str(sector))], source_pos] = 1.0
    fd_lookup = {key: pos for pos, key in enumerate(expected_Y_columns)}
    H = np.zeros((len(expected_Y_columns), len(source_Y.columns)), dtype="float64")
    for source_pos, (region, category) in enumerate(source_Y.columns):
        H[fd_lookup[(assignment[str(region)], str(category))], source_pos] = 1.0

    np.testing.assert_allclose(out_Z.toarray(), R @ source_Z.toarray() @ R.T)
    np.testing.assert_allclose(out_x.to_numpy(), R @ source_x.to_numpy())
    np.testing.assert_allclose(out_Y.to_numpy(), R @ source_Y.to_numpy() @ H.T)
    assert metadata["dataset"] == "synthetic"
    region_meta = cast(dict[str, object], metadata["region_aggregation"])
    assert region_meta["target_regions"] == ["EU", "RoW"]

    # The transformation is non-mutating and A remains an explicit post-step.
    np.testing.assert_array_equal(source_Z.toarray(), Z_before.toarray())
    pd.testing.assert_series_equal(source_x, x_before)
    pd.testing.assert_frame_equal(source_Y, Y_before)
    A = derive_A_from_Z_and_x(out_Z, out_x, basis_index=out_basis)
    zero_output = out_x.to_numpy(dtype="float64") == 0.0
    assert zero_output.any()
    assert np.all(A.toarray()[:, zero_output] == 0.0)


def test_complete_region_aggregation_requires_complete_exact_primitive_bundle() -> None:
    matrices, basis = _synthetic_bundle()
    mapping = pd.DataFrame(
        [["EU", "EU test", "AT", 1, True, ""], ["EU", "EU test", "DE", 2, True, ""]],
        columns=["scope_code", "scope_label", "member_region", "member_order", "enabled", "notes"],
    )
    partial = compile_region_aggregation(
        source_regions=("AT", "DE", "US"), mapping=mapping, scopes=("EU",)
    )
    with pytest.raises(ValueError, match="every source region.*missing=.*US"):
        aggregate_mrio_regions(matrices, basis_index=basis, plan=partial)

    with pytest.raises(ValueError, match="requires exactly primitive Z, x, and Y"):
        aggregate_mrio_regions(
            {"Z": matrices["Z"], "x": matrices["x"]},
            basis_index=basis,
            plan=_eu_row_plan(),
        )
    with pytest.raises(ValueError, match="Derived A/y inputs are not accepted"):
        aggregate_mrio_regions(
            {**matrices, "A": matrices["Z"]},
            basis_index=basis,
            plan=_eu_row_plan(),
        )


def test_complete_region_aggregation_rejects_nonrectangular_source_axes() -> None:
    matrices, basis = _synthetic_bundle()
    bad_basis = pd.MultiIndex.from_tuples(
        [("AT", "A"), ("AT", "B"), ("DE", "B"), ("DE", "A"), ("US", "A"), ("US", "B")],
        names=["region", "sector"],
    )
    source_x = matrices["x"]
    source_Y = matrices["Y"]
    assert isinstance(source_x, pd.Series)
    assert isinstance(source_Y, pd.DataFrame)
    bad_x = pd.Series(source_x.to_numpy(), index=bad_basis, name="x")
    bad_Y = pd.DataFrame(
        source_Y.to_numpy(), index=bad_basis, columns=source_Y.columns
    )
    with pytest.raises(ValueError, match="same unique ordered sector basis"):
        aggregate_mrio_regions(
            {"Z": matrices["Z"], "x": bad_x, "Y": bad_Y},
            basis_index=bad_basis,
            plan=_eu_row_plan(),
        )

    bad_columns = pd.MultiIndex.from_tuples(
        [("AT", "HH"), ("AT", "GOV"), ("DE", "GOV"), ("DE", "HH"), ("US", "HH"), ("US", "GOV")],
        names=["region", "category"],
    )
    Y_bad_columns = pd.DataFrame(
        source_Y.to_numpy(), index=basis, columns=bad_columns
    )
    with pytest.raises(ValueError, match="same unique ordered final-demand category basis"):
        aggregate_mrio_regions(
            {"Z": matrices["Z"], "x": matrices["x"], "Y": Y_bad_columns},
            basis_index=basis,
            plan=_eu_row_plan(),
        )


def test_complete_region_aggregation_matches_pymrio_region_aggregation() -> None:
    io = pymrio.load_test()
    io.calc_all()
    assert isinstance(io.Z.index, pd.MultiIndex)
    assert isinstance(io.Y.columns, pd.MultiIndex)
    basis = io.Z.index.set_names(["region", "sector"])
    Y = io.Y.copy()
    Y.index = basis
    Y.columns = Y.columns.set_names(["region", "category"])
    x = io.x.iloc[:, 0].astype("float64").copy()
    x.index = basis
    matrices = {
        "Z": csr_matrix(io.Z.to_numpy(dtype="float64")),
        "x": x.rename("x"),
        "Y": Y.astype("float64"),
    }
    regions = tuple(map(str, io.get_regions()))
    split = max(1, len(regions) // 2)
    rows = []
    for pos, region in enumerate(regions):
        scope = "G1" if pos < split else "G2"
        rows.append([scope, scope, region, pos + 1, True, ""])
    mapping = pd.DataFrame(
        rows,
        columns=["scope_code", "scope_label", "member_region", "member_order", "enabled", "notes"],
    )
    plan = compile_region_aggregation(source_regions=regions, mapping=mapping)
    actual, actual_basis, _ = aggregate_mrio_regions(
        matrices, basis_index=basis, plan=plan
    )

    region_vector = ["G1" if pos < split else "G2" for pos in range(len(regions))]
    reference = io.aggregate(region_agg=region_vector, inplace=False)
    assert reference is not None
    actual_Z = actual["Z"]
    actual_x = actual["x"]
    actual_Y = actual["Y"]
    assert isinstance(actual_Z, csr_matrix)
    assert isinstance(actual_x, pd.Series)
    assert isinstance(actual_Y, pd.DataFrame)
    assert actual_basis.equals(reference.Z.index)
    assert actual_Y.columns.equals(reference.Y.columns)
    np.testing.assert_allclose(
        actual_Z.toarray(), reference.Z.to_numpy(dtype="float64"), rtol=1e-12, atol=1e-9
    )
    np.testing.assert_allclose(
        actual_Y.to_numpy(), reference.Y.to_numpy(dtype="float64"), rtol=1e-12, atol=1e-9
    )
    np.testing.assert_allclose(
        actual_x.to_numpy(), reference.x.iloc[:, 0].to_numpy(dtype="float64"), rtol=1e-12, atol=1e-9
    )





def test_complete_region_aggregation_refuses_to_overwrite_provenance() -> None:
    matrices, basis = _synthetic_bundle()
    with pytest.raises(ValueError, match="refuses to overwrite existing region_aggregation provenance"):
        aggregate_mrio_regions(
            matrices,
            basis_index=basis,
            plan=_eu_row_plan(),
            metadata={"region_aggregation": {"target_regions": ["old"]}},
        )

def test_aggregate_mrio_regions_is_public_unique_and_does_not_derive_A() -> None:
    from ecolecon_mrio import aggregate_mrio_regions as public_aggregate

    assert public_aggregate is aggregate_mrio_regions
    source = Path("src/ecolecon_mrio/region_aggregation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "aggregate_mrio_regions"
    ]
    assert len(matches) == 1
    called = {
        node.func.id
        for node in ast.walk(matches[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "derive_A_from_Z_and_x" not in called
    assert "technical_coefficients" not in called
