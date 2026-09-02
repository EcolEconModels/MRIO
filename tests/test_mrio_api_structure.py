from __future__ import annotations

import ast
from pathlib import Path

import inspect

import ecolecon_mrio.data as data
import ecolecon_mrio.exiobase3 as exiobase3
import ecolecon_mrio.region_aggregation as region_aggregation


ROOT = Path(__file__).resolve().parents[1]


def _top_level_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def test_data_api_definitions_are_unique() -> None:
    names = _top_level_names(ROOT / "src/ecolecon_mrio/data.py")
    for name in (
        "load_mrio",
        "load_mrio_iot_matrices",
        "derive_A_from_Z_and_x",
        "derive_nonexport_final_demand_from_Y",
        "_normalize_iot_matrix_requests",
        "_validate_same_iot_basis",
    ):
        assert names.count(name) == 1, name


def test_scope_api_definitions_are_unique() -> None:
    names = _top_level_names(ROOT / "src/ecolecon_mrio/region_aggregation.py")
    for name in ("MrioScopeSlice", "RegionScope", "RegionAggregationPlan", "load_mrio_iot_scope_slice"):
        assert names.count(name) == 1, name


def test_deleted_loader_surfaces_are_absent() -> None:
    assert not hasattr(data, "load_mrio_matrices")
    assert not hasattr(data, "load_mrio_matrix_result")
    assert not hasattr(data, "MrioMatrixResult")
    assert not hasattr(exiobase3, "load_exiobase3_scope_slice")
    assert not hasattr(exiobase3, "load_exiobase3_region_slice")
    assert not hasattr(exiobase3, "load_exiobase3_region_slices")
    assert not hasattr(exiobase3, "ExioRegionSlice")
    assert not hasattr(region_aggregation, "MrioSource")
    assert not hasattr(region_aggregation, "load_mrio_scope_slice")
    assert callable(region_aggregation.load_mrio_iot_scope_slice)


def test_generic_full_loader_has_no_dead_exio_cache_knobs() -> None:
    params = inspect.signature(data.load_mrio).parameters
    for name in (
        "delete_source_zip_after_parse",
        "exio_full_cache",
        "exio_save_text_cache",
        "exio_system",
    ):
        assert name not in params
