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

def test_scope_result_exposes_no_legacy_region_slice_aliases() -> None:
    for name in ("sectors", "system", "Z_domestic", "Y_region", "x_region"):
        assert name not in region_aggregation.MrioScopeSlice.__dict__, name

def test_flat_diagnostics_scope_loader_uses_current_api(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from scripts.diagnostics import _flat_diagnostics_common as helper

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_loader(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    region_mod = SimpleNamespace(
        RegionScope=region_aggregation.RegionScope,
        load_mrio_iot_scope_slice=fake_loader,
    )
    data_mod = SimpleNamespace(default_data_root=lambda: tmp_path)

    def fake_import_repo_module(_repo_root: Path, module_name: str):
        if module_name == "ecolecon_mrio.region_aggregation":
            return region_mod
        if module_name == "ecolecon_mrio.data":
            return data_mod
        raise AssertionError(module_name)

    monkeypatch.setattr(helper, "import_repo_module", fake_import_repo_module)
    result = helper.load_exio_slice(ROOT, region="AT", year=2020, flow_name="imports")

    assert result is sentinel
    assert captured["args"] == ("exiobase",)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs == {
        "year": 2020,
        "scope": region_aggregation.RegionScope(code="AT", members=("AT",)),
        "exio_system": "pxp",
        "flow": "external_imports",
        "data_root": tmp_path,
    }
