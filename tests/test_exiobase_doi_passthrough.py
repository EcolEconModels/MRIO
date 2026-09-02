from __future__ import annotations

import ast
import inspect
import zipfile
from pathlib import Path

import pytest
from ecolecon_mrio import data
from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI


EXPECTED_DOI = "10.5281/zenodo.20051562"


def test_default_exiobase_doi_is_new_zenodo_record() -> None:
    assert DEFAULT_EXIOBASE3_DOI == EXPECTED_DOI
    assert (
        inspect.signature(data.ensure_exiobase_year_zip).parameters["doi"].default
        == DEFAULT_EXIOBASE3_DOI
    )
    assert (
        inspect.signature(data.load_mrio_iot_matrices).parameters["doi"].default
        == DEFAULT_EXIOBASE3_DOI
    )


def test_exiobase_doi_url_normalizes_to_zenodo_record_fallback(monkeypatch) -> None:
    def fail_url_discovery(**_kwargs: object) -> object:
        raise RuntimeError("offline")

    monkeypatch.setattr(data, "_get_url_datafiles", fail_url_discovery)
    url = data._resolve_exiobase3_url(
        2020,
        system="pxp",
        doi="https://doi.org/10.5281/zenodo.20051562",
    )
    assert url == "https://zenodo.org/records/20051562/files/IOT_2020_pxp.zip"


def test_split_script_forwards_exio_doi_to_all_region_slice_calls() -> None:
    script = Path("scripts/split_eurostat_energy_by_exiobase.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_exiobase3_region_slice"
    ]
    assert calls
    missing = [node.lineno for node in calls if not any(kw.arg == "doi" for kw in node.keywords)]
    assert missing == []



def _write_minimal_exio_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")


def test_existing_exiobase_zip_requires_matching_download_log_doi(tmp_path: Path) -> None:
    data_dir = tmp_path / "exiobase3_2020"
    zip_path = data_dir / "IOT_2020_pxp.zip"
    _write_minimal_exio_zip(zip_path)
    (data_dir / "download_log.json").write_text(
        '{"version": "https://doi.org/10.5281/zenodo.20051562"}\n',
        encoding="utf-8",
    )

    out = data.ensure_exiobase_year_zip(
        2020,
        data_dir=data_dir,
        system="pxp",
        allow_download_missing=False,
        doi="10.5281/zenodo.20051562",
    )

    assert out == zip_path


def test_existing_exiobase_zip_rejects_mismatched_download_log_doi(tmp_path: Path) -> None:
    data_dir = tmp_path / "exiobase3_2020"
    zip_path = data_dir / "IOT_2020_pxp.zip"
    _write_minimal_exio_zip(zip_path)
    (data_dir / "download_log.json").write_text(
        '{"version": "10.5281/zenodo.3583070"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DOI/version mismatch"):
        data.ensure_exiobase_year_zip(
            2020,
            data_dir=data_dir,
            system="pxp",
            allow_download_missing=False,
            doi="10.5281/zenodo.20051562",
        )


def test_existing_exiobase_zip_rejects_missing_download_log_doi(tmp_path: Path) -> None:
    data_dir = tmp_path / "exiobase3_2020"
    zip_path = data_dir / "IOT_2020_pxp.zip"
    _write_minimal_exio_zip(zip_path)

    with pytest.raises(FileNotFoundError, match="no readable download_log.json DOI/version"):
        data.ensure_exiobase_year_zip(
            2020,
            data_dir=data_dir,
            system="pxp",
            allow_download_missing=False,
            doi="10.5281/zenodo.20051562",
        )


def test_split_script_has_no_standalone_exio_doi_tuple_assignments() -> None:
    script = Path("scripts/split_eurostat_energy_by_exiobase.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    bad_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "doi" for target in node.targets):
            continue
        value = node.value
        if not isinstance(value, ast.Tuple) or len(value.elts) != 1:
            continue
        elt = value.elts[0]
        if (
            isinstance(elt, ast.Attribute)
            and elt.attr == "exio_doi"
            and isinstance(elt.value, ast.Name)
            and elt.value.id == "args"
        ):
            bad_lines.append(node.lineno)
    assert bad_lines == []
