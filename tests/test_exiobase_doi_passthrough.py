from __future__ import annotations

import ast
import inspect
from pathlib import Path

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
        inspect.signature(data.load_mrio_matrices).parameters["doi"].default
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
