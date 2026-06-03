from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest


def _fake_matrix_zip(tmp_path: Path, year: int) -> Path:
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    a_header = "region		AT	AT
sector		a	b
region	sector		
"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("x.txt", "region	sector	indout
AT	a	10
AT	b	20
")
        zf.writestr("A.txt", a_header + "AT	a	0.1	0
AT	b	0.2	0.3
")
        zf.writestr(
            "Y.txt",
            "region		AT	AT
"
            "category		Final consumption expenditure by households	Exports: Total (fob)
"
            "region	sector		
"
            "AT	a	7	1
"
            "AT	b	8	2
",
        )
    return zip_path


def _fake_matrix_zip_with_swapped_a_rows(tmp_path: Path, year: int) -> Path:
    zip_path = _fake_matrix_zip(tmp_path, year)
    a_header = "region		AT	AT
sector		a	b
region	sector		
"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("x.txt", "region	sector	indout
AT	a	10
AT	b	20
")
        zf.writestr("A.txt", a_header + "AT	b	0.2	0.3
AT	a	0.1	0
")
        zf.writestr(
            "Y.txt",
            "region		AT	AT
category		Final consumption expenditure by households	Exports: Total (fob)
region	sector		
AT	a	7	1
AT	b	8	2
",
        )
    return zip_path


def test_resolve_exiobase_cache_paths_zip_only(tmp_path):
    from ecolecon_mrio.data import resolve_exiobase_cache_paths

    paths = resolve_exiobase_cache_paths(2021, system="pxp", data_root=tmp_path)
    assert set(paths) == {"data_dir", "zip"}
    assert paths["zip"].name == "IOT_2021_pxp.zip"


def test_load_mrio_matrices_streams_a_x_y_from_zip(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2090
    zip_path = _fake_matrix_zip(tmp_path, year)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    monkeypatch.setattr(
        data_mod.pymrio,
        "parse_exiobase3",
        lambda _path: (_ for _ in ()).throw(AssertionError("parse_exiobase3 should not run in zip-stream mode")),
    )

    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        cache_matrices=False,
        read_existing_cache=False,
        verbose=False,
    )

    expected_index = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    pd.testing.assert_index_equal(index, expected_index)
    assert A.shape == (2, 2)
    assert A.nnz == 3
    pd.testing.assert_series_equal(x, pd.Series([10.0, 20.0], index=expected_index, name="x"))
    pd.testing.assert_series_equal(y, pd.Series([7.0, 8.0], index=expected_index, name="y"))
    assert metadata["cache_status"] == "streamed_direct_from_zip"
    assert metadata["zip_only_cache"] is True
    assert not (tmp_path / f"exiobase3_{year}" / f"matrix_cache_{year}_pxp").exists()


def test_load_mrio_matrices_rejects_derived_cache_writes(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2088
    zip_path = _fake_matrix_zip(tmp_path, year)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="matrix cache writes are disabled"):
        data_mod.load_mrio_matrices(year, data_root=tmp_path, cache_matrices=True)


def test_load_mrio_matrices_direct_parser_validates_a_row_order(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2089
    zip_path = _fake_matrix_zip_with_swapped_a_rows(tmp_path, year)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)

    with pytest.raises(ValueError, match="A.txt row labels do not match"):
        data_mod.load_mrio_matrices(
            year,
            data_root=tmp_path,
            allow_download_missing=False,
            cache_matrices=False,
            read_existing_cache=False,
            verbose=False,
        )
