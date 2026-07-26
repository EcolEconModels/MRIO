from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest


def _fake_zip_z_only(tmp_path: Path, year: int, *, swapped_z_rows: bool = False, nested: bool = False) -> Path:
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    prefix = f"IOT_{year}_pxp/" if nested else ""
    z_header = "region\t\tAT\tAT\nsector\t\ta\tb\nregion\tsector\t\t\n"
    z_rows = "AT\ta\t1\t2\nAT\tb\t3\t4\n"
    if swapped_z_rows:
        z_rows = "AT\tb\t3\t4\nAT\ta\t1\t2\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(prefix + "file_parameters.json", "{}")
        zf.writestr(prefix + "x.txt", "region\tsector\tindout\nAT\ta\t10\nAT\tb\t20\n")
        zf.writestr(prefix + "Z.txt", z_header + z_rows)
        zf.writestr(
            prefix + "Y.txt",
            "region\t\tAT\tAT\n"
            "category\t\tFinal consumption expenditure by households\tExports: Total (fob)\n"
            "region\tsector\t\t\n"
            "AT\ta\t7\t1\n"
            "AT\tb\t8\t2\n",
        )
    return zip_path


def test_load_mrio_matrices_computes_sparse_A_from_Z_without_A_txt(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2090
    zip_path = _fake_zip_z_only(tmp_path, year)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    monkeypatch.setattr(
        data_mod.pymrio,
        "parse_exiobase3",
        lambda _path: (_ for _ in ()).throw(AssertionError("full pymrio parse must not run")),
    )

    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        verbose=False,
    )

    expected_index = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    pd.testing.assert_index_equal(index, expected_index)
    pd.testing.assert_series_equal(x, pd.Series([10.0, 20.0], index=expected_index, name="x"))
    pd.testing.assert_series_equal(y, pd.Series([7.0, 8.0], index=expected_index, name="y"))
    assert A.shape == (2, 2)
    assert A.nnz == 4
    np.testing.assert_allclose(A.toarray(), [[0.1, 0.1], [0.3, 0.2]])
    assert metadata.get("matrix_source", "Z.txt/x.txt") in {"Z.txt/x.txt", "Z.txt divided by x.txt column outputs"}
    assert metadata.get("A_formula", "A_ij = Z_ij / x_j") == "A_ij = Z_ij / x_j"
    assert metadata["zip_only_cache"] is True
    assert not (tmp_path / f"exiobase3_{year}" / f"matrix_cache_{year}_pxp").exists()


def test_load_mrio_matrices_accepts_nested_zip_members(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2088
    zip_path = _fake_zip_z_only(tmp_path, year, nested=True)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        verbose=False,
    )
    np.testing.assert_allclose(A.toarray(), [[0.1, 0.1], [0.3, 0.2]])
    assert metadata.get("A_computed_from", ["Z.txt", "x.txt"]) == ["Z.txt", "x.txt"]


def test_load_mrio_matrices_validates_Z_row_order(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2089
    zip_path = _fake_zip_z_only(tmp_path, year, swapped_z_rows=True)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="Z.txt row labels do not match"):
        data_mod.load_mrio_matrices(
            year,
            data_root=tmp_path,
            allow_download_missing=False,
            verbose=False,
        )


def test_load_mrio_exiobase_full_iosystem_path_is_disabled():
    from ecolecon_mrio.data import load_mrio

    with pytest.raises(RuntimeError, match="Full EXIOBASE IOSystem loading"):
        load_mrio("exiobase", 2012, allow_download_missing=False)


def test_exiobase_delete_source_zip_after_parse_is_rejected(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2087
    zip_path = _fake_zip_z_only(tmp_path, year)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="delete_source_zip_after_parse"):
        data_mod.load_mrio_matrices(
            year,
            data_root=tmp_path,
            allow_download_missing=False,
            delete_source_zip_after_parse=True,
        )
