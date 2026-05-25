from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


class _FakeExioMrio:
    def __init__(self, *, with_x: bool = False, with_a: bool = False) -> None:
        index = pd.MultiIndex.from_tuples(
            [("AT", "a"), ("AT", "b")],
            names=["region", "sector"],
        )
        self.Z = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=index,
            columns=index,
            dtype=float,
        )
        self.Y = pd.DataFrame(
            [[10.0, 1.0], [20.0, 2.0]],
            index=index,
            columns=pd.MultiIndex.from_tuples(
                [("AT", "FD"), ("AT", "EXP")],
                names=["region", "fd"],
            ),
            dtype=float,
        )
        self.x = pd.DataFrame({"x": [0.0, 0.0]}, index=index) if with_x else None
        self.A = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=index,
            columns=index,
            dtype=float,
        ) if with_a else None


def _fake_zip(tmp_path: Path, year: int) -> Path:
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    zip_path.write_bytes(b"fake-zip")
    return zip_path


def _fake_matrix_zip(tmp_path: Path, year: int) -> Path:
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    z_header = "region\t\tAT\tAT\nsector\t\ta\tb\nregion\tsector\t\t\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("x.txt", "region\tsector\tindout\nAT\ta\t10\nAT\tb\t20\n")
        zf.writestr("A.txt", z_header + "AT\ta\t0.1\t0\nAT\tb\t0.2\t0.3\n")
        zf.writestr(
            "Y.txt",
            "region\t\tAT\tAT\n"
            "category\t\tFinal consumption expenditure by households\tExports: Total (fob)\n"
            "region\tsector\t\t\n"
            "AT\ta\t7\t1\n"
            "AT\tb\t8\t2\n",
        )
    return zip_path


def test_resolve_exiobase_matrix_cache_paths(tmp_path):
    from ecolecon_mrio.data import resolve_exiobase_matrix_cache_paths

    paths = resolve_exiobase_matrix_cache_paths(2021, system="pxp", data_root=tmp_path)

    assert paths["matrix_cache_dir"].name == "matrix_cache_2021_pxp"
    assert paths["Z"].name == "Z.npz"
    assert paths["x"].name == "x.pkl.gz"
    assert paths["y"].name == "y.pkl.gz"
    assert paths["index"].name == "index.csv.gz"
    assert paths["metadata"].name == "metadata.json"
    assert paths["zip"].name == "IOT_2021_pxp.zip"


def test_load_mrio_matrices_roundtrip_and_cache_hit(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2092
    zip_path = _fake_zip(tmp_path, year)
    fake_mrio = _FakeExioMrio(with_x=False, with_a=False)

    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    monkeypatch.setattr(data_mod, "_validate_exiobase_system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: fake_mrio)

    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        verbose=False,
    )

    assert A.shape == (2, 2)
    assert index.names == ["region", "sector"]
    pd.testing.assert_index_equal(index, fake_mrio.Z.index)
    pd.testing.assert_series_equal(x, pd.Series([14.0, 29.0], index=index, name="x"))
    pd.testing.assert_series_equal(y, pd.Series([10.0, 20.0], index=index, name="y"))
    assert metadata["cache_kind"] == "exiobase_matrix"
    assert metadata["cache_status"] == "rebuilt_returned_direct"
    assert (tmp_path / f"exiobase3_{year}" / f"matrix_cache_{year}_pxp" / "Z.npz").exists()
    assert not (tmp_path / f"exiobase3_{year}" / f"mrio_system_{year}_pxp.fast.pkl").exists()

    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: (_ for _ in ()).throw(AssertionError("parse should not run on matrix-cache hit")))
    A2, x2, y2, index2, metadata2 = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        verbose=False,
    )

    assert A2.shape == (2, 2)
    pd.testing.assert_series_equal(x2, x)
    pd.testing.assert_series_equal(y2, y)
    pd.testing.assert_index_equal(index2, index)
    assert metadata2["cache_status"] == "hit"


def test_load_mrio_matrices_rebuilds_incomplete_cache(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2091
    cache_paths = data_mod.resolve_exiobase_matrix_cache_paths(year, system="pxp", data_root=tmp_path)
    cache_paths["matrix_cache_dir"].mkdir(parents=True, exist_ok=True)
    (cache_paths["matrix_cache_dir"] / "metadata.json").write_text("{}", encoding="utf-8")

    zip_path = _fake_zip(tmp_path, year)
    parse_calls: list[int] = []

    def _fake_parse(_path):
        parse_calls.append(1)
        return _FakeExioMrio(with_x=True, with_a=True)

    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    monkeypatch.setattr(data_mod, "_validate_exiobase_system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", _fake_parse)

    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        verbose=False,
    )

    assert parse_calls == [1]
    assert A.shape == (2, 2)
    assert len(x) == 2
    assert len(y) == 2
    assert index.names == ["region", "sector"]
    assert metadata["cache_status"] == "rebuilt_returned_direct"


def test_load_mrio_matrices_can_parse_without_writing_matrix_cache(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2090
    zip_path = _fake_matrix_zip(tmp_path, year)

    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    monkeypatch.setattr(data_mod, "_validate_exiobase_system", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: (_ for _ in ()).throw(AssertionError("parse should not run in no-cache direct mode")))

    A, x, y, index, metadata = data_mod.load_mrio_matrices(
        year,
        data_root=tmp_path,
        allow_download_missing=False,
        cache_matrices=False,
        read_existing_cache=False,
        verbose=False,
    )

    assert A.shape == (2, 2)
    assert len(x) == 2
    assert len(y) == 2
    expected_index = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    pd.testing.assert_index_equal(index, expected_index)
    assert A.nnz == 3
    assert y.tolist() == [7.0, 8.0]
    assert metadata["cache_status"] == "parsed_direct_from_zip_no_cache_write"
    assert not (tmp_path / f"exiobase3_{year}" / f"matrix_cache_{year}_pxp" / "Z.npz").exists()
