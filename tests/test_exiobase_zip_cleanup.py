from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ecolecon_mrio.data import load_mrio


class _FakeExioMrio:
    def __init__(self, complete_cache: bool) -> None:
        self.Y = pd.DataFrame({"FD": [1.0, 2.0]}, index=["a", "b"])
        self.G = None
        self._complete_cache = complete_cache

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._complete_cache:
            (path / "file_parameters.json").write_text("{}", encoding="utf-8")

    def calc_all(self, include_ghosh: bool = True) -> None:  # pragma: no cover - defensive
        self.G = pd.DataFrame([[1.0]])


def _setup_zip(tmp_path: Path, year: int) -> Path:
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    zip_path.write_bytes(b"fake-zip")
    return zip_path


def _exio_named_cache_paths(tmp_path: Path, year: int, system: str = "pxp") -> tuple[Path, Path]:
    data_dir = tmp_path / f"exiobase3_{year}"
    tag = f"{year}_{system}"
    return data_dir / f"mrio_system_{tag}", data_dir / f"mrio_system_{tag}.fast.pkl"


def test_load_mrio_exiobase_deletes_source_zip_after_successful_parse(monkeypatch, tmp_path):
    year = 2098
    zip_path = _setup_zip(tmp_path, year)

    import ecolecon_mrio.data as data_mod

    monkeypatch.setattr(
        data_mod,
        "ensure_exiobase_year_zip",
        lambda **kwargs: zip_path,
    )
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: _FakeExioMrio(complete_cache=True))

    mrio, y = load_mrio(
        "exiobase",
        year,
        data_root=tmp_path,
        calc_all=False,
        allow_download_missing=False,
        delete_source_zip_after_parse=True,
        verbose=False,
    )

    assert isinstance(mrio.Y, pd.DataFrame)
    assert list(y.index) == ["a", "b"]
    assert not zip_path.exists()


def test_load_mrio_exiobase_keeps_zip_if_cache_incomplete(monkeypatch, tmp_path):
    year = 2097
    zip_path = _setup_zip(tmp_path, year)

    import ecolecon_mrio.data as data_mod

    monkeypatch.setattr(
        data_mod,
        "ensure_exiobase_year_zip",
        lambda **kwargs: zip_path,
    )
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: _FakeExioMrio(complete_cache=False))

    _mrio, _y = load_mrio(
        "exiobase",
        year,
        data_root=tmp_path,
        calc_all=False,
        allow_download_missing=False,
        delete_source_zip_after_parse=True,
        exio_save_text_cache=True,
        verbose=False,
    )

    assert zip_path.exists()


def test_load_mrio_exiobase_prefers_fast_cache_when_present(monkeypatch, tmp_path):
    year = 2096
    cache_dir, fast_path = _exio_named_cache_paths(tmp_path, year, system="pxp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "file_parameters.json").write_text("{}", encoding="utf-8")

    cached_obj = _FakeExioMrio(complete_cache=True)
    fast_path.write_bytes(pickle.dumps(cached_obj, protocol=pickle.HIGHEST_PROTOCOL))

    import ecolecon_mrio.data as data_mod

    def _should_not_run(_path):
        raise AssertionError("pymrio.load should not be called when fast cache exists")

    monkeypatch.setattr(data_mod.pymrio, "load", _should_not_run)

    mrio, y = load_mrio(
        "exiobase",
        year,
        data_root=tmp_path,
        calc_all=False,
        allow_download_missing=False,
        verbose=False,
    )

    assert isinstance(mrio.Y, pd.DataFrame)
    assert list(y.index) == ["a", "b"]


def test_load_mrio_exiobase_writes_fast_cache_after_text_load(monkeypatch, tmp_path):
    year = 2095
    cache_dir, fast_path = _exio_named_cache_paths(tmp_path, year, system="pxp")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "file_parameters.json").write_text("{}", encoding="utf-8")
    assert not fast_path.exists()

    import ecolecon_mrio.data as data_mod

    monkeypatch.setattr(data_mod.pymrio, "load", lambda _path: _FakeExioMrio(complete_cache=True))

    _mrio, _y = load_mrio(
        "exiobase",
        year,
        data_root=tmp_path,
        calc_all=False,
        allow_download_missing=False,
        verbose=False,
    )

    assert fast_path.exists()
    loaded = pickle.loads(fast_path.read_bytes())
    assert isinstance(loaded.Y, pd.DataFrame)


def test_load_mrio_exiobase_default_parse_writes_fast_only(monkeypatch, tmp_path):
    year = 2094
    zip_path = _setup_zip(tmp_path, year)

    import ecolecon_mrio.data as data_mod

    monkeypatch.setattr(
        data_mod,
        "ensure_exiobase_year_zip",
        lambda **kwargs: zip_path,
    )
    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", lambda _path: _FakeExioMrio(complete_cache=True))

    cache_dir, fast_path = _exio_named_cache_paths(tmp_path, year, system="pxp")

    _mrio, _y = load_mrio(
        "exiobase",
        year,
        data_root=tmp_path,
        calc_all=False,
        allow_download_missing=False,
        verbose=False,
    )

    assert fast_path.exists()
    assert not cache_dir.exists()
    assert zip_path.exists()
