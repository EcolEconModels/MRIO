from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

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
        verbose=False,
    )

    assert zip_path.exists()


def test_load_mrio_exiobase_timeout_writes_debug_snapshot(monkeypatch, tmp_path):
    year = 2096
    zip_path = _setup_zip(tmp_path, year)

    import ecolecon_mrio.data as data_mod

    if not hasattr(data_mod.signal, "setitimer"):
        pytest.skip("SIGALRM/setitimer unavailable on this platform")

    monkeypatch.setattr(
        data_mod,
        "ensure_exiobase_year_zip",
        lambda **kwargs: zip_path,
    )

    def _slow_parse(_path):
        time.sleep(0.05)
        return _FakeExioMrio(complete_cache=True)

    monkeypatch.setattr(data_mod.pymrio, "parse_exiobase3", _slow_parse)
    debug_dir = tmp_path / "debug_snapshots"
    monkeypatch.setenv("MRIO_EXIO_STAGE_TIMEOUT_S", "0.01")
    monkeypatch.setenv("MRIO_EXIO_DEBUG_DIR", str(debug_dir))

    with pytest.raises(RuntimeError, match="stage='parse_exiobase3'"):
        load_mrio(
            "exiobase",
            year,
            data_root=tmp_path,
            calc_all=False,
            allow_download_missing=False,
            delete_source_zip_after_parse=False,
            verbose=False,
        )

    snaps = sorted(debug_dir.glob("exio_load_debug_*.json"))
    assert snaps, "expected debug snapshot file on timeout"
    payload = json.loads(snaps[-1].read_text(encoding="utf-8"))
    assert payload["dataset"] == "exiobase"
    assert payload["year"] == year
    assert payload["stage"] == "parse_exiobase3"
    assert payload["event"] == "exception"
    assert "error" in payload
