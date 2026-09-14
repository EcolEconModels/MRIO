from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ecolecon_mrio.data as data


def _basis() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("AT", "p01"), ("DE", "p02")], names=["region", "sector"]
    )


def test_exiobase_alias_predicate_is_exact() -> None:
    assert data.is_exiobase_dataset("exiobase")
    assert data.is_exiobase_dataset(" EXIOBASE3 ")
    assert not data.is_exiobase_dataset("wiod16")


def test_data_root_override_is_resolved(tmp_path: Path) -> None:
    nested = tmp_path / "a" / ".." / "data"
    assert data.get_mrio_data_root(nested) == nested.resolve()


def test_source_input_paths_delegate_to_exiobase_cache_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = tmp_path / "IOT_2022_pxp.zip"

    def fake_paths(year: int, *, system: str, data_root: Path) -> dict[str, Path]:
        assert year == 2022
        assert system == "pxp"
        assert data_root == tmp_path.resolve()
        return {"data_dir": tmp_path, "zip": expected}

    monkeypatch.setattr(data, "resolve_exiobase_cache_paths", fake_paths)
    assert data.mrio_source_input_paths_for_year(
        dataset="exiobase3", year=2022, data_root=tmp_path, exio_system="pxp"
    ) == [expected]
    assert data.mrio_source_input_paths_for_year(
        dataset="wiod16", year=2014, data_root=tmp_path
    ) == []


def test_output_vector_uses_x_only_primitive_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    basis = _basis()
    expected = pd.Series([2.0, 3.0], index=basis, name="x")
    calls: list[tuple[str, int, dict[str, object]]] = []

    def fake_loader(dataset: str, year: int, **kwargs: object):
        calls.append((dataset, year, dict(kwargs)))
        return {"x": expected.copy()}, basis, {"dataset": "exiobase"}

    monkeypatch.setattr(data, "load_mrio_iot_matrices", fake_loader)
    out = data.load_mrio_output_vector(
        "exiobase3",
        2022,
        data_root=tmp_path,
        exio_system="pxp",
        exio_doi="10.5281/zenodo.test",
        allow_download_missing_exiobase_source_zips=False,
    )
    pd.testing.assert_series_equal(out, expected)
    assert len(calls) == 1
    dataset, year, kwargs = calls[0]
    assert dataset == "exiobase3" and year == 2022
    assert kwargs["matrices"] == ("x",)
    assert kwargs["allow_download_missing"] is False
    assert kwargs["exio_system"] == "pxp"
    assert kwargs["doi"] == "10.5281/zenodo.test"


def test_output_vector_non_exio_preserves_generic_loader_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """EXIO-only controls must not override another provider's loader defaults."""

    basis = _basis()
    expected = pd.Series([4.0, 5.0], index=basis, name="x")
    calls: list[tuple[str, int, dict[str, object]]] = []

    def fake_loader(dataset: str, year: int, **kwargs: object):
        calls.append((dataset, year, dict(kwargs)))
        return {"x": expected.copy()}, basis, {"dataset": dataset}

    monkeypatch.setattr(data, "load_mrio_iot_matrices", fake_loader)
    out = data.load_mrio_output_vector(
        "wiod16",
        2014,
        data_root=tmp_path,
        exio_system="ixi",
        exio_doi="ignored-for-non-exio",
        allow_download_missing_exiobase_source_zips=True,
        exio_force_download=True,
    )
    pd.testing.assert_series_equal(out, expected)
    assert len(calls) == 1
    dataset, year, kwargs = calls[0]
    assert dataset == "wiod16" and year == 2014
    assert kwargs == {
        "matrices": ("x",),
        "data_root": tmp_path.resolve(),
        "verbose": False,
    }


def test_index_is_the_x_only_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    basis = _basis()
    monkeypatch.setattr(
        data,
        "load_mrio_output_vector",
        lambda *args, **kwargs: pd.Series([1.0, 2.0], index=basis, name="x"),
    )
    assert data.load_mrio_index("exiobase", 2022).equals(basis)


def test_source_zip_refresh_is_explicit_and_retries_forcefully(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    result = tmp_path / "IOT_2022_pxp.zip"

    def fake_ensure(year: int, **kwargs: object) -> Path:
        assert year == 2022
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError("EXIOBASE DOI/version mismatch for cached source")
        return result

    monkeypatch.setattr(data, "ensure_exiobase_year_zip", fake_ensure)
    out = data.ensure_exiobase_source_zip(
        year=2022,
        data_root=tmp_path,
        exio_doi="10.5281/zenodo.test",
        refresh_mismatched=True,
    )
    assert out == result
    assert len(calls) == 2
    assert calls[0]["force_download"] is False
    assert calls[1]["force_download"] is True
    assert calls[1]["allow_download_missing"] is True
