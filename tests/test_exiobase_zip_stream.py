from __future__ import annotations

from pathlib import Path
import zipfile

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




def test_load_mrio_iot_matrices_validates_Z_row_order(monkeypatch, tmp_path):
    import ecolecon_mrio.data as data_mod

    year = 2089
    zip_path = _fake_zip_z_only(tmp_path, year, swapped_z_rows=True)
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="Z.txt row labels do not match"):
        data_mod.load_mrio_iot_matrices(
            "exiobase", year, matrices=("Z", "x", "Y"),
            data_root=tmp_path,
            allow_download_missing=False,
            verbose=False,
        )


def test_load_mrio_exiobase_full_iosystem_path_is_disabled():
    from ecolecon_mrio.data import load_mrio

    with pytest.raises(RuntimeError, match="Full EXIOBASE IOSystem loading"):
        load_mrio("exiobase", 2012, allow_download_missing=False)


def test_exiobase_iot_loader_has_no_delete_source_zip_compatibility_kwarg(tmp_path):
    import inspect
    import ecolecon_mrio.data as data_mod

    assert "delete_source_zip_after_parse" not in inspect.signature(
        data_mod.load_mrio_iot_matrices
    ).parameters
    with pytest.raises(TypeError, match="delete_source_zip_after_parse"):
        data_mod.load_mrio_iot_matrices(
            "exiobase", 2087, matrices=("x",),
            data_root=tmp_path,
            allow_download_missing=False,
            delete_source_zip_after_parse=True,
        )


def _write_integrity_core_zip(
    tmp_path: Path,
    year: int,
    *,
    x_value: str = "10",
    z_value: str = "1",
    y_values: str = "7\t1",
) -> Path:
    """Write a one-sector source zip for fail-closed monetary parsing tests."""
    data_dir = tmp_path / f"exiobase3_{year}"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"IOT_{year}_pxp.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("x.txt", f"region\tsector\tindout\nAT\ta\t{x_value}\n")
        zf.writestr(
            "Z.txt",
            "region\t\tAT\nsector\t\ta\nregion\tsector\t\n"
            f"AT\ta\t{z_value}\n",
        )
        zf.writestr(
            "Y.txt",
            "region\t\tAT\tAT\n"
            "category\t\tFinal consumption expenditure by households\tExports: Total (fob)\n"
            "region\tsector\t\t\n"
            f"AT\ta\t{y_values}\n",
        )
    return zip_path


def test_zip_stream_rejects_blank_x_instead_of_zero(monkeypatch, tmp_path: Path) -> None:
    """Blank EXIO monetary output must fail before coefficient construction."""
    import ecolecon_mrio.data as data_mod

    year = 2086
    zip_path = _write_integrity_core_zip(tmp_path, year, x_value="")
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="x.txt contains blank"):
        data_mod.load_mrio_iot_matrices("exiobase", year, matrices=("Z", "x", "Y"), data_root=tmp_path, allow_download_missing=False)


def test_zip_stream_rejects_nonfinite_Z(monkeypatch, tmp_path: Path) -> None:
    """Non-finite EXIO Z source values must never enter sparse A derivation."""
    import ecolecon_mrio.data as data_mod

    year = 2085
    zip_path = _write_integrity_core_zip(tmp_path, year, z_value="nan")
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="Z.txt contains non-finite"):
        data_mod.load_mrio_iot_matrices("exiobase", year, matrices=("Z", "x", "Y"), data_root=tmp_path, allow_download_missing=False)


def test_zip_stream_rejects_truncated_Y_row(monkeypatch, tmp_path: Path) -> None:
    """A short EXIO Y row is missing monetary evidence, not an implicit zero tail."""
    import ecolecon_mrio.data as data_mod

    year = 2084
    zip_path = _write_integrity_core_zip(tmp_path, year, y_values="7")
    monkeypatch.setattr(data_mod, "ensure_exiobase_year_zip", lambda **kwargs: zip_path)
    with pytest.raises(ValueError, match="Y.txt row width mismatch"):
        data_mod.load_mrio_iot_matrices("exiobase", year, matrices=("Z", "x", "Y"), data_root=tmp_path, allow_download_missing=False)
