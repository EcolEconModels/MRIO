from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio import exiobase3
from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI


def _write_tiny_exio_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr(
            "x.txt",
            "region\tsector\tindout\n"
            "AT\tA\t100\n"
            "AT\tB\t200\n"
            "DE\tA\t300\n"
            "DE\tB\t400\n",
        )
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\tDE\tDE\n"
            "\t\tA\tB\tA\tB\n"
            "\t\tunit\tunit\tunit\tunit\n"
            "AT\tA\t1\t2\t3\t4\n"
            "AT\tB\t5\t6\t7\t8\n"
            "DE\tA\t9\t10\t11\t12\n"
            "DE\tB\t13\t14\t15\t16\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\tAT\tDE\n"
            "\t\tHFCE\tGFCF\tHFCE\n"
            "\t\tunit\tunit\tunit\n"
            "AT\tA\t100\t1\t9\n"
            "AT\tB\t200\t2\t8\n"
            "DE\tA\t300\t3\t7\n"
            "DE\tB\t400\t4\t6\n",
        )


def _prepare_cache_root(tmp_path: Path) -> Path:
    cache_root = tmp_path
    data_dir = cache_root / "exiobase3_2020"
    _write_tiny_exio_zip(data_dir / "IOT_2020_pxp.zip")
    (data_dir / "download_log.json").write_text(
        json.dumps({"version": DEFAULT_EXIOBASE3_DOI}) + "\n",
        encoding="utf-8",
    )
    return cache_root


@pytest.fixture(autouse=True)
def _identity_sector_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exiobase3, "_exio_label_to_code_lookup", lambda system: {"A": "CPA_A", "B": "CPA_B"})


def test_region_slice_streams_domestic_block_without_cache_or_full_mrio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    def fail_full_mrio(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("load_exiobase3/full MRIO loader must not be called")

    monkeypatch.setattr(exiobase3, "load_exiobase3", fail_full_mrio)

    out = exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        flow="domestic",
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert list(out.sectors) == ["CPA_A", "CPA_B"]
    pd.testing.assert_frame_equal(
        out.Z_domestic,
        pd.DataFrame(
            [[1.0, 2.0], [5.0, 6.0]],
            index=pd.Index(["CPA_A", "CPA_B"], name="sector"),
            columns=pd.Index(["CPA_A", "CPA_B"]),
        ),
    )
    pd.testing.assert_frame_equal(
        out.Y_region,
        pd.DataFrame(
            [[100.0, 1.0], [200.0, 2.0]],
            index=pd.Index(["CPA_A", "CPA_B"], name="sector"),
            columns=pd.Index(["HFCE", "GFCF"]),
        ),
    )
    pd.testing.assert_series_equal(
        out.x_region,
        pd.Series([100.0, 200.0], index=pd.Index(["CPA_A", "CPA_B"], name="sector"), name="x_region"),
    )
    assert not (cache_root / "exiobase3_2020" / "slices").exists()


def test_region_slice_streams_import_and_total_flows(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    imports = exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        flow="imports",
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    total = exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        flow="total",
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )

    assert imports.Z_domestic.loc["CPA_A", "CPA_A"] == 9.0
    assert imports.Z_domestic.loc["CPA_B", "CPA_B"] == 14.0
    assert imports.Y_region.loc["CPA_A", "HFCE"] == 300.0
    assert total.Z_domestic.loc["CPA_A", "CPA_A"] == 10.0
    assert total.Z_domestic.loc["CPA_B", "CPA_B"] == 20.0
    assert total.Y_region.loc["CPA_A", "GFCF"] == 4.0


def test_region_slice_optional_cache_writes_and_reuses_valid_metadata(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)

    first = exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        flow="domestic",
        cache_region_slice=True,
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    slice_dir = cache_root / "exiobase3_2020" / "slices" / "AT" / "pxp"
    assert (slice_dir / "Z_domestic.pkl").exists()
    assert (slice_dir / "Y_region.pkl").exists()
    assert (slice_dir / "x_region.pkl").exists()
    meta = json.loads((slice_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["exio_doi"] == DEFAULT_EXIOBASE3_DOI
    assert meta["flow"] == "domestic"

    second = exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        flow="domestic",
        cache_region_slice=True,
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    pd.testing.assert_frame_equal(first.Z_domestic, second.Z_domestic)
    pd.testing.assert_frame_equal(first.Y_region, second.Y_region)
    pd.testing.assert_series_equal(first.x_region, second.x_region)


def test_region_slice_optional_cache_rejects_stale_metadata(tmp_path: Path) -> None:
    cache_root = _prepare_cache_root(tmp_path)
    exiobase3.load_exiobase3_region_slice(
        year=2020,
        region="AT",
        system="pxp",
        cache_region_slice=True,
        cache_root=cache_root,
        doi=DEFAULT_EXIOBASE3_DOI,
    )
    meta_path = cache_root / "exiobase3_2020" / "slices" / "AT" / "pxp" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["exio_doi"] = "10.5281/zenodo.3583070"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata does not match"):
        exiobase3.load_exiobase3_region_slice(
            year=2020,
            region="AT",
            system="pxp",
            cache_region_slice=True,
            cache_root=cache_root,
            doi=DEFAULT_EXIOBASE3_DOI,
        )


def _write_region_integrity_zip(
    path: Path, *, x_b: str = "200", y_b_row: str = "AT\tB\t20\n"
) -> None:
    """Write a two-sector domestic slice for monetary source-integrity tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.txt", f"region\tsector\tindout\nAT\tA\t100\nAT\tB\t{x_b}\n")
        zf.writestr(
            "Z.txt",
            "\t\tAT\tAT\n\t\tA\tB\n\t\tunit\tunit\n"
            "AT\tA\t1\t2\nAT\tB\t3\t4\n",
        )
        zf.writestr(
            "Y.txt",
            "\t\tAT\n\t\tHFCE\n\t\tunit\n"
            "AT\tA\t10\n"
            + y_b_row,
        )


def test_region_slice_rejects_blank_output_value(tmp_path: Path) -> None:
    """The fuels region loader must not turn a blank producer output into zero."""
    source = tmp_path / "blank_x.zip"
    _write_region_integrity_zip(source, x_b="")
    with pytest.raises(ValueError, match="x.txt contains blank"):
        exiobase3._load_exio_region_slice_direct_from_zip(
            year=2020, system="pxp", region="AT", source_zip=source, flow="domestic"
        )


def test_region_slice_rejects_missing_Y_sector_row(tmp_path: Path) -> None:
    """Z/Y source row bases must match instead of filling a missing Y row with zero."""
    source = tmp_path / "missing_y_row.zip"
    _write_region_integrity_zip(source, y_b_row="")
    with pytest.raises(ValueError, match="Y.txt row basis does not match Z.txt"):
        exiobase3._load_exio_region_slice_direct_from_zip(
            year=2020, system="pxp", region="AT", source_zip=source, flow="domestic"
        )


def test_region_slice_rejects_nonfinite_selected_Y_value(tmp_path: Path) -> None:
    """A non-finite destination-use value cannot enter fuels donor profiles."""
    source = tmp_path / "nonfinite_y.zip"
    _write_region_integrity_zip(source, y_b_row="AT\tB\tnan\n")
    with pytest.raises(ValueError, match="Y.txt contains non-finite monetary values"):
        exiobase3._load_exio_region_slice_direct_from_zip(
            year=2020, system="pxp", region="AT", source_zip=source, flow="domestic"
        )
