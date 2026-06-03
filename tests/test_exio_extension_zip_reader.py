from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.temporal import _read_exio_extension_table_from_zip


def _basis() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("R1", "S1"), ("R1", "S2"), ("R2", "S3")],
        names=["region", "sector"],
    )


def _write_zip(path: Path, content: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("air_emissions/F.txt", content)


def test_extension_reader_handles_two_label_rows_and_blank_numeric_cells(tmp_path: Path) -> None:
    zpath = tmp_path / "IOT_mock.zip"
    content = "\t\tR1\tR1\tR2\n\t\tS1\tS2\tS3\n\t\tkg\tkg\tkg\nCO2\tcombustion\t1\t\t3\nCH4\tprocess\t\t2\t\n"
    _write_zip(zpath, content)

    frame = _read_exio_extension_table_from_zip(zpath, "air_emissions.F", _basis())

    assert list(frame.index) == ["CO2 | combustion", "CH4 | process"]
    assert frame.shape == (2, 3)
    assert frame.iloc[0].tolist() == [1.0, 0.0, 3.0]
    assert frame.iloc[1].tolist() == [0.0, 2.0, 0.0]
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_extension_reader_handles_one_label_data_rows_with_two_label_headers(tmp_path: Path) -> None:
    zpath = tmp_path / "IOT_mock.zip"
    # This reproduces the real failure mode: the header has two leading fields,
    # but data rows have only one label before the numeric vector.  The old
    # np.fromstring(split(..., 2)) parser consumed the first numeric value as a
    # second label and reported one fewer data column.
    content = "\t\tR1\tR1\tR2\n\t\tS1\tS2\tS3\n\t\tkg\tkg\tkg\nCO2 combustion\t1\t\t3\n"
    _write_zip(zpath, content)

    frame = _read_exio_extension_table_from_zip(zpath, "air_emissions.F", _basis())

    assert list(frame.index) == ["CO2 combustion"]
    assert frame.iloc[0].tolist() == [1.0, 0.0, 3.0]


def test_extension_reader_still_fails_on_true_width_mismatch(tmp_path: Path) -> None:
    zpath = tmp_path / "IOT_mock.zip"
    content = "\t\tR1\tR1\tR2\n\t\tS1\tS2\tS3\n\t\tkg\tkg\tkg\nCO2\tcombustion\t1\t2\n"
    _write_zip(zpath, content)

    with pytest.raises(ValueError, match="tab-field width mismatch|data-field width mismatch"):
        _read_exio_extension_table_from_zip(zpath, "air_emissions.F", _basis())
