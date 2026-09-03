from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import ecolecon_mrio.data as data


def _zip(tmp_path: Path, year: int) -> Path:
    path = tmp_path / f"IOT_{year}_pxp.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("file_parameters.json", "{}")
        zf.writestr("x.txt", "region\tsector\tindout\nAT\ta\t10\nAT\tb\t20\n")
        zf.writestr("Z.txt", "region\t\tAT\tAT\nsector\t\ta\tb\nregion\tsector\t\t\nAT\ta\t1\t2\nAT\tb\t3\t4\n")
        zf.writestr("Y.txt", "region\t\tAT\tAT\ncategory\t\tHFCE\tExports: Total (fob)\nregion\tsector\t\t\nAT\ta\t7\t1\nAT\tb\t8\t2\n")
    return path


def test_selective_exio_iot_loader_and_explicit_derivations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    year = 2091
    path = _zip(tmp_path, year)
    monkeypatch.setattr(data, "ensure_exiobase_year_zip", lambda **_kwargs: path)
    matrices, basis, meta = data.load_mrio_iot_matrices("exiobase", year, matrices=("Z", "x", "Y"), data_root=tmp_path, allow_download_missing=False)
    assert tuple(matrices) == ("Z", "x", "Y")
    assert isinstance(matrices["Z"], csr_matrix)
    assert isinstance(matrices["x"], pd.Series)
    assert isinstance(matrices["Y"], pd.DataFrame)
    A = data.derive_A_from_Z_and_x(matrices["Z"], matrices["x"], basis_index=basis)
    np.testing.assert_allclose(A.toarray(), [[0.1, 0.1], [0.3, 0.2]])
    y = data.derive_nonexport_final_demand_from_Y(matrices["Y"])
    np.testing.assert_allclose(y.to_numpy(), [7.0, 8.0])
    assert meta["requested_matrices"] == ["Z", "x", "Y"]


def test_selective_loader_does_not_require_x_for_Y_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    year = 2092
    path = _zip(tmp_path, year)
    monkeypatch.setattr(data, "ensure_exiobase_year_zip", lambda **_kwargs: path)
    monkeypatch.setattr(data, "_read_exio_x_from_zip", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("x.txt must not be read")))
    matrices, basis, _ = data.load_mrio_iot_matrices("exiobase", year, matrices=("Y",), data_root=tmp_path, allow_download_missing=False)
    assert list(matrices) == ["Y"]
    assert list(basis.names) == ["region", "sector"]


def test_obsolete_matrix_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="allowed"):
        data._normalize_iot_matrix_requests(("A",))
    with pytest.raises(ValueError, match="allowed"):
        data._normalize_iot_matrix_requests(("y",))



def test_derive_A_default_does_not_mutate_Z() -> None:
    basis = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    Z = csr_matrix([[1.0, 2.0], [3.0, 4.0]])
    before = Z.copy()
    x = pd.Series([10.0, 20.0], index=basis, name="x")
    _ = data.derive_A_from_Z_and_x(Z, x, basis_index=basis)
    np.testing.assert_array_equal(Z.indptr, before.indptr)
    np.testing.assert_array_equal(Z.indices, before.indices)
    np.testing.assert_allclose(
        cast(Any, np.asarray(Z.data)), cast(Any, np.asarray(before.data))
    )


def test_full_object_adapter_calculates_requested_missing_x() -> None:
    import pymrio

    matrices, basis, metadata = data.load_mrio_iot_matrices(
        "test", 2010, matrices=("Z", "x", "Y")
    )
    assert isinstance(matrices["x"], pd.Series)
    reference = pymrio.load_test()
    reference.calc_system(include_ghosh=False)
    assert reference.x is not None
    expected = cast(pd.DataFrame, reference.x).iloc[:, 0].astype("float64")
    expected.index = basis
    pd.testing.assert_series_equal(
        matrices["x"], expected.rename("x"), check_names=True
    )
    assert metadata["loader"] == "pymrio_full_object_adapter"

def test_derive_A_rejects_zero_output_with_nonzero_buyer_inputs() -> None:
    basis = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    Z = csr_matrix([[0.0, 2.0], [0.0, 0.0]])
    x = pd.Series([0.0, 0.0], index=basis, name="x")
    with pytest.raises(ValueError, match="zero output with non-zero intermediate inputs"):
        data.derive_A_from_Z_and_x(Z, x, basis_index=basis)


def test_derive_A_allows_zero_output_only_for_zero_buyer_column() -> None:
    basis = pd.MultiIndex.from_tuples([("AT", "a"), ("AT", "b")], names=["region", "sector"])
    Z = csr_matrix([[0.0, 2.0], [0.0, 4.0]])
    x = pd.Series([0.0, 20.0], index=basis, name="x")
    A = data.derive_A_from_Z_and_x(Z, x, basis_index=basis)
    np.testing.assert_allclose(A.toarray(), [[0.0, 0.1], [0.0, 0.2]])


def test_derive_A_rejects_negative_output() -> None:
    basis = pd.MultiIndex.from_tuples([("AT", "a")], names=["region", "sector"])
    with pytest.raises(ValueError, match="negative"):
        data.derive_A_from_Z_and_x(csr_matrix([[0.0]]), pd.Series([-1.0], index=basis), basis_index=basis)


def test_nonexport_final_demand_rejects_nan_instead_of_skipping_it() -> None:
    idx = pd.MultiIndex.from_tuples([("AT", "a")], names=["region", "sector"])
    cols = pd.MultiIndex.from_tuples([("AT", "HFCE"), ("AT", "Exports")], names=["region", "category"])
    Y = pd.DataFrame([[np.nan, 1.0]], index=idx, columns=cols)
    with pytest.raises(ValueError, match="non-finite"):
        data.derive_nonexport_final_demand_from_Y(Y)


def test_non_exio_adapter_rejects_nonfinite_Y(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.MultiIndex.from_tuples([("AT", "a")], names=["region", "sector"])
    fake: Any = type("FakeMrio", (), {})()
    fake.Y = pd.DataFrame([[np.nan]], index=idx, columns=pd.Index(["FD"]))
    fake.Z = pd.DataFrame([[0.0]], index=idx, columns=idx)
    fake.x = pd.DataFrame([[1.0]], index=idx, columns=pd.Index(["indout"]))
    monkeypatch.setattr(data, "load_mrio", lambda *_a, **_k: fake)
    with pytest.raises(ValueError, match="non-finite final-demand"):
        data.load_mrio_iot_matrices("test", 2010, matrices=("Y",))


# --- Stage-2 non-EXIO x-shape regression ---
def test_non_exio_adapter_rejects_multicolumn_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider cannot silently choose the first of several incompatible output columns."""
    idx = pd.MultiIndex.from_tuples([("AT", "a")], names=["region", "sector"])
    fake: Any = type("FakeMrio", (), {})()
    fake.x = pd.DataFrame([[1.0, 2.0]], index=idx, columns=pd.Index(["x1", "x2"]))
    monkeypatch.setattr(data, "load_mrio", lambda *_a, **_k: fake)
    with pytest.raises(ValueError, match="exactly one output column"):
        data.load_mrio_iot_matrices("test", 2010, matrices=("x",))
