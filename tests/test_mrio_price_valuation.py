"""Regression tests for optional common-EUR MRIO year-state valuation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio import mrio_extension_sda, mrio_sda


def _index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("AA", "s1"), ("BB", "s1")],
        names=["region", "sector"],
    )


def _state(*, valuation_basis: str, price_base_year: int | None) -> mrio_sda.MrioYearState:
    idx = _index()
    return mrio_sda.MrioYearState(
        year=2020,
        A=csr_matrix((2, 2), dtype=float),
        x=pd.Series([1.0, 1.0], index=idx),
        y=pd.Series([1.0, 1.0], index=idx),
        index=idx,
        monetary_unit="MEUR",
        monetary_unit_source="test",
        metadata={"valuation_basis": valuation_basis, "price_base_year": price_base_year},
    )


def test_load_year_state_adjusts_primitives_before_deriving_a_and_y(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ecolecon_mrio import data, exiobase_price_factors, price_adjustment

    idx = _index()
    raw_Z = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    raw_x = pd.Series([10.0, 20.0], index=idx, name="x")
    raw_Y = pd.DataFrame([[5.0], [6.0]], index=idx, columns=pd.Index(["fd"], name="category"))
    monkeypatch.setattr(
        data,
        "load_mrio_iot_matrices",
        lambda *args, **kwargs: ({"Z": raw_Z, "x": raw_x, "Y": raw_Y}, idx, {"fixture": True}),
    )
    seen: dict[str, Any] = {}

    def fake_derive_a(Z, x, *, basis_index, copy=False):
        del copy
        seen["Z"] = Z.toarray().copy()
        seen["x"] = x.to_numpy(dtype=float).copy()
        assert basis_index.equals(idx)
        return csr_matrix((2, 2), dtype=float)

    def fake_derive_y(Y):
        seen["Y"] = Y.to_numpy(dtype=float).copy()
        return Y.sum(axis=1).astype(float)

    monkeypatch.setattr(data, "derive_A_from_Z_and_x", fake_derive_a)
    monkeypatch.setattr(data, "derive_nonexport_final_demand_from_Y", fake_derive_y)
    monkeypatch.setattr(
        exiobase_price_factors,
        "build_exiobase3_eur_price_factors",
        lambda **kwargs: (
            pd.DataFrame(
                [
                    {"year": 2020, "exio_region": "AA", "price_adjustment_factor": 2.0},
                    {"year": 2020, "exio_region": "BB", "price_adjustment_factor": 3.0},
                ]
                + [
                    {"year": 2020, "exio_region": f"X{pos:02d}", "price_adjustment_factor": 1.0}
                    for pos in range(47)
                ]
            ),
            {"factor_table_sha256": "fixture-sha"},
        ),
    )
    monkeypatch.setattr(
        price_adjustment,
        "expand_region_price_factors",
        lambda basis_index, region_factors: pd.Series(
            [float(region_factors.loc["AA"]), float(region_factors.loc["BB"])],
            index=basis_index,
            name="price_adjustment_factor",
        ),
    )

    state = mrio_sda.load_mrio_year_state(
        2020,
        monetary_unit="MEUR",
        monetary_unit_source="test",
        price_base_year=2022,
        price_factor_storage_folder=tmp_path,
    )
    np.testing.assert_allclose(seen["Z"], np.array([[2.0, 4.0], [9.0, 12.0]]))
    np.testing.assert_allclose(seen["x"], np.array([20.0, 60.0]))
    np.testing.assert_allclose(seen["Y"], np.array([[10.0], [18.0]]))
    assert state.metadata["valuation_basis"] == "country_gdp_deflator_common_eur"
    assert state.metadata["price_base_year"] == 2022
    assert state.metadata["price_factor_table_sha256"] == "fixture-sha"


def test_default_year_state_remains_current_price(monkeypatch: pytest.MonkeyPatch) -> None:
    from ecolecon_mrio import data

    idx = _index()
    monkeypatch.setattr(
        data,
        "load_mrio_iot_matrices",
        lambda *args, **kwargs: (
            {
                "Z": csr_matrix((2, 2), dtype=float),
                "x": pd.Series([1.0, 2.0], index=idx),
                # Pandas accepts this literal column list; pandas-stubs rejects its Sequence protocol.
                "Y": pd.DataFrame([[1.0], [2.0]], index=idx, columns=["fd"]),  # pyright: ignore[reportArgumentType]
            },
            idx,
            {},
        ),
    )
    monkeypatch.setattr(data, "derive_A_from_Z_and_x", lambda Z, x, *, basis_index, copy=False: csr_matrix((2, 2), dtype=float))
    monkeypatch.setattr(data, "derive_nonexport_final_demand_from_Y", lambda Y: Y.sum(axis=1).astype(float))
    state = mrio_sda.load_mrio_year_state(2020, monetary_unit="MEUR", monetary_unit_source="test")
    assert state.metadata["valuation_basis"] == "current_price"
    assert state.metadata["price_base_year"] is None


def test_sda_state_pair_rejects_mixed_valuation_before_solving() -> None:
    raw = _state(valuation_basis="current_price", price_base_year=None)
    adjusted = _state(valuation_basis="country_gdp_deflator_common_eur", price_base_year=2022)
    with pytest.raises(ValueError, match="valuation bases do not match"):
        mrio_sda.compute_output_sda_pair_result_from_states(raw, adjusted)


def test_extension_loader_forwards_price_controls_only_to_monetary_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    idx = _index()
    monetary_state = _state(valuation_basis="country_gdp_deflator_common_eur", price_base_year=2022)
    calls: dict[str, Any] = {}

    def fake_mrio_state(year: int, **kwargs: Any):
        calls["mrio"] = (year, kwargs)
        return monetary_state

    monkeypatch.setattr(mrio_sda, "load_mrio_year_state", fake_mrio_state)
    extension_row = SimpleNamespace(values=pd.Series([2.0, 3.0], index=idx))

    def fake_extension_loader(year: int, **kwargs: Any):
        calls["extension"] = (year, kwargs)
        return extension_row

    monkeypatch.setattr(mrio_extension_sda, "load_exiobase3_extension_row_with_unit_for_year", fake_extension_loader)
    intensity_state = SimpleNamespace(name="intensity")
    monkeypatch.setattr(mrio_extension_sda, "compute_extension_intensity_state", lambda **kwargs: intensity_state)

    bundle = mrio_extension_sda.load_mrio_extension_year_state(
        2020,
        extension="air_emissions",
        price_base_year=2022,
        price_factor_storage_folder=tmp_path,
        price_factor_force_download=True,
        price_factor_timeout_s=77,
    )
    _, mrio_kwargs = calls["mrio"]
    assert mrio_kwargs["price_base_year"] == 2022
    assert mrio_kwargs["price_factor_storage_folder"] == tmp_path
    assert mrio_kwargs["price_factor_force_download"] is True
    assert mrio_kwargs["price_factor_timeout_s"] == 77
    _, extension_kwargs = calls["extension"]
    assert "price_base_year" not in extension_kwargs
    assert "price_factor_storage_folder" not in extension_kwargs
    assert bundle.extension_row is extension_row
    assert bundle.intensity_state is intensity_state
