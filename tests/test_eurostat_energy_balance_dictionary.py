from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio import eurostat_energy_balances as eb


def _fake_loader(*, dataset, nrg_bal, siec, **kwargs):
    if dataset == eb.DEFAULT_ENERGY_BALANCE_DATASET:
        dims = {
            "nrg_bal": {
                "NRG_PR_E": "Production",
                "FC_TRA_ROAD_E": "Final consumption - transport - road",
            },
            "siec": {
                "TOTAL": "Total",
                "R5220P": "Pure biodiesels",
                "R5220B": "Blended biodiesels",
            },
        }
    elif dataset == eb.DEFAULT_ELECTRICITY_HEAT_MIX_DATASET:
        dims = {
            "nrg_bal": {
                "GEP": "Gross electricity production",
                "GHP": "Gross heat production",
            },
            "siec": {
                "TOTAL": "Total",
                "N900H": "Nuclear heat",
                "RA300": "Wind",
            },
        }
    else:  # pragma: no cover
        raise AssertionError(dataset)
    return pd.DataFrame([[0.0]], index=["dummy"], columns=["dummy"]), {"dimensions": dims}


def test_build_eurostat_energy_balance_dictionary_from_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eb, "load_eurostat_energy_balance_slice", _fake_loader)

    out = eb.build_eurostat_energy_balance_dictionary(geo="AT", time=2021)

    assert list(out.columns) == ["dataset", "dimension", "code", "label", "description", "provenance_source"]
    assert not out.empty
    assert not out["code"].astype(str).str.strip().eq("").any()
    assert not out["label"].astype(str).str.strip().eq("").any()
    assert not out["description"].astype(str).str.strip().eq("").any()
    assert not out["provenance_source"].astype(str).str.strip().eq("").any()
    assert not out.duplicated(subset=["dataset", "dimension", "code"]).any()
    assert "R5220P" in set(out.loc[out["dimension"].eq("siec"), "code"])
    assert "GEP" in set(out.loc[out["dimension"].eq("nrg_bal"), "code"])


def test_write_eurostat_energy_balance_dictionary_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(eb, "load_eurostat_energy_balance_slice", _fake_loader)
    out_path = tmp_path / "eurostat_energy_balances_dictionary.csv"

    out = eb.write_eurostat_energy_balance_dictionary_csv(out_path, geo="AT", time=2021)
    reread = pd.read_csv(out_path, dtype=str)

    assert len(out) == len(reread)
    assert set(reread.columns) == {"dataset", "dimension", "code", "label", "description", "provenance_source"}


def test_generated_dictionary_rejects_empty_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_loader(**kwargs):
        return pd.DataFrame(), {"dimensions": {}}

    monkeypatch.setattr(eb, "load_eurostat_energy_balance_slice", empty_loader)

    with pytest.raises(ValueError, match="dictionary is empty"):
        eb.build_eurostat_energy_balance_dictionary(geo="AT", time=2021)
