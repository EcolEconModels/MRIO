from __future__ import annotations

import pandas as pd

from ecolecon_mrio.disaggregation import SplitProfiles
from ecolecon_mrio.exio_share_split import (
    emitted_sector_to_exio_members_from_split_profiles,
    map_exio_emission_intensity,
    raw_exio_members_from_split_profiles,
)


def _one_child_profile() -> dict[str, SplitProfiles]:
    idx = pd.Index(["p24.d@CPA_C20"])
    share = pd.Series([1.0], index=idx, dtype="float64")
    return {
        "CPA_C20": SplitProfiles(
            aggregate_sector="CPA_C20",
            subsectors=("p24.d@CPA_C20",),
            output_share=share.copy(),
            intermediate_sales_share=share.copy(),
            intermediate_input_share=share.copy(),
            final_demand_share=share.copy(),
            subsector_to_exio_sectors={"p24.d@CPA_C20": ["p24.d"]},
        )
    }


def test_load_eurostat_iot_ignores_derived_parsed_cache(tmp_path, monkeypatch):
    from ecolecon_mrio import eurostat_iot

    base = tmp_path / "eurostat" / "naio_10_cp1700" / "request_hash"
    parsed_dir = base / "parsed"
    parsed_dir.mkdir(parents=True)
    raw_path = base / "raw.json"
    raw_path.write_text("{}", encoding="utf-8")

    # Stale legacy derived cache lacks V/M. load_eurostat_iot must not read it.
    pd.DataFrame([[999.0]], index=["stale"], columns=["stale"]).to_pickle(parsed_dir / "Z.pkl")
    pd.DataFrame([[999.0]], index=["stale"], columns=["P3"]).to_pickle(parsed_dir / "Y.pkl")
    (parsed_dir / "meta.json").write_text('{"stale": true}', encoding="utf-8")

    fresh = eurostat_iot.ParsedEurostatIOT(
        Z=pd.DataFrame([[1.0]], index=["CPA_A"], columns=["CPA_A"]),
        Y=pd.DataFrame([[2.0]], index=["CPA_A"], columns=["P3"]),
        V=pd.DataFrame([[3.0]], index=["B1G"], columns=["CPA_A"]),
        M=pd.DataFrame([[4.0]], index=["B1G"], columns=["P3"]),
        meta={"fresh": True},
    )

    def fake_download(*args, **kwargs):
        return eurostat_iot.EurostatDataset(
            dataset="naio_10_cp1700",
            filters={},
            request_url="memory://test",
            raw_path=raw_path,
        )

    def fake_read(path):
        assert path == raw_path
        return {"fresh_payload": True}

    def fake_parse(payload, *, row_dim, col_dim):
        assert payload == {"fresh_payload": True}
        assert row_dim == "prd_use"
        assert col_dim == "prd_uses"
        return fresh

    monkeypatch.setattr(eurostat_iot, "download_eurostat_dataset", fake_download)
    monkeypatch.setattr(eurostat_iot, "read_eurostat_raw", fake_read)
    monkeypatch.setattr(eurostat_iot, "parse_eurostat_iot", fake_parse)

    result = eurostat_iot.load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={"time": "2020"},
        row_dim="prd_use",
        col_dim="prd_uses",
        storage_folder=tmp_path,
    )

    pd.testing.assert_frame_equal(result.Z, fresh.Z)
    pd.testing.assert_frame_equal(result.Y, fresh.Y)
    assert result.V is not None
    assert result.M is not None
    pd.testing.assert_frame_equal(result.V, fresh.V)
    pd.testing.assert_frame_equal(result.M, fresh.M)
    assert result.meta == {"fresh": True}


def test_emitted_edge_mapping_and_raw_member_helper():
    profiles = _one_child_profile()
    assert raw_exio_members_from_split_profiles(profiles) == ["p24.d"]
    assert emitted_sector_to_exio_members_from_split_profiles(profiles) == {
        "CPA_C20": {"p24.d@CPA_C20": ["p24.d"]}
    }


def test_edge_child_co2_intensity_uses_raw_exio_member_and_edge_weight():
    profiles = _one_child_profile()
    emitted_mapping = emitted_sector_to_exio_members_from_split_profiles(profiles)
    exio_co2 = pd.DataFrame(
        [
            {
                "exio_sector": "p24.d",
                "extension": "air_emissions",
                "stressor": "CO2 - combustion - air",
                "unit": "kg",
                "emission": 100.0,
                "monetary_output": 200.0,
            }
        ]
    )

    out = map_exio_emission_intensity(
        exio_emissions_and_intensity=exio_co2,
        split_profiles=profiles,
        sector_to_exio_sectors=emitted_mapping,
        eurostat_sector_mapping={
            "CPA_C20": {"p24.d": 0.25},
            "CPA_C21": {"p24.d": 0.75},
        },
    )

    assert out is not None
    assert len(out) == 1
    row = out.iloc[0]
    assert row["disaggregated_sector"] == "p24.d@CPA_C20"
    assert row["emission"] == 25.0
    assert row["monetary_output"] == 50.0
    assert row["emission_intensity"] == 0.5


def test_split_child_edges_metadata_parses_materialized_edge_child():
    from scripts.split_eurostat_energy_by_exiobase import _build_split_child_edge_metadata

    meta = _build_split_child_edge_metadata(
        selected_children={"CPA_C20": ["p24.d@CPA_C20"]},
        sector_mapping={
            "CPA_C20": {"p24.d": 0.25},
            "CPA_C21": {"p24.d": 0.75},
        },
        variant_dim="stk_flow",
        variant_value="DOM",
        exio_flow="domestic",
    )

    assert len(meta) == 1
    row = meta.iloc[0]
    assert row["current_child_code"] == "p24.d@CPA_C20"
    assert row["proposed_edge_child_code"] == "p24.d@CPA_C20"
    assert row["source_code"] == "p24.d"
    assert row["concordance_target_code"] == "CPA_C20"
    assert row["source_n_targets"] == 2
    assert bool(row["is_many_target_source"]) is True
    assert row["edge_weight_current_reference"] == 0.25
    assert row["phase"] == "phase4_edge_output"
