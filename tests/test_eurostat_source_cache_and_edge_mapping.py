from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.disaggregation import SplitProfiles
from ecolecon_mrio.exio_share_split import (
    emitted_sector_to_exio_members_from_split_profiles,
    project_environmental_extensions_to_split_sectors,
    raw_exio_members_from_split_profiles,
)
from ecolecon_mrio.iot_flow_semantics import flow_use_total


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
    pd.DataFrame([[999.0]], index=pd.Index(["stale"]), columns=pd.Index(["stale"])).to_pickle(parsed_dir / "Z.pkl")
    pd.DataFrame([[999.0]], index=pd.Index(["stale"]), columns=pd.Index(["P3"])).to_pickle(parsed_dir / "Y.pkl")
    (parsed_dir / "meta.json").write_text('{"stale": true}', encoding="utf-8")

    fresh = eurostat_iot.ParsedEurostatIOT(
        Z=pd.DataFrame([[1.0]], index=pd.Index(["CPA_A"]), columns=pd.Index(["CPA_A"])),
        Y=pd.DataFrame([[2.0]], index=pd.Index(["CPA_A"]), columns=pd.Index(["P3"])),
        V=pd.DataFrame([[3.0]], index=pd.Index(["B1G"]), columns=pd.Index(["CPA_A"])),
        M=pd.DataFrame([[4.0]], index=pd.Index(["B1G"]), columns=pd.Index(["P3"])),
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


def test_edge_child_environmental_extension_uses_edge_weight_then_eurostat_output():
    profiles = _one_child_profile()
    exio = pd.DataFrame(
        [
            {
                "source_label": "p24.d",
                "extension": "air_emissions",
                "kind": "F",
                "stressor": "CO2 - combustion - air",
                "unit": "kg",
                "value": 100.0,
                "monetary_denominator": 200.0,
                "intensity": 0.5,
            },
        ]
    )

    out = project_environmental_extensions_to_split_sectors(
        exio_extensions=exio,
        split_profiles=profiles,
        eurostat_output=pd.Series(
            {"p24.d@CPA_C20": 80.0}, dtype="float64"
        ),
        eurostat_sector_mapping={
            "CPA_C20": {"p24.d": 0.25},
            "CPA_C21": {"p24.d": 0.75},
        },
    )

    assert out is not None
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sector"] == "p24.d@CPA_C20"
    # P first yields 25 kg / 50 monetary units = 0.5 donor intensity;
    # the Eurostat satellite total is then 0.5 * 80, not the EXIO 25 kg total.
    assert float(row["value"]) == 40.0
    assert float(row["monetary_output"]) == 80.0
    assert float(row["intensity"]) == 0.5

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


def test_load_eurostat_table_2d_ignores_derived_parsed_cache(tmp_path, monkeypatch):
    from ecolecon_mrio import eurostat_iot

    base = tmp_path / "eurostat" / "demo_table" / "request_hash"
    parsed_dir = base / "parsed"
    parsed_dir.mkdir(parents=True)
    raw_path = base / "raw.json"
    raw_path.write_text("{}", encoding="utf-8")
    pd.DataFrame([[999.0]], index=pd.Index(["stale"]), columns=pd.Index(["stale"])).to_pickle(
        parsed_dir / "table_row_x_col.pkl"
    )
    (parsed_dir / "table_row_x_col_meta.json").write_text('{"stale": true}', encoding="utf-8")

    fresh_df = pd.DataFrame([[1.0]], index=pd.Index(["fresh_row"]), columns=pd.Index(["fresh_col"]))
    fresh_meta = {"fresh": True}

    def fake_download(*args, **kwargs):
        return eurostat_iot.EurostatDataset(
            dataset="demo_table",
            filters={},
            request_url="memory://table",
            raw_path=raw_path,
        )

    def fake_read(path):
        assert path == raw_path
        return {"fresh_payload": "table"}

    def fake_parse(payload, *, row_dim, col_dim, value_key="value"):
        assert payload == {"fresh_payload": "table"}
        assert row_dim == "row"
        assert col_dim == "col"
        assert value_key == "value"
        return fresh_df, fresh_meta

    monkeypatch.setattr(eurostat_iot, "download_eurostat_dataset", fake_download)
    monkeypatch.setattr(eurostat_iot, "read_eurostat_raw", fake_read)
    monkeypatch.setattr(eurostat_iot, "parse_eurostat_table_2d", fake_parse)

    result_df, result_meta = eurostat_iot.load_eurostat_table_2d(
        dataset="demo_table",
        filters={"time": "2020"},
        row_dim="row",
        col_dim="col",
        storage_folder=tmp_path,
    )

    pd.testing.assert_frame_equal(result_df, fresh_df)
    assert result_meta == fresh_meta


def test_load_eurostat_series_1d_ignores_derived_parsed_cache(tmp_path, monkeypatch):
    from ecolecon_mrio import eurostat_iot

    base = tmp_path / "eurostat" / "demo_series" / "request_hash"
    parsed_dir = base / "parsed"
    parsed_dir.mkdir(parents=True)
    raw_path = base / "raw.json"
    raw_path.write_text("{}", encoding="utf-8")
    pd.Series([999.0], index=["stale"]).to_pickle(parsed_dir / "series_dim.pkl")
    (parsed_dir / "series_dim_meta.json").write_text('{"stale": true}', encoding="utf-8")

    fresh_series = pd.Series([1.0], index=["fresh"], dtype="float64")
    fresh_meta = {"fresh": True}

    def fake_download(*args, **kwargs):
        return eurostat_iot.EurostatDataset(
            dataset="demo_series",
            filters={},
            request_url="memory://series",
            raw_path=raw_path,
        )

    def fake_read(path):
        assert path == raw_path
        return {"fresh_payload": "series"}

    def fake_parse(payload, *, dim, value_key="value"):
        assert payload == {"fresh_payload": "series"}
        assert dim == "dim"
        assert value_key == "value"
        return fresh_series, fresh_meta

    monkeypatch.setattr(eurostat_iot, "download_eurostat_dataset", fake_download)
    monkeypatch.setattr(eurostat_iot, "read_eurostat_raw", fake_read)
    monkeypatch.setattr(eurostat_iot, "parse_eurostat_series_1d", fake_parse)

    result_series, result_meta = eurostat_iot.load_eurostat_series_1d(
        dataset="demo_series",
        filters={"time": "2020"},
        dim="dim",
        storage_folder=tmp_path,
    )

    pd.testing.assert_series_equal(result_series, fresh_series)
    assert result_meta == fresh_meta


def test_projection_matrix_matches_explicit_pzp_and_py_toy(tmp_path):
    from ecolecon_mrio.region_aggregation import MrioScopeSlice, RegionScope
    from ecolecon_mrio.exio_share_split import (
        build_projection_matrix_for_split_profiles,
        project_exio_donor_blocks_for_split_profiles,
        split_iot_using_exiobase_split_profiles,
    )

    exio_Z = pd.DataFrame(
        [[10.0, 30.0], [20.0, 40.0]],
        index=pd.Index(["S", "A"], name="sector"),
        columns=pd.Index(["S", "A"], name="sector"),
    )
    exio_Y = pd.DataFrame(
        [[100.0], [200.0]],
        index=pd.Index(["S", "A"], name="sector"),
        columns=pd.Index(["P3"], name="final_demand"),
    )
    exio_x = exio_Z.sum(axis=1) + exio_Y.sum(axis=1)
    exio_slice = MrioScopeSlice(scope=RegionScope(code=str('AT'), members=(str('AT'),)), Z=exio_Z, Y=exio_Y, x_scope=exio_x, flow='internal', metadata={"dataset": "exiobase", "year": int(2020), "system": str('pxp'), "source_zip": str(tmp_path / 'dummy.zip')})

    eurostat_Z = pd.DataFrame(
        [[100.0, 80.0], [60.0, 40.0]],
        index=pd.Index(["CPA_C20", "CPA_C21"]),
        columns=pd.Index(["CPA_C20", "CPA_C21"]),
    )
    eurostat_Y = pd.DataFrame(
        [[50.0], [30.0]],
        index=pd.Index(["CPA_C20", "CPA_C21"]),
        columns=pd.Index(["P3"]),
    )
    sector_to_exio = {
        "CPA_C20": ["S", "A"],
        "CPA_C21": ["S"],
    }
    mapping = {
        "CPA_C20": {"S": 0.25, "A": 1.0},
        "CPA_C21": {"S": 0.75},
    }

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors=sector_to_exio,
        eurostat_sector_mapping=mapping,
        create_io_system=False,
    )

    P = build_projection_matrix_for_split_profiles(
        res.split_profiles,
        eurostat_sector_mapping=mapping,
        raw_exio_sectors=["S", "A"],
    )
    expected_P = pd.DataFrame(
        [[0.25, 0.0], [0.0, 1.0], [0.75, 0.0]],
        index=pd.Index(["S@CPA_C20", "A", "S@CPA_C21"]),
        columns=pd.Index(["S", "A"]),
    )
    pd.testing.assert_frame_equal(P, expected_P)

    _, donor_Z, donor_Y = project_exio_donor_blocks_for_split_profiles(
        exio_Z=exio_Z,
        exio_Y=exio_Y,
        split_profiles=res.split_profiles,
        eurostat_sector_mapping=mapping,
        raw_exio_sectors=["S", "A"],
    )
    explicit_Z = pd.DataFrame(
        expected_P.to_numpy() @ exio_Z.to_numpy() @ expected_P.to_numpy().T,
        index=expected_P.index,
        columns=expected_P.index,
    )
    explicit_Y = pd.DataFrame(
        expected_P.to_numpy() @ exio_Y.to_numpy(),
        index=expected_P.index,
        columns=exio_Y.columns,
    )
    pd.testing.assert_frame_equal(donor_Z, explicit_Z)
    pd.testing.assert_frame_equal(donor_Y, explicit_Y)

    c20_labels = ["S@CPA_C20", "A"]
    c21_labels = ["S@CPA_C21"]
    expected_c20_c20 = explicit_Z.loc[c20_labels, c20_labels]
    expected_c20_c20 = expected_c20_c20 * (100.0 / float(expected_c20_c20.to_numpy().sum()))
    pd.testing.assert_frame_equal(res.Z.loc[c20_labels, c20_labels], expected_c20_c20)

    expected_c20_c21 = explicit_Z.loc[c20_labels, c21_labels]
    expected_c20_c21 = expected_c20_c21 * (80.0 / float(expected_c20_c21.to_numpy().sum()))
    pd.testing.assert_frame_equal(res.Z.loc[c20_labels, c21_labels], expected_c20_c21)

    expected_y_c20 = explicit_Y.loc[c20_labels, "P3"]
    expected_y_c20 = expected_y_c20 * (50.0 / float(expected_y_c20.sum()))
    pd.testing.assert_series_equal(res.Y.loc[c20_labels, "P3"], expected_y_c20)
def test_aggregation_mapping_from_split_profiles_recovers_edge_split_z_y_v_m():
    from ecolecon_mrio.exio_share_split import aggregation_mapping_from_split_profiles
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    children_c20 = ["S@CPA_C20", "A"]
    children_c21 = ["S@CPA_C21"]
    all_children = pd.Index(children_c20 + children_c21)
    profiles = {
        "CPA_C20": SplitProfiles(
            aggregate_sector="CPA_C20",
            subsectors=tuple(children_c20),
            output_share=pd.Series([0.25, 0.75], index=children_c20, dtype="float64"),
            intermediate_sales_share=pd.Series([0.25, 0.75], index=children_c20, dtype="float64"),
            intermediate_input_share=pd.Series([0.25, 0.75], index=children_c20, dtype="float64"),
            final_demand_share=pd.Series([0.25, 0.75], index=children_c20, dtype="float64"),
            subsector_to_exio_sectors={"S@CPA_C20": ["S"], "A": ["A"]},
        ),
        "CPA_C21": SplitProfiles(
            aggregate_sector="CPA_C21",
            subsectors=tuple(children_c21),
            output_share=pd.Series([1.0], index=children_c21, dtype="float64"),
            intermediate_sales_share=pd.Series([1.0], index=children_c21, dtype="float64"),
            intermediate_input_share=pd.Series([1.0], index=children_c21, dtype="float64"),
            final_demand_share=pd.Series([1.0], index=children_c21, dtype="float64"),
            subsector_to_exio_sectors={"S@CPA_C21": ["S"]},
        ),
    }
    mapping = aggregation_mapping_from_split_profiles(profiles)
    assert mapping == {"S@CPA_C20": "CPA_C20", "A": "CPA_C20", "S@CPA_C21": "CPA_C21"}

    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        index=all_children,
        columns=all_children,
    )
    Y = pd.DataFrame([[10.0], [20.0], [30.0]], index=all_children, columns=pd.Index(["P3"]))
    V = pd.DataFrame([[100.0, 200.0, 300.0]], index=pd.Index(["B1G"]), columns=all_children)
    M = pd.DataFrame([[5.0]], index=pd.Index(["B1G"]), columns=pd.Index(["P3"]))

    result = aggregate_iot(
        Z,
        Y,
        mapping,
        V=V,
        M=M,
        output=flow_use_total(Z, Y),
        keep_unmapped=False,
        create_io_system=False,
    )

    expected_Z = pd.DataFrame(
        [[12.0, 9.0], [15.0, 9.0]],
        index=pd.Index(["CPA_C20", "CPA_C21"]),
        columns=pd.Index(["CPA_C20", "CPA_C21"]),
    )
    expected_Y = pd.DataFrame([[30.0], [30.0]], index=pd.Index(["CPA_C20", "CPA_C21"]), columns=pd.Index(["P3"]))
    expected_x = expected_Z.sum(axis=1) + expected_Y.sum(axis=1)
    expected_V = pd.DataFrame([[300.0, 300.0]], index=pd.Index(["B1G"]), columns=pd.Index(["CPA_C20", "CPA_C21"]))
    expected_M = M.copy()

    pd.testing.assert_frame_equal(result.Z, expected_Z)
    pd.testing.assert_frame_equal(result.Y, expected_Y)
    pd.testing.assert_series_equal(result.x, expected_x.rename("x"))
    assert result.V is not None
    assert result.M is not None
    pd.testing.assert_frame_equal(result.V, expected_V)
    pd.testing.assert_frame_equal(result.M, expected_M)


def test_aggregation_cli_augments_mapping_from_split_child_edges_metadata():
    from scripts.aggregate_iot_by_mapping import _augment_mapping_from_split_child_edges

    mapping = {"A": "CPA_C20"}
    mapping_df = pd.DataFrame(
        [
            {
                "source_code": "A",
                "target_code": "CPA_C20",
                "from_sector": "A",
                "to_sector": "CPA_C20",
                "enabled": "1",
                "kind": "aggregate",
                "notes": "manual",
            }
        ]
    )
    edge_metadata = pd.DataFrame(
        [
            {
                "current_child_code": "S@CPA_C20",
                "source_code": "S",
                "parent_code": "CPA_C20",
                "concordance_target_code": "CPA_C20",
            }
        ]
    )

    new_mapping, new_df, audit = _augment_mapping_from_split_child_edges(
        mapping=mapping,
        mapping_df=mapping_df,
        edge_metadata=edge_metadata,
    )

    assert new_mapping["S@CPA_C20"] == "CPA_C20"
    assert "S@CPA_C20" in set(new_df["from_sector"].astype(str))
    assert len(audit) == 1
    assert audit.iloc[0]["mapping_source"] == "identity_to_concordance_target"


def test_phase7_validator_recomputes_edge_reaggregation_outputs(tmp_path):
    from ecolecon_mrio.iot_aggregation import (
        aggregate_environmental_extensions,
        aggregate_iot,
    )
    from scripts.aggregate_iot_by_mapping import (
        _augment_mapping_from_split_child_edges,
        _complete_environmental_extension_sector_grid,
    )
    from scripts.validate_many_many_pipeline_outputs import validate_pipeline_outputs

    split_dir = tmp_path / "split"
    agg_dir = tmp_path / "agg"
    split_dir.mkdir()
    agg_dir.mkdir()

    children = pd.Index(["S@CPA_C20", "A", "S@CPA_C21"])
    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        index=children,
        columns=children,
    )
    Y = pd.DataFrame([[10.0], [20.0], [30.0]], index=children, columns=pd.Index(["P3"]))
    V = pd.DataFrame([[100.0, 200.0, 300.0]], index=pd.Index(["B1G"]), columns=children)
    M = pd.DataFrame([[5.0]], index=pd.Index(["B1G"]), columns=pd.Index(["P3"]))
    x = Z.sum(axis=1) + Y.sum(axis=1)

    Z.to_csv(split_dir / "Z_split.csv")
    Y.to_csv(split_dir / "Y_split.csv")
    V.to_csv(split_dir / "V_split.csv")
    M.to_csv(split_dir / "M_split.csv")
    x.to_csv(split_dir / "x_split.csv")
    (split_dir / "exio_selected_sectors.json").write_text(
        json.dumps({"CPA_C20": ["S@CPA_C20"], "CPA_C21": ["S@CPA_C21"]}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "parent_code": "CPA_C20",
                "current_child_code": "S@CPA_C20",
                "source_code": "S",
                "concordance_target_code": "CPA_C20",
            }
        ]
    ).to_csv(split_dir / "split_child_edges.csv", index=False)

    environmental = pd.DataFrame(
        [
            {
                "boundary": "scope_production",
                "denominator_role": "target_domestic_output",
                "sector": "S@CPA_C20",
                "extension": "air_emissions",
                "stressor": "CO2",
                "unit": "kg",
                "value": 8.0,
                "monetary_output": float(x.loc["S@CPA_C20"]),
                "intensity": 0.5,
            },
            # Positive-output edge children must carry supported DOM evidence.
            # S@CPA_C21 uses the same raw-S donor coefficient as S@CPA_C20.
            {
                "boundary": "scope_production",
                "denominator_role": "target_domestic_output",
                "sector": "S@CPA_C21",
                "extension": "air_emissions",
                "stressor": "CO2",
                "unit": "kg",
                "value": 0.5 * float(x.loc["S@CPA_C21"]),
                "monetary_output": float(x.loc["S@CPA_C21"]),
                "intensity": 0.5,
            },
            {
                "boundary": "scope_production",
                "denominator_role": "target_domestic_output",
                "sector": "A",
                "extension": "air_emissions",
                "stressor": "CO2",
                "unit": "kg",
                "value": float(x.loc["A"]),
                "monetary_output": float(x.loc["A"]),
                "intensity": 1.0,
            }
        ]
    )
    environmental.to_csv(split_dir / "environmental_extensions.csv", index=False)
    raw_environmental = pd.DataFrame(
        [
            {
                "source_label": "S",
                "extension": "air_emissions",
                "kind": "F",
                "stressor": "CO2",
                "unit": "kg",
                "value": 8.0,
                "monetary_denominator": 16.0,
                "intensity": 0.5,
            },
            {
                "source_label": "P3",
                "extension": "air_emissions",
                "kind": "F_Y",
                "stressor": "CO2",
                "unit": "kg",
                "value": 1.0,
                "monetary_denominator": 2.0,
                "intensity": 0.5,
            },
        ]
    )
    raw_environmental.to_csv(
        split_dir / "exio_environmental_extensions.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "disaggregated_sector": "S@CPA_C20",
                "extension": "energy",
                "stressor": "quantity",
                "unit": "TJ",
                "quantity": 2.0,
                "monetary_output": 10.0,
                "value_per_physical_input": 5.0,
            }
        ]
    ).to_csv(split_dir / "disaggregated_sector_value_per_physical_input.csv", index=False)

    mapping_csv = tmp_path / "aggregation_mapping.csv"
    mapping_df = pd.DataFrame(
        [
            {"source_code": "S@CPA_C21", "target_code": "CPA_C21", "enabled": "1"},
        ]
    )
    mapping_df.to_csv(mapping_csv, index=False)
    mapping = {"S@CPA_C21": "CPA_C21"}
    edge_meta = pd.read_csv(split_dir / "split_child_edges.csv")
    augmented_mapping, _, audit = _augment_mapping_from_split_child_edges(
        mapping=mapping,
        mapping_df=mapping_df.assign(
            from_sector=mapping_df["source_code"],
            to_sector=mapping_df["target_code"],
        ),
        edge_metadata=edge_meta,
    )
    assert not audit.empty

    expected = aggregate_iot(
        Z,
        Y,
        augmented_mapping,
        V=V,
        M=M,
        output=flow_use_total(Z, Y),
        keep_unmapped=True,
        create_io_system=False,
    )
    emitted_order = pd.Index(list(reversed(expected.Z.index.astype(str).tolist())))
    expected.Z.loc[emitted_order, emitted_order].to_csv(agg_dir / "Z_agg.csv")
    expected.Y.loc[emitted_order, :].to_csv(agg_dir / "Y_agg.csv")
    expected.x.reindex(emitted_order).to_csv(agg_dir / "x_agg.csv")
    assert expected.V is not None
    assert expected.M is not None
    expected.V.loc[:, emitted_order].to_csv(agg_dir / "V_agg.csv")
    expected.M.to_csv(agg_dir / "M_agg.csv")
    audit.to_csv(agg_dir / "aggregation_edge_mapping_audit.csv", index=False)

    expected_environmental = aggregate_environmental_extensions(
        environmental,
        augmented_mapping,
        output=x,
        keep_unmapped=True,
    )
    assert expected_environmental is not None
    completed_environmental = _complete_environmental_extension_sector_grid(
        expected_environmental,
        output=expected.x.reindex(emitted_order),
        production_groups=raw_environmental.loc[
            raw_environmental["kind"].eq("F"),
            ["extension", "stressor", "unit"],
        ].assign(
            boundary="scope_production",
            denominator_role="target_domestic_output",
        ),
    )
    assert completed_environmental is not None
    completed_environmental.to_csv(
        agg_dir / "environmental_extensions.csv", index=False
    )
    raw_environmental.to_csv(
        agg_dir / "exio_environmental_extensions.csv", index=False
    )

    report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=mapping_csv,
    )

    assert report["ok"] is True
    names = {check["name"] for check in report["checks"]}
    assert "Z_agg_matches_recomputed_reaggregation" in names
    assert "V_agg_matches_recomputed_reaggregation" in names
    assert "environmental_totals_match_reaggregation" in names
    assert "environmental_final_grid_contract" in names
    assert "aggregation_mapping_covers_selected_split_children" in names

    missing_mapping_csv = tmp_path / "missing_child_mapping.csv"
    pd.DataFrame(
        [{"source_code": "A", "target_code": "A", "enabled": "1"}]
    ).to_csv(missing_mapping_csv, index=False)
    missing_report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=missing_mapping_csv,
    )
    closure = next(
        check
        for check in missing_report["checks"]
        if check["name"] == "aggregation_mapping_covers_selected_split_children"
    )
    assert closure["ok"] is False
    assert closure["missing_sample"] == ["S@CPA_C21"]


def test_output_consistency_runs_for_dom_imp_total_and_uses_non_export_boundary():
    from scripts.split_eurostat_energy_by_exiobase import (
        _assert_output_consistency,
        _build_output_consistency_frame,
        _should_check_output,
    )

    # The consistency check must run for all three Eurostat stk_flow variants,
    # and it must not silently skip just because a sensitivity run uses a
    # different EXIO year.
    for flow in ("DOM", "IMP", "TOTAL", "TOT"):
        assert _should_check_output(flow, eurostat_year=2021, exio_year=2020)

    from ecolecon_mrio.region_aggregation import MrioScopeSlice, RegionScope

    # x_scope is deliberately unrelated to this destination-use check. The
    # consistency code must use the selected scope Z + Y flow instead, especially
    # for external-import and total slices where x_scope is producer-side output.
    exio_slice = MrioScopeSlice(
        scope=RegionScope(code="AT", members=("AT",)),
        Z=pd.DataFrame([[4.0]], index=pd.Index(["exio_a"]), columns=pd.Index(["exio_a"])),
        Y=pd.DataFrame([[6.0]], index=pd.Index(["exio_a"]), columns=pd.Index(["P3"])),
        x_scope=pd.Series({"exio_a": 999.0}, dtype="float64"),
        flow="total",
        metadata={"system": "pxp"},
    )

    mapping = {"CPA_A": {"exio_a": 1.0}}
    Z = pd.DataFrame([[7.0]], index=pd.Index(["CPA_A"]), columns=pd.Index(["CPA_A"]))
    # P6 is export final demand. The mapped-output check must use Z + non-export
    # final demand (= 7 + 3), not Z + P3 + P6 (= 110).
    Y = pd.DataFrame([[3.0, 100.0]], index=pd.Index(["CPA_A"]), columns=pd.Index(["P3", "P6"]))

    frame, exio_total, euro_total = _build_output_consistency_frame(
        sector_mapping=mapping,
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_slice=exio_slice,
    )

    assert exio_total == 10.0
    assert euro_total == 10.0
    assert frame.loc[0, "eurostat_output"] == 10.0

    _assert_output_consistency(
        sector_mapping=mapping,
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_slice=exio_slice,
        rtol=0.0,
        atol=0.0,
        label="boundary-test",
    )


def test_mapped_output_flow_identity_frame_checks_dom_imp_total():
    from scripts.split_eurostat_energy_by_exiobase import _build_mapped_output_flow_identity_frame

    good = {
        "DOM": {"eurostat_output_total": 7.0, "exio_mapped_output_total": 70.0},
        "IMP": {"eurostat_output_total": 3.0, "exio_mapped_output_total": 30.0},
        "TOTAL": {"eurostat_output_total": 10.0, "exio_mapped_output_total": 100.0},
    }
    frame = _build_mapped_output_flow_identity_frame(good, rtol=0.0, atol=0.0)
    assert list(frame["quantity"]) == ["eurostat_output_total", "exio_mapped_output_total"]
    assert frame["ok"].tolist() == [True, True]

    bad = {
        "DOM": {"eurostat_output_total": 7.0, "exio_mapped_output_total": 70.0},
        "IMP": {"eurostat_output_total": 3.0, "exio_mapped_output_total": 30.0},
        "TOTAL": {"eurostat_output_total": 11.0, "exio_mapped_output_total": 100.0},
    }
    bad_frame = _build_mapped_output_flow_identity_frame(bad, rtol=0.0, atol=0.0)
    assert not bool(bad_frame.loc[bad_frame["quantity"] == "eurostat_output_total", "ok"].iloc[0])
    assert bool(bad_frame.loc[bad_frame["quantity"] == "exio_mapped_output_total", "ok"].iloc[0])


def test_edge_mapping_augmentation_prefers_target_compatible_raw_source_mapping():
    from scripts.aggregate_iot_by_mapping import _augment_mapping_from_split_child_edges

    mapping = {"p24.d": "CPA_C20_others"}
    mapping_df = pd.DataFrame(
        [
            {
                "source_code": "p24.d",
                "target_code": "CPA_C20_others",
                "enabled": "1",
            }
        ]
    )
    edge_metadata = pd.DataFrame(
        [
            {
                "current_child_code": "p24.d@CPA_C20",
                "source_code": "p24.d",
                "parent_code": "CPA_C20",
                "concordance_target_code": "CPA_C20",
            }
        ]
    )

    new_mapping, _, audit = _augment_mapping_from_split_child_edges(
        mapping=mapping,
        mapping_df=mapping_df,
        edge_metadata=edge_metadata,
    )

    assert new_mapping["p24.d@CPA_C20"] == "CPA_C20_others"
    assert audit.iloc[0]["mapping_source"] == "existing_mapping_for_source_code_p24.d"


def test_phase7_validator_fails_on_extra_aggregate_labels(tmp_path):
    from scripts.validate_many_many_pipeline_outputs import validate_pipeline_outputs

    split_dir = tmp_path / "split"
    agg_dir = tmp_path / "agg"
    split_dir.mkdir()
    agg_dir.mkdir()

    pd.DataFrame([[1.0]], index=pd.Index(["A"]), columns=pd.Index(["A"])).to_csv(split_dir / "Z_split.csv")
    pd.DataFrame([[2.0]], index=pd.Index(["A"]), columns=pd.Index(["P3"])).to_csv(split_dir / "Y_split.csv")
    pd.Series([3.0], index=["A"], name="x").to_csv(split_dir / "x_split.csv")

    pd.DataFrame([{"source_code": "A", "target_code": "A", "enabled": "1"}]).to_csv(
        tmp_path / "mapping.csv",
        index=False,
    )

    # Same expected A block, but with an extra aggregate label B. This must fail;
    # silently reindexing observed outputs to expected labels would hide it.
    pd.DataFrame(
        [[1.0, 0.0], [0.0, 0.0]],
        index=pd.Index(["A", "B"]),
        columns=pd.Index(["A", "B"]),
    ).to_csv(agg_dir / "Z_agg.csv")
    pd.DataFrame([[2.0], [0.0]], index=pd.Index(["A", "B"]), columns=pd.Index(["P3"])).to_csv(agg_dir / "Y_agg.csv")
    pd.Series([3.0, 0.0], index=["A", "B"], name="x").to_csv(agg_dir / "x_agg.csv")

    report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=tmp_path / "mapping.csv",
    )

    assert report["ok"] is False
    z_check = next(check for check in report["checks"] if check["name"] == "Z_agg_matches_recomputed_reaggregation")
    assert z_check["max_abs_gap"] == float("inf")


def test_validator_numeric_comparisons_preserve_elementwise_relative_tolerance(tmp_path):
    """Validator reductions retain the rtol contract and report the worst cell."""
    from scripts.validate_many_many_pipeline_outputs import (
        _check_ratio_file,
        _compare_environmental_extensions,
        _compare_numeric_frames,
    )

    keys = {
        "sector": ["large", "small"],
        "boundary": ["scope_production", "scope_production"],
        "denominator_role": ["target_domestic_output", "target_domestic_output"],
        "extension": ["air", "air"],
        "stressor": ["CO2", "CO2"],
        "unit": ["kg", "kg"],
    }
    expected = pd.DataFrame(
        {
            **keys,
            "value": [1.0e11, 10.0],
            "monetary_output": [1.0e11, 10.0],
        }
    )
    actual = expected.copy()
    actual.loc[0, "value"] += 1.52587890625e-05
    passing = _compare_environmental_extensions(
        actual, expected, rtol=1e-9, atol=1e-6
    )
    assert passing["ok"] is True
    assert passing["max_abs_gap"] == 1.52587890625e-05

    actual.loc[1, "value"] += 2.0e-5
    failing = _compare_environmental_extensions(
        actual, expected, rtol=1e-9, atol=1e-6
    )
    assert failing["ok"] is False
    assert failing["worst_label"] == [
        "small", "scope_production", "target_domestic_output", "air", "CO2", "kg"
    ]
    assert failing["worst_field"] == "value"
    assert failing["worst_tolerance"] == pytest.approx(1.01e-6)

    mismatched_keys = actual.copy()
    mismatched_keys.loc[1, "sector"] = "other"
    assert not _compare_environmental_extensions(
        mismatched_keys, expected, rtol=1e-9, atol=1e-6
    )["ok"]
    mismatched_nan = actual.copy()
    mismatched_nan.loc[1, "value"] = np.nan
    assert not _compare_environmental_extensions(
        mismatched_nan, expected, rtol=1e-9, atol=1e-6
    )["ok"]
    for field, changed in (
        ("boundary", "external_imports_direct"),
        ("denominator_role", "target_import_use"),
    ):
        mismatched_group = actual.copy()
        mismatched_group.loc[1, field] = changed
        assert not _compare_environmental_extensions(
            mismatched_group, expected, rtol=1e-9, atol=1e-6
        )["ok"]

    ratio_path = tmp_path / "ratio.csv"
    pd.DataFrame(
        {
            "numerator": [1.0e11],
            "denominator": [1.0],
            "ratio": [1.0e11 + 1.52587890625e-05],
        }
    ).to_csv(ratio_path, index=False)
    assert _check_ratio_file(
        ratio_path,
        numerator_col="numerator",
        denominator_col="denominator",
        ratio_col="ratio",
        rtol=1e-9,
        atol=1e-6,
    )["ok"]

    frame_actual = pd.DataFrame([[1.0e11 + 1.52587890625e-05]], index=["A"], columns=["A"])
    frame_expected = pd.DataFrame([[1.0e11]], index=["A"], columns=["A"])
    assert _compare_numeric_frames(
        frame_actual, frame_expected, rtol=1e-9, atol=1e-6
    )["ok"]
    exact = _compare_numeric_frames(
        pd.DataFrame([[0.0]], index=["A"], columns=["A"]),
        pd.DataFrame([[0.0]], index=["A"], columns=["A"]),
        rtol=0.0,
        atol=0.0,
    )
    assert exact["ok"] and exact["max_normalized_gap"] == 0.0
    assert not _compare_numeric_frames(
        pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
        pd.DataFrame([[0.0]], index=["A"], columns=["A"]),
        rtol=0.0,
        atol=0.0,
    )["ok"]


def test_environmental_artifact_grid_validator_rejects_alignment_and_na_defects(
    tmp_path,
):
    """Phase-7 storage validation must inspect undefined rows, not filter them away."""
    from scripts.aggregate_iot_by_mapping import (
        _complete_environmental_extension_sector_grid,
    )
    from scripts.validate_many_many_pipeline_outputs import (
        _validate_environmental_grid,
    )

    output = pd.Series({"A": 10.0, "B": 0.0, "C": 0.0, "D": 0.0})
    actual = pd.DataFrame(
        {
            "sector": ["A", "B"],
            "boundary": "scope_production",
            "denominator_role": "target_domestic_output",
            "extension": ["air_emissions", "air_emissions"],
            "stressor": ["CO2", "CO2"],
            "unit": ["kg", "kg"],
            "value": [2.0, 0.0],
            "monetary_output": [10.0, 0.0],
            "intensity": [0.2, float("nan")],
        }
    )
    raw = pd.DataFrame(
        [
            {
                "source_label": "S",
                "extension": "air_emissions",
                "kind": "F",
                "stressor": "CO2",
                "unit": "kg",
                "value": 1.0,
                "monetary_denominator": 2.0,
                "intensity": 0.5,
            },
            {
                "source_label": "P3",
                "extension": "use_phase_only",
                "kind": "F_Y",
                "stressor": "CH4",
                "unit": "kg",
                "value": 1.0,
                "monetary_denominator": 2.0,
                "intensity": 0.5,
            },
        ]
    )
    completed = _complete_environmental_extension_sector_grid(
        actual,
        output=output,
        production_groups=raw.loc[
            raw["kind"].eq("F"),
            ["extension", "stressor", "unit"],
        ].assign(
            boundary="scope_production",
            denominator_role="target_domestic_output",
        ),
    )
    assert completed is not None
    grid_path = tmp_path / "environmental_extensions.csv"
    raw_path = tmp_path / "exio_environmental_extensions.csv"
    raw.to_csv(raw_path, index=False)

    def validate(frame: pd.DataFrame) -> dict[str, object]:
        frame.to_csv(grid_path, index=False)
        return _validate_environmental_grid(
            environmental_path=grid_path,
            raw_path=raw_path,
            output=output,
            rtol=1e-9,
            atol=1e-6,
        )

    valid_result = validate(completed)
    assert valid_result["ok"] is True
    assert valid_result["n_nonzero_denominator_unavailable_rows"] == 0

    # Phase-7 independently protects the same state machine as the producer.
    positive_output = output.copy()
    positive_output.loc["C"] = 5.0
    positive_missing = _complete_environmental_extension_sector_grid(
        actual,
        output=positive_output,
        production_groups=raw.loc[
            raw["kind"].eq("F"), ["extension", "stressor", "unit"]
        ].assign(
            boundary="scope_production",
            denominator_role="target_domestic_output",
        ),
    )
    assert positive_missing is not None
    positive_missing.to_csv(grid_path, index=False)
    positive_result = _validate_environmental_grid(
        environmental_path=grid_path, raw_path=raw_path, output=positive_output,
        rtol=1e-9, atol=1e-6,
    )
    assert positive_result["ok"] is False
    assert any(
        "positive target domestic output requires available environmental evidence" in e
        for e in positive_result["errors"]
    )
    assert positive_result["n_nonzero_denominator_unavailable_rows"] == 1
    assert positive_result["n_nonzero_denominator_unavailable_sectors"] == 1
    assert positive_result["nonzero_denominator_unavailable_sector_sample"] == ["C"]
    assert positive_result["nonzero_denominator_unavailable_abs_monetary_share"] == pytest.approx(5.0/15.0)

    # Restore the base valid grid for the remaining mutation checks.
    completed.to_csv(grid_path, index=False)
    missing_row = completed.loc[~completed["sector"].eq("C")].copy()
    assert validate(missing_row)["ok"] is False

    bad_position = completed.copy()
    bad_position.loc[bad_position["sector"].eq("C"), "sector_position"] = 99
    assert validate(bad_position)["ok"] is False

    bad_denominator = completed.copy()
    bad_denominator.loc[bad_denominator["sector"].eq("C"), "monetary_output"] = 6.0
    assert validate(bad_denominator)["ok"] is False

    bad_missingness = completed.copy()
    bad_missingness.loc[bad_missingness["sector"].eq("C"), "intensity"] = 0.0
    assert validate(bad_missingness)["ok"] is False

    bad_zero_value = completed.copy()
    bad_zero_value.loc[bad_zero_value["sector"].eq("B"), "value"] = 1.0
    assert validate(bad_zero_value)["ok"] is False

    bad_zero_intensity = completed.copy()
    bad_zero_intensity.loc[bad_zero_intensity["sector"].eq("B"), "intensity"] = 0.0
    assert validate(bad_zero_intensity)["ok"] is False



def test_environmental_grid_reports_nonzero_import_unavailability_without_failing(tmp_path):
    from scripts.validate_many_many_pipeline_outputs import _validate_environmental_grid

    path = tmp_path / "environmental_extensions.csv"
    output = pd.Series({"A": 20.0, "B": -5.0, "C": 0.0}, dtype="float64")
    frame = pd.DataFrame({
        "sector_position": [1,2,3], "sector": ["A","B","C"],
        "boundary": ["external_imports_direct"]*3,
        "denominator_role": ["target_import_use"]*3,
        "extension": ["air"]*3, "stressor": ["CO2"]*3, "unit": ["kg"]*3,
        "value": [float("nan"), -2.5, float("nan")],
        "monetary_output": [20.0, -5.0, 0.0],
        "intensity": [float("nan"), 0.5, float("nan")],
    })
    frame.to_csv(path,index=False)
    groups=frame.loc[:,["boundary","denominator_role","extension","stressor","unit"]].drop_duplicates()
    result=_validate_environmental_grid(
        environmental_path=path,raw_path=None,output=output,rtol=1e-9,atol=1e-6,
        production_groups=groups,
    )
    assert result["ok"] is True
    assert result["n_nonzero_denominator_unavailable_rows"] == 1
    assert result["n_nonzero_denominator_unavailable_sectors"] == 1
    assert result["nonzero_denominator_unavailable_sector_sample"] == ["A"]
    assert result["nonzero_denominator_unavailable_abs_monetary_mass"] == 20.0
    assert result["total_abs_monetary_mass"] == 25.0
    assert result["nonzero_denominator_unavailable_abs_monetary_share"] == pytest.approx(0.8)


def test_flow_identity_tolerance_is_strict_for_internal_additivity():
    from scripts.split_eurostat_energy_by_exiobase import _build_mapped_output_flow_identity_frame

    # Cross-dataset EXIO-vs-Eurostat diagnostics may use a loose 20% tolerance,
    # but internal Eurostat/EXIO flow additivity should be checked with its own
    # strict tolerance. DOM + IMP = TOTAL is an accounting identity, not an
    # approximate concordance-quality diagnostic.
    totals = {
        "DOM": {"eurostat_output_total": 7.0, "exio_mapped_output_total": 70.0},
        "IMP": {"eurostat_output_total": 3.0, "exio_mapped_output_total": 30.0},
        "TOTAL": {"eurostat_output_total": 11.0, "exio_mapped_output_total": 101.0},
    }

    strict = _build_mapped_output_flow_identity_frame(totals, rtol=1e-9, atol=1e-6)

    assert strict["ok"].tolist() == [False, False]


def _tiny_complete_eurostat_iot_payload() -> dict[str, object]:
    """Build a 2-product JSON-stat IOT with every core observation explicit."""
    return {
        "id": ["geo", "prd_ava", "prd_use"],
        "size": [1, 2, 3],
        "dimension": {
            "geo": {"category": {"index": {"AT": 0}, "label": {"AT": "Austria"}}},
            "prd_ava": {
                "category": {
                    "index": {"CPA_A": 0, "CPA_B": 1},
                    "label": {"CPA_A": "A", "CPA_B": "B"},
                }
            },
            "prd_use": {
                "category": {
                    "index": {"CPA_A": 0, "CPA_B": 1, "P3": 2},
                    "label": {"CPA_A": "A", "CPA_B": "B", "P3": "Final consumption"},
                }
            },
        },
        # Flat positions are row * 3 + column because geo is a singleton.
        "value": {"0": 1.0, "1": 0.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0},
    }


def _tiny_export_residual_eurostat_iot_payload(
    *, geo: str = "AT"
) -> dict[str, object]:
    """Build two product rows so one residual cell can be sparse while its column exists."""
    use_codes = [
        "CPA_A", "CPA_B", "P6", "P6_B0", "P6_D0", "P6_U2", "P6_U3"
    ]
    values: dict[str, float] = {}
    rows = {
        "CPA_A": [1.0, 0.0, 10.0, 6.0, 4.0, 2.0, 4.0],
        "CPA_B": [0.0, 1.0, 20.0, 12.0, 8.0, 5.0, 7.0],
    }
    for row_pos, row in enumerate(("CPA_A", "CPA_B")):
        for col_pos, value in enumerate(rows[row]):
            values[str(row_pos * len(use_codes) + col_pos)] = value
    return {
        "id": ["geo", "prd_ava", "prd_use"],
        "size": [1, 2, len(use_codes)],
        "dimension": {
            "geo": {"category": {"index": {geo: 0}}},
            "prd_ava": {
                "category": {"index": {"CPA_A": 0, "CPA_B": 1}}
            },
            "prd_use": {
                "category": {"index": {code: pos for pos, code in enumerate(use_codes)}}
            },
        },
        "value": values,
    }


def test_parse_eurostat_iot_derives_only_missing_export_display_residual_cells() -> None:
    from ecolecon_mrio.eurostat_iot import parse_eurostat_iot

    payload = _tiny_export_residual_eurostat_iot_payload()
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    # CPA_A P6_U3 is absent but P6_B0 and P6_U2 are explicit: 6 - 2 = 4.
    del values["6"]
    # CPA_B P6_D0 is absent but P6 and P6_B0 are explicit: 20 - 12 = 8.
    del values["11"]
    payload["value"] = values
    parsed = parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")

    assert float(parsed.Y.loc["CPA_A", "P6_U3"]) == 4.0
    assert float(parsed.Y.loc["CPA_B", "P6_D0"]) == 8.0
    assert parsed.meta["derived_final_demand_residual_cells"] == [
        {"row": "CPA_B", "column": "P6_D0", "formula": "P6 - P6_B0"},
        {"row": "CPA_A", "column": "P6_U3", "formula": "P6_B0 - P6_U2"},
    ]


def test_parse_eurostat_iot_does_not_invent_missing_residual_operands() -> None:
    from ecolecon_mrio.eurostat_iot import EurostatParseError, parse_eurostat_iot

    payload = _tiny_export_residual_eurostat_iot_payload()
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    del values["6"]  # CPA_A P6_U3 residual
    del values["5"]  # CPA_A P6_U2 required operand
    payload["value"] = values
    with pytest.raises(EurostatParseError, match="absent SDMX observations"):
        parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")


def test_parse_eurostat_iot_still_rejects_missing_nonresidual_final_demand() -> None:
    from ecolecon_mrio.eurostat_iot import EurostatParseError, parse_eurostat_iot

    payload = _tiny_export_residual_eurostat_iot_payload()
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    del values["3"]  # CPA_A P6_B0 is a published control, not a residual.
    payload["value"] = values
    with pytest.raises(EurostatParseError, match="absent SDMX observations"):
        parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")


def test_parse_eurostat_iot_materialises_consolidated_eu_intra_eu_export_structural_zeros() -> None:
    from ecolecon_mrio.eurostat_iot import parse_eurostat_iot

    payload = _tiny_export_residual_eurostat_iot_payload(geo="EU27_2020")
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    sizes_obj = payload["size"]
    assert isinstance(sizes_obj, list)
    use_count = int(sizes_obj[2])
    # Reproduce the real EU27 pattern: P6_D0 remains an explicit extra-EU export
    # control equal to P6; P6_B0 and P6_U2 are globally absent; P6_U3 is explicit
    # zero for one row and absent for the other.
    p6_pos, p6_b0_pos, p6_d0_pos, p6_u2_pos, p6_u3_pos = 2, 3, 4, 5, 6
    for row_pos in (0, 1):
        values[str(row_pos * use_count + p6_d0_pos)] = values[str(row_pos * use_count + p6_pos)]
        values.pop(str(row_pos * use_count + p6_b0_pos), None)
        values.pop(str(row_pos * use_count + p6_u2_pos), None)
    values[str(0 * use_count + p6_u3_pos)] = 0.0
    values.pop(str(1 * use_count + p6_u3_pos), None)
    payload["value"] = values

    parsed = parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")

    for code in ("P6_B0", "P6_U2", "P6_U3"):
        assert code in parsed.Y.columns
        assert parsed.Y.loc[:, code].eq(0.0).all()
    pd.testing.assert_series_equal(
        parsed.Y.loc[:, "P6_D0"],
        parsed.Y.loc[:, "P6"],
        check_names=False,
    )
    assert parsed.meta["structural_zero_final_demand_columns"] == [
        "P6_B0", "P6_U2", "P6_U3"
    ]


def test_parse_eurostat_iot_rejects_nonzero_consolidated_eu_intra_eu_export_control() -> None:
    from ecolecon_mrio.eurostat_iot import EurostatParseError, parse_eurostat_iot

    payload = _tiny_export_residual_eurostat_iot_payload(geo="EU27_2020")
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    values["3"] = 1.0  # CPA_A P6_B0 contradicts the consolidated-EU boundary.
    payload["value"] = values

    with pytest.raises(EurostatParseError, match="structural zero"):
        parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")

def test_parse_eurostat_iot_preserves_explicit_zero_in_complete_core() -> None:
    """An explicit SDMX zero remains zero when every actual Z/Y core cell is observed."""
    from ecolecon_mrio.eurostat_iot import parse_eurostat_iot

    parsed = parse_eurostat_iot(
        _tiny_complete_eurostat_iot_payload(), row_dim="prd_ava", col_dim="prd_use"
    )
    assert parsed.Z.shape == (2, 2)
    assert parsed.Y.shape == (2, 1)
    assert float(parsed.Z.loc["CPA_A", "CPA_B"]) == 0.0


def test_parse_eurostat_iot_rejects_absent_observation_inside_core() -> None:
    """Sparse JSON-stat absence inside identified Z/Y is missing evidence, not zero."""
    from ecolecon_mrio.eurostat_iot import EurostatParseError, parse_eurostat_iot

    payload = _tiny_complete_eurostat_iot_payload()
    values_obj = payload["value"]
    assert isinstance(values_obj, dict)
    values = dict(values_obj)
    del values["4"]  # CPA_B -> CPA_B is a real core cell, not a structural non-cell.
    payload["value"] = values
    with pytest.raises(EurostatParseError, match="absent SDMX observations"):
        parse_eurostat_iot(payload, row_dim="prd_ava", col_dim="prd_use")

def test_phase7_environmental_grid_validates_negative_import_use_ratio(tmp_path) -> None:
    from scripts.validate_many_many_pipeline_outputs import _validate_environmental_grid

    path = tmp_path / "environmental_extensions.csv"
    frame = pd.DataFrame(
        {
            "sector_position": [1],
            "sector": ["A"],
            "boundary": ["external_imports_direct"],
            "denominator_role": ["target_import_use"],
            "extension": ["air"],
            "stressor": ["CO2"],
            "unit": ["kg"],
            "value": [-10.0],
            "monetary_output": [-5.0],
            "intensity": [2.0],
        }
    )
    groups = frame.loc[
        :, ["boundary", "denominator_role", "extension", "stressor", "unit"]
    ].copy()
    frame.to_csv(path, index=False)
    good = _validate_environmental_grid(
        environmental_path=path,
        raw_path=None,
        output=pd.Series({"A": -5.0}),
        rtol=1e-9,
        atol=1e-6,
        production_groups=groups,
    )
    assert good["ok"] is True

    frame.loc[0, "intensity"] = 999.0
    frame.to_csv(path, index=False)
    bad = _validate_environmental_grid(
        environmental_path=path,
        raw_path=None,
        output=pd.Series({"A": -5.0}),
        rtol=1e-9,
        atol=1e-6,
        production_groups=groups,
    )
    assert bad["ok"] is False
    assert "nonzero-denominator intensity does not equal value/output" in bad["errors"]


@pytest.mark.parametrize("role", ["target_import_use", "target_total_use"])
def test_environmental_grid_allows_signed_zero_denominator_cancellation(tmp_path, role) -> None:
    from scripts.validate_many_many_pipeline_outputs import _validate_environmental_grid

    path = tmp_path / "environmental_extensions.csv"
    frame = pd.DataFrame(
        {
            "sector_position": [1], "sector": ["A"], "boundary": ["signed_use"],
            "denominator_role": [role], "extension": ["air"], "stressor": ["CO2"],
            "unit": ["kg"], "value": [3.0], "monetary_output": [0.0], "intensity": [np.nan],
        }
    )
    frame.to_csv(path, index=False)
    groups = frame.loc[:, ["boundary", "denominator_role", "extension", "stressor", "unit"]]
    assert _validate_environmental_grid(
        environmental_path=path, raw_path=None, output=pd.Series({"A": 0.0}),
        rtol=1e-9, atol=1e-6, production_groups=groups,
    )["ok"]
    frame.loc[0, "denominator_role"] = "target_domestic_output"
    frame.to_csv(path, index=False)
    assert not _validate_environmental_grid(
        environmental_path=path, raw_path=None, output=pd.Series({"A": 0.0}),
        rtol=1e-9, atol=1e-6, production_groups=frame.loc[:, ["boundary", "denominator_role", "extension", "stressor", "unit"]],
    )["ok"]
