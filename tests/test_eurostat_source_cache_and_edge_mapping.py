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


def test_load_eurostat_table_2d_ignores_derived_parsed_cache(tmp_path, monkeypatch):
    from ecolecon_mrio import eurostat_iot

    base = tmp_path / "eurostat" / "demo_table" / "request_hash"
    parsed_dir = base / "parsed"
    parsed_dir.mkdir(parents=True)
    raw_path = base / "raw.json"
    raw_path.write_text("{}", encoding="utf-8")
    pd.DataFrame([[999.0]], index=["stale"], columns=["stale"]).to_pickle(
        parsed_dir / "table_row_x_col.pkl"
    )
    (parsed_dir / "table_row_x_col_meta.json").write_text('{"stale": true}', encoding="utf-8")

    fresh_df = pd.DataFrame([[1.0]], index=["fresh_row"], columns=["fresh_col"])
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
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
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
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(["S", "A"], name="sector"),
        Z_domestic=exio_Z,
        Y_region=exio_Y,
        x_region=exio_x,
        source_zip=tmp_path / "dummy.zip",
        flow="domestic",
    )

    eurostat_Z = pd.DataFrame(
        [[100.0, 80.0], [60.0, 40.0]],
        index=["CPA_C20", "CPA_C21"],
        columns=["CPA_C20", "CPA_C21"],
    )
    eurostat_Y = pd.DataFrame(
        [[50.0], [30.0]],
        index=["CPA_C20", "CPA_C21"],
        columns=["P3"],
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
        index=["S@CPA_C20", "A", "S@CPA_C21"],
        columns=["S", "A"],
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
def test_supplementary_rows_use_row_specific_proxy_hierarchy():
    from ecolecon_mrio.exio_share_split import _split_supplementary_rows_columnwise

    children = pd.Index(["child_output_heavy", "child_input_heavy"])
    profiles = {
        "CPA_X": SplitProfiles(
            aggregate_sector="CPA_X",
            subsectors=tuple(children),
            output_share=pd.Series([0.8, 0.2], index=children, dtype="float64"),
            intermediate_sales_share=pd.Series([0.5, 0.5], index=children, dtype="float64"),
            intermediate_input_share=pd.Series([0.25, 0.75], index=children, dtype="float64"),
            final_demand_share=pd.Series([0.5, 0.5], index=children, dtype="float64"),
            subsector_to_exio_sectors={
                "child_output_heavy": ["exio_a"],
                "child_input_heavy": ["exio_b"],
            },
        )
    }
    V = pd.DataFrame(
        [[100.0], [100.0]],
        index=["B1G", "OTHER_ROW"],
        columns=["CPA_X"],
    )
    M = pd.DataFrame([[7.0]], index=["B1G"], columns=["P3"])

    V_split, M_out, diag = _split_supplementary_rows_columnwise(
        eurostat_V=V,
        eurostat_M=M,
        sectors_old=["CPA_X"],
        sectors_new=list(children),
        split_profiles=profiles,
    )

    assert V_split is not None
    assert M_out is not None
    assert diag is not None
    pd.testing.assert_series_equal(V_split.sum(axis=1), V["CPA_X"], check_names=False)
    assert V_split.loc["B1G", "child_output_heavy"] == 80.0
    assert V_split.loc["B1G", "child_input_heavy"] == 20.0
    assert V_split.loc["OTHER_ROW", "child_output_heavy"] == 25.0
    assert V_split.loc["OTHER_ROW", "child_input_heavy"] == 75.0
    assert "output_share_proxy_for_value_added_row" in set(diag["allocation_method"])
    assert "projected_exio_column_use_proxy" in set(diag["allocation_method"])
    pd.testing.assert_frame_equal(M_out, M.astype("float64"))


def test_aggregation_mapping_from_split_profiles_recovers_edge_split_z_y_v_m():
    from ecolecon_mrio.exio_share_split import aggregation_mapping_from_split_profiles
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    children_c20 = ["S@CPA_C20", "A"]
    children_c21 = ["S@CPA_C21"]
    all_children = children_c20 + children_c21
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
    Y = pd.DataFrame([[10.0], [20.0], [30.0]], index=all_children, columns=["P3"])
    V = pd.DataFrame([[100.0, 200.0, 300.0]], index=["B1G"], columns=all_children)
    M = pd.DataFrame([[5.0]], index=["B1G"], columns=["P3"])

    result = aggregate_iot(Z, Y, mapping, V=V, M=M, keep_unmapped=False, create_io_system=False)

    expected_Z = pd.DataFrame(
        [[12.0, 9.0], [15.0, 9.0]],
        index=["CPA_C20", "CPA_C21"],
        columns=["CPA_C20", "CPA_C21"],
    )
    expected_Y = pd.DataFrame([[30.0], [30.0]], index=["CPA_C20", "CPA_C21"], columns=["P3"])
    expected_V = pd.DataFrame([[300.0, 300.0]], index=["B1G"], columns=["CPA_C20", "CPA_C21"])
    expected_x = expected_Z.sum(axis=1) + expected_Y.sum(axis=1)

    pd.testing.assert_frame_equal(result.Z, expected_Z)
    pd.testing.assert_frame_equal(result.Y, expected_Y)
    pd.testing.assert_series_equal(result.x, expected_x)
    assert result.V is not None
    assert result.M is not None
    pd.testing.assert_frame_equal(result.V, expected_V)
    pd.testing.assert_frame_equal(result.M, M)


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
    from ecolecon_mrio.iot_aggregation import aggregate_iot
    from scripts.aggregate_iot_by_mapping import _augment_mapping_from_split_child_edges
    from scripts.validate_many_many_pipeline_outputs import validate_pipeline_outputs

    split_dir = tmp_path / "split"
    agg_dir = tmp_path / "agg"
    split_dir.mkdir()
    agg_dir.mkdir()

    children = ["S@CPA_C20", "A", "S@CPA_C21"]
    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        index=children,
        columns=children,
    )
    Y = pd.DataFrame([[10.0], [20.0], [30.0]], index=children, columns=["P3"])
    V = pd.DataFrame([[100.0, 200.0, 300.0]], index=["B1G"], columns=children)
    M = pd.DataFrame([[5.0]], index=["B1G"], columns=["P3"])
    x = Z.sum(axis=1) + Y.sum(axis=1)

    Z.to_csv(split_dir / "Z_split_balanced.csv")
    Y.to_csv(split_dir / "Y_split_balanced.csv")
    V.to_csv(split_dir / "V_split_balanced.csv")
    M.to_csv(split_dir / "M_split_balanced.csv")
    x.to_csv(split_dir / "x_split_balanced.csv")

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
    pd.DataFrame(
        [
            {
                "disaggregated_sector": "S@CPA_C20",
                "extension": "air_emissions",
                "stressor": "CO2",
                "unit": "kg",
                "emission": 2.0,
                "monetary_output": 4.0,
                "emission_intensity": 0.5,
            }
        ]
    ).to_csv(split_dir / "disaggregated_sector_co2_intensity.csv", index=False)
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
            {"source_code": "A", "target_code": "CPA_C20", "enabled": "1"},
            {"source_code": "S@CPA_C21", "target_code": "CPA_C21", "enabled": "1"},
        ]
    )
    mapping_df.to_csv(mapping_csv, index=False)
    mapping = {"A": "CPA_C20", "S@CPA_C21": "CPA_C21"}
    edge_meta = pd.read_csv(split_dir / "split_child_edges.csv")
    augmented_mapping, _, audit = _augment_mapping_from_split_child_edges(
        mapping=mapping,
        mapping_df=mapping_df.assign(from_sector=mapping_df["source_code"], to_sector=mapping_df["target_code"]),
        edge_metadata=edge_meta,
    )
    assert not audit.empty

    expected = aggregate_iot(Z, Y, augmented_mapping, V=V, M=M, keep_unmapped=False, create_io_system=False)
    expected.Z.to_csv(agg_dir / "Z_agg.csv")
    expected.Y.to_csv(agg_dir / "Y_agg.csv")
    expected.x.to_csv(agg_dir / "x_agg.csv")
    assert expected.V is not None
    assert expected.M is not None
    expected.V.to_csv(agg_dir / "V_agg.csv")
    expected.M.to_csv(agg_dir / "M_agg.csv")
    audit.to_csv(agg_dir / "aggregation_edge_mapping_audit.csv", index=False)

    report = validate_pipeline_outputs(
        split_dir=split_dir,
        agg_dir=agg_dir,
        mapping_csv=mapping_csv,
    )

    assert report["ok"] is True
    names = {check["name"] for check in report["checks"]}
    assert "Z_agg_matches_recomputed_reaggregation" in names
    assert "V_agg_matches_recomputed_reaggregation" in names
    assert "ratio_file::disaggregated_sector_co2_intensity.csv" in names


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

    class DummyExioSlice:
        # x_region is deliberately wrong for this destination-use check. The
        # consistency code must use Z_domestic + Y_region instead, especially for
        # IMP and TOTAL flow slices where x_region is producer-side output.
        x_region = pd.Series({"exio_a": 999.0}, dtype="float64")
        Z_domestic = pd.DataFrame([[4.0]], index=["exio_a"], columns=["exio_a"])
        Y_region = pd.DataFrame([[6.0]], index=["exio_a"], columns=["P3"])

    mapping = {"CPA_A": {"exio_a": 1.0}}
    Z = pd.DataFrame([[7.0]], index=["CPA_A"], columns=["CPA_A"])
    # P6 is export final demand. The mapped-output check must use Z + non-export
    # final demand (= 7 + 3), not Z + P3 + P6 (= 110).
    Y = pd.DataFrame([[3.0, 100.0]], index=["CPA_A"], columns=["P3", "P6"])

    frame, exio_total, euro_total = _build_output_consistency_frame(
        sector_mapping=mapping,
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_slice=DummyExioSlice(),
    )

    assert exio_total == 10.0
    assert euro_total == 10.0
    assert frame.loc[0, "eurostat_output"] == 10.0

    _assert_output_consistency(
        sector_mapping=mapping,
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_slice=DummyExioSlice(),
        rtol=0.0,
        atol=0.0,
        label="boundary-test",
    )
