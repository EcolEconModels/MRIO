from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _canonical_base(kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "source_dataset": "SRC",
        "source_system": "pxp",
        "source_version": "1",
        "source_level": "product",
        "target_dataset": "TGT",
        "target_system": "pxp",
        "target_version": "1",
        "target_level": "product",
        "source_code": "",
        "source_label": "",
        "target_code": "",
        "target_label": "",
        "order": "",
        "enabled": "1",
        "provenance": "methodological",
        "certainty": "derived",
        "allocation_rule": "test",
        "source_name": "sample.csv",
        "official_source": "",
        "notes": "",
        "source_balance_role": "",
        "target_sector_role": "",
        "include_in_direct_energy": "",
    }


def test_validate_mapping_weights_passes() -> None:
    from ecolecon_mrio.mapping_csv import validate_mapping_weights

    mapping = {
        "CPA_A": {"S1": 0.4, "S2": 0.6},
        "CPA_B": {"S1": 0.6, "S2": 0.4},
        "CPA_C": {"S3": 1.0},
    }
    validate_mapping_weights(mapping)


def test_validate_mapping_weights_fails_on_sum() -> None:
    from ecolecon_mrio.mapping_csv import validate_mapping_weights

    mapping = {"CPA_A": {"S1": 0.7}, "CPA_B": {"S1": 0.2}}
    with pytest.raises(ValueError):
        validate_mapping_weights(mapping)


def test_read_weighted_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import (
        load_weighted_mapping_frame,
        read_weighted_mapping,
    )

    rows = []
    for target, source, order in [
        ("CPA_A", "S1", "1"),
        ("CPA_B", "S1", "2"),
        ("CPA_A", "S2", "3"),
        ("CPA_B", "S2", "4"),
    ]:
        row = _canonical_base("weighted")
        row["target_code"] = target
        row["target_label"] = target
        row["source_code"] = source
        row["source_label"] = source
        row["order"] = order
        row["allocation_rule"] = "split-by-output-share"
        rows.append(row)
    csv_path = tmp_path / "weighted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_weighted_mapping_frame(csv_path)
    mapping = read_weighted_mapping(
        csv_path, target_output_by_code={"CPA_A": 2.0, "CPA_B": 6.0}
    )
    assert set(frame["aggregate_sector"]) == {"CPA_A", "CPA_B"}
    assert mapping["CPA_A"]["S1"] == pytest.approx(0.25)
    assert mapping["CPA_B"]["S2"] == pytest.approx(0.75)


def test_read_weighted_mapping_can_resolve_split_shares_from_year_outputs(
    tmp_path: Path,
) -> None:
    from ecolecon_mrio.mapping_csv import read_weighted_mapping

    rows = []
    for target, weight in [("CPA_A", "1"), ("CPA_B", "2")]:
        row = _canonical_base("weighted")
        row["target_code"] = target
        row["target_label"] = target
        row["source_code"] = "S1"
        row["source_label"] = "S1"
        row["order"] = weight
        row["certainty"] = "derived-output-share-split"
        row["allocation_rule"] = "split-by-output-share"
        rows.append(row)

    single = _canonical_base("weighted")
    single["target_code"] = "CPA_C"
    single["target_label"] = "CPA_C"
    single["source_code"] = "S2"
    single["source_label"] = "S2"
    single["order"] = ""
    single["certainty"] = "derived-single-target"
    rows.append(single)

    csv_path = tmp_path / "weighted_split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mapping = read_weighted_mapping(
        csv_path, target_output_by_code={"CPA_A": 2.0, "CPA_B": 6.0}
    )

    assert mapping["CPA_A"]["S1"] == pytest.approx(0.25)
    assert mapping["CPA_B"]["S1"] == pytest.approx(0.75)
    assert mapping["CPA_C"]["S2"] == pytest.approx(1.0)


def test_read_split_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_split_mapping_frame, read_split_mapping

    rows = []
    for source in ["Electricity by coal", "Electricity by gas"]:
        row = _canonical_base("split")
        row["target_code"] = "CPA_D"
        row["target_label"] = "CPA_D"
        row["source_code"] = source
        row["source_label"] = source
        rows.append(row)
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_split_mapping_frame(csv_path)
    mapping = read_split_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_D"}
    assert mapping["CPA_D"]["Electricity by coal"] == ["Electricity by coal"]
    assert mapping["CPA_D"]["Electricity by gas"] == ["Electricity by gas"]


def test_read_split_mapping_uses_code_columns(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_split_mapping_frame, read_split_mapping

    rows = []
    for source, label in [
        ("p40.11.a", "Electricity by coal"),
        ("p40.11.b", "Electricity by gas"),
    ]:
        row = _canonical_base("split")
        row["source_code"] = source
        row["source_label"] = label
        row["target_code"] = "CPA_D"
        row["target_label"] = "Electricity, gas, steam and air conditioning"
        rows.append(row)
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_split_mapping_frame(csv_path)
    mapping = read_split_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_D"}
    assert mapping == {"CPA_D": {"p40.11.a": ["p40.11.a"], "p40.11.b": ["p40.11.b"]}}


def test_read_split_manifest_uses_exio_to_eurostat_concordance(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import read_split_mapping

    concordance_path = tmp_path / "exio_to_eurostat_cpa_products.csv"
    concordance_rows = []
    for source_code, target_code in [
        ("p40.11.a", "CPA_D"),
        ("p40.11.b", "CPA_D"),
        ("p23.1.a", "CPA_C19"),
    ]:
        row = _canonical_base("weighted")
        row["source_code"] = source_code
        row["source_label"] = source_code
        row["target_code"] = target_code
        row["target_label"] = target_code
        row["weight"] = "1"
        row["enabled"] = "1"
        row["notes"] = ""
        concordance_rows.append(row)
    pd.DataFrame(concordance_rows).to_csv(concordance_path, index=False)

    manifest_path = tmp_path / "energy_green_brown_split.csv"
    pd.DataFrame(
        [
            {
                "source_code": "CPA_D",
                "source_label": "Electricity, gas, steam and air conditioning",
            },
            {
                "source_code": "CPA_C19",
                "source_label": "Coke and refined petroleum products",
            },
        ]
    ).to_csv(manifest_path, index=False)

    mapping = read_split_mapping(
        manifest_path, exio_to_eurostat_mapping_csv=concordance_path
    )
    assert mapping == {
        "CPA_D": ["p40.11.a", "p40.11.b"],
        "CPA_C19": ["p23.1.a"],
    }


def test_read_aggregation_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import (
        load_aggregation_mapping_frame,
        read_aggregation_mapping,
    )

    rows = []
    for source, target in [("A", "X"), ("B", "X")]:
        row = _canonical_base("aggregate")
        row["source_code"] = source
        row["source_label"] = source
        row["target_code"] = target
        row["target_label"] = target
        rows.append(row)
    csv_path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_aggregation_mapping_frame(csv_path)
    mapping, returned_frame = read_aggregation_mapping(csv_path)
    assert set(frame["from_sector"]) == {"A", "B"}
    assert mapping == {"A": "X", "B": "X"}
    assert len(returned_frame) == 2


def test_read_aggregation_manifest(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import (
        load_aggregation_mapping_frame,
        read_aggregation_mapping,
    )

    csv_path = tmp_path / "energy_green_brown_reaggregation.csv"
    pd.DataFrame(
        [
            {
                "source_code": "p40.11.d",
                "source_label": "Electricity by hydro",
                "target_code": "ENERGY_GREEN",
                "target_label": "Energy Green",
                "enabled": "True",
                "notes": "",
            },
            {
                "source_code": "p40.11.a",
                "source_label": "Electricity by coal",
                "target_code": "ENERGY_BROWN",
                "target_label": "Energy Brown",
                "enabled": "True",
                "notes": "",
            },
        ]
    ).to_csv(csv_path, index=False)

    frame = load_aggregation_mapping_frame(csv_path)
    mapping, returned_frame = read_aggregation_mapping(csv_path)
    assert set(frame["from_sector"]) == {"p40.11.d", "p40.11.a"}
    assert mapping == {"p40.11.d": "ENERGY_GREEN", "p40.11.a": "ENERGY_BROWN"}
    assert len(returned_frame) == 2


def test_read_category_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_code"] = "Industry"
    row["source_label"] = "Industry"
    row["target_code"] = "Chemicals"
    row["target_label"] = "Chemicals"
    row["source_balance_role"] = "final_use"
    row["target_sector_role"] = "end_use_sector"
    row["include_in_direct_energy"] = "1"
    csv_path = tmp_path / "category.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    frame = load_category_mapping_frame(csv_path)
    assert frame.loc[0, "Category"] == "Industry"
    assert frame.loc[0, "TargetSector"] == "Chemicals"
    assert frame.loc[0, "TargetCode"] == "Chemicals"
    assert frame.loc[0, "AllocationRule"] == "test"
    assert frame.loc[0, "SourceBalanceRole"] == "final_use"
    assert frame.loc[0, "TargetSectorRole"] == "end_use_sector"
    assert bool(frame.loc[0, "IncludeInDirectEnergy"])


def test_read_category_mapping_requires_explicit_semantic_roles(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_code"] = "Industry"
    row["source_label"] = "Industry"
    row["target_code"] = "Chemicals"
    row["target_label"] = "Chemicals"
    csv_path = tmp_path / "category_missing_semantics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="source_balance_role"):
        load_category_mapping_frame(csv_path)


def test_read_category_mapping_parses_semantic_roles(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_code"] = "TI_RPI_DU_E"
    row["source_label"] = (
        "Transformation input - refineries and petrochemical industry - direct use - energy use"
    )
    row["target_code"] = "CPA_C19"
    row["target_label"] = "Coke and refined petroleum products"
    row["order"] = "1"
    row["allocation_rule"] = "fixed-share"
    row["source_balance_role"] = "transformation_input"
    row["target_sector_role"] = "transformation_sector"
    row["include_in_direct_energy"] = "1"
    csv_path = tmp_path / "category_semantics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    frame = load_category_mapping_frame(csv_path)
    assert frame.loc[0, "SourceBalanceRole"] == "transformation_input"
    assert frame.loc[0, "TargetSectorRole"] == "transformation_sector"
    assert bool(frame.loc[0, "IncludeInDirectEnergy"])


def test_static_mappings_reject_duplicate_source_codes(tmp_path: Path) -> None:
    from ecolecon_mrio.canonical_mapping_csv import load_canonical_mapping_frame

    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(
        [
            {"code": "A", "label": "A", "description": "A"},
            {"code": "B", "label": "B", "description": "B"},
        ]
    ).to_csv(dictionary_path, index=False)

    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        [
            {
                "source_code": "A",
                "source_label": "A",
                "target_code": "B",
                "target_label": "B",
                "order": 1,
                "enabled": True,
                "notes": "ok",
            },
            {
                "source_code": "A",
                "source_label": "A",
                "target_code": "A",
                "target_label": "A",
                "order": 2,
                "enabled": True,
                "notes": "dup",
            },
        ]
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="duplicate source codes"):
        load_canonical_mapping_frame(
            mapping_path, dictionary_path, validate_target_codes=False
        )


def test_resolve_weighted_mapping_frame_accepts_dataframe_and_runtime_reference() -> (
    None
):
    from ecolecon_mrio.mapping_csv import resolve_weighted_mapping_frame

    rows = []
    for target in ["CPA_A", "CPA_B"]:
        row = _canonical_base("weighted")
        row["source_code"] = "S1"
        row["source_label"] = "Source 1"
        row["target_code"] = target
        row["target_label"] = target
        row["allocation_rule"] = "split-by-output-share"
        rows.append(row)
    frame = pd.DataFrame(rows)

    resolved = resolve_weighted_mapping_frame(
        frame, target_output_by_code={"CPA_A": 2.0, "CPA_B": 6.0}
    )

    weights = dict(zip(resolved["target_code"], resolved["weight"], strict=False))
    assert weights == {"CPA_A": pytest.approx(0.25), "CPA_B": pytest.approx(0.75)}
