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
        "source_key": "",
        "source_code": "",
        "source_label": "",
        "target_key": "",
        "target_code": "",
        "target_label": "",
        "subtarget_key": "",
        "weight": "",
        "share": "",
        "enabled": "1",
        "provenance": "methodological",
        "certainty": "derived",
        "transformation_rule": "test",
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
    from ecolecon_mrio.mapping_csv import load_weighted_mapping_frame, read_weighted_mapping

    rows = []
    for target, source, weight in [("CPA_A", "S1", "0.4"), ("CPA_B", "S1", "0.6"), ("CPA_A", "S2", "0.6"), ("CPA_B", "S2", "0.4")]:
        row = _canonical_base("weighted")
        row["target_key"] = target
        row["source_key"] = source
        row["weight"] = weight
        rows.append(row)
    csv_path = tmp_path / "weighted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_weighted_mapping_frame(csv_path)
    mapping = read_weighted_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_A", "CPA_B"}
    assert mapping["CPA_A"]["S1"] == 0.4
    assert mapping["CPA_B"]["S2"] == 0.4


def test_read_split_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_split_mapping_frame, read_split_mapping

    rows = []
    for source, subtarget in [("Electricity by coal", "Electricity generation"), ("Electricity by gas", "Electricity generation")]:
        row = _canonical_base("split")
        row["target_key"] = "CPA_D"
        row["source_key"] = source
        row["subtarget_key"] = subtarget
        rows.append(row)
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_split_mapping_frame(csv_path)
    mapping = read_split_mapping(csv_path)
    assert set(frame["aggregate_sector"]) == {"CPA_D"}
    assert mapping["CPA_D"]["Electricity generation"] == ["Electricity by coal", "Electricity by gas"]


def test_read_split_mapping_expands_manifest_with_exio_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_split_manifest_frame, read_split_mapping

    eurostat_dictionary = tmp_path / "eurostat_cpa65_products_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "CPA_C19", "label": "Coke and refined petroleum products", "description": "Eurostat"},
            {"code": "CPA_D", "label": "Electricity, gas, steam and air conditioning", "description": "Eurostat"},
        ]
    ).to_csv(eurostat_dictionary, index=False)

    exio_dictionary = tmp_path / "exiobase_products_dictionary.csv"
    pd.DataFrame(
        [
            {"code": "p23.1", "label": "Coke oven products", "description": "EXIO"},
            {"code": "p23.2", "label": "Refined Petroleum", "description": "EXIO"},
            {"code": "p40.11.a", "label": "Electricity by coal", "description": "EXIO"},
            {"code": "p40.11.b", "label": "Electricity by gas", "description": "EXIO"},
        ]
    ).to_csv(exio_dictionary, index=False)

    manifest = tmp_path / "split_manifest.csv"
    pd.DataFrame(
        [
            {"source_code": "CPA_C19", "source_label": "Coke and refined petroleum products"},
            {"source_code": "CPA_D", "source_label": "Electricity, gas, steam and air conditioning"},
        ]
    ).to_csv(manifest, index=False)

    exio_mapping = tmp_path / "exio_to_eurostat_cpa_products.csv"
    rows = []
    for source_code, source_label, target_code, target_label in [
        ("p23.1", "Coke oven products", "CPA_C19", "Coke and refined petroleum products"),
        ("p23.2", "Refined Petroleum", "CPA_C19", "Coke and refined petroleum products"),
        ("p40.11.a", "Electricity by coal", "CPA_D", "Electricity, gas, steam and air conditioning"),
        ("p40.11.b", "Electricity by gas", "CPA_D", "Electricity, gas, steam and air conditioning"),
    ]:
        row = _canonical_base("weighted")
        row.update(
            {
                "source_key": source_label,
                "source_code": source_code,
                "source_label": source_label,
                "target_key": target_code,
                "target_code": target_code,
                "target_label": target_label,
                "weight": "1.0",
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(exio_mapping, index=False)

    frame = load_split_manifest_frame(manifest, eurostat_dictionary)
    mapping = read_split_mapping(
        manifest,
        exio_to_eurostat_mapping_csv=exio_mapping,
        source_dictionary_path=eurostat_dictionary,
        target_dictionary_path=eurostat_dictionary,
    )

    assert list(frame["source_code"]) == ["CPA_C19", "CPA_D"]
    assert mapping == {"CPA_C19": ["p23.1", "p23.2"], "CPA_D": ["p40.11.a", "p40.11.b"]}


def test_read_aggregation_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_aggregation_mapping_frame, read_aggregation_mapping

    rows = []
    for source, target in [("A", "X"), ("B", "X")]:
        row = _canonical_base("aggregate")
        row["source_key"] = source
        row["target_key"] = target
        rows.append(row)
    csv_path = tmp_path / "aggregate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame = load_aggregation_mapping_frame(csv_path)
    mapping, returned_frame = read_aggregation_mapping(csv_path)
    assert set(frame["from_sector"]) == {"A", "B"}
    assert mapping == {"A": "X", "B": "X"}
    assert len(returned_frame) == 2


def test_read_category_mapping(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_key"] = "Industry"
    row["source_label"] = "Industry"
    row["target_key"] = "Chemicals"
    row["target_label"] = "Chemicals"
    row["share"] = "0.75"
    row["source_balance_role"] = "final_use"
    row["target_sector_role"] = "end_use_sector"
    row["include_in_direct_energy"] = "1"
    csv_path = tmp_path / "category.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    frame = load_category_mapping_frame(csv_path)
    assert frame.loc[0, "Category"] == "Industry"
    assert frame.loc[0, "TargetSector"] == "Chemicals"
    assert frame.loc[0, "TargetCode"] == ""
    assert frame.loc[0, "DeclaredShare"] == 0.75
    assert frame.loc[0, "AllocationRule"] == "test"
    assert frame.loc[0, "SourceBalanceRole"] == "final_use"
    assert frame.loc[0, "TargetSectorRole"] == "end_use_sector"
    assert frame.loc[0, "IncludeInDirectEnergy"] == True


def test_read_category_mapping_requires_explicit_semantic_roles(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_key"] = "Industry"
    row["source_label"] = "Industry"
    row["target_key"] = "Chemicals"
    row["target_label"] = "Chemicals"
    row["share"] = "0.75"
    csv_path = tmp_path / "category_missing_semantics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="source_balance_role"):
        load_category_mapping_frame(csv_path)


def test_read_category_mapping_parses_semantic_roles(tmp_path: Path) -> None:
    from ecolecon_mrio.mapping_csv import load_category_mapping_frame

    row = _canonical_base("category-target")
    row["source_key"] = "Transformation input - refineries and petrochemical industry - direct use - energy use"
    row["source_code"] = "TI_RPI_DU_E"
    row["target_key"] = "Coke and refined petroleum products"
    row["target_code"] = "CPA_C19"
    row["target_label"] = "Coke and refined petroleum products"
    row["share"] = "1"
    row["transformation_rule"] = "fixed-share"
    row["source_balance_role"] = "transformation_input"
    row["target_sector_role"] = "transformation_sector"
    row["include_in_direct_energy"] = "1"
    csv_path = tmp_path / "category_semantics.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    frame = load_category_mapping_frame(csv_path)
    assert frame.loc[0, "SourceBalanceRole"] == "transformation_input"
    assert frame.loc[0, "TargetSectorRole"] == "transformation_sector"
    assert frame.loc[0, "IncludeInDirectEnergy"] == True