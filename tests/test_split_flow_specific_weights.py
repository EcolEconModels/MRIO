from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_target_product_schema import CPA65_CODES
from scripts.split_eurostat_energy_by_exiobase import (
    _eurostat_non_export_use_reference_from_parsed,
    _mapping_weights_frame,
    _resolve_exio_to_eurostat_mapping_for_parsed,
)


def _canonical_base(kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "source_dataset": "EXIOBASE",
        "source_system": "pxp",
        "source_version": "3",
        "source_level": "product",
        "target_dataset": "Eurostat",
        "target_system": "CPA",
        "target_version": "65",
        "target_level": "product",
        "source_code": "",
        "source_label": "",
        "target_code": "",
        "target_label": "",
        "order": "",
        "weight": "",
        "enabled": "1",
        "provenance": "methodological",
        "certainty": "derived-output-share-split",
        "allocation_rule": "split-by-output-share",
        "source_name": "test.csv",
        "official_source": "",
        "notes": "",
        "source_balance_role": "",
        "target_sector_role": "",
        "include_in_direct_energy": "",
    }


def _write_weighted_mapping(path: Path) -> None:
    rows = []
    for order, target in enumerate(CPA65_CODES, start=1):
        row = _canonical_base("weighted")
        row["source_code"] = "p24.d" if target in {"CPA_C20", "CPA_C21"} else f"source_{target}"
        row["source_label"] = "Chemicals nec" if target in {"CPA_C20", "CPA_C21"} else target
        row["target_code"] = target
        row["target_label"] = target
        row["order"] = str(order)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _parsed_iot(*, c20_non_export_y: float, c21_non_export_y: float) -> ParsedEurostatIOT:
    codes = pd.Index(CPA65_CODES)
    z = pd.DataFrame(
        0.0,
        index=codes,
        columns=codes,
    )
    y = pd.DataFrame(
        {
            "P3": [c20_non_export_y if code == "CPA_C20" else c21_non_export_y if code == "CPA_C21" else 0.0 for code in codes],
            "P6": [9999.0] * len(codes),
        },
        index=codes,
    )
    return ParsedEurostatIOT(Z=z, Y=y, meta={})


def test_non_export_reference_excludes_exports() -> None:
    parsed = _parsed_iot(c20_non_export_y=90.0, c21_non_export_y=10.0)

    reference = _eurostat_non_export_use_reference_from_parsed(parsed)

    assert reference["CPA_C20"] == pytest.approx(90.0)
    assert reference["CPA_C21"] == pytest.approx(10.0)


def test_flow_specific_weights_use_each_flows_own_reference(tmp_path: Path) -> None:
    mapping_path = tmp_path / "weighted.csv"
    _write_weighted_mapping(mapping_path)

    dom_mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(
        mapping_path,
        _parsed_iot(c20_non_export_y=90.0, c21_non_export_y=10.0),
    )
    imp_mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(
        mapping_path,
        _parsed_iot(c20_non_export_y=40.0, c21_non_export_y=60.0),
    )
    total_mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(
        mapping_path,
        _parsed_iot(c20_non_export_y=130.0, c21_non_export_y=70.0),
    )

    assert dom_mapping["CPA_C20"]["p24.d"] == pytest.approx(0.9)
    assert imp_mapping["CPA_C20"]["p24.d"] == pytest.approx(0.4)
    assert total_mapping["CPA_C20"]["p24.d"] == pytest.approx(0.65)


def test_mapping_weights_frame_records_reference_basis() -> None:
    frame = _mapping_weights_frame(
        {"CPA_C20": {"p24.d": 0.65}, "CPA_C21": {"p24.d": 0.35}},
        variant_dim="stk_flow",
        variant_value="TOTAL",
        exio_flow="total",
    )

    assert set(frame["target_code"]) == {"CPA_C20", "CPA_C21"}
    assert set(frame["reference_basis"]) == {"eurostat_Z_plus_non_export_Y"}
    assert frame.loc[frame["target_code"] == "CPA_C20", "weight"].iloc[0] == pytest.approx(0.65)
