"""Regression tests for exact Eurostat target-product semantic boundaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_target_product_schema import (
    CPA64_CODES,
    CPA65_CODES,
    CONSOLIDATED64_SOURCE_CODES,
    detect_target_product_schema,
    project_reviewed_cpa65_to_cpa64,
)
from scripts.split_eurostat_energy_by_exiobase import (
    _mapping_weights_frame,
    _resolve_exio_to_eurostat_mapping_for_parsed,
)


def _mapping_frame() -> pd.DataFrame:
    """Make a complete reviewed-style CPA65 mapping with p70 as the only L68 split."""
    rows: list[dict[str, str]] = []
    for order, target in enumerate(CPA65_CODES, start=1):
        rows.append(
            {
                "order": str(order), "kind": "weighted", "source_dataset": "EXIOBASE",
                "source_system": "pxp", "source_version": "3", "source_level": "product",
                "target_dataset": "Eurostat", "target_system": "CPA", "target_version": "CPA_65",
                "target_level": "product", "source_code": "p70" if target.startswith("CPA_L68") else f"source_{target}",
                "source_label": "Real estate" if target.startswith("CPA_L68") else target,
                "target_code": target, "target_label": target, "enabled": "true",
                "provenance": "reviewed", "certainty": "reviewed",
                "allocation_rule": "split-by-output-share" if target.startswith("CPA_L68") else "direct-single-target",
                "source_name": "test", "official_source": "test", "notes": "reviewed",
                "source_balance_role": "", "target_sector_role": "", "include_in_direct_energy": "",
            }
        )
    return pd.DataFrame(rows)


def _parsed(codes: tuple[str, ...]) -> ParsedEurostatIOT:
    """Construct a complete source axis while keeping L68 use values identifiable."""
    axis = pd.Index(codes)
    z = pd.DataFrame(0.0, index=axis, columns=axis)
    y = pd.DataFrame({"P3": [30.0 if code.endswith("L68A") else 70.0 if code.endswith("L68B") else 1.0 for code in codes]}, index=axis)
    return ParsedEurostatIOT(Z=z, Y=y, meta={})


def test_exact_cpa65_schema_keeps_real_estate_split_and_runtime_weights(tmp_path: Path) -> None:
    """The national 65-product source axis must retain its two native targets."""
    path = tmp_path / "reviewed.csv"
    _mapping_frame().to_csv(path, index=False)
    schema = detect_target_product_schema(CPA65_CODES)
    assert schema.schema_id == "CPA65"

    mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(path, _parsed(CPA65_CODES))
    assert mapping["CPA_L68A"]["p70"] == pytest.approx(0.3)
    assert mapping["CPA_L68B"]["p70"] == pytest.approx(0.7)


def test_consolidated_schema_resolves_canonical_l68_back_to_native_carrier(tmp_path: Path) -> None:
    """The consolidated source carrier must receive the unsplit CPA64 p70 edge."""
    path = tmp_path / "reviewed.csv"
    _mapping_frame().to_csv(path, index=False)
    parsed = _parsed(CONSOLIDATED64_SOURCE_CODES)
    schema = detect_target_product_schema(parsed.Z.index)
    assert schema.schema_id == "EUROSTAT_CONSOLIDATED_CPA64"
    assert "CPA_L68" in schema.canonicalize_reference(
        parsed.Z.sum(axis=1) + parsed.Y.sum(axis=1)
    ).index
    assert "CPA_L68B" not in schema.canonical_codes

    mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(path, parsed)
    assert mapping["CPA_L68B"]["p70"] == pytest.approx(1.0)
    assert "CPA_L68" not in mapping
    assert "CPA_L68B" in parsed.Z.index and "CPA_L68" not in parsed.Z.index

    diagnostic = _mapping_weights_frame(mapping, target_schema=schema)
    row = diagnostic.loc[diagnostic["source_code"].eq("p70")].iloc[0]
    assert row["target_code"] == "CPA_L68B"
    assert row["canonical_target_code"] == "CPA_L68"
    assert row["target_product_schema"] == "EUROSTAT_CONSOLIDATED_CPA64"


def test_unregistered_axes_fail_before_concordance_resolution(tmp_path: Path) -> None:
    """Neither a one-target map nor a coincidental missing L68A can bypass schema validation."""
    path = tmp_path / "one_target.csv"
    _mapping_frame().iloc[:1].to_csv(path, index=False)
    malformed = tuple(code for code in CONSOLIDATED64_SOURCE_CODES if code != "CPA_M71")
    with pytest.raises(ValueError, match="Unrecognised Eurostat target-product source axis"):
        _resolve_exio_to_eurostat_mapping_for_parsed(path, _parsed(malformed))


def test_every_other_consolidated_target_is_required() -> None:
    """The registered 64-product domain is exact, not a generic '65 minus L68A' rule."""
    for code in CPA64_CODES:
        if code == "CPA_L68":
            continue
        malformed = tuple(value for value in CONSOLIDATED64_SOURCE_CODES if value != code)
        with pytest.raises(ValueError, match="Unrecognised Eurostat target-product source axis"):
            detect_target_product_schema(malformed)


def test_reviewed_projection_preserves_non_real_estate_production_corrections() -> None:
    """CPA64 selection must inherit reviewed p23.3, p24.*, and p33 decisions."""
    reviewed = pd.read_csv(
        Path("scripts/mappings/exio_to_eurostat_cpa65_products_official_direct_reviewed.csv"),
        dtype=str,
    ).fillna("")
    projected = project_reviewed_cpa65_to_cpa64(reviewed)
    targets = lambda source: set(projected.loc[projected["source_code"].eq(source), "target_code"])
    assert targets("p70") == {"CPA_L68"}
    assert targets("p23.3") == {"CPA_C24"}
    assert targets("p24.a") == {"CPA_C20"}
    assert targets("p24.d") == {"CPA_C20", "CPA_C21"}
    assert targets("p33") == {"CPA_C26"}
