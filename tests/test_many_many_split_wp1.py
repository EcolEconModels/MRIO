"""WP1 regression tests for conservative many-to-many Eurostat splitting."""

from __future__ import annotations

from pathlib import Path
import inspect

import pandas as pd
import pytest

from ecolecon_mrio.disaggregation import SplitProfiles
from ecolecon_mrio.exio_share_split import (
    _aggregate_rows_to_original,
    _validate_emitted_children,
    _validate_split_reconstruction,
    split_iot_using_exiobase_shares,
    split_iot_using_exiobase_split_profiles,
)
from ecolecon_mrio.exiobase3 import ExioRegionSlice
from ecolecon_mrio.mapping_csv import resolve_weighted_mapping_frame


def _core_fixture(
    tmp_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, ExioRegionSlice, dict[str, list[str]]]:
    """Build asymmetric parents, donors, and controls that expose axis or edge loss."""
    parents = ["A", "CPA_C20", "CPA_C21"]
    z = pd.DataFrame(
        [[2.0, 7.0, 11.0], [13.0, 17.0, 19.0], [23.0, 29.0, 31.0]],
        index=parents,
        columns=parents,
    )
    y = pd.DataFrame(
        {
            "P3_S13": [3.0, 5.0, 7.0],
            "P3_S14": [11.0, 13.0, 17.0],
            "P3_S15": [19.0, 23.0, 29.0],
            "P3": [33.0, 41.0, 53.0],
            "P51G": [2.0, 3.0, 5.0],
            "P52": [7.0, 11.0, 13.0],
            "P53": [17.0, 19.0, 23.0],
            "P5M": [24.0, 30.0, 36.0],
            "P5": [26.0, 33.0, 41.0],
            "P6": [31.0, 37.0, 41.0],
            "TFU": [90.0, 111.0, 135.0],
            "TU": [110.0, 160.0, 218.0],
        },
        index=parents,
    )
    donors = ["p24.d", "p24.g", "p24.h"]
    donor_z = pd.DataFrame(
        [[3.0, 2.0, 1.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]],
        index=donors,
        columns=donors,
    )
    donor_y = pd.DataFrame({"Households": [2.0, 3.0, 5.0]}, index=donors)
    slice_ = ExioRegionSlice(
        year=2021,
        system="pxp",
        region="AT",
        flow="domestic",
        sectors=pd.Index(donors),
        Z_domestic=donor_z,
        Y_region=donor_y,
        x_region=(donor_z.sum(axis=1) + donor_y.sum(axis=1)),
        source_zip=tmp_path / "donors.zip",
    )
    return z, y, slice_, {"CPA_C20": ["p24.d", "p24.g"], "CPA_C21": ["p24.d", "p24.h"]}


def test_wp1_keeps_zero_weight_edge_children_and_reconstructs_cells(
    tmp_path: Path,
) -> None:
    """A zero DOM edge stays labelled while each asymmetric source Z cell reconstructs."""
    z, y, donor_slice, split_mapping = _core_fixture(tmp_path)
    dom_edges = {
        "CPA_C20": {"p24.d": 0.0, "p24.g": 1.0},
        "CPA_C21": {"p24.d": 1.0, "p24.h": 1.0},
    }
    imp_edges = {
        "CPA_C20": {"p24.d": 0.5, "p24.g": 1.0},
        "CPA_C21": {"p24.d": 0.5, "p24.h": 1.0},
    }

    def export_loader(**_kwargs: object) -> pd.Series:
        """Supply an explicit zero export donor without requiring an EXIO zip."""
        return pd.Series(0.0, index=donor_slice.sectors)

    dom = split_iot_using_exiobase_split_profiles(
        eurostat_Z=z,
        eurostat_Y=y,
        exio_slice=donor_slice,
        sector_to_exio_sectors=split_mapping,
        eurostat_sector_mapping=dom_edges,
        export_vector_loader=export_loader,
        create_io_system=False,
    )
    imp = split_iot_using_exiobase_split_profiles(
        eurostat_Z=z,
        eurostat_Y=y,
        exio_slice=donor_slice,
        sector_to_exio_sectors=split_mapping,
        eurostat_sector_mapping=imp_edges,
        export_vector_loader=export_loader,
        create_io_system=False,
    )

    assert {"p24.d@CPA_C20", "p24.d@CPA_C21"}.issubset(dom.Z.index)
    assert list(dom.Z.index) == list(imp.Z.index)
    assert dom.Y["P5M"].equals(dom.Y["P52"] + dom.Y["P53"])
    assert dom.Y["P5"].equals(dom.Y["P51G"] + dom.Y["P5M"])
    pd.testing.assert_frame_equal(
        _aggregate_rows_to_original(
            dom.Y.loc[:, ["P3_S13", "P51G", "P52", "P53"]],
            split_profiles=dom.split_profiles,
        ).reindex(index=y.index),
        y.loc[:, ["P3_S13", "P51G", "P52", "P53"]],
    )
    for row in z.index:
        for column in z.columns:
            row_children = (
                [row]
                if row == "A"
                else [
                    child
                    for child in dom.Z.index
                    if child.endswith(f"@{row}")
                    or (row == "CPA_C20" and child == "p24.g")
                    or (row == "CPA_C21" and child == "p24.h")
                ]
            )
            col_children = (
                [column]
                if column == "A"
                else [
                    child
                    for child in dom.Z.columns
                    if child.endswith(f"@{column}")
                    or (column == "CPA_C20" and child == "p24.g")
                    or (column == "CPA_C21" and child == "p24.h")
                ]
            )
            assert dom.Z.loc[
                row_children, col_children
            ].to_numpy().sum() == pytest.approx(z.loc[row, column])


def test_wp1_rejects_unauthorised_duplicate_mapping_and_missing_donor(
    tmp_path: Path,
) -> None:
    """Duplicate mapping rows cannot authorise splitting, and donor absence reports its edge."""
    rows = pd.DataFrame(
        {
            "kind": ["weighted", "weighted"],
            "source_code": ["p24.d", "p24.d"],
            "target_code": ["CPA_C20", "CPA_C21"],
            "enabled": ["1", "1"],
        }
    )
    with pytest.raises(
        ValueError,
        match="requires explicit numerical weights or a recognised allocation_rule",
    ):
        resolve_weighted_mapping_frame(
            rows, target_output_by_code={"CPA_C20": 1.0, "CPA_C21": 1.0}
        )

    z, y, donor_slice, split_mapping = _core_fixture(tmp_path)
    missing_slice = ExioRegionSlice(
        year=2021,
        system="pxp",
        region="AT",
        flow="imports",
        sectors=pd.Index(["p24.g", "p24.h"]),
        Z_domestic=donor_slice.Z_domestic.loc[["p24.g", "p24.h"], ["p24.g", "p24.h"]],
        Y_region=donor_slice.Y_region.loc[["p24.g", "p24.h"]],
        x_region=donor_slice.x_region.loc[["p24.g", "p24.h"]],
        source_zip=tmp_path / "missing.zip",
    )
    with pytest.raises(
        ValueError,
        match="missing_exio_code=.*p24.d.*eurostat_parent=.*CPA_C20.*emitted_child=.*p24.d",
    ):
        split_iot_using_exiobase_split_profiles(
            eurostat_Z=z,
            eurostat_Y=y,
            exio_slice=missing_slice,
            sector_to_exio_sectors=split_mapping,
            create_io_system=False,
        )


def test_wp1_accepts_explicit_weights_and_rejects_cellwise_or_p5m_drift(
    tmp_path: Path,
) -> None:
    """Valid static splits work, while offsetting Z cells and derived P5M cannot pass."""
    explicit = pd.DataFrame(
        {
            "kind": ["weighted", "weighted"],
            "source_code": ["p24.d", "p24.d"],
            "target_code": ["CPA_C20", "CPA_C21"],
            "weight": [0.25, 0.75],
            "enabled": ["1", "1"],
        }
    )
    resolved = resolve_weighted_mapping_frame(explicit)
    assert resolved["weight"].tolist() == pytest.approx([0.25, 0.75])

    z, y, donor_slice, split_mapping = _core_fixture(tmp_path)
    edges = {
        "CPA_C20": {"p24.d": 0.5, "p24.g": 1.0},
        "CPA_C21": {"p24.d": 0.5, "p24.h": 1.0},
    }
    result = split_iot_using_exiobase_split_profiles(
        eurostat_Z=z,
        eurostat_Y=y,
        exio_slice=donor_slice,
        sector_to_exio_sectors=split_mapping,
        eurostat_sector_mapping=edges,
        export_vector_loader=lambda **_kwargs: pd.Series(
            0.0, index=donor_slice.sectors
        ),
        create_io_system=False,
    )
    mutated_z = result.Z.copy()
    # This 2x2 cycle preserves every expanded row and column total, so only the
    # required labelled-cell reconstruction check can expose the defect.
    mutated_z.loc["A", "A"] += 1.0
    mutated_z.loc["p24.g", "p24.g"] += 1.0
    mutated_z.loc["A", "p24.g"] -= 1.0
    mutated_z.loc["p24.g", "A"] -= 1.0
    with pytest.raises(ValueError, match="Z reconstruction failed: parent_row"):
        _validate_split_reconstruction(
            eurostat_Z=z,
            eurostat_Y=y,
            split_profiles=result.split_profiles,
            Z_split=mutated_z,
            Y_split=result.Y,
            x_split=result.x,
        )

    corrupted_y = result.Y.copy()
    corrupted_y.loc["p24.g", "P5M"] += 1.0
    with pytest.raises(ValueError, match="Generated P5M control failed"):
        _validate_split_reconstruction(
            eurostat_Z=z,
            eurostat_Y=y,
            split_profiles=result.split_profiles,
            Z_split=result.Z,
            Y_split=corrupted_y,
            x_split=result.x,
        )


def test_wp1_rejects_child_collisions_and_removes_inactive_balancing_api() -> None:
    """A child owns one parent, and donor-block APIs expose no inactive RAS controls."""
    share = pd.Series([1.0], index=["edge"], dtype="float64")
    profiles = {
        "CPA_C20": SplitProfiles("CPA_C20", ("edge",), share, share, share, share),
        "CPA_C21": SplitProfiles("CPA_C21", ("edge",), share, share, share, share),
    }
    with pytest.raises(ValueError, match="not unique.*edge"):
        _validate_emitted_children(
            sectors_old=["A", "CPA_C20", "CPA_C21"], split_profiles=profiles
        )

    for function in (
        split_iot_using_exiobase_split_profiles,
        split_iot_using_exiobase_shares,
    ):
        parameters = inspect.signature(function).parameters
        assert {"balancing_scope", "ras_max_iter", "ras_tol"}.isdisjoint(parameters)
