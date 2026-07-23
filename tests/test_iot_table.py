from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.iot_table import BlockName, CanonicalIOT, MatrixBlock


def _block(
    name: BlockName,
    frame: pd.DataFrame,
    applicability_mask: pd.DataFrame | None = None,
) -> MatrixBlock:
    return MatrixBlock(
        name=name,
        values=frame,
        row_dimension="row",
        column_dimension="column",
        applicability_mask=applicability_mask,
    )


def test_canonical_iot_excludes_unselected_published_controls_from_output() -> None:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    Y = pd.DataFrame(
        {
            "P3": [99.0, 99.0],
            "P3_S13": [1.0, 2.0],
            "P3_S14": [2.0, 3.0],
            "P3_S15": [3.0, 4.0],
        },
        index=idx,
    )
    table = CanonicalIOT(
        Z=_block("Z", Z),
        Y=_block("Y", Y),
        final_demand_columns=("P3_S13", "P3_S14", "P3_S15"),
    )

    expected = Z.sum(axis=1) + Y[["P3_S13", "P3_S14", "P3_S15"]].sum(axis=1)
    pd.testing.assert_series_equal(table.output(), expected.rename("x").astype("float64"))
    assert list(table.computational_final_demand().columns) == ["P3_S13", "P3_S14", "P3_S15"]


def test_matrix_block_preserves_zero_and_non_applicability_as_distinct_states() -> None:
    frame = pd.DataFrame(
        [[0.0, np.nan]],
        index=pd.Index(["row"]),
        columns=pd.Index(["zero", "not_applicable"]),
    )
    block = _block("V", frame)

    assert block.applicability_mask is not None
    assert bool(block.applicability_mask.loc["row", "zero"])
    assert not bool(block.applicability_mask.loc["row", "not_applicable"])
    assert block.values.loc["row", "zero"] == 0.0
    assert np.isnan(block.values.loc["row", "not_applicable"])


def test_matrix_block_rejects_values_outside_explicit_applicability_mask() -> None:
    frame = pd.DataFrame([[1.0]], index=pd.Index(["row"]), columns=pd.Index(["column"]))
    mask = pd.DataFrame([[False]], index=frame.index, columns=frame.columns)
    with pytest.raises(ValueError, match="non-applicable cells must be represented by NaN"):
        _block("M", frame, mask)


def test_canonical_iot_requires_complete_supplementary_domains() -> None:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame(0.0, index=idx, columns=idx)
    Y = pd.DataFrame({"FD": [1.0, 2.0]}, index=idx)
    V = pd.DataFrame([[1.0]], index=pd.Index(["P7"]), columns=pd.Index(["a"]))

    with pytest.raises(ValueError, match="V columns must exactly match"):
        CanonicalIOT(
            Z=_block("Z", Z),
            Y=_block("Y", Y),
            V=_block("V", V),
            final_demand_columns=("FD",),
        )


def test_canonical_iot_rejects_non_applicable_selected_final_demand() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[0.0]], index=idx, columns=idx)
    Y = pd.DataFrame([[np.nan]], index=idx, columns=pd.Index(["FD"]))

    with pytest.raises(ValueError, match="Selected final-demand cells must all be applicable"):
        CanonicalIOT(
            Z=_block("Z", Z),
            Y=_block("Y", Y),
            final_demand_columns=("FD",),
        )
