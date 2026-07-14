from __future__ import annotations

import pandas as pd
import pytest

from ecolecon_mrio.sector_allocation import allocate_parent_matrix_by_output


def test_allocate_parent_matrix_uses_final_iot_output_and_conserves_parents() -> None:
    values = pd.DataFrame({"stock": [100.0, 50.0], "p5": [20.0, 10.0]}, index=["CPA_C19", "CPA_D"])
    output = pd.Series([30.0, 70.0, 50.0], index=["CPA_C19: Coal", "CPA_C19: Oil", "CPA_D: Electricity"])

    allocated, audit = allocate_parent_matrix_by_output(values, final_output=output)

    assert allocated["stock"].tolist() == pytest.approx([30.0, 70.0, 50.0])
    by_parent = allocated.assign(parent_sector=audit["parent_sector"].to_numpy()).groupby("parent_sector")["p5"].sum()
    assert by_parent.tolist() == pytest.approx([20.0, 10.0])
    assert audit["share"].tolist() == pytest.approx([0.3, 0.7, 1.0])


def test_allocate_parent_matrix_rejects_incomplete_product_coverage() -> None:
    values = pd.DataFrame({"stock": [100.0]}, index=["CPA_C19"])
    output = pd.Series([30.0, 10.0], index=["CPA_C19: Coal", "CPA_D: Electricity"])

    with pytest.raises(ValueError, match="coverage mismatch"):
        allocate_parent_matrix_by_output(values, final_output=output)
