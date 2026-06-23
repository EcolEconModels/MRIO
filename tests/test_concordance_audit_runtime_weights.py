from __future__ import annotations

import pandas as pd
import pytest

from ecolecon_mrio.exio_eurostat_concordance_audit import build_concordance_audit


def test_concordance_audit_uses_eurostat_target_values_for_unweighted_splits() -> None:
    mapping_df = pd.DataFrame(
        [
            {
                "kind": "weighted",
                "source_code": "S1",
                "source_label": "Source 1",
                "target_code": "T1",
                "target_label": "Target 1",
                "enabled": "1",
                "allocation_rule": "split-by-output-share",
            },
            {
                "kind": "weighted",
                "source_code": "S1",
                "source_label": "Source 1",
                "target_code": "T2",
                "target_label": "Target 2",
                "enabled": "1",
                "allocation_rule": "split-by-output-share",
            },
        ]
    )
    exio_z = pd.DataFrame([[0.0]], index=["S1"], columns=["S1"])
    exio_y = pd.DataFrame({"FD": [100.0]}, index=["S1"])
    exio_x = pd.Series({"S1": 100.0})

    eurostat_z = pd.DataFrame(0.0, index=["T1", "T2"], columns=["T1", "T2"])
    eurostat_y = pd.DataFrame({"FD": [30.0, 70.0]}, index=["T1", "T2"])

    result = build_concordance_audit(
        country="AT",
        year=2021,
        exio_x=exio_x,
        exio_Z=exio_z,
        exio_Y=exio_y,
        eurostat_Z=eurostat_z,
        eurostat_Y=eurostat_y,
        mapping_df=mapping_df,
        exio_labels={"S1": "Source 1"},
        eurostat_labels={"T1": "Target 1", "T2": "Target 2"},
    )

    assert result.mapped_y is not None
    assert result.mapped_y["T1"] == pytest.approx(30.0)
    assert result.mapped_y["T2"] == pytest.approx(70.0)
    coverage = result.mapping_coverage.set_index("source_code")
    assert coverage.loc["S1", "sum_mapping_weight"] == pytest.approx(1.0)
