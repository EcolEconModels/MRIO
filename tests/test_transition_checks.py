from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio.transition_checks import (
    annualized_growth_rate,
    compound_endpoint,
    compute_closure_growth,
    percent_deviation,
    summarize_scope_totals,
)


def test_scope_totals_split_households_correctly() -> None:
    frame = pd.DataFrame(
        {
            "Sector": ["Industry A", "Industry B", "Households"],
            "Ren": [1000.0, 500.0, 250.0],
            "Nren": [400.0, 100.0, 50.0],
        }
    )

    totals = summarize_scope_totals(frame, ["Ren", "Nren"])
    assert totals.all_total == 2.3
    assert totals.industry_total == 2.0
    assert totals.household_total == 0.3


def test_compound_endpoint_round_trip() -> None:
    eps = annualized_growth_rate(10.0, 20.0, 4)
    assert compound_endpoint(10.0, eps, 5) == pytest.approx(20.0)


def test_percent_deviation_handles_zero_reference() -> None:
    assert np.isnan(percent_deviation(2.0, 0.0))


def test_compute_closure_growth_residual_others_hits_identity() -> None:
    weights = np.array([20.0, 30.0, 50.0, 100.0])
    out = compute_closure_growth(
        option="residual-others",
        eps_ren=0.1,
        eps_nren=-0.2,
        aggregate_growth=0.05,
        weights=weights,
        renewable_indices=[0],
        nonrenewable_indices=[1],
    )
    assert out.weighted_total_prev == 200.0
    assert out.weighted_total_next == 210.0
    assert out.g_ren == 0.1
    assert out.g_nren == -0.2


def test_compute_closure_growth_fixed_others_hits_identity() -> None:
    weights = np.array([20.0, 30.0, 50.0, 100.0])
    out = compute_closure_growth(
        option="fixed-others",
        eps_ren=0.1,
        eps_nren=-0.2,
        aggregate_growth=0.05,
        weights=weights,
        renewable_indices=[0],
        nonrenewable_indices=[1],
    )
    assert out.weighted_total_next == 210.0
    assert out.g_other == 0.05


def test_check_transition_closure_script_payload(tmp_path: Path) -> None:
    weights_csv = tmp_path / "weights.csv"
    pd.DataFrame(
        {
            "sector": ["ren", "nren", "other"],
            "weight": [20.0, 30.0, 150.0],
            "block": ["ren", "nren", "other"],
        }
    ).to_csv(weights_csv, index=False)

    from scripts.diagnostics.check_transition_closure_targets import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--weights-csv",
                str(weights_csv),
                "--years",
                "5",
                "--ren-start",
                "10",
                "--ren-end",
                "20",
                "--nren-start",
                "10",
                "--nren-end",
                "5",
                "--aggregate-growth",
                "0.03",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["closure_option"] == "residual-others"
    assert payload["weighted_total_prev"] == 200.0


def test_check_energy_workbook_scope_script_payload(tmp_path: Path) -> None:
    workbook = tmp_path / "scope.xlsx"
    pd.DataFrame(
        {
            "Sector": ["Industry A", "Households"],
            "REN": [1000.0, 200.0],
            "NREN": [500.0, 300.0],
        }
    ).to_excel(workbook, sheet_name="Energy", index=False)

    from scripts.diagnostics.check_energy_workbook_scope import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--workbook",
                str(workbook),
                "--sheet",
                "Energy",
                "--group-name",
                "total",
                "--group-cols",
                "REN,NREN",
                "--reference-total",
                "2.0",
            ]
        )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["group_name"] == "total"
    assert payload["all_total"] == 2.0
    assert payload["household_total"] == 0.5