"""Protect the complete certified factor panel against source-vintage revisions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_exiobase_price_factor_rebase.py"
SPEC = importlib.util.spec_from_file_location("price_factor_rebase_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("defect", ["revision", "missing_region_year", "duplicate"])
def test_full_certified_panel_rejects_revisions_and_incomplete_support(defect: str) -> None:
    """A changed factor or any lost/duplicated observation invalidates certification."""
    reference = pd.read_csv(audit.REFERENCE, float_precision="round_trip")
    candidate = reference.copy()
    if defect == "revision":
        candidate.loc[0, "price_adjustment_factor"] += 1e-12
    elif defect == "missing_region_year":
        candidate = candidate.iloc[1:]
    else:
        candidate = pd.concat([candidate, candidate.iloc[:1]], ignore_index=True)
    with pytest.raises((AssertionError, ValueError)):
        audit.compare_reference(reference, candidate)


def test_reference_comparison_aligns_labels_and_allows_later_years() -> None:
    """The unchanged old domain passes after reordering and adding a later year."""
    reference = pd.read_csv(audit.REFERENCE, float_precision="round_trip")
    later = reference.loc[reference["year"].eq(2022)].assign(year=2023)
    candidate = pd.concat([later, reference.iloc[::-1]], ignore_index=True)
    audit.compare_reference(reference, candidate)
