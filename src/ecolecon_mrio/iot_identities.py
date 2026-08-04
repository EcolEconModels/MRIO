from __future__ import annotations

import re

def normalise_supplementary_row_label(row_code: object) -> str:
    label = " ".join(str(row_code).strip().casefold().replace("\\", "/").split())
    return re.sub(r"\s*/\s*", " /", label)


def supplementary_accounting_identity_kind(row_code: object) -> str | None:
    code = str(row_code).strip().upper()
    label = normalise_supplementary_row_label(row_code)
    if code in {"P1", "OUTPUT"} or label == "output":
        return "output"
    if code in {"P2_ADJ", "P2ADJ"} or label in {
        "total intermediate consumption /final use",
        "total intermediate consumption /final demand",
        "total intermediate consumption /final uses",
        "total intermediate use /final use",
    }:
        return "intermediate_use"
    if code in {"B1G", "B1GQ", "VALUE_ADDED_GROSS"} or label == "value added, gross":
        return "value_added"
    if code in {"TS_BP", "TOTAL_SUPPLY_AT_BASIC_PRICES"} or label == "total supply at basic prices":
        return "total_supply"
    return None


def is_imports_row(row_code: object) -> bool:
    """Recognise imports controls used by the read-only accounting audit."""
    code = str(row_code).strip().upper()
    label = normalise_supplementary_row_label(row_code)
    return code in {"P7", "P7_CIF", "P7_FOB", "IMPORTS"} or label in {
        "imports",
        "imports of goods and services",
        "imports cif",
        "imports fob",
    }
