"""Validated Eurostat target-product schemas for concordance resolution.

Eurostat's parsed matrices retain their published product codes.  This module
only supplies the semantic target space in which an EXIOBASE concordance is
resolved.  In particular, the consolidated 64-product table publishes its
merged real-estate aggregate under the source carrier ``CPA_L68B``.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

import pandas as pd


CPA64_CODES: tuple[str, ...] = (
    "CPA_A01", "CPA_A02", "CPA_A03", "CPA_B", "CPA_C10-12", "CPA_C13-15",
    "CPA_C16", "CPA_C17", "CPA_C18", "CPA_C19", "CPA_C20", "CPA_C21",
    "CPA_C22", "CPA_C23", "CPA_C24", "CPA_C25", "CPA_C26", "CPA_C27",
    "CPA_C28", "CPA_C29", "CPA_C30", "CPA_C31_32", "CPA_C33", "CPA_D",
    "CPA_E36", "CPA_E37-39", "CPA_F", "CPA_G45", "CPA_G46", "CPA_G47",
    "CPA_H49", "CPA_H50", "CPA_H51", "CPA_H52", "CPA_H53", "CPA_I",
    "CPA_J58", "CPA_J59_60", "CPA_J61", "CPA_J62_63", "CPA_K64", "CPA_K65",
    "CPA_K66", "CPA_L68", "CPA_M69_70", "CPA_M71", "CPA_M72", "CPA_M73",
    "CPA_M74_75", "CPA_N77", "CPA_N78", "CPA_N79", "CPA_N80-82", "CPA_O",
    "CPA_P", "CPA_Q86", "CPA_Q87_88", "CPA_R90-92", "CPA_R93", "CPA_S94",
    "CPA_S95", "CPA_S96", "CPA_T", "CPA_U",
)

CPA65_CODES: tuple[str, ...] = (
    *CPA64_CODES[:43], "CPA_L68A", "CPA_L68B", *CPA64_CODES[44:],
)

CONSOLIDATED64_SOURCE_CODES: tuple[str, ...] = (
    *CPA64_CODES[:43], "CPA_L68B", *CPA64_CODES[44:],
)


@dataclass(frozen=True)
class TargetProductSchema:
    """Exact source axis and canonical target axis used by a concordance."""

    schema_id: str
    source_codes: tuple[str, ...]
    canonical_codes: tuple[str, ...]
    source_to_canonical: Mapping[str, str]
    canonical_to_source: Mapping[str, str]

    def canonicalize_reference(self, reference: pd.Series) -> pd.Series:
        """Relabel a complete native use reference without changing its values."""
        native = reference.copy()
        native.index = native.index.map(str)
        if tuple(native.index) != self.source_codes:
            raise ValueError(
                f"Schema {self.schema_id!r} requires its complete ordered source axis; "
                f"got {tuple(native.index)!r}."
            )
        native.index = pd.Index(
            [self.source_to_canonical[code] for code in self.source_codes],
            dtype="object",
        )
        if native.index.has_duplicates:
            raise ValueError(f"Schema {self.schema_id!r} canonicalizes to duplicate targets.")
        return native

    def source_target(self, canonical_code: str) -> str:
        """Return the published matrix-axis code for one canonical target."""
        try:
            return self.canonical_to_source[str(canonical_code)]
        except KeyError as exc:
            raise ValueError(
                f"Canonical target {canonical_code!r} is outside schema {self.schema_id!r}."
            ) from exc


_CPA65 = TargetProductSchema(
    schema_id="CPA65",
    source_codes=CPA65_CODES,
    canonical_codes=CPA65_CODES,
    source_to_canonical={code: code for code in CPA65_CODES},
    canonical_to_source={code: code for code in CPA65_CODES},
)
_CONSOLIDATED64 = TargetProductSchema(
    schema_id="EUROSTAT_CONSOLIDATED_CPA64",
    source_codes=CONSOLIDATED64_SOURCE_CODES,
    canonical_codes=CPA64_CODES,
    source_to_canonical={
        **{code: code for code in CONSOLIDATED64_SOURCE_CODES},
        "CPA_L68B": "CPA_L68",
    },
    canonical_to_source={
        **{code: code for code in CPA64_CODES},
        "CPA_L68": "CPA_L68B",
    },
)


def detect_target_product_schema(source_codes: Iterable[object]) -> TargetProductSchema:
    """Recognize only complete, ordered CPA65 or consolidated-CPA64 axes.

    Geography and an incidental missing ``CPA_L68A`` observation are deliberately
    not evidence of the consolidated source semantic.  The complete ordered
    domain must match one registered source schema exactly.
    """
    codes = tuple(map(str, source_codes))
    if codes == _CPA65.source_codes:
        return _CPA65
    if codes == _CONSOLIDATED64.source_codes:
        return _CONSOLIDATED64
    raise ValueError(
        "Unrecognised Eurostat target-product source axis; expected exactly the "
        "registered ordered CPA65 or consolidated CPA64 domain."
    )


def project_reviewed_cpa65_to_cpa64(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a complete reviewed CPA65 concordance into canonical CPA64.

    Every reviewed edge is retained, except that the two CPA65 real-estate
    targets become the one canonical ``CPA_L68`` target.  Duplicate edges caused
    solely by that collapse are represented once, retaining combined provenance.
    """
    required = {"source_code", "target_code"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reviewed CPA65 mapping is missing required columns: {missing!r}.")
    projected = frame.copy()
    targets = set(projected["target_code"].astype(str).str.strip())
    if targets != set(CPA65_CODES):
        raise ValueError(
            "Reviewed mapping cannot be projected because its target domain is not "
            "the complete CPA65 schema."
        )
    projected["target_code"] = projected["target_code"].astype(str).str.strip().replace(
        {"CPA_L68A": "CPA_L68", "CPA_L68B": "CPA_L68"}
    )
    if "target_version" in projected.columns:
        projected["target_version"] = "CPA_64"
    if "target_label" in projected.columns:
        projected.loc[projected["target_code"].eq("CPA_L68"), "target_label"] = "Real estate services"
    if "notes" in projected.columns:
        mask = projected["target_code"].eq("CPA_L68")
        projected.loc[mask, "notes"] = (
            projected.loc[mask, "notes"].astype(str).str.strip()
            + " Canonical CPA64 projection of reviewed CPA65 real-estate semantics."
        )
    # CPA_L68A/L68B create a duplicate p70->CPA_L68 edge.  The numeric resolver
    # must see one direct edge, while the retained row still documents its source.
    projected = projected.drop_duplicates(subset=["source_code", "target_code"], keep="first")
    if set(projected["target_code"].astype(str).str.strip()) != set(CPA64_CODES):
        raise ValueError("Reviewed CPA65 projection did not produce the complete CPA64 domain.")
    return projected
