from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd


def _normalize_text(value: str) -> str:
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_NACE2_SECTIONS: dict[str, list[int]] = {
    "A": list(range(1, 4)),
    "B": list(range(5, 10)),
    "C": list(range(10, 34)),
    "D": [35],
    "E": list(range(36, 40)),
    "F": list(range(41, 44)),
    "G": list(range(45, 48)),
    "H": list(range(49, 54)),
    "I": list(range(55, 57)),
    "J": list(range(58, 64)),
    "K": list(range(64, 67)),
    "L": [68],
    "M": list(range(69, 76)),
    "N": list(range(77, 83)),
    "O": [84],
    "P": [85],
    "Q": list(range(86, 89)),
    "R": list(range(90, 94)),
    "S": list(range(94, 97)),
    "T": list(range(97, 99)),
    "U": [99],
}


_NACE11_TO_NACE2: dict[int, list[int]] = {
    1: [1],
    2: [2],
    5: [3],
    10: [5],
    11: [6],
    12: [7],
    13: [7],
    14: [8],
    15: [10],
    16: [12],
    17: [13],
    18: [14],
    19: [15],
    20: [16],
    21: [17],
    22: [18, 58],
    23: [19],
    24: [20, 21],
    25: [22],
    26: [23],
    27: [24],
    28: [25],
    29: [28, 33],
    30: [26],
    31: [27],
    32: [26],
    33: [26],
    34: [29],
    35: [30],
    36: [31, 32],
    37: [38],
    40: [35],
    41: [36],
    45: [41, 42, 43],
    50: [45],
    51: [46],
    52: [47],
    55: [55, 56],
    60: [49],
    61: [50],
    62: [51],
    63: [52],
    64: [53, 61],
    65: [64],
    66: [65],
    67: [66],
    70: [68],
    71: [77],
    72: [62, 63],
    73: [72],
    74: [69, 70, 71, 73, 74, 75, 78, 79, 80, 81, 82],
    75: [84],
    80: [85],
    85: [86, 87, 88],
    90: [37, 38, 39],
    91: [94],
    92: [59, 60, 90, 91, 92, 93],
    93: [95, 96],
    95: [97, 98],
    99: [99],
}


_OFFICIAL_NACE2_DIVISION_OVERRIDES: tuple[tuple[re.Pattern[str], tuple[int, ...], str], ...] = (
    (
        re.compile(r"^p23\.1", re.IGNORECASE),
        (19,),
        "official-exio-nace:C.19.1 coke oven products",
    ),
    (
        re.compile(r"^p23\.20", re.IGNORECASE),
        (19,),
        "official-exio-nace:C.19.2 refined petroleum products",
    ),
    (
        re.compile(r"^p23\.3$", re.IGNORECASE),
        (24,),
        "official-exio-nace:C.24.46 nuclear fuel processing",
    ),
)


def _parse_cpa_divisions(code: str) -> list[int]:
    if not code.startswith("CPA_"):
        return []
    body = code[4:]
    if not body:
        return []
    section = body[0]
    rest = body[1:]
    if not rest:
        return list(_NACE2_SECTIONS.get(section, []))

    # Drop trailing letters (e.g. L68A -> 68)
    rest = rest.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if not rest:
        return list(_NACE2_SECTIONS.get(section, []))

    if "-" in rest:
        start, end = rest.split("-", 1)
        if start.isdigit() and end.isdigit():
            return list(range(int(start), int(end) + 1))
    if "_" in rest:
        parts = [p for p in rest.split("_") if p.isdigit()]
        return [int(p) for p in parts]
    if rest.isdigit():
        return [int(rest)]
    return []


def _build_division_to_cpa_codes(cpa_codes: Iterable[str]) -> dict[int, list[str]]:
    div_to_codes: dict[int, list[str]] = {}
    for code in cpa_codes:
        for div in _parse_cpa_divisions(str(code)):
            div_to_codes.setdefault(div, []).append(str(code))
    return div_to_codes


def build_cpa_division_index(cpa_codes: Iterable[str]) -> dict[int, list[str]]:
    """
    Build a lookup from NACE/CPA divisions -> CPA codes present in the target table.
    """
    return _build_division_to_cpa_codes(cpa_codes)


def _exio_division_from_code(exio_code: str) -> int | None:
    m = re.match(r"p(\d{2})", str(exio_code).strip().lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _official_nace2_divisions_for_exio(exio_code: str | None) -> tuple[list[int], str | None]:
    code = str(exio_code or "").strip()
    if not code:
        return [], None
    for pattern, divisions, note in _OFFICIAL_NACE2_DIVISION_OVERRIDES:
        if pattern.search(code):
            return list(divisions), note
    return [], None


def _generic_nace2_divisions_for_exio(exio_code: str | None) -> list[int]:
    if not exio_code:
        return []
    div = _exio_division_from_code(exio_code)
    if div is None:
        return []
    return list(_NACE11_TO_NACE2.get(div, [div]))


def _candidate_nace2_divisions_for_exio(exio_code: str | None) -> tuple[list[int], str]:
    official_divisions, official_note = _official_nace2_divisions_for_exio(exio_code)
    if official_divisions:
        return official_divisions, official_note or "official-exio-nace"
    generic_divisions = _generic_nace2_divisions_for_exio(exio_code)
    if generic_divisions:
        return generic_divisions, "nace11-to-nace2-fallback"
    return [], "unmapped"


def _candidate_cpa_codes_for_exio(
    exio_code: str | None,
    div_to_cpa: dict[int, list[str]],
) -> list[str]:
    nace2_divs, _ = _candidate_nace2_divisions_for_exio(exio_code)
    if not nace2_divs:
        return []
    candidates: list[str] = []
    for d in nace2_divs:
        candidates.extend(div_to_cpa.get(d, []))
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
def _filter_candidates_by_keywords(exio_name: str, candidates: list[str]) -> list[str]:
    return candidates


def candidate_cpa_codes_for_exio(
    exio_code: str | None,
    *,
    division_index: dict[int, list[str]],
) -> list[str]:
    """
    Return candidate CPA codes for a given EXIO sector code using NACE division mapping.
    """
    return _candidate_cpa_codes_for_exio(exio_code, division_index)


def filter_cpa_candidates_for_exio(exio_name: str, candidates: list[str]) -> list[str]:
    """
    Apply keyword-based filters to candidate CPA codes for a given EXIO sector name.
    """
    return _filter_candidates_by_keywords(exio_name, candidates)


def build_exio_to_eurostat_mapping(
    exio_names: Iterable[str],
    cpa_codes: Iterable[str],
    cpa_label_map: dict[str, str],
    *,
    exio_code_map: Mapping[str, str] | None = None,
    rule_overrides: list[tuple[str, str]] | None = None,
    min_similarity: float = 0.6,
    min_tfidf_similarity: float = 0.15,
) -> pd.DataFrame:
    exio_list = [str(x).strip() for x in exio_names if str(x).strip()]
    cpa_set = {str(c).strip() for c in cpa_codes if str(c).strip()}
    div_to_cpa = _build_division_to_cpa_codes(cpa_set)

    rows = []
    seen = set()
    for exio_name in exio_list:
        exio_code = exio_code_map.get(exio_name) if exio_code_map else None
        candidates = _candidate_cpa_codes_for_exio(exio_code, div_to_cpa)
        candidates = _filter_candidates_by_keywords(exio_name, candidates)
        candidates = [c for c in candidates if c in cpa_set]
        target = candidates[0] if candidates else None
        _, note = _candidate_nace2_divisions_for_exio(exio_code)
        if not target or target not in cpa_set:
            continue
        key = (target, exio_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "aggregate_sector": target,
                "exio_sector": exio_name,
                "enabled": "1",
                "notes": note,
            }
        )

    return pd.DataFrame(rows)
