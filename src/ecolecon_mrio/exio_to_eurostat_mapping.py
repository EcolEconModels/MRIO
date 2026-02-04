from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd

try:  # Optional dependency in case sklearn isn't installed in some environments.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - fallback path only
    TfidfVectorizer = None
    cosine_similarity = None
    _HAVE_SKLEARN = False


def _normalize_text(value: str) -> str:
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(value: str) -> set[str]:
    return {t for t in _normalize_text(value).split(" ") if t}


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _default_rule_overrides() -> list[tuple[str, str]]:
    return []


def _rule_based_cpa(exio_name: str, overrides: list[tuple[str, str]]) -> str | None:
    s = _normalize_text(exio_name)
    for pat, code in overrides:
        if re.search(pat, s):
            return code
    return None


def _label_match_score(exio_name: str, label: str, *, min_similarity: float) -> float:
    n_exio = _normalize_text(exio_name)
    n_label = _normalize_text(label)
    if not n_exio or not n_label:
        return 0.0
    if n_exio in n_label or n_label in n_exio:
        return 1.0
    score = _similarity(exio_name, label)
    if score >= min_similarity:
        return score
    return 0.0


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


def _candidate_cpa_codes_for_exio(
    exio_code: str | None,
    div_to_cpa: dict[int, list[str]],
) -> list[str]:
    if not exio_code:
        return []
    div = _exio_division_from_code(exio_code)
    if div is None:
        return []
    nace2_divs = _NACE11_TO_NACE2.get(div, [div])
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


_PHARMA_KEYWORDS = (
    "pharm",
    "pharmaceutical",
    "drug",
    "medicine",
    "medic",
    "vaccine",
    "vaccin",
    "medical",
)

_PHARMA_PROXY_KEYWORDS = (
    "chemicals nec",
    "chemical products nec",
    "chemical products n e c",
    "other chemicals",
)

_FUEL_KEYWORDS = (
    "petroleum",
    "refined",
    "coke",
    "gasoline",
    "diesel",
    "kerosene",
    "jet fuel",
    "biofuel",
    "biodiesel",
    "biogasoline",
    "charcoal",
    "fuel",
    "crude oil",
    "natural gas",
)


def _filter_candidates_by_keywords(exio_name: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return candidates
    s = _normalize_text(exio_name)
    if "CPA_C19" in candidates and any(k in s for k in _FUEL_KEYWORDS):
        return ["CPA_C19"]
    if "CPA_C21" in candidates and not any(k in s for k in _PHARMA_KEYWORDS):
        if any(k in s for k in _PHARMA_PROXY_KEYWORDS):
            return candidates
        return [c for c in candidates if c != "CPA_C21"]
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


def _build_tfidf_similarity(
    exio_names: list[str],
    cpa_label_map: dict[str, str],
) -> tuple[list[str], list[str], list[list[float]]]:
    if not _HAVE_SKLEARN:
        return exio_names, list(cpa_label_map.keys()), []

    cpa_codes = list(cpa_label_map.keys())
    cpa_labels = [cpa_label_map[c] for c in cpa_codes]
    docs = list(exio_names) + list(cpa_labels)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(docs)
    exio_mat = matrix[: len(exio_names)]
    cpa_mat = matrix[len(exio_names) :]
    sims = cosine_similarity(exio_mat, cpa_mat)
    return exio_names, cpa_codes, sims.tolist()


def _best_cpa_by_label(
    exio_name: str,
    cpa_label_map: dict[str, str],
    *,
    min_similarity: float,
) -> str | None:
    best_code = None
    best_score = 0.0
    for code, label in cpa_label_map.items():
        score = _label_match_score(exio_name, label, min_similarity=min_similarity)
        if score > best_score:
            best_score = score
            best_code = code
    if best_score > 0.0:
        return best_code
    return None


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

    overrides = rule_overrides if rule_overrides is not None else _default_rule_overrides()
    div_to_cpa = _build_division_to_cpa_codes(cpa_set)

    tfidf_exio, tfidf_cpa_codes, tfidf_sims = _build_tfidf_similarity(exio_list, cpa_label_map)
    tfidf_index = {name: i for i, name in enumerate(tfidf_exio)}
    tfidf_code_index = {code: i for i, code in enumerate(tfidf_cpa_codes)}

    rows = []
    seen = set()
    for exio_name in exio_list:
        exio_code = exio_code_map.get(exio_name) if exio_code_map else None
        target = _rule_based_cpa(exio_name, overrides) if overrides else None
        note = "rule-override" if target is not None else "label-match"
        if target is None and exio_code:
            candidates = _candidate_cpa_codes_for_exio(exio_code, div_to_cpa)
            candidates = _filter_candidates_by_keywords(exio_name, candidates)
            candidates = [c for c in candidates if c in cpa_set]
            if candidates:
                if len(candidates) == 1:
                    target = candidates[0]
                    note = "code-map"
                elif tfidf_sims:
                    exio_idx = tfidf_index.get(exio_name)
                    if exio_idx is not None:
                        best_code = None
                        best_score = -1.0
                        for code in candidates:
                            c_idx = tfidf_code_index.get(code)
                            if c_idx is None:
                                continue
                            score = tfidf_sims[exio_idx][c_idx]
                            if score > best_score:
                                best_score = score
                                best_code = code
                        if best_code is not None:
                            target = best_code
                            note = "code-map+label"
        if target is None and tfidf_sims:
            exio_idx = tfidf_index.get(exio_name)
            if exio_idx is not None:
                best_code = None
                best_score = -1.0
                for code, c_idx in tfidf_code_index.items():
                    score = tfidf_sims[exio_idx][c_idx]
                    if score > best_score:
                        best_score = score
                        best_code = code
                if best_code is not None and best_score >= min_tfidf_similarity:
                    target = best_code
                    note = "tfidf-label"
        if target is None:
            target = _best_cpa_by_label(exio_name, cpa_label_map, min_similarity=min_similarity)
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
