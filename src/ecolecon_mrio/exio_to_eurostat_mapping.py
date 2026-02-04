from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


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


def _best_cpa_by_label(
    exio_name: str,
    cpa_label_map: dict[str, str],
    *,
    min_similarity: float,
) -> str | None:
    best_code = None
    best_score = 0.0
    for code, label in cpa_label_map.items():
        score = _similarity(exio_name, label)
        if score > best_score:
            best_score = score
            best_code = code
    if best_score >= min_similarity:
        return best_code
    return None


def build_exio_to_eurostat_mapping(
    exio_names: Iterable[str],
    cpa_codes: Iterable[str],
    cpa_label_map: dict[str, str],
    *,
    rule_overrides: list[tuple[str, str]] | None = None,
    min_similarity: float = 0.25,
) -> pd.DataFrame:
    exio_list = [str(x).strip() for x in exio_names if str(x).strip()]
    cpa_set = {str(c).strip() for c in cpa_codes if str(c).strip()}

    overrides = rule_overrides if rule_overrides is not None else _default_rule_overrides()

    rows = []
    seen = set()
    for exio_name in exio_list:
        target = _rule_based_cpa(exio_name, overrides) if overrides else None
        note = "rule-override" if target is not None else "label-match"
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
