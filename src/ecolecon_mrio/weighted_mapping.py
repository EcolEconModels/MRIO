from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _parse_boolish(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s == "":
        return True
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {x!r}")


def validate_mapping_weights(
    sector_mapping: dict[str, dict[str, float]],
    *,
    atol: float = 1e-9,
) -> None:
    """
    Validate weighted mappings. For each EXIO sector, weights across CPA codes must sum to 1.
    """
    exio_weight_sum: dict[str, float] = {}
    for agg, weights in sector_mapping.items():
        if not isinstance(weights, dict):
            raise ValueError(f"Weighted mapping for {agg!r} must be a dict of exio->weight.")
        for exio, w in weights.items():
            if w is None or not np.isfinite(float(w)):
                raise ValueError(f"Weight for exio {exio!r} in {agg!r} is not finite: {w!r}")
            w_f = float(w)
            if w_f < 0:
                raise ValueError(f"Negative weight for exio {exio!r} in {agg!r}: {w_f}")
            exio_weight_sum[exio] = exio_weight_sum.get(exio, 0.0) + w_f

    bad = {exio: total for exio, total in exio_weight_sum.items() if abs(total - 1.0) > atol}
    if bad:
        sample = ", ".join(f"{k}={v:.6g}" for k, v in list(bad.items())[:8])
        raise ValueError(
            "EXIO weights must sum to 1.0 across mapped CPA codes. "
            f"Example mismatches: {sample}"
        )


def read_weighted_mapping_csv(
    path: str | Path,
    *,
    aggregate_col: str = "aggregate_sector",
    exio_col: str = "exio_sector",
    enabled_col: str = "enabled",
    weight_col: str = "weight",
) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in (aggregate_col, exio_col, weight_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV {Path(path)} missing required columns: {missing}")

    if enabled_col in df.columns:
        mask = df[enabled_col].map(_parse_boolish).astype(bool)
        df = df[mask].copy()

    df[aggregate_col] = df[aggregate_col].astype(str).str.strip()
    df[exio_col] = df[exio_col].astype(str).str.strip()
    df = df[(df[aggregate_col] != "") & (df[exio_col] != "")]

    df[weight_col] = pd.to_numeric(df[weight_col], errors="raise")

    mapping: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        agg = str(row[aggregate_col])
        exio = str(row[exio_col])
        w = float(row[weight_col])
        mapping.setdefault(agg, {})
        mapping[agg][exio] = mapping[agg].get(exio, 0.0) + w

    validate_mapping_weights(mapping)
    return mapping


def flatten_exio_sectors_from_mapping(
    sector_mapping: dict[str, list[str]] | dict[str, dict[str, list[str]]] | dict[str, dict[str, float]],
) -> list[str]:
    out: list[str] = []
    for members in sector_mapping.values():
        if isinstance(members, dict):
            if members and any(isinstance(v, list) for v in members.values()):
                for v in members.values():
                    if isinstance(v, list):
                        out.extend(list(map(str, v)))
            else:
                out.extend(list(map(str, members.keys())))
        else:
            out.extend(list(map(str, members)))
    return list(dict.fromkeys(out))
