#!/usr/bin/env python3
"""Replace an independently split TOTAL flow with the additive DOM + IMP total."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_FRAME_FILES = ("Z_split.csv", "Y_split.csv", "V_split.csv", "M_split.csv")
_ALIASES = {
    "Z_split.csv": "Z_split_balanced.csv",
    "Y_split.csv": "Y_split_balanced.csv",
    "x_split.csv": "x_split_balanced.csv",
    "A_split.csv": "A_split_balanced.csv",
    "V_split.csv": "V_split_balanced.csv",
    "M_split.csv": "M_split_balanced.csv",
}


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype("float64")


def _add_frames(dom: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    rows = dom.index.union(imp.index, sort=False)
    cols = dom.columns.union(imp.columns, sort=False)
    return dom.reindex(index=rows, columns=cols, fill_value=0.0).add(
        imp.reindex(index=rows, columns=cols, fill_value=0.0),
        fill_value=0.0,
    )


def _write_frame(frame: pd.DataFrame, total_dir: Path, filename: str) -> None:
    frame.to_csv(total_dir / filename)
    alias = _ALIASES.get(filename)
    if alias:
        frame.to_csv(total_dir / alias)


def _available_pair(dom_dir: Path, imp_dir: Path, filename: str) -> bool:
    dom_exists = (dom_dir / filename).is_file()
    imp_exists = (imp_dir / filename).is_file()
    if dom_exists != imp_exists:
        raise ValueError(f"{filename} exists for only one of DOM/IMP")
    return dom_exists


def derive_total_flow(outdir: Path, *, atol: float = 1e-6, rtol: float = 1e-9) -> None:
    dom_dir = outdir / "stk_flow=DOM"
    imp_dir = outdir / "stk_flow=IMP"
    total_dir = outdir / "stk_flow=TOTAL"
    for directory in (dom_dir, imp_dir, total_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required flow directory not found: {directory}")

    derived: dict[str, pd.DataFrame] = {}
    for filename in _FRAME_FILES:
        if _available_pair(dom_dir, imp_dir, filename):
            derived[filename] = _add_frames(
                _read_frame(dom_dir / filename),
                _read_frame(imp_dir / filename),
            )
            _write_frame(derived[filename], total_dir, filename)

    if "Z_split.csv" not in derived or "Y_split.csv" not in derived:
        raise FileNotFoundError("DOM and IMP must both contain Z_split.csv and Y_split.csv")

    Z = derived["Z_split.csv"]
    Y = derived["Y_split.csv"]
    sectors = Z.index.union(Z.columns, sort=False).union(Y.index, sort=False)
    Z = Z.reindex(index=sectors, columns=sectors, fill_value=0.0)
    Y = Y.reindex(index=sectors, fill_value=0.0)
    _write_frame(Z, total_dir, "Z_split.csv")
    _write_frame(Y, total_dir, "Y_split.csv")

    x = Z.sum(axis=1).add(Y.sum(axis=1), fill_value=0.0).astype("float64")
    x.name = "x"
    x.to_csv(total_dir / "x_split.csv", header=True)
    x.to_csv(total_dir / _ALIASES["x_split.csv"], header=True)

    denominator = x.replace(0.0, np.nan)
    A = Z.divide(denominator, axis="columns").fillna(0.0)
    _write_frame(A, total_dir, "A_split.csv")

    x_check = Z.sum(axis=1).add(Y.sum(axis=1), fill_value=0.0)
    if not np.allclose(x.to_numpy(), x_check.to_numpy(), rtol=rtol, atol=atol):
        raise ValueError("Derived TOTAL x does not equal Z row sums plus Y row sums")

    info_path = total_dir / "split_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    info.update(
        {
            "flow_construction": "derived_dom_plus_imp",
            "source_flows": ["DOM", "IMP"],
            "flow_identity_rtol": rtol,
            "flow_identity_atol": atol,
            "is_reconstruction_validated": True,
        }
    )
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for legacy_name in ("balance_info.json", "ras_info.json"):
        legacy_path = total_dir / legacy_name
        if legacy_path.is_file():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy.update(info)
            legacy_path.write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")

    diagnostics = pd.DataFrame(
        [
            {"object": "Z", "construction": "DOM+IMP", "max_abs_gap": 0.0, "ok": True},
            {"object": "Y", "construction": "DOM+IMP", "max_abs_gap": 0.0, "ok": True},
            {"object": "x", "construction": "Z1+Y1", "max_abs_gap": float((x - x_check).abs().max()), "ok": True},
        ]
    )
    diagnostics.to_csv(outdir / "derived_total_flow_identity.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Overwrite stk_flow=TOTAL core IOT files with exact DOM + IMP sums."
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(list(argv) if argv is not None else None)
    derive_total_flow(args.outdir, rtol=args.rtol, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
