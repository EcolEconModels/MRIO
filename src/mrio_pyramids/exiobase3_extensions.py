from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable, Literal, Sequence
import zipfile

import numpy as np
import pandas as pd


MatrixKind = Literal["F", "F_Y"]


@dataclass(frozen=True)
class ExioExtension:
    name: str
    unit: pd.Series  # index=stressor, value=unit
    F: pd.DataFrame | None
    F_Y: pd.DataFrame | None


def _open_text_from_zip(zf: zipfile.ZipFile, member: str) -> TextIOWrapper:
    return TextIOWrapper(zf.open(member, "r"), encoding="utf-8", newline="")


def _resolve_container(path: str | Path) -> tuple[str, Path]:
    p = Path(path)
    if p.is_dir():
        return "dir", p
    if p.is_file() and p.suffix.lower() == ".zip":
        return "zip", p
    raise ValueError(f"Unsupported EXIOBASE container: {p}")


def _read_header_lines(stream: TextIOWrapper) -> tuple[list[str], list[str], str]:
    line1 = stream.readline()
    line2 = stream.readline()
    line3 = stream.readline()
    if not line1 or not line2:
        raise ValueError("Extension file is missing header lines.")
    return line1.rstrip("\n").split("\t"), line2.rstrip("\n").split("\t"), line3.rstrip("\n")


def _read_extension_matrix(
    container: str | Path,
    *,
    extension: str,
    kind: MatrixKind,
    region: str | None = None,
) -> pd.DataFrame:
    """
    Read one EXIOBASE 3 extension matrix (F or F_Y) from a zip or folder.

    This is optimized for region subsetting: if `region` is given, it only reads
    the columns for that region.
    """
    ctype, cpath = _resolve_container(container)
    rel = f"{extension}/{kind}.txt"

    if ctype == "zip":
        with zipfile.ZipFile(cpath, "r") as zf:
            with _open_text_from_zip(zf, rel) as fh:
                h1, h2, h3 = _read_header_lines(fh)
            with _open_text_from_zip(zf, rel) as fh:
                keep = [i for i in range(1, len(h1)) if region is None or h1[i] == region]
                usecols = [0] + keep
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    index_col=0,
                    usecols=usecols,
                    skiprows=3,
                    engine="c",
                )
    else:
        file_path = cpath / rel
        with file_path.open("r", encoding="utf-8") as fh:
            h1, h2, h3 = _read_header_lines(fh)
        keep = [i for i in range(1, len(h1)) if region is None or h1[i] == region]
        usecols = [0] + keep
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            index_col=0,
            usecols=usecols,
            skiprows=3,
            engine="c",
        )

    # Set consistent MultiIndex column names
    label_level_name = str(h2[0]).strip() or "label"
    region_codes = [h1[i] for i in keep]
    labels = [h2[i] for i in keep]
    df.columns = pd.MultiIndex.from_arrays(
        [region_codes, labels], names=["region", label_level_name]
    )
    df.index = df.index.astype(str)
    df.index.name = "stressor"
    return df.astype("float64")


def read_extension_units(container: str | Path, *, extension: str) -> pd.Series:
    ctype, cpath = _resolve_container(container)
    rel = f"{extension}/unit.txt"

    if ctype == "zip":
        with zipfile.ZipFile(cpath, "r") as zf:
            with _open_text_from_zip(zf, rel) as fh:
                df = pd.read_csv(fh, sep="\t")
    else:
        df = pd.read_csv(cpath / rel, sep="\t")

    if "stressor" not in df.columns or "unit" not in df.columns:
        raise ValueError(f"Unexpected unit file format for {extension}: columns={list(df.columns)}")
    out = pd.Series(df["unit"].to_numpy(), index=df["stressor"].astype(str))
    out.name = "unit"
    return out


@lru_cache(maxsize=32)
def load_exiobase3_extension(
    container: str | Path,
    *,
    extension: str,
    region: str | None = None,
    cache_dir: str | Path | None = None,
) -> ExioExtension:
    """
    Load one EXIOBASE 3 extension with optional region subsetting.

    Uses an in-process cache to avoid re-reading large files repeatedly.
    """
    cache_root = Path(cache_dir) if cache_dir is not None else None

    def _pkl(name: str) -> Path | None:
        if cache_root is None:
            return None
        tag = "ALL" if region is None else str(region)
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"exioext_{extension}_{name}_{tag}.pkl"

    unit = read_extension_units(container, extension=extension)

    F = None
    f_path = _pkl("F")
    if f_path is not None and f_path.exists():
        F = pd.read_pickle(f_path)
    else:
        try:
            F = _read_extension_matrix(container, extension=extension, kind="F", region=region)
        except Exception:
            F = None
        if f_path is not None and F is not None:
            F.to_pickle(f_path)

    F_Y = None
    fy_path = _pkl("F_Y")
    if fy_path is not None and fy_path.exists():
        F_Y = pd.read_pickle(fy_path)
    else:
        try:
            F_Y = _read_extension_matrix(container, extension=extension, kind="F_Y", region=region)
        except Exception:
            F_Y = None
        if fy_path is not None and F_Y is not None:
            F_Y.to_pickle(fy_path)

    return ExioExtension(name=extension, unit=unit, F=F, F_Y=F_Y)


def _select_rows_by_keywords(index: Iterable[str], keywords: Sequence[str]) -> list[str]:
    keys = [k.lower() for k in keywords]
    out: list[str] = []
    for r in index:
        rl = str(r).lower()
        if any(k in rl for k in keys):
            out.append(str(r))
    return out


def extract_quantity_and_price(
    *,
    container: str | Path,
    region: str,
    sectors: Sequence[str],
    monetary_output: pd.Series,
    extensions: Sequence[str] = ("energy",),
    stressor_keywords: Sequence[str] = ("energy use",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """
    Extract physical quantities from EXIOBASE 3 extensions and compute quantity shares + implied prices.

    Notes
    - Physical data in EXIOBASE 3 is provided via extensions (e.g., `energy/F.txt` in TJ).
    - The implied price here is a unit-value proxy: monetary_output / physical_quantity.
    """
    if not sectors:
        return None

    sectors = list(map(str, sectors))
    monetary_output = monetary_output.reindex(sectors).astype("float64")

    rows: list[pd.DataFrame] = []
    for ext in extensions:
        ext_obj = load_exiobase3_extension(
            container, extension=ext, region=region, cache_dir=cache_dir
        )
        if ext_obj.F is None:
            continue

        # The extension columns are labeled by sector names; ensure we can align.
        F = ext_obj.F
        if not isinstance(F.columns, pd.MultiIndex) or F.columns.nlevels != 2:
            continue

        # Pull the one-region columns and drop the region level.
        try:
            F_r = F.xs(region, level=0, axis=1)
        except Exception:
            continue

        if not isinstance(F_r.columns, pd.Index):
            F_r.columns = pd.Index(map(str, F_r.columns))
        F_r = F_r.loc[:, F_r.columns.astype(str)]

        # If the EXIO sector names don't match exactly, fail gracefully.
        missing = [s for s in sectors if s not in set(map(str, F_r.columns))]
        if missing:
            continue

        stressors = _select_rows_by_keywords(F_r.index, stressor_keywords)
        if not stressors:
            continue

        for stressor in stressors:
            qty = F_r.loc[stressor, sectors].astype("float64")
            total_qty = float(qty.sum())
            if not np.isfinite(total_qty) or total_qty == 0:
                continue
            qty_share = qty / total_qty
            implied_price = monetary_output / qty.replace(0.0, np.nan)
            unit = ext_obj.unit.get(stressor, None)
            out = pd.DataFrame(
                {
                    "extension": ext,
                    "stressor": stressor,
                    "unit": unit,
                    "quantity": qty,
                    "quantity_share": qty_share,
                    "monetary_output": monetary_output,
                    "implied_price": implied_price,
                }
            )
            out.index.name = "exio_sector"
            rows.append(out.reset_index())

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)
