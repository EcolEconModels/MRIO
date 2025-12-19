from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    import pymrio
except Exception:  # pragma: no cover
    pymrio = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExioRegionSlice:
    year: int
    system: str
    region: str
    sectors: pd.Index
    Z_domestic: pd.DataFrame
    Y_region: pd.DataFrame
    x_region: pd.Series
    source_zip: Path
    flow: str = "domestic"


def _coerce_single_column_df_to_series(x: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(x)}")
    if x.shape[1] != 1:
        raise ValueError(f"{name} must have exactly 1 column when provided as DataFrame, got shape={x.shape}")
    s = x.iloc[:, 0]
    # Keep the original index, but ensure a stable name for downstream code.
    s.name = s.name or name
    return s


def list_local_exiobase3_years(*, data_root: str | Path | None = None) -> list[int]:
    if data_root is None:
        from ecolecon_mrio.data import default_data_root

        root = default_data_root()
    else:
        root = Path(data_root)
    years: list[int] = []
    for p in root.glob("exiobase3_*"):
        try:
            years.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(years))


def _normalize_flow(flow: str) -> str:
    flow_norm = str(flow).strip().lower()
    if flow_norm in {"dom", "domestic"}:
        return "domestic"
    if flow_norm in {"imp", "import", "imports"}:
        return "imports"
    if flow_norm in {"tot", "total"}:
        return "total"
    raise ValueError("flow must be one of: domestic, imports, total")


def build_exio_region_use_slice_from_mrio(
    *,
    year: int,
    system: str,
    region: str,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    source_zip: Path,
    flow: str = "domestic",
) -> ExioRegionSlice:
    """
    Build an EXIO-like single-region *use* slice from a full MRIO (Z,Y).

    Definitions (destination = `region`):
    - flow='domestic': origin == region
    - flow='imports' : sum over origins != region
    - flow='total'   : sum over all origins

    Returns an `ExioRegionSlice` whose `Z_domestic`/`Y_region` contain the chosen use matrices.
    """
    flow_norm = _normalize_flow(flow)

    if not isinstance(Z.index, pd.MultiIndex) or not isinstance(Z.columns, pd.MultiIndex):
        raise ValueError("Expected MRIO Z to have MultiIndex rows/cols (region, sector).")
    if not isinstance(Y.index, pd.MultiIndex) or not isinstance(Y.columns, pd.MultiIndex):
        raise ValueError("Expected MRIO Y to have MultiIndex rows/cols.")

    # Slice use side (columns) to destination region (drops region level in columns).
    Z_to_r = Z.xs(region, level=0, axis=1)
    Y_to_r = Y.xs(region, level=0, axis=1)

    # Domestic use (origin == region).
    Z_dom = Z_to_r.xs(region, level=0, axis=0)
    Y_dom = Y_to_r.xs(region, level=0, axis=0)

    # Import use (origin != region), summed over origins -> by-sector.
    Z_foreign = Z_to_r.loc[Z_to_r.index.get_level_values(0) != region]
    Y_foreign = Y_to_r.loc[Y_to_r.index.get_level_values(0) != region]
    Z_imp = Z_foreign.groupby(level=1).sum()
    Y_imp = Y_foreign.groupby(level=1).sum()

    # Total use, summed over all origins -> by-sector.
    Z_tot = Z_to_r.groupby(level=1).sum()
    Y_tot = Y_to_r.groupby(level=1).sum()

    if flow_norm == "domestic":
        Z_use, Y_use = Z_dom, Y_dom
    elif flow_norm == "imports":
        Z_use, Y_use = Z_imp, Y_imp
    else:
        Z_use, Y_use = Z_tot, Y_tot

    sectors = pd.Index(Z_use.index.astype(str))
    x_use = (Z_use.sum(axis=1) + Y_use.sum(axis=1)).astype("float64")

    return ExioRegionSlice(
        year=year,
        system=system,
        region=region,
        sectors=sectors,
        Z_domestic=Z_use,
        Y_region=Y_use,
        x_region=x_use,
        source_zip=source_zip,
        flow=flow_norm,
    )


def load_exiobase3(
    year: int,
    *,
    system: str = "pxp",
    force_download: bool = False,
    cache_root: str | Path | None = None,
    calc_all: bool = True,
    verbose: bool = False,
):
    """
    Load EXIOBASE 3 using the repo pymrio-based caching loader.

    Notes
    - This loads the full EXIO MRIO into memory. For disaggregation tasks that only
      need one region block, prefer `load_exiobase3_region_slice`, which caches the slice.
    """
    if pymrio is None:
        raise ImportError("pymrio is required to load EXIOBASE 3.")
    if system != "pxp":
        raise ValueError("Only system='pxp' is supported by the repo loader currently.")

    from ecolecon_mrio.data import load_mrio

    try:
        mrio, _ = load_mrio(
            "exiobase",
            year,
            force_download=force_download,
            data_root=cache_root,
            calc_all=calc_all,
            verbose=verbose,
        )
        return mrio
    except Exception as e:
        local = list_local_exiobase3_years(data_root=cache_root)
        hint = f"Local cached years: {local}" if local else "No local EXIOBASE cache found under data/."
        raise RuntimeError(f"Failed to load EXIOBASE 3 for year={year}. {hint}") from e


def load_exiobase3_region_slice(
    *,
    year: int,
    region: str = "AT",
    system: str = "pxp",
    flow: str = "domestic",
    force_download: bool = False,
    force_rebuild_slice: bool = False,
    cache_root: str | Path | None = None,
    verbose: bool = False,
) -> ExioRegionSlice:
    """
    Load (and cache) a single-region slice needed for domestic split profiles.

    Caches `Z_domestic`, `Y_region`, and `x_region` as pickles to avoid re-loading the full MRIO.
    """
    if cache_root is None:
        from ecolecon_mrio.data import default_data_root

        cache_root = default_data_root()
    cache_root = Path(cache_root)
    flow_norm = _normalize_flow(flow)

    slice_dir = cache_root / f"exiobase3_{year}" / "slices" / region / system
    slice_dir.mkdir(parents=True, exist_ok=True)

    if flow_norm == "domestic":
        z_path = slice_dir / "Z_domestic.pkl"
        y_path = slice_dir / "Y_region.pkl"
    elif flow_norm == "imports":
        z_path = slice_dir / "Z_imports.pkl"
        y_path = slice_dir / "Y_imports.pkl"
    else:  # total
        z_path = slice_dir / "Z_total.pkl"
        y_path = slice_dir / "Y_total.pkl"
    meta_path = slice_dir / "meta.json"

    source_zip = cache_root / f"exiobase3_{year}" / f"IOT_{year}_{system}.zip"
    if not source_zip.exists():
        # The repo loader names it IOT_{year}_pxp.zip (no system in suffix)
        alt = cache_root / f"exiobase3_{year}" / f"IOT_{year}_pxp.zip"
        if alt.exists():
            source_zip = alt

    if not force_rebuild_slice and z_path.exists() and y_path.exists():
        if verbose:
            print(f"Loading cached EXIOBASE region slice from {slice_dir} [flow={flow_norm}] ...")
        Z_use = pd.read_pickle(z_path)
        Y_use = pd.read_pickle(y_path)
        sectors = pd.Index(Z_use.index.astype(str))
        x_use = (Z_use.sum(axis=1) + Y_use.sum(axis=1)).astype("float64")
        return ExioRegionSlice(
            year=year,
            system=system,
            region=region,
            sectors=sectors,
            Z_domestic=Z_use,
            Y_region=Y_use,
            x_region=x_use,
            source_zip=source_zip,
            flow=flow_norm,
        )

    if verbose:
        print(
            f"Building EXIOBASE region slice for region={region} year={year} system={system} "
            f"(requires loading the full MRIO once)..."
        )
    mrio = load_exiobase3(
        year,
        system=system,
        force_download=force_download,
        cache_root=cache_root,
        calc_all=False,
        verbose=verbose,
    )

    Z = mrio.Z
    Y = mrio.Y

    if not isinstance(Z.index, pd.MultiIndex) or not isinstance(Z.columns, pd.MultiIndex):
        raise ValueError("Expected EXIOBASE Z to have MultiIndex rows/cols (region, sector).")
    if not isinstance(Y.index, pd.MultiIndex) or not isinstance(Y.columns, pd.MultiIndex):
        raise ValueError("Expected EXIOBASE Y to have MultiIndex rows and columns.")

    slice_obj = build_exio_region_use_slice_from_mrio(
        year=year,
        system=system,
        region=region,
        Z=Z,
        Y=Y,
        source_zip=source_zip,
        flow=flow_norm,
    )

    if verbose:
        print(f"Writing EXIOBASE region slice cache to {slice_dir} ...")
    slice_obj.Z_domestic.to_pickle(z_path)
    slice_obj.Y_region.to_pickle(y_path)
    meta_path.write_text(
        f'{{"year":{year},"system":"{system}","region":"{region}","flow":"{flow_norm}"}}',
        encoding="utf-8",
    )
    return slice_obj


def load_exiobase3_region_slices(
    *,
    year: int,
    regions: Sequence[str],
    system: str = "pxp",
    force_download: bool = False,
    force_rebuild_slice: bool = False,
    cache_root: str | Path | None = None,
) -> dict[str, ExioRegionSlice]:
    """
    Load (and cache) multiple region slices, ensuring the full MRIO is loaded at most once.
    """
    if cache_root is None:
        from ecolecon_mrio.data import default_data_root

        cache_root = default_data_root()
    cache_root = Path(cache_root)
    regions = list(dict.fromkeys(map(str, regions)))
    if not regions:
        return {}

    def _slice_paths(r: str) -> tuple[Path, Path, Path, Path]:
        slice_dir = cache_root / f"exiobase3_{year}" / "slices" / r / system
        return (
            slice_dir / "Z_domestic.pkl",
            slice_dir / "Y_region.pkl",
            slice_dir / "x_region.pkl",
            slice_dir / "meta.json",
        )

    missing_any = False
    if force_rebuild_slice:
        missing_any = True
    else:
        for r in regions:
            z_path, y_path, x_path, meta_path = _slice_paths(r)
            if not (z_path.exists() and y_path.exists() and x_path.exists()):
                missing_any = True
                break

    source_zip = cache_root / f"exiobase3_{year}" / f"IOT_{year}_{system}.zip"
    if not source_zip.exists():
        alt = cache_root / f"exiobase3_{year}" / f"IOT_{year}_pxp.zip"
        if alt.exists():
            source_zip = alt

    if missing_any:
        mrio = load_exiobase3(
            year,
            system=system,
            force_download=force_download,
            cache_root=cache_root,
            calc_all=False,
        )
        Z = mrio.Z
        Y = mrio.Y
        x = mrio.x

        if not isinstance(Z.index, pd.MultiIndex) or not isinstance(Z.columns, pd.MultiIndex):
            raise ValueError("Expected EXIOBASE Z to have MultiIndex rows/cols (region, sector).")
        if not isinstance(Y.index, pd.MultiIndex) or not isinstance(Y.columns, pd.MultiIndex):
            raise ValueError("Expected EXIOBASE Y to have MultiIndex rows and columns.")

        for r in regions:
            z_path, y_path, x_path, meta_path = _slice_paths(r)
            z_path.parent.mkdir(parents=True, exist_ok=True)
            Z_rr = Z.xs(r, level=0, axis=0).xs(r, level=0, axis=1)
            Y_r = Y.xs(r, level=0, axis=0).xs(r, level=0, axis=1)
            x_r = x.xs(r, level=0, axis=0)
            Z_rr.to_pickle(z_path)
            Y_r.to_pickle(y_path)
            x_r.to_pickle(x_path)
            meta_path.write_text(
                f'{{"year":{year},"system":"{system}","region":"{r}"}}', encoding="utf-8"
            )

    out: dict[str, ExioRegionSlice] = {}
    for r in regions:
        out[r] = load_exiobase3_region_slice(
            year=year,
            region=r,
            system=system,
            force_download=force_download,
            force_rebuild_slice=False,
            cache_root=cache_root,
        )
    return out


def select_by_patterns(values: Iterable[str], patterns: Sequence[str]) -> list[str]:
    patterns_l = [p.lower() for p in patterns]
    hits: list[str] = []
    for v in values:
        v_l = str(v).lower()
        if any(p in v_l for p in patterns_l):
            hits.append(str(v))
    return hits


def build_split_profiles_from_exio_domestic_block(
    exio_slice: ExioRegionSlice,
    *,
    aggregate_sector: str,
    exio_sector_patterns: Sequence[str],
    prefix_labels_with: str | None = None,
    selector: str = "patterns",
):
    """
    Build 1D split profiles for disaggregation using EXIOBASE domestic block totals.

    Returns (SplitProfiles, matched_exio_sectors).
    """
    from ecolecon_mrio.disaggregation import SplitProfiles, normalize_share

    Z_rr = exio_slice.Z_domestic
    Y_r = exio_slice.Y_region
    # Use-side "output" for the slice (matches Eurostat-style Z+Y accounting).
    x_use = (Y_r.sum(axis=1) + Z_rr.sum(axis=1)).astype("float64")

    exio_sector_index = pd.Index(Z_rr.index.astype(str))
    if selector == "patterns":
        matched = select_by_patterns(exio_sector_index, exio_sector_patterns)
    elif selector == "exact":
        available = set(map(str, exio_sector_index))
        matched = []
        missing = []
        for s in exio_sector_patterns:
            s_str = str(s)
            if s_str not in available:
                missing.append(s_str)
            else:
                matched.append(s_str)
        if missing:
            raise ValueError(
                f"Missing EXIO sector labels for aggregate_sector={aggregate_sector!r} in region={exio_slice.region}: "
                f"{missing}"
            )
    else:
        raise ValueError("selector must be one of: 'patterns', 'exact'")
    if not matched:
        raise ValueError(
            f"No EXIO sectors matched patterns={list(exio_sector_patterns)} in region={exio_slice.region}."
        )

    subsectors = [
        f"{prefix_labels_with}{s}" if prefix_labels_with is not None else s for s in matched
    ]

    matched_idx = pd.Index(matched)
    x_sub = x_use.loc[matched_idx].astype("float64")
    domestic_sales_intermediate = Z_rr.loc[matched_idx, :].sum(axis=1).astype("float64")
    domestic_inputs = Z_rr.loc[:, matched_idx].sum(axis=0).astype("float64")
    domestic_final_demand = Y_r.loc[matched_idx, :].sum(axis=1).astype("float64")

    output_share = normalize_share(x_sub, name=f"{aggregate_sector}.output_share")
    sales_share = normalize_share(
        domestic_sales_intermediate, name=f"{aggregate_sector}.intermediate_sales_share"
    )
    input_share = normalize_share(
        domestic_inputs, name=f"{aggregate_sector}.intermediate_input_share"
    )

    try:
        final_demand_share = normalize_share(
            domestic_final_demand, name=f"{aggregate_sector}.final_demand_share"
        )
    except ValueError:
        final_demand_share = sales_share.copy()

    rename = dict(zip(matched, subsectors))
    profiles = SplitProfiles(
        aggregate_sector=aggregate_sector,
        subsectors=tuple(subsectors),
        output_share=output_share.rename(index=rename),
        intermediate_sales_share=sales_share.rename(index=rename),
        intermediate_input_share=input_share.rename(index=rename),
        final_demand_share=final_demand_share.rename(index=rename),
    )
    return profiles, matched


def load_exiobase(
    *,
    year: int,
    regions: Sequence[str] | None = None,
    system: str = "pxp",
    force_download: bool = False,
    force_rebuild_slice: bool = False,
    cache_root: str | Path = "data",
):
    """
    Convenience wrapper:
    - If `regions` is None: returns the full `pymrio.IOSystem`.
    - Else: returns a dict mapping region -> `ExioRegionSlice` (cached on disk).
    """
    if regions is None:
        return load_exiobase3(year, system=system, force_download=force_download)
    return load_exiobase3_region_slices(
        year=year,
        regions=regions,
        system=system,
        force_download=force_download,
        force_rebuild_slice=force_rebuild_slice,
        cache_root=cache_root,
    )
