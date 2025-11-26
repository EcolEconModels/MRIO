from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    import pymrio
except ImportError:  # pragma: no cover - optional dependency until installed
    pymrio = None


def load_exiobase3(
    storage_folder: Path | str = "data/exiobase3",
    years: Iterable[int] | int = (2011,),
    force_download: bool = False,
):
    """
    Download and parse EXIOBASE3 with pymrio.

    Returns a pymrio MRIOSystem. This function defers the actual download/parse
    until the user installs pymrio and explicitly calls it in a notebook or
    script. Network access may be restricted in some environments.
    """
    if pymrio is None:
        raise ImportError("pymrio is required to download or load EXIOBASE3")

    storage = Path(storage_folder)
    storage.mkdir(parents=True, exist_ok=True)
    pkl_path = storage / "mrio_system.pkl"

    if isinstance(years, (int, str)):
        years = [years]
    first_year = list(years)[0]
    zip_name = f"IOT_{first_year}_pxp.zip"
    zip_path = storage / zip_name

    if pkl_path.exists() and not force_download:
        mrio = pymrio.read_mrio(pkl_path)
        mrio.calc_all()
        return mrio

    if not zip_path.exists() or force_download:
        pymrio.download_exiobase3(
            storage_folder=storage,
            years=years,
            system="pxp",
            overwrite_existing=force_download,
        )

    if not zip_path.exists():
        raise FileNotFoundError(
            f"EXIOBASE archive {zip_path} not found after download; check permissions/network."
        )

    mrio = pymrio.parse_exiobase3(zip_path)
    mrio.calc_all()
    mrio.save(pkl_path)
    return mrio


def load_test_mrio() -> "pymrio.MRIOSystem":
    """Return the tiny built-in pymrio test MRIO (no download needed)."""
    if pymrio is None:
        raise ImportError("pymrio is required to load the test MRIO")
    mrio = pymrio.load_test()
    mrio.calc_all()
    return mrio


def load_oecd_icio(
    storage_folder: Path | str = "data/oecd_icio",
    years: Iterable[int] | int = (2018,),
    version: str = "v2023",
    force_download: bool = False,
):
    """Download/parse OECD ICIO and cache."""
    if pymrio is None:
        raise ImportError("pymrio is required to download or load OECD ICIO")

    storage = Path(storage_folder)
    storage.mkdir(parents=True, exist_ok=True)

    if isinstance(years, (int, str)):
        years = [years]
    first_year = list(years)[0]
    mrio_dir = storage / f"mrio_system_{first_year}"

    # Look for any existing saved MRIO directory containing file_parameters.json
    existing = list(storage.glob(f"**/file_parameters.json"))
    existing = [p.parent for p in existing if str(first_year) in str(p.parent)]
    if force_download and mrio_dir.exists():
        import shutil

        shutil.rmtree(mrio_dir)
    elif existing and not force_download:
        mrio = pymrio.load(existing[0])
        mrio.calc_all()
        return mrio

    pymrio.download_oecd(
        storage_folder=storage,
        version=version,
        years=years,
        overwrite_existing=force_download,
    )
    mrio = pymrio.parse_oecd(storage, year=first_year)
    mrio.calc_all()
    mrio.save(mrio_dir)
    return mrio


def load_wiod2013(
    storage_folder: Path | str = "data/wiod13_2011",
    years: Iterable[int] | int = (2011,),
    force_download: bool = False,
):
    """Download/parse WIOD 2013 (with environmental extensions) and cache.

    Parameters
    ----------
    storage_folder : Path or str
        Destination for downloads and cache.
    years : iterable or int
        Years to download/parse (two-digit accepted). Defaults to 2011.
    force_download : bool
        If True, redownload even when cache exists.
    """
    if pymrio is None:
        raise ImportError("pymrio is required to download or load WIOD 2013")

    storage = Path(storage_folder)
    storage.mkdir(parents=True, exist_ok=True)

    if isinstance(years, (int, str)):
        years = [years]

    # Use first year in path naming for cache; multi-year users can extend.
    first_year = list(years)[0]
    pkl_path = storage / f"mrio_system_{first_year}.pkl"

    if pkl_path.exists() and not force_download:
        mrio = pymrio.read_mrio(pkl_path)
        mrio.calc_all()
        return mrio

    # Download IO + satellite data
    pymrio.download_wiod2013(storage_folder=storage, years=years)

    # Ensure IO table exists (WIOD downloads may require credentials; if missing, prompt user)
    wiot_files = [
        fl
        for fl in storage.iterdir()
        if fl.name.lower().startswith("wiot") and fl.suffix.lower() in {".csv", ".txt", ".xlsx"}
    ]
    if not wiot_files:
        raise FileNotFoundError(
            f"No WIOT IO table found in {storage}. "
            "WIOD 2013 IO tables often require licensed access; "
            "please place the WIOT file for the requested year in this folder and rerun."
        )
    if len(wiot_files) > 1 and not any(str(first_year)[-2:] in f.name for f in wiot_files):
        raise RuntimeError(
            f"Multiple WIOT files detected in {storage} without a clear match for year {first_year}. "
            "Specify or clean the folder, then rerun."
        )

    # Parse WIOD IO tables (pymrio handles satellite parsing internally)
    mrio = pymrio.parse_wiod(storage, year=first_year)
    mrio.calc_all()
    mrio.save(pkl_path)
    return mrio


def final_demand_vector(
    mrio, demand_categories: Optional[Iterable[str]] = None
) -> pd.Series:
    """
    Build a final demand vector summed over selected categories.

    Parameters
    ----------
    mrio : pymrio.MRIOSystem
        Must expose a final demand table `Y` with sector index.
    demand_categories : iterable, optional
        Column subset in mrio.Y to include (e.g., households, exports).
    """
    Y = mrio.Y
    if demand_categories is not None:
        demand_categories = list(demand_categories)
        missing = set(demand_categories) - set(Y.columns)
        if missing:
            raise KeyError(f"demand categories not found: {missing}")
        Y = Y[demand_categories]
    return Y.sum(axis=1)


def satellite_intensity(
    satellite_df: pd.DataFrame, output: pd.Series
) -> pd.Series:
    """
    Convert satellite totals to intensity per unit output.

    satellite_df : DataFrame
        Rows indexed by sector, columns as satellite measures (choose one).
    output : Series
        Sectoral gross output to normalise satellite accounts.
    """
    if not satellite_df.index.equals(output.index):
        raise ValueError("satellite_df index must align with output index")
    total = satellite_df.sum(axis=1)
    return total.divide(output).rename("intensity")
