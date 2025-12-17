from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd
import pymrio


def default_data_root() -> Path:
    """
    Return the default data root directory for caches (Eurostat/EXIOBASE/etc.).

    Prefers the env var `MRIO_DATA_ROOT` if set; otherwise anchors to the repo root
    (so running from `src/` doesn't accidentally create `src/data/...`).
    """
    env = os.environ.get("MRIO_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/mrio_pyramids/data.py -> repo_root is two parents up from this file.
    return Path(__file__).resolve().parents[2] / "data"


def load_mrio(
    dataset: str,
    year: int,
    *,
    force_download: bool = False,
    data_root: str | Path | None = None,
    calc_all: bool = True,
    include_ghosh: bool = True,
    verbose: bool = False,
) -> Tuple[pymrio.IOSystem, pd.Series]:
    """
    Load MRIO data using native pymrio download and parsing capabilities.
    Returns the mrio system and the calculated final demand vector y.
    
    Parameters
    ----------
    dataset : str
        One of "test", "oecd", "wiod", "exiobase".
    year : int
        Year of analysis.
        
    Returns
    -------
    mrio : pymrio.IOSystem
        The loaded input-output system with calculated inverses.
    y_full : pd.Series
        The total final demand vector (summed across categories).
    """
    if dataset == "test":
        if verbose:
            print("Loading internal pymrio test dataset (no download)...")
        mrio = pymrio.load_test()
        y_full = mrio.Y.sum(axis=1)

    elif dataset == "oecd":
        # OECD 2018 edition covers 2005-2015.
        # OECD 2016 edition covers 1995-2011.
        # We target the 2018 edition for year 2010 as it's a common replication point.
        # If year is outside 2005-2015, we might need a different version.
        
        # Pymrio defaults: version='v2021' (which uses 5-year bundles).
        # We explicitly request v2018 for individual year handling flexibility or simplicity if applicable.
        # Or we use v2021 if that's what the user has.
        # Based on file listing (ICIO2023), it seems user has v2021 (published 2023 maybe).
        
        # Let's try standard behavior:
        # If we want 2010, we can try detecting what's available or just ask pymrio to handle it.
        
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"oecd_icio_{year}"
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            
        if verbose:
            print(f"Ensuring OECD data for {year} in {data_dir}...")
        
        # Note: download_oecd verifies existence internally.
        # We specify version='v2018' to ensure we get the yearly files expected for 2010 replication if possible,
        # or stick to the simple API.
        # However, 2020 paper used 2010 data from Eurostat. OECD is the fallback.
        # Let's use the 'v2018' version logic which is stable for 2010.
        
        try:
             pymrio.download_oecd(storage_folder=data_dir, years=[year], version="v2018")
        except Exception as e:
             print(f"Download warning/check: {e}")

        # parser looks for files in the specific path
        mrio = pymrio.parse_oecd(path=data_dir, year=year)
        
        # Filter for Y columns that are not exports (if needed, or just sum)
        # OECD columns usually: "AUS_HFCE", etc.
        # Exports might be "AUS_EXP".
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    elif dataset == "wiod":
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"wiod13_{year}"
        if verbose:
            print(f"Ensuring WIOD 2013 data for {year} in {data_dir}...")
        
        if not (data_dir / "metadata.json").exists(): # Simple check, pymrio does its own
            pymrio.download_wiod2013(storage_folder=data_dir, years=[year])
            
        if (data_dir / f"mrio_system_{year}.pkl").exists():
            mrio = pymrio.load(data_dir / f"mrio_system_{year}.pkl")
        else:
            mrio = pymrio.parse_wiod(path=data_dir, year=year)
            mrio.save(data_dir / f"mrio_system_{year}.pkl")
        
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    else:
        # Exiobase
        root = Path(data_root) if data_root is not None else default_data_root()
        data_dir = root / f"exiobase3_{year}"
        data_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensuring EXIOBASE 3 data for {year} in {data_dir}...")
        
        # Pymrio saves to a *directory*; avoid giving it a directory name with an
        # extension (e.g., "mrio_system.pkl") because `MRIOMetaData.save()` will
        # treat that as a filename and fail with IsADirectoryError.
        mrio_cache_dir = data_dir / "mrio_system"
        legacy_cache_dir = data_dir / "mrio_system.pkl"

        if not mrio_cache_dir.exists() and legacy_cache_dir.is_dir():
            # Backward-compat: previous runs may have created a directory named
            # "mrio_system.pkl". Prefer migrating it to the extension-less name.
            if verbose:
                print(f"Found legacy EXIOBASE cache dir at {legacy_cache_dir}")
            try:
                legacy_cache_dir.rename(mrio_cache_dir)
                if verbose:
                    print(f"Migrated legacy cache dir to {mrio_cache_dir}")
            except Exception:
                # If migration fails (e.g., permissions), just use the legacy dir
                # for reading; avoid writing to it again.
                mrio_cache_dir = legacy_cache_dir

        # Optional: force re-download/re-parse
        if force_download and mrio_cache_dir.exists():
            ts = int(time.time())
            mrio_cache_dir.rename(data_dir / f"mrio_system.forced.{ts}")

        # Check if cached system exists to avoid reparsing
        if mrio_cache_dir.exists():
            if verbose:
                print(f"Loading cached EXIOBASE IOSystem from {mrio_cache_dir} ...")
            mrio = pymrio.load(mrio_cache_dir)
        else:
             # Download if needed
             zip_path = data_dir / f"IOT_{year}_pxp.zip"
             if force_download and zip_path.exists():
                 ts = int(time.time())
                 zip_path.rename(data_dir / f"IOT_{year}_pxp.zip.corrupt_or_forced.{ts}")

             def _zip_looks_ok(path: Path) -> bool:
                 try:
                     with zipfile.ZipFile(path, "r") as zf:
                         names = set(zf.namelist())
                     return "file_parameters.json" in names
                 except Exception:
                     return False

             if zip_path.exists() and not _zip_looks_ok(zip_path):
                 ts = int(time.time())
                 zip_path.rename(data_dir / f"IOT_{year}_pxp.zip.corrupt.{ts}")

             if not zip_path.exists():
                 if verbose:
                     print(f"EXIOBASE zip not found at {zip_path}; downloading via pymrio...")
                 pymrio.download_exiobase3(storage_folder=data_dir, years=[year], system="pxp")
             
             if verbose:
                 print(f"Parsing EXIOBASE zip {zip_path} (this can take a while)...")
             mrio = pymrio.parse_exiobase3(zip_path)
             # If a previous partial run created a directory at this path, move it aside.
             if mrio_cache_dir.exists():
                 ts = int(time.time())
                 mrio_cache_dir.rename(data_dir / f"mrio_system.stale.{ts}")
             if verbose:
                 print(f"Saving parsed EXIOBASE IOSystem to {mrio_cache_dir} ...")
             mrio.save(mrio_cache_dir)

        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)
    
    # Ensure Ghosh is calculated
    if calc_all and getattr(mrio, "G", None) is None:
        if verbose:
            print("Computing MRIO derived matrices (calc_all)...")
        mrio.calc_all(include_ghosh=include_ghosh)
        
    return mrio, y_full
