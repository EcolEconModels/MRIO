from pathlib import Path
from typing import Tuple

import pandas as pd
import pymrio


def load_mrio(dataset: str, year: int) -> Tuple[pymrio.IOSystem, pd.Series]:
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
        
        data_dir = Path(f"data/oecd_icio_{year}")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            
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
        data_dir = Path(f"data/wiod13_{year}")
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
        data_dir = Path(f"data/exiobase3_{year}")
        print(f"Ensuring EXIOBASE 3 data for {year} in {data_dir}...")
        
        # Check if pickled system exists to avoid reparsing
        if (data_dir / "mrio_system.pkl").exists():
             mrio = pymrio.load(data_dir / "mrio_system.pkl")
        else:
             # Download if needed
             if not (data_dir / f"IOT_{year}_pxp.zip").exists():
                 pymrio.download_exiobase3(storage_folder=data_dir, years=[year], system="pxp")
             
             mrio = pymrio.parse_exiobase3(data_dir / f"IOT_{year}_pxp.zip")
             mrio.save(data_dir / "mrio_system.pkl")

        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)
    
    # Ensure Ghosh is calculated
    if getattr(mrio, 'G', None) is None:
        mrio.calc_all(include_ghosh=True)
        
    return mrio, y_full
