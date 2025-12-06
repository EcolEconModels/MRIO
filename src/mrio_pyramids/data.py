from pathlib import Path
from typing import Tuple

import pandas as pd
import pymrio


def load_mrio(dataset: str, year: int) -> Tuple[pymrio.IOSystem, pd.Series]:
    """
    Load MRIO data with basic caching logic for standard datasets.
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
        data_dir = Path(f"data/oecd_icio_{year}")
        # Check for file_parameters.json as a marker of existing dataset
        param_file = list(data_dir.glob("**/file_parameters.json"))
        if not param_file:
            print(f"Downloading OECD {year} data...")
            pymrio.download_oecd(storage_folder=data_dir.parent, years=[year])
        
        mrio = pymrio.parse_oecd(path=data_dir, year=year)
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    elif dataset == "wiod":
        data_dir = Path(f"data/wiod13_{year}")
        if not (data_dir / f"mrio_system_{year}.pkl").exists():
             print(f"Downloading WIOD {year} data...")
             pymrio.download_wiod2013(storage_folder=data_dir, years=[year])
             mrio = pymrio.parse_wiod(path=data_dir, year=year)
             mrio.save(data_dir / f"mrio_system_{year}.pkl")
        else:
             mrio = pymrio.load(data_dir / f"mrio_system_{year}.pkl")
        
        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)

    else:
        # Exiobase
        data_dir = Path(f"data/exiobase3_{year}")
        if not (data_dir / "mrio_system.pkl").exists():
             print(f"Downloading EXIOBASE {year} data...")
             pymrio.download_exiobase3(storage_folder=data_dir, years=[year], system="pxp")
             mrio = pymrio.parse_exiobase3(data_dir / f"IOT_{year}_pxp.zip")
             mrio.save(data_dir / "mrio_system.pkl")
        else:
             mrio = pymrio.load(data_dir / "mrio_system.pkl")

        y_cols = [c for c in mrio.Y.columns if "EXP" not in c]
        y_full = mrio.Y[y_cols].sum(axis=1)
    
    mrio.calc_all(include_ghosh=True)
    return mrio, y_full
