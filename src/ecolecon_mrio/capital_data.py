"""
Capital stock data loading for stranding analysis.

Loads capital stock data from WIOD Socio-Economic Accounts or other sources.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import urllib.request

from ecolecon_mrio.aggregation import map_sector_to_2021_paper_code

logger = logging.getLogger(__name__)


WIOD16_SEA_DATAVERSE_URL = "https://dataverse.nl/api/access/datafile/199095"


def _looks_like_xlsx(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def ensure_wiod16_sea(
    data_dir: Path,
    *,
    force_download: bool = False,
) -> Path:
    """
    Ensure the WIOD 2016 Socio-Economic Accounts (SEA) file is present locally.

    The dataset metadata is at `doi:10.34894/PJ2M1C` and the SEA xlsx file is
    available via Dataverse API.

    Returns the path to `Socio_Economic_Accounts.xlsx`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    sea_path = data_dir / "Socio_Economic_Accounts.xlsx"
    if sea_path.exists() and _looks_like_xlsx(sea_path) and not force_download:
        return sea_path

    tmp_path = sea_path.with_suffix(".xlsx.tmp")
    action = "Re-downloading" if sea_path.exists() else "Downloading"
    logger.info(f"{action} WIOD 2016 SEA to {sea_path} ...")
    try:
        urllib.request.urlretrieve(WIOD16_SEA_DATAVERSE_URL, tmp_path)  # nosec B310
    except Exception as e:
        raise RuntimeError(
            "Failed to download WIOD 2016 Socio-Economic Accounts.\n"
            f"URL: {WIOD16_SEA_DATAVERSE_URL}\n"
            f"Either download it manually to {sea_path} or retry with network access."
        ) from e
    if not _looks_like_xlsx(tmp_path):
        # Most common failure mode: HTML error page stored on disk.
        raise RuntimeError(f"Downloaded file does not look like a valid .xlsx: {tmp_path}")
    tmp_path.replace(sea_path)
    return sea_path


def load_wiod16_sea_series(
    *,
    sea_path: Path,
    variable: str,
    year: int,
) -> pd.Series:
    """
    Load a single WIOD SEA variable for a given year as a Series indexed by (country, code).

    Variables include:
    - K: Nominal capital stock (local currency)
    - GO: Gross output (local currency)
    """
    # Only read the columns we need to keep this fast.
    df = pd.read_excel(
        sea_path,
        sheet_name="DATA",
        engine="openpyxl",
        usecols=lambda c: c in {"country", "variable", "code", year},
    )
    df = df[df["variable"] == variable].copy()
    df = df.drop(columns=["variable"])
    df = df.rename(columns={year: "value"})
    df["code"] = df["code"].astype(str).str.strip().str.replace("–", "-", regex=False)
    return df.set_index(["country", "code"])["value"]


def generate_mock_capital_stock(
    output: pd.Series,
    seed: int = 42,
    mean_intensity: float = 1.5,
    std_intensity: float = 0.5
) -> pd.Series:
    """
    Generate mock capital stock data for testing.
    
    Uses random capital intensities around a mean value.
    
    Args:
        output: Gross output by sector (from MRIO)
        seed: Random seed for reproducibility
        mean_intensity: Mean capital intensity (k/x)
        std_intensity: Std dev of capital intensity
        
    Returns:
        Capital stock series aligned with output index
    """
    import numpy as np
    
    np.random.seed(seed)
    
    # Generate random capital intensities
    n = len(output)
    intensities = np.random.normal(mean_intensity, std_intensity, n)
    intensities = np.clip(intensities, 0.1, 3.0)  # Reasonable range
    
    # Capital = intensity * output
    capital = output * intensities
    
    logger.warning(f"Generated MOCK capital stock data (n={n}, mean κ={intensities.mean():.2f})")
    
    return capital


def load_capital_stock_data(
    mrio,
    year: int,
    source: str = "mock",
    data_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.Series:
    """
    Load capital stock data from specified source.
    
    Args:
        mrio: pymrio IOSystem
        year: Year for capital data
        source: One of "wiod", "mock", "penn" (Penn World Table)
        data_dir: Optional data directory path
        
    Returns:
        Capital stock series indexed by (region, sector) matching mrio structure
    """
    output = mrio.x.squeeze()
    
    if source == "mock":
        return generate_mock_capital_stock(output)
    
    elif source in {"wiod", "wiod16"}:
        # Use WIOD 2016 Socio-Economic Accounts to build capital intensities κ = K/GO
        # in a currency-consistent way (both K and GO are in local currency). We then
        # scale by MRIO output to obtain an approximate capital stock in MRIO units.
        if data_dir is None:
            data_dir = Path("data/wiod16")

        try:
            sea_path = ensure_wiod16_sea(Path(data_dir), force_download=force_download)
            K = load_wiod16_sea_series(sea_path=sea_path, variable="K", year=year)
            GO = load_wiod16_sea_series(sea_path=sea_path, variable="GO", year=year)

            kappa_raw = (K / GO.replace(0, pd.NA)).dropna()
            if kappa_raw.empty:
                raise RuntimeError("WIOD SEA κ series is empty after cleaning.")

            # Aggregate κ to paper sector codes for robust alignment.
            kappa_df = (
                kappa_raw.rename("kappa")
                .reset_index()
                .assign(paper=lambda d: d["code"].map(map_sector_to_2021_paper_code))
                .groupby(["country", "paper"], as_index=True)["kappa"]
                .mean()
            )

            if not isinstance(output.index, pd.MultiIndex):
                logger.warning("MRIO output index is not a MultiIndex; cannot align WIOD SEA by country/sector. Using mock data.")
                return generate_mock_capital_stock(output)

            out_df = output.rename("output").reset_index()
            out_df.columns = ["country", "sector", "output"]
            out_df["paper"] = out_df["sector"].map(map_sector_to_2021_paper_code)

            # Lookup κ by (country, paper); fall back to country mean, then global mean.
            kappa_lookup = kappa_df
            kappa_country_mean = kappa_df.groupby(level=0).mean()
            kappa_global_mean = float(kappa_df.mean())

            def _lookup(row) -> float:
                key = (row["country"], row["paper"])
                if key in kappa_lookup.index:
                    return float(kappa_lookup.loc[key])
                if row["country"] in kappa_country_mean.index:
                    return float(kappa_country_mean.loc[row["country"]])
                return kappa_global_mean

            out_df["kappa"] = out_df.apply(_lookup, axis=1)
            capital = out_df["kappa"].to_numpy() * out_df["output"].to_numpy()
            capital = pd.Series(
                capital,
                index=pd.MultiIndex.from_frame(out_df[["country", "sector"]]),
                name="capital_stock",
            )
            return capital.reindex(output.index)

        except Exception as e:
            logger.error(f"Failed to load WIOD SEA capital data ({type(e).__name__}): {e}")
            logger.info("Falling back to mock capital stock data.")
            return generate_mock_capital_stock(output)
    
    elif source == "penn":
        logger.warning("Penn World Table capital stock integration not yet implemented.")
        logger.info("Falling back to mock capital stock data.")
        return generate_mock_capital_stock(output)
    
    else:
        raise ValueError(f"Unknown capital stock source: {source}")


def calculate_capital_output_ratio(
    capital: pd.Series,
    output: pd.Series
) -> pd.DataFrame:
    """
    Calculate capital-output ratios by sector.
    
    Args:
        capital: Capital stock
        output: Gross output
        
    Returns:
        DataFrame with sectors and their capital-output ratios
    """
    ratio = capital / output.replace(0, float('nan'))
    ratio = ratio.fillna(0)
    
    result = pd.DataFrame({
        'capital': capital,
        'output': output,
        'capital_intensity': ratio
    })
    
    return result.sort_values('capital_intensity', ascending=False)
