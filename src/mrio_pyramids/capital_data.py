"""
Capital stock data loading for stranding analysis.

Loads capital stock data from WIOD Socio-Economic Accounts or other sources.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging
import requests
import zipfile
import io

logger = logging.getLogger(__name__)


def load_wiod_capital_stock(
    data_dir: Path = Path("data/wiod_sea"),
    year: int = 2014
) -> pd.DataFrame:
    """
    Load capital stock data from WIOD Socio-Economic Accounts (SEA).
    
    WIOD 2016 release contains capital stock data for 2000-2014.
    Data available at: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/PJ2M1C
    
    Args:
        data_dir: Directory to store/load WIOD SEA data
        year: Year to load (2000-2014 for WIOD 2016)
        
    Returns:
        DataFrame with capital stock by country-sector (MultiIndex)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    capital_file = data_dir / f"CAP_{year}.csv"
    
    if capital_file.exists():
        logger.info(f"Loading capital stock data from {capital_file}")
        capital = pd.read_csv(capital_file, index_col=[0, 1])
        return capital
    
    # Otherwise, provide download instructions
    logger.warning(f"Capital stock data not found at {capital_file}")
    logger.info("To download WIOD SEA capital stock data:")
    logger.info("1. Visit: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/PJ2M1C")
    logger.info("2. Download 'CAP.zip' or individual CAP_{year}.csv files")
    logger.info(f"3. Extract to: {data_dir}")
    logger.info("4. Rerun the script")
    
    raise FileNotFoundError(
        f"Capital stock data not found. Please download WIOD SEA data manually.\n"
        f"Expected file: {capital_file}\n"
        f"Download from: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/PJ2M1C"
    )


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
    data_dir: Optional[Path] = None
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
    
    elif source == "wiod":
        if data_dir is None:
            data_dir = Path("data/wiod_sea")
        
        try:
            wiod_capital = load_wiod_capital_stock(data_dir, year)
            
            # Align with mrio structure
            # WIOD SEA typically has (Country, Sector) index
            # Need to match with mrio.x index
            
            if isinstance(output.index, pd.MultiIndex):
                # Reindex to match mrio
                capital = wiod_capital.reindex(output.index)
                
                # Fill missing with mock
                if capital.isna().any():
                    logger.warning(f"Missing capital data for {capital.isna().sum()} sectors. Filling with mock.")
                    mock = generate_mock_capital_stock(output[capital.isna()])
                    capital = capital.fillna(mock)
                
                return capital
            else:
                logger.warning("MRIO index structure doesn't match WIOD SEA. Using mock data.")
                return generate_mock_capital_stock(output)
                
        except FileNotFoundError as e:
            logger.error(str(e))
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
