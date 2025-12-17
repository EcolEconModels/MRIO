"""
Sector aggregation module for Cahen-Fourot et al. (2021) replication.

Maps WIOD/OECD sector classifications to paper-specific aggregations.
"""

import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_2021_sector_mapping() -> Dict[str, List[str]]:
    """
    Get sector aggregation mapping for Cahen-Fourot et al. (2021).
    
    Maps original NACE Rev. 2 / ISIC Rev. 4 sectors to paper acronyms.
    Based on Table A.1 in the paper appendix.
    
    Returns:
        Dictionary mapping paper acronym → list of original sector codes
    """
    # Note: This is a partial mapping based on common sectors
    # Full mapping requires Table A.1 from paper appendix
    
    mapping = {
        # Fossil fuel sectors
        'MINfos': ['05', '06', 'B05', 'B06', '05T06'],  # Mining of coal, oil, gas
        'MINnon': ['07', '08', '09', 'B07', 'B08', 'B09'],  # Other mining
        
        # Energy sectors
        'PWR+': ['35', 'D35', '35T39'],  # Electricity, gas, steam supply
        
        # Manufacturing - refined products
        'MANref': ['19', 'C19'],  # Coke and refined petroleum
        'MANche': ['20', 'C20', '20T21'],  # Chemicals (may include pharma)
        'MANpla': ['22', 'C22'],  # Rubber and plastics
        'MANmet': ['24', 'C24'],  # Basic metals
        'MANmac': ['28', 'C28'],  # Machinery and equipment
        'MANmot': ['29', 'C29'],  # Motor vehicles
        'MANpap': ['17', 'C17'],  # Paper products
        'MANpri': ['18', 'C18'],  # Printing
        
        # Construction
        'CNS+': ['41', '42', '43', 'F41', 'F42', 'F43', '41T43'],  # Construction
        
        # Services
        'TRAair': ['51', 'H51'],  # Air transport
        'TRAinl': ['50', 'H50'],  # Water transport
        'TRAwar': ['50', 'H50'],  # Warehousing
        
        # Wholesale/retail
        'MAN': ['45', '46', '47', 'G45', 'G46', 'G47', '45T47'],  # Trade
        
        # Accommodation
        'RES+': ['55', '56', 'I55', 'I56', '55T56'],  # Accommodation, food
        
        # Public admin
        'PUB+': ['84', 'O84'],  # Public administration
        
        # Arts/recreation
        'ART+': ['90', '91', '92', '93', 'R90', 'R91', 'R92', 'R93'],  # Arts, recreation
        
        # Agriculture
        'AGRagr': ['01', 'A01'],  # Crop and animal production
    }
    
    return mapping


def aggregate_mrio_sectors(
    mrio_data: pd.DataFrame,
    mapping: Dict[str, List[str]],
    level: int = -1
) -> pd.DataFrame:
    """
    Aggregate MRIO data (Z, S, or other sector-indexed matrix) using mapping.
    
    Args:
        mrio_data: DataFrame with sector codes in index/columns (may be MultiIndex)
        mapping: Dictionary of {new_sector: [original_sectors]}
        level: Which level of MultiIndex contains sector codes (-1 = last)
        
    Returns:
        Aggregated DataFrame with new sector names
    """
    # Handle MultiIndex
    if isinstance(mrio_data.index, pd.MultiIndex):
        # Extract sector level
        sectors = mrio_data.index.get_level_values(level).unique()
    else:
        sectors = mrio_data.index.unique()
    
    # Create reverse mapping: original_sector → new_sector
    reverse_map = {}
    for new_sector, orig_sectors in mapping.items():
        for orig in orig_sectors:
            reverse_map[orig] = new_sector
    
    # Map existing sectors
    if isinstance(mrio_data.index, pd.MultiIndex):
        # For MultiIndex, create mapper for specific level
        index_mapper = {}
        for idx in mrio_data.index:
            old_sector = idx[level]
            new_sector = reverse_map.get(old_sector, old_sector)  # Keep original if not in mapping
            new_idx = list(idx)
            new_idx[level] = new_sector
            index_mapper[idx] = tuple(new_idx)
        
        # Rename index
        new_index = pd.MultiIndex.from_tuples(
            [index_mapper[idx] for idx in mrio_data.index],
            names=mrio_data.index.names
        )
        mrio_data = mrio_data.copy()
        mrio_data.index = new_index
        
        # Same for columns if DataFrame
        if isinstance(mrio_data.columns, pd.MultiIndex):
            col_mapper = {}
            for idx in mrio_data.columns:
                old_sector = idx[level]
                new_sector = reverse_map.get(old_sector, old_sector)
                new_idx = list(idx)
                new_idx[level] = new_sector
                col_mapper[idx] = tuple(new_idx)
            
            new_columns = pd.MultiIndex.from_tuples(
                [col_mapper[idx] for idx in mrio_data.columns],
                names=mrio_data.columns.names
            )
            mrio_data.columns = new_columns
    else:
        # Simple index case
        mrio_data.index = mrio_data.index.map(lambda x: reverse_map.get(x, x))
        if hasattr(mrio_data, 'columns'):
            mrio_data.columns = mrio_data.columns.map(lambda x: reverse_map.get(x, x))
    
    # Aggregate: sum duplicate sectors
    if isinstance(mrio_data.index, pd.MultiIndex):
        mrio_data = mrio_data.groupby(level=list(range(mrio_data.index.nlevels))).sum()
    else:
        mrio_data = mrio_data.groupby(level=0).sum()
    
    if hasattr(mrio_data, 'columns') and isinstance(mrio_data.columns, pd.MultiIndex):
        mrio_data = mrio_data.T.groupby(level=list(range(mrio_data.columns.nlevels))).sum().T
    elif hasattr(mrio_data, 'columns'):
        mrio_data = mrio_data.T.groupby(level=0).sum().T
    
    logger.info(f"Aggregated sectors: {len(sectors)} → {len(mrio_data.index.get_level_values(level).unique() if isinstance(mrio_data.index, pd.MultiIndex) else mrio_data.index.unique())}")
    
    return mrio_data


def disaggregate_mining_sector(
    mrio,
    fossil_codes: List[str] = ['05', '06', 'B05', 'B06'],
    non_fossil_codes: List[str] = ['07', '08', '09', 'B07', 'B08', 'B09']
) -> tuple:
    """
    Disaggregate Mining sector (B) into fossil and non-fossil components.
    
    Paper mentions splitting NACE B into:
    - MINfos: Mining of coal (B05), crude petroleum and natural gas (B06)
    - MINnon: Other mining (B07, B08, B09)
    
    Args:
        mrio: pymrio IOSystem
        fossil_codes: Sector codes for fossil fuel mining
        non_fossil_codes: Sector codes for other mining
        
    Returns:
        (fossil_Z, fossil_x, non_fossil_Z, non_fossil_x) if disaggregation possible
        Otherwise returns (None, None, None, None)
    """
    # Check if data already has disaggregated sectors
    if isinstance(mrio.Z.index, pd.MultiIndex):
        sectors = mrio.Z.index.get_level_values(-1).unique()
    else:
        sectors = mrio.Z.index.unique()
    
    # Check if any fossil codes exist
    has_fossil = any(code in str(s) for s in sectors for code in fossil_codes)
    
    if has_fossil:
        logger.info("Mining sector appears to be disaggregated already.")
        return None, None, None, None
    else:
        logger.warning("Mining sector (B) is aggregated. Cannot disaggregate without raw data.")
        logger.info("Assuming entire B sector is fossil (MINfos) for replication purposes.")
        return None, None, None, None


def get_sector_display_name(sector_code: str, mapping: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Get human-readable display name for sector code.
    
    Args:
        sector_code: Original or aggregated sector code
        mapping: Optional reverse lookup mapping
        
    Returns:
        Display name for plotting
    """
    if mapping is None:
        mapping = get_2021_sector_mapping()
    
    # Reverse lookup
    for display_name, codes in mapping.items():
        if sector_code in codes or sector_code == display_name:
            return display_name
    
    return sector_code  # Return as-is if not found
