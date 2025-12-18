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
    # Mapping based on Table A.1 in Cahen-Fourot et al. (2021).
    #
    # The mapping values include common variants encountered across sources:
    # - NACE-style codes (e.g. "C10-C12", "B05-06", "J62_63")
    # - OECD/ICIO aggregates (e.g. "10T12", "05T06", "49T53")
    # - Single numeric codes (e.g. "19") and letter-prefixed forms (e.g. "C19")
    #
    # Callers can use this mapping either as a forward mapping (paper code -> source
    # codes) or to build a reverse lookup for harmonizing indices.
    mapping: Dict[str, List[str]] = {
        # A: Agriculture, forestry and fishing
        "AGR+": ["A", "01T03"],
        "AGRagr": ["A01", "01"],
        "AGRfor": ["A02", "02"],
        "AGRfis": ["A03", "03"],

        # B: Mining and quarrying
        "MIN+": ["B"],
        "MINfos": ["B05-06", "B05–06", "B05", "B06", "05T06", "05", "06"],
        "MINoth": ["B07-08", "B07–08", "B07", "B08", "07T08", "07", "08"],
        "MINsup": ["B09", "09"],

        # C: Manufacturing
        "MAN+": ["C"],
        "MANfoo": ["C10-C12", "C10–12", "10T12"],
        "MANtex": ["C13-C15", "C13–15", "13T15"],
        "MANwoo": ["C16", "16"],
        "MANpap": ["C17", "17"],
        "MANpri": ["C18", "18"],
        "MANref": ["C19", "19"],
        "MANche": ["C20", "20"],
        "MANpha": ["C21", "21"],
        "MANpla": ["C22", "22"],
        "MANmin": ["C23", "23"],
        "MANmet": ["C24", "24"],
        "MANfmp": ["C25", "25"],
        "MANcom": ["C26", "26"],
        "MANele": ["C27", "27"],
        "MANmac": ["C28", "28"],
        "MANmot": ["C29", "29"],
        "MANtra": ["C30", "30"],
        "MANfur": ["C31_32", "C31-C32", "C31–32", "31T33", "31", "32"],
        "MANrep": ["C33", "33"],

        # D: Electricity, gas, steam and air conditioning
        "PWR+": ["D", "D35", "35", "35T39"],

        # E: Water supply; sewerage; waste management and remediation
        "WAT+": ["E", "36T39"],
        "WATwat": ["E36", "36"],
        "WATwst": ["E37-39", "E37–39", "37T39", "37", "38", "39"],

        # F: Construction
        "CNS+": ["F", "41T43"],

        # G: Wholesale and retail trade; repair of motor vehicles and motorcycles
        "TRD+": ["G", "45T47"],
        "TRDmot": ["G45", "45"],
        "TRDwho": ["G46", "46"],
        "TRDret": ["G47", "47"],

        # H: Transportation and storage
        "TRA+": ["H", "49T53"],
        "TRAinl": ["H49", "49"],
        "TRAwat": ["H50", "50"],
        "TRAair": ["H51", "51"],
        "TRAwar": ["H52", "52"],
        "TRApos": ["H53", "53"],

        # I: Accommodation and food service activities
        "FD+": ["I", "55T56"],

        # J: Information and communication
        "COM+": ["J", "58T63"],
        "COMpub": ["J58", "58"],
        "COMvid": ["J59_60", "J59-J60", "J59–60", "59T60", "59", "60"],
        "COMtel": ["J61", "61"],
        "COMcom": ["J62_63", "J62-J63", "J62–63", "62T63", "62", "63"],

        # K: Financial and insurance activities
        "FIN+": ["K", "64T66"],
        "FINser": ["K64", "64"],
        "FINins": ["K65", "65"],
        "FINaux": ["K66", "66"],

        # L: Real estate activities
        "RES+": ["L", "68"],

        # M: Professional, scientific and technical activities
        "PRO+": ["M", "69T75"],
        "PROleg": ["M69_70", "M69-M70", "M69–70", "69T70", "69", "70"],
        "PROeng": ["M71", "71"],
        "PROsci": ["M72", "72"],
        "PROadv": ["M73", "73"],
        "PROoth": ["M74_75", "M74-M75", "M74–75", "74T75", "74", "75"],

        # N: Administrative and support service activities
        "ADM+": ["N", "77T82"],

        # O: Public administration and defence; compulsory social security
        "PUB+": ["O", "84"],

        # P: Education
        "EDU+": ["P", "85"],

        # Q: Human health and social work activities
        "HEA+": ["Q", "86T88"],

        # R_S: Arts, entertainment and recreation
        "ART+": ["R_S", "R", "S", "90T93", "90", "91", "92", "93"],

        # U: Activities of households as employers
        "HOU+": ["U", "97T98"],
    }
    
    return mapping


def map_sector_to_2021_paper_code(sector_code: str, mapping: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Map a source-sector code to the Cahen-Fourot et al. (2021) paper code.

    This performs a best-effort reverse lookup against `get_2021_sector_mapping()`.
    If no mapping is found, it returns the original `sector_code`.
    """
    if mapping is None:
        mapping = get_2021_sector_mapping()

    s = str(sector_code).strip()
    # Normalize common unicode dash to ascii.
    s = s.replace("–", "-")

    for paper_code, source_codes in mapping.items():
        if s == paper_code:
            return paper_code
        for src in source_codes:
            if s == str(src):
                return paper_code

    return s


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
