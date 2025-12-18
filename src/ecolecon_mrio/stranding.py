"""
Stranding Analysis Module for Cahen-Fourot et al. (2021) Replication.

Implements capital stranding multiplier calculations and cascade network analysis.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calc_capital_intensity(output: pd.Series, capital_stock: pd.Series) -> pd.Series:
    """
    Calculate capital intensity κ = k / x for each sector.
    
    Args:
        output: Gross output by sector (x)
        capital_stock: Physical capital stock by sector (k)
        
    Returns:
        Capital intensity κ by sector
    """
    # Align indices
    capital_stock = capital_stock.reindex(output.index, fill_value=0)
    
    # Avoid division by zero
    kappa = capital_stock / output.replace(0, np.nan)
    kappa = kappa.fillna(0)
    
    return kappa


def calc_stranding_multipliers(
    G: pd.DataFrame,
    capital_intensity: pd.Series
) -> pd.DataFrame:
    """
    Calculate stranding multiplier matrix S = κ̂ G^T.
    
    S_ij represents the capital at risk in sector i from a $1 reduction 
    in primary inputs of sector j.
    
    Args:
        G: Ghosh inverse matrix (I - B)^(-1)
        capital_intensity: Capital intensity κ by sector
        
    Returns:
        Stranding multiplier matrix S
    """
    # Ensure alignment
    capital_intensity = capital_intensity.reindex(G.index, fill_value=0)
    
    # κ̂ = diag(κ)
    kappa_hat = np.diag(capital_intensity.values)
    
    # S = κ̂ G^T
    S = pd.DataFrame(
        kappa_hat @ G.T.values,
        index=G.index,
        columns=G.columns
    )
    
    logger.info(f"Calculated stranding multipliers: shape {S.shape}")
    logger.info(f"Total stranding multiplier (fossil col sum): {S.sum().max():.3f}")
    
    return S


def decompose_stranding_rounds(
    S: pd.DataFrame,
    G: pd.DataFrame,
    fossil_sector: str,
    q: int = 3,
    max_rounds: int = 3
) -> Dict[int, Dict[str, float]]:
    """
    Decompose stranding effects into "rounds" (direct, 1-step, 2-step, etc.).
    
    Algorithm:
    - Round 1: Top q sectors directly affected by fossil sector
    - Round 2: Top q sectors affected by Round 1 sectors (excluding already seen)
    - Round 3+: Continue iteratively
    
    Args:
        S: Stranding multiplier matrix
        G: Ghosh inverse matrix (for network structure)
        fossil_sector: Name of the fossil fuel sector (e.g., 'MINfos')
        q: Number of top linkages to retain per layer
        max_rounds: Maximum number of rounds to compute
        
    Returns:
        Dictionary mapping round number to {sector: stranding_value}
    """
    rounds = {}
    seen_sectors = {fossil_sector}
    current_layer = [fossil_sector]
    
    # Get fossil column from S (total stranding from fossil)
    fossil_col = S[fossil_sector].copy()
    
    for round_num in range(1, max_rounds + 1):
        round_sectors = {}
        next_layer = []
        
        for origin_sector in current_layer:
            # Get stranding linkages from origin_sector
            linkages = S[origin_sector].copy()
            
            # Exclude self and already seen sectors
            linkages = linkages.drop(labels=list(seen_sectors), errors='ignore')
            
            # Get top q sectors
            top_q = linkages.nlargest(q)
            
            for sector, value in top_q.items():
                if sector not in seen_sectors:
                    # Accumulate stranding value for this round
                    round_sectors[sector] = round_sectors.get(sector, 0) + value
                    next_layer.append(sector)
                    seen_sectors.add(sector)
        
        if not round_sectors:
            logger.warning(f"Round {round_num}: No new sectors found, stopping.")
            break
            
        rounds[round_num] = round_sectors
        current_layer = list(set(next_layer))  # Unique sectors for next iteration
        
        logger.info(f"Round {round_num}: {len(round_sectors)} sectors, "
                   f"total stranding = {sum(round_sectors.values()):.3f}")
    
    return rounds


def aggregate_stranding_by_rounds(
    S: pd.DataFrame,
    fossil_sector: str,
    rounds: Dict[int, Dict[str, float]]
) -> pd.DataFrame:
    """
    Aggregate stranding values by sector across rounds.
    
    Returns a DataFrame with columns:
    - 'Round 1', 'Round 2', 'Round 3', 'Further Rounds', 'Total'
    
    Args:
        S: Stranding multiplier matrix
        fossil_sector: Name of fossil sector
        rounds: Round decomposition from decompose_stranding_rounds
        
    Returns:
        DataFrame with stranding values by round
    """
    # Collect all affected sectors
    all_sectors = set()
    for round_dict in rounds.values():
        all_sectors.update(round_dict.keys())
    
    result = pd.DataFrame(index=sorted(all_sectors))
    
    # Populate round columns
    for round_num, round_dict in rounds.items():
        col_name = f"Round {round_num}" if round_num <= 3 else "Further Rounds"
        
        if col_name not in result.columns:
            result[col_name] = 0.0
            
        for sector, value in round_dict.items():
            result.loc[sector, col_name] += value
    
    # Calculate total
    result['Total'] = result.sum(axis=1)
    
    # Fill NaN with 0
    result = result.fillna(0)
    
    # Sort by total descending
    result = result.sort_values('Total', ascending=False)
    
    return result


def calc_external_stranding_multiplier(S: pd.DataFrame) -> pd.Series:
    """
    Calculate external stranding multiplier for each sector.
    
    External stranding = sum of stranding effects on OTHER sectors (excluding self).
    
    Args:
        S: Stranding multiplier matrix
        
    Returns:
        Series of external stranding multipliers by sector
    """
    # For each column (originating sector), sum all rows except diagonal
    external = pd.Series(index=S.columns, dtype=float)
    
    for sector in S.columns:
        col = S[sector].copy()
        col = col.drop(sector, errors='ignore')  # Exclude self-stranding
        external[sector] = col.sum()
    
    return external


def calc_exposure_network(
    S: pd.DataFrame,
    focal_sectors: List[str],
    fossil_sectors: List[str],
    threshold: float = 0.01
) -> nx.DiGraph:
    """
    Create exposure network showing strongest stranding links.
    
    Args:
        S: Stranding multiplier matrix
        focal_sectors: List of sectors to focus on (e.g., most exposed sectors)
        fossil_sectors: List of fossil fuel sectors (origins)
        threshold: Minimum stranding value to include an edge
        
    Returns:
        Directed graph with nodes={sectors, countries} and weighted edges
    """
    G = nx.DiGraph()
    
    # Add nodes
    for sector in focal_sectors:
        G.add_node(sector, node_type='focal')
    for sector in fossil_sectors:
        G.add_node(sector, node_type='fossil')
    
    # Add edges based on threshold
    for fossil in fossil_sectors:
        if fossil not in S.columns:
            continue
            
        fossil_col = S[fossil]
        
        for focal in focal_sectors:
            if focal not in fossil_col.index:
                continue
                
            value = fossil_col[focal]
            if value >= threshold:
                G.add_edge(fossil, focal, weight=value)
    
    logger.info(f"Exposure network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def compute_country_to_country_effects_from_stranding(
    S: pd.DataFrame,
    *,
    fossil_cols: List[tuple],
) -> pd.DataFrame:
    """
    Compute country-to-country stranding effects from a full (region, sector) S matrix.

    Returns a DataFrame `origin_country x affected_country` where each cell is the
    total stranding effect in the affected country resulting from a unit shock in
    the origin country's fossil sector(s) (as provided in `fossil_cols`).

    This is the input needed to reproduce Figure 5/6-style dot plots.
    """
    if not isinstance(S.index, pd.MultiIndex) or not isinstance(S.columns, pd.MultiIndex):
        raise ValueError("S must have MultiIndex for both index and columns: (region, sector).")

    if not fossil_cols:
        raise ValueError("fossil_cols must be a non-empty list of S columns.")

    # Subset to fossil-origin columns, then aggregate rows to affected-country totals.
    sub = S.loc[:, fossil_cols]
    affected_country_by_fossil_col = sub.groupby(level=0).sum()

    # Aggregate fossil columns to origin-country totals (sum if multiple fossil columns per origin).
    affected_country_by_origin_country = affected_country_by_fossil_col.T.groupby(level=0).sum().T

    # Return origin x affected for convenience.
    origin_to_affected = affected_country_by_origin_country.T
    origin_to_affected.index.name = "origin_country"
    origin_to_affected.columns.name = "affected_country"
    return origin_to_affected
