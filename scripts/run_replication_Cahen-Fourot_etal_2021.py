"""
Replication script for Cahen-Fourot et al. (2021) "Capital stranding cascades".

This script:
1. Loads WIOD data
2. Calculates Ghosh inverse
3. Loads/generates capital stock data
4. Calculates stranding multipliers S = κ̂ G^T
5. Decomposes into cascade rounds
6. Generates figures (3, 4, 5, 9, A.3)
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from mrio_pyramids.data import load_mrio
from mrio_pyramids.capital_data import load_capital_stock_data
from mrio_pyramids.network_viz import plot_cascade_network
from mrio_pyramids.stranding import (
    calc_capital_intensity,
    calc_stranding_multipliers,
    decompose_stranding_rounds,
    aggregate_stranding_by_rounds,
    calc_external_stranding_multiplier
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Replicate Cahen-Fourot et al. (2021)")
    parser.add_argument("--dataset", type=str, default="wiod", help="MRIO dataset (wiod, oecd)")
    parser.add_argument("--year", type=int, default=2014, help="Year for analysis")
    parser.add_argument("--capital-source", type=str, default="mock", help="Capital stock source (wiod, mock)")
    parser.add_argument("--fossil-sector", type=str, default="B", help="Fossil fuel sector code")
    parser.add_argument("--q", type=int, default=3, help="Top q sectors per round for cascade")
    parser.add_argument("--max-rounds", type=int, default=4, help="Maximum cascade rounds")
    args = parser.parse_args()
    
    # 1. Load MRIO data
    logger.info(f"Loading {args.dataset} data for {args.year}...")
    mrio, meta = load_mrio(dataset=args.dataset, year=args.year)
    
    # 2. Calculate Ghosh inverse
    logger.info("Calculating Ghosh inverse...")
    if not hasattr(mrio, 'G') or mrio.G is None:
        mrio.calc_all(include_ghosh=True)
    
    G = mrio.G
    logger.info(f"Ghosh matrix shape: {G.shape}")
    
    # 3. Load capital stock data
    logger.info(f"Loading capital stock data (source={args.capital_source})...")
    capital_stock = load_capital_stock_data(mrio, args.year, source=args.capital_source)
    
    # 4. Calculate capital intensity κ = k/x
    logger.info("Calculating capital intensity...")
    output = mrio.x.squeeze()
    kappa = calc_capital_intensity(output, capital_stock)
    
    logger.info(f"Capital intensity stats:")
    logger.info(f"  Mean: {kappa.mean():.3f}")
    logger.info(f"  Median: {kappa.median():.3f}")
    logger.info(f"  Range: [{kappa.min():.3f}, {kappa.max():.3f}]")
    
    # 5. Calculate stranding multipliers S = κ̂ G^T
    logger.info("Calculating stranding multipliers...")
    S = calc_stranding_multipliers(G, kappa)
    
    # 6. Identify fossil sector columns
    # S has MultiIndex (Region, Sector) for both rows and columns
    if isinstance(S.columns, pd.MultiIndex):
        # Extract all (region, sector) tuples where sector matches fossil pattern
        fossil_cols = [col for col in S.columns 
                      if args.fossil_sector in str(col[-1])]  # Last level is sector
    else:
        fossil_cols = [col for col in S.columns 
                      if args.fossil_sector in str(col)]
    
    if not fossil_cols:
        logger.error(f"Fossil sector '{args.fossil_sector}' not found in data.")
        logger.info(f"Sample column structure: {S.columns[:5].tolist()}")
        if isinstance(S.columns, pd.MultiIndex):
            logger.info(f"Sample sectors (last level): {S.columns.get_level_values(-1).unique()[:10].tolist()}")
        return
    
    logger.info(f"Found {len(fossil_cols)} fossil sector columns: {fossil_cols[:3]}")
    
    # Aggregate fossil sectors across all regions (sum columns)
    fossil_col_aggregate = S[fossil_cols].sum(axis=1)
    
    logger.info(f"Fossil stranding column stats:")
    logger.info(f"  Total: {fossil_col_aggregate.sum():.3f}")
    logger.info(f"  Max sector: {fossil_col_aggregate.idxmax()} = {fossil_col_aggregate.max():.3f}")
    
    # 7. For round decomposition, flatten to sector-level (aggregate regions)
    logger.info("Flattening multi-region data to sector level...")
    
    if isinstance(S.index, pd.MultiIndex):
        # Aggregate: sum over regions for each sector
        S_sector = S.groupby(level=-1).sum().T.groupby(level=-1).sum().T
        G_sector = G.groupby(level=-1).sum().T.groupby(level=-1).sum().T
    else:
        S_sector = S
        G_sector = G
    
    logger.info(f"Sector-level S matrix shape: {S_sector.shape}")
    
    # Use aggregated fossil sector name
    fossil_sector_name = args.fossil_sector
    
    if fossil_sector_name not in S_sector.index:
        logger.error(f"Fossil sector '{fossil_sector_name}' not in aggregated data.")
        logger.info(f"Available sectors: {S_sector.index.tolist()[:10]}")
        return
    
    # 8. Calculate external stranding multiplier
    external_sm = calc_external_stranding_multiplier(S_sector)
    logger.info(f"External stranding multiplier for {fossil_sector_name}: {external_sm[fossil_sector_name]:.3f}")
    
    # 9. Decompose into cascade rounds
    logger.info(f"Decomposing stranding into rounds (q={args.q})...")
    
    rounds = decompose_stranding_rounds(
        S_sector, G_sector,
        fossil_sector=fossil_sector_name,
        q=args.q,
        max_rounds=args.max_rounds
    )
    
    # 10. Aggregate by rounds
    logger.info("Aggregating stranding by rounds...")
    stranding_by_rounds = aggregate_stranding_by_rounds(S_sector, fossil_sector_name, rounds)
    
    # Display top 10 most affected sectors
    logger.info("\nTop 10 sectors by total stranding multiplier:")
    logger.info(stranding_by_rounds.head(10).to_string())
    
    # 10. Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 11. Generate cascade network visualization (Figure 3 style)
    logger.info("Generating cascade network visualization...")
    cascade_fig = plot_cascade_network(
        rounds,
        fossil_sector=fossil_sector_name,
        title=f"Stranding Cascade from {fossil_sector_name} (q={args.q})",
        save_path=output_dir / f"cascade_network_{args.year}.png"
    )
    plt.close(cascade_fig)
    
    # 12. Save results
    
    # Save stranding multipliers
    S.to_csv(output_dir / f"stranding_multipliers_{args.year}.csv")
    logger.info(f"Saved stranding multipliers to {output_dir / f'stranding_multipliers_{args.year}.csv'}")
    
    # Save round aggregation
    stranding_by_rounds.to_csv(output_dir / f"stranding_by_rounds_{args.year}.csv")
    logger.info(f"Saved round aggregation to {output_dir / f'stranding_by_rounds_{args.year}.csv'}")
    
    # 11. Generate basic visualization (placeholder for Figure 4)
    logger.info("Generating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot top 10 sectors as stacked bar chart
    top_10 = stranding_by_rounds.head(10)
    round_cols = [col for col in top_10.columns if 'Round' in col]
    
    top_10[round_cols].plot(kind='barh', stacked=True, ax=ax, 
                             colormap='viridis')
    ax.set_xlabel('Stranding Multiplier')
    ax.set_ylabel('Sector')
    ax.set_title(f'Top 10 Global Sectors by Fossil Stranding Exposure (q={args.q})')
    ax.legend(title='Round', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_file = output_dir / f"stranding_top10_{args.year}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    
    logger.info("\nReplication complete!")


if __name__ == "__main__":
    main()
