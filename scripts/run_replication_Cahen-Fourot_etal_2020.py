"""
Replication script for Cahen-Fourot et al. (2020) "The role of foundational sectors in the economy".
Focus: Inverted Pyramid visualization and Forward Linkages.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pymrio

from ecolecon_mrio.data import load_mrio
from ecolecon_mrio.graph import build_dependency_graph
from ecolecon_mrio.plotting import plot_inverted_pyramid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Replicate Cahen-Fourot et al. (2020)")
    parser.add_argument("--dataset", type=str, default="oecd", help="MRIO dataset to use (test, oecd, wiod)")
    parser.add_argument("--year", type=int, default=2010, help="Year for the dataset")
    parser.add_argument("--country", type=str, default="DEU", help="Country code to plot (e.g., DEU, FRA)")
    parser.add_argument("--sector", type=str, default="B", help="Root sector for pyramid (default: B Mining)")
    parser.add_argument("--q", type=float, default=10.0, help="Percentile threshold q (e.g., 10.0 for top 10%%)")
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading MRIO data: {args.dataset} ({args.year})")
    mrio, _ = load_mrio(dataset=args.dataset, year=args.year)
    
    # 2. Construct National "Total" System (Domestic + Imports)
    logger.info(f"Constructing National Total System for {args.country}...")
    
    # Extract Z block for the country (Columns = country inputs, Rows = All origins)
    df_Z = mrio.Z
    country_cols = [c for c in df_Z.columns if c[0] == args.country] # MultiIndex (Region, Sector)
    
    if not country_cols:
        logger.error(f"No columns found for country {args.country}")
        return

    # Z_national Input: Inputs *into* the country
    # We sum over all origin regions to get "Total Input of Product i (from world) to Sector j (in country)"
    Z_national_input = df_Z[country_cols].groupby(level=1).sum() 
    Z_national_input.columns = Z_national_input.columns.get_level_values(1)
    
    # Calculate Imports intermediate + final to get Total Supply x
    country_input_cols = [c for c in df_Z.columns if c[0] == args.country] # same as country_cols
    
    # Intermediate Imports
    # This sums rows NOT from country, into country cols.
    imp = mrio.Z.loc[mrio.Z.index.get_level_values(0) != args.country, country_input_cols].groupby(level=1).sum().sum(axis=1)
    
    # Final Demand Imports
    y_country_cols = [c for c in mrio.Y.columns if c[0] == args.country]
    imp_y = mrio.Y.loc[mrio.Y.index.get_level_values(0) != args.country, y_country_cols].groupby(level=1).sum().sum(axis=1)
    
    total_imports = imp.add(imp_y, fill_value=0)
    
    # Domestic Output
    x_domestic = mrio.x.loc[country_cols]
    x_domestic.index = x_domestic.index.get_level_values(1)
    
    # Total Supply
    x_total_supply = x_domestic.iloc[:,0].add(total_imports, fill_value=0)
    x_total_supply = x_total_supply.reindex(Z_national_input.index).fillna(1)
    
    # Create x DataFrame
    x_df = pd.DataFrame(x_total_supply, columns=['output'])
    
    national_sys = pymrio.IOSystem(
        Z=Z_national_input, 
        x=x_df, 
        meta=getattr(mrio, 'meta', None)
    )
    
    # 3. Calculate Ghosh Inverse using pymrio native method
    logger.info("Computing Ghosh Inverse using pymrio native method...")
    national_sys.calc_all(include_ghosh=True) 
    
    G_national = national_sys.G
    
    # 4. Filter Graph
    # Use Global Quantile logic (Top Q% of off-diagonal edges)
    # q=10 means top 10%. quantile should be 0.90.
    quantile_threshold = 1.0 - (args.q / 100.0)
    logger.info(f"Building Graph with Global Quantile: {quantile_threshold:.3f} (Top {args.q}%)")
    
    G_graph = build_dependency_graph(
        G_national, 
        normalize_by_column=False, 
        global_quantile=quantile_threshold
    )
    
    # 5. Plot
    root_node = args.sector 
    if root_node not in G_graph:
        logger.warning(f"Root node {root_node} not in graph! Available: {list(G_graph.nodes)[:5]}...")
        # Check standard sectors
        return

    logger.info(f"Plotting Inverted Pyramid for {root_node}...")
    fig_ax = plot_inverted_pyramid(
        G_graph, 
        base_node=root_node,
        title=f"Inverted Pyramid: {args.country} (Root: {root_node}, top {args.q}%)"
    )
    
    # Save
    output_file = f"inverted_pyramid_{args.country}_{root_node}.png"
    plt.savefig(output_file)
    logger.info(f"Saved plot to {output_file}")
    
    if Path(output_file).exists():
        print(f"SUCCESS: Created {output_file}")
    else:
        print("ERROR: Failed to create plot file")

if __name__ == "__main__":
    main()
