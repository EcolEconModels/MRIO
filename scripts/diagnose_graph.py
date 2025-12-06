
import sys
from pathlib import Path
import networkx as nx
import pandas as pd
import logging
import numpy as np

# Add src to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from mrio_pyramids.data import load_mrio
from mrio_pyramids.graph import build_dependency_graph, build_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose(dataset="oecd", year=2010, country="DEU", sector="05T06"):
    logger.info(f"Diagnosing graph for {dataset} {year} {country} {sector}")
    mrio, _ = load_mrio(dataset, year)
    
    # Construct National Total System
    df_Z = mrio.Z
    country_cols = [c for c in df_Z.columns if c[0] == country]
    Z_national_input = df_Z[country_cols].groupby(level=1).sum()
    Z_national_input.columns = Z_national_input.columns.get_level_values(1)
    
    # Calculate Total Supply (x_total) = Domestic Output + Imports
    # mrio.x is Domestic Output.
    # Imports into DEU from world:
    # Z has columns for DEU. Rows are all origins.
    # Imports = Sum of Z rows (Rest of World) -> DEU.
    # Actually, we need Imports of PRODUCT i.
    # Import of Mining = Sum of Mining rows (from RoW) into DEU columns?
    # No. Total Supply of Mining available in DEU = Domestic Output of Mining + Imports of Mining.
    # Imports of Mining = Sum of flows FROM Foreign Mining TO DEU (Intermediate + Final).
    # pymrio: mrio.Z is [Region x Sector] x [Region x Sector].
    # Imports vector: mrio.calc_all() creates mrio.imports (aggregated?).
    # Let's try raw calculation to be sure.
    
    country_input_cols = [c for c in df_Z.columns if c[0] == country]
    # Z_national_input (Total Use) was calculated by summing origins.
    # This matrix is [Sector x Sector].
    # Row i = Total amount of Input i (Dom+Imp) used by DEU sectors.
    # But this doesn't include Final Demand use.
    # Total Supply i = Total Intermediate Use + Total Final Demand Use?
    # Or Output + Imports?
    # Let's usage Output + Imports.
    
    # Imports of product i: Sum of row i (from foreign regions) across all DEU columns (Z + Y).
    # But we aggreagted Z to National (summing origins).
    # Z_national_input row sum = Total Intermediate Use of i (Dom+Imp).
    
    # Let's take specific approach:
    # mrio.imports (if flexible)
    # Or:
    imp = mrio.Z.loc[mrio.Z.index.get_level_values(0) != country, country_input_cols].groupby(level=1).sum().sum(axis=1) # Intermediate imports
    # Plus Final Demand imports?
    y_country_cols = [c for c in mrio.Y.columns if c[0] == country]
    imp_y = mrio.Y.loc[mrio.Y.index.get_level_values(0) != country, y_country_cols].groupby(level=1).sum().sum(axis=1)
    total_imports = imp.add(imp_y, fill_value=0)
    
    x_domestic = mrio.x.loc[country_cols]
    x_domestic.index = x_domestic.index.get_level_values(1)
    
    x_total_supply = x_domestic.iloc[:,0].add(total_imports, fill_value=0)
    
    # Recalculate coefficients B for National System
    # B = Z_national / x_total_supply (row-wise division)
    # national_sys will calculate B=Z/x. So passing x=x_total_supply is correct.
    
    # Need to align indices
    x_total_supply = x_total_supply.reindex(Z_national_input.index).fillna(1) # avoid div/0?
    
    import pymrio
    # Pass x as DataFrame
    x_df = pd.DataFrame(x_total_supply, columns=['output'])
    
    national_sys = pymrio.IOSystem(Z=Z_national_input, x=x_df, meta=getattr(mrio, 'meta', None))
    national_sys.calc_all(include_ghosh=True)
    G = national_sys.G
    
    # Diagnostic: Check Root Node Row in G
    if sector in G.index:
         row = G.loc[sector]
         print(f"Root {sector} linkages summary:")
         print(f"  Mean: {row.mean():.4f}, Max: {row.max():.4f}, Sum (FL): {row.sum():.4f}")
         print(f"  Top 5 targets: {row.sort_values(ascending=False).head(5).to_dict()}")
         
         # Global stats (Off-Diagonal)
         # Create mask for off-diagonal
         G_np = G.values
         np.fill_diagonal(G_np, 0) # Zero out diagonal for quantile calc if we assume only positive linkages matter
         # Or better, exclude them.
         # Re-read: "removing self-loops".
         flat_G = G_np.flatten()
         flat_G_off = flat_G[flat_G > 1e-6] # Only consider non-zero off-diagonals
         
         if len(flat_G_off) > 0:
             p99 = pd.Series(flat_G_off).quantile(0.99)
             p90 = pd.Series(flat_G_off).quantile(0.90)
             print(f"  Global 99th percentile (Off-Diag > 0) value: {p99:.4f}")
             print(f"  Global 90th percentile (Off-Diag > 0) value: {p90:.4f}")
             
             row_vals = row.values.copy()
             # Set diagonal to 0
             # We need to know which index is sector.
             # Ideally usage named index.
             # row is a Series.
             if sector in row:
                 row_drop_self = row.drop(sector)
             else:
                 row_drop_self = row
                 
             print(f"  Root outgoing edges > {p99:.4f} (q=1): {(row_drop_self > p99).sum()}")
             print(f"  Root outgoing edges > {p90:.4f} (q=10): {(row_drop_self > p90).sum()}")

    # Test different global q values with off-diagonal approximation
    # build_dependency_graph doesn't remove diagonal yet.
    # We will assume graph builder will be fixed.
    # For diagnostic, let's just interpret q.
    
    # We will simulate the fix by filtering edges manually here?
    # No, let's just rely on the printout above to guide the decision.
    # But if we want to see the Layers, we need to pass the correct threshold to graph builder.
    
    # Let's try passing the calculated threshold explicitly to verifying
    # Modify loop to usage specific thresholds.
    
    thresholds = {'q=1%': p99, 'q=10%': p90}
    
    for label, thresh in thresholds.items():
        graph = build_dependency_graph(G, normalize_by_column=False, threshold=thresh)
        # Remove self loops manually
        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        if sector not in graph:
             logger.warning(f"{label}: Root {sector} NOT in graph")
             continue
             
        layers = build_layers(graph, sector)
        max_depth = max(layers.values()) if layers else 0
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        descendants = nx.descendants(graph, sector)
        
        print(f"--- {label} (Threshold {thresh:.4f}) ---")
        print(f"Nodes: {node_count}, Edges: {edge_count}")
        print(f"Root Outgoing Degree: {graph.out_degree(sector) if sector in graph else 0}")
        print(f"Max Depth: {max_depth}")
        print(f"Reachable: {len(descendants)}")
        print(f"Layers: {pd.Series(list(layers.values())).value_counts().sort_index().to_dict()}")
        print("")

if __name__ == "__main__":
    diagnose()
