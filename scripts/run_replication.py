"""
Scriptable replication runner for Cahen-Fourot-style inverted pyramid analysis.

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/run_replication.py --dataset oecd --year 2018 --region DEU --base-sector B
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import networkx as nx

import mrio_pyramids as mp
from mrio_pyramids.mapping import sector_label

matplotlib.use("Agg")  # non-interactive backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["test", "oecd", "exiobase", "wiod"], default="oecd")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--top", type=int, default=15, help="Top sectors to display")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Edge weight threshold for graph pruning (after normalisation)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        dest="top_k",
        help="Top edges per downstream sector to retain",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.9,
        help="Quantile (0-1) to retain edges per origin (paper keeps top q percentile)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region/country code to subset (for multi-region datasets like OECD)",
    )
    parser.add_argument(
        "--base-sector",
        type=str,
        default=None,
        help="Sector code to use as pyramid base (e.g., mining/quarrying code)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/replication"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "test":
        mrio = mp.load_test_mrio()
    elif args.dataset == "oecd":
        mrio = mp.load_oecd_icio(storage_folder=Path(f"data/oecd_icio_{args.year}"), years=[args.year])
    elif args.dataset == "wiod":
        mrio = mp.load_wiod2013(storage_folder=Path(f"data/wiod13_{args.year}"), years=[args.year])
    else:
        mrio = mp.load_exiobase3(storage_folder=Path(f"data/exiobase3_{args.year}"), years=[args.year])

    y_full = mp.data.final_demand_vector(
        mrio, demand_categories=[c for c in mrio.Y.columns if "EXP" not in c]
    )
    L_full = mp.compute_leontief(mrio.Z, mrio.x)
    G_full = mp.compute_ghosh(mrio.Z, mrio.x)

    # Optional region subset (for multi-region datasets)
    if args.region is not None and isinstance(G_full.index, pd.MultiIndex):
        idx = G_full.index.get_level_values(0) == args.region
        G = pd.DataFrame(G_full.values[idx][:, idx], index=G_full.index[idx], columns=G_full.columns[idx])
        y = y_full.loc[G.index]
        L = pd.DataFrame(L_full.values[idx][:, idx], index=L_full.index[idx], columns=L_full.columns[idx])
    else:
        G, y, L = G_full, y_full, L_full

    inputs = mp.PyramidInputs(leontief=L, final_demand=y)

    monetary_base = mp.embodied_pressure(inputs)
    print("\nTop upstream contributors (Leontief):")
    print(mp.rank_foundational_sectors(monetary_base, top_n=args.top))

    ghosh_response = (G @ y.values)
    ghosh_base = pd.Series(ghosh_response, index=G.index, name="ghosh_base")
    print("\nTop upstream contributors (Ghosh supply-driven):")
    print(mp.rank_foundational_sectors(ghosh_base, top_n=args.top))

    dep_graph = mp.build_dependency_graph(
        pd.DataFrame(G, index=G.index, columns=G.columns),
        top_k_per_downstream=args.top_k,
        threshold=args.threshold,
        quantile_per_origin=args.quantile,
        normalize_by_column=True,
    )
    # remove self-loops
    dep_graph.remove_edges_from(nx.selfloop_edges(dep_graph))
    undirected = mp.symmetrize(dep_graph, agg="max")
    communities = mp.detect_communities(undirected)
    print(f"\nCommunities detected: {len(set(communities.values()))}")

    if not args.no_plots:
        import matplotlib.pyplot as plt  # lazy import

        # Bar plots
        ax1 = mp.plot_foundations(
            monetary_base, top_n=args.top, title="Leontief foundation (shares)"
        )
        leopath = args.output_dir / "foundations_leontief.png"
        plt.savefig(leopath, dpi=200)
        plt.close(ax1.figure)

        ax2 = mp.plot_foundations(
            ghosh_base, top_n=args.top, title="Ghosh foundation (supply-driven)"
        )
        ghoshpath = args.output_dir / "foundations_ghosh.png"
        plt.savefig(ghoshpath, dpi=200)
        plt.close(ax2.figure)

        # Inverted pyramid style (stacked largest-at-bottom)
        ax3 = mp.plot_inverted_pyramid(
            monetary_base, top_n=args.top, title="Inverted pyramid (Leontief shares)"
        )
        invp = args.output_dir / "inverted_pyramid_leontief.png"
        plt.savefig(invp, dpi=200)
        plt.close(ax3.figure)

        ax4 = mp.plot_inverted_pyramid(
            ghosh_base, top_n=args.top, title="Inverted pyramid (Ghosh supply-driven)"
        )
        invg = args.output_dir / "inverted_pyramid_ghosh.png"
        plt.savefig(invg, dpi=200)
        plt.close(ax4.figure)

        # Layered pyramid from Ghosh graph (approximate Fig. 3 styling)
        base_node = None
        if args.base_sector is not None:
            # Try to find node matching base sector (consider MultiIndex)
            for n in dep_graph.nodes():
                if isinstance(n, tuple) and len(n) > 1:
                    if n[-1] == args.base_sector and (args.region is None or n[0] == args.region):
                        base_node = n
                        break
                elif n == args.base_sector:
                    base_node = n
                    break
        if base_node is None:
            base_node = next(iter(dep_graph.nodes()))
        labels = {
            n: sector_label(n[1], include_code=True, region=n[0]) if isinstance(n, tuple) else sector_label(n)
            for n in dep_graph.nodes()
        }
        ax5 = mp.plot_layered_pyramid(dep_graph, base_node=base_node, labels=labels, title="Layered pyramid (supply-driven)")
        pyramidpath = args.output_dir / "layered_pyramid.png"
        plt.savefig(pyramidpath, dpi=300)
        plt.close(ax5.figure)

        # Network plot
        ax6 = mp.plot_dependency_graph(dep_graph, community_map=communities)
        graphpath = args.output_dir / "dependency_graph.png"
        plt.savefig(graphpath, dpi=300)
        plt.close(ax6.figure)

        print(f"Saved plots to {args.output_dir}")

    # Hierarchical clustering across countries (only for multi-region datasets)
    if isinstance(G_full.index, pd.MultiIndex):
        regions = sorted(set(G_full.index.get_level_values(0)))
        sectors = list(G_full.index.get_level_values(1).unique())
        data = {}
        for reg in regions:
            idx = G_full.index.get_level_values(0) == reg
            G_reg = pd.DataFrame(G_full.values[idx][:, idx], index=G_full.index[idx], columns=G_full.columns[idx])
            y_reg = y_full.loc[G_reg.index]
            resp = (G_reg.values @ y_reg.values)
            data[reg] = pd.Series(resp, index=G_reg.index.get_level_values(1))
        matrix = pd.DataFrame(data)
        clusters = mp.sector_clustering(matrix)
        clusters.to_csv(args.output_dir / "sector_clusters.csv")
        print("Saved sector clustering (PCA + hierarchical) to sector_clusters.csv")


if __name__ == "__main__":
    main()
