import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from mrio_pyramids.plotting import (
    plot_foundations,
    plot_inverted_pyramid,
    plot_dependency_graph,
    plot_layered_pyramid
)

class TestPlotting(unittest.TestCase):
    
    def setUp(self):
        self.series = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        self.graph = MagicMock()
        self.graph.nodes.return_value = ["A", "B"]
        self.graph.edges.return_value = [("A", "B")]
        
    @patch("mrio_pyramids.plotting.plt")
    @patch("mrio_pyramids.plotting.sns")
    def test_plot_foundations(self, mock_sns, mock_plt):
        """Test bar plot generation."""
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        
        ax = plot_foundations(self.series)
        
        mock_sns.barplot.assert_called_once()
        self.assertEqual(ax, mock_ax)

    @patch("mrio_pyramids.plotting.plt")
    @patch("mrio_pyramids.plotting.sns")
    def test_plot_inverted_pyramid(self, mock_sns, mock_plt):
        """Test inverted pyramid generation."""
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        
        ax = plot_inverted_pyramid(self.series)
        
        # Should create barh
        mock_ax.barh.assert_called()
        self.assertEqual(ax, mock_ax)
        
    @patch("mrio_pyramids.plotting.nx")
    @patch("mrio_pyramids.plotting.plt")
    def test_plot_dependency_graph(self, mock_plt, mock_nx):
        """Test network graph plotting."""
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        # Mock layout to avoid nx computation
        mock_nx.spring_layout.return_value = {"A": (0,0), "B": (1,1)}
        
        # Real nx graph but small
        import networkx as nx
        real_graph = nx.DiGraph()
        real_graph.add_edge("A", "B", weight=0.5)
        
        ax = plot_dependency_graph(real_graph)
        
        mock_nx.draw_networkx_nodes.assert_called()
        mock_nx.draw_networkx_edges.assert_called()
        self.assertEqual(ax, mock_ax)

if __name__ == "__main__":
    unittest.main()
