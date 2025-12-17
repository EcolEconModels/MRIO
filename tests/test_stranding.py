"""
Unit tests for stranding analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from ecolecon_mrio.stranding import (
    calc_capital_intensity,
    calc_stranding_multipliers,
    decompose_stranding_rounds,
    aggregate_stranding_by_rounds,
    calc_external_stranding_multiplier
)


class TestStrandingCalculations(unittest.TestCase):
    """Test core stranding multiplier calculations."""
    
    def setUp(self):
        """Create simple test data."""
        # 3-sector economy: Fossil, Power, Manufacturing
        sectors = ['Fossil', 'Power', 'Mfg']
        
        # Mock Ghosh inverse (identity-like for simplicity)
        self.G = pd.DataFrame(
            [[1.0, 0.1, 0.05],
             [0.2, 1.0, 0.1],
             [0.1, 0.2, 1.0]],
            index=sectors,
            columns=sectors
        )
        
        # Mock capital intensity (k/x)
        self.kappa = pd.Series(
            [0.5, 2.0, 1.5],  # Power is capital-intensive
            index=sectors
        )
        
        # Mock output and capital stock
        self.output = pd.Series([100, 50, 75], index=sectors)
        self.capital = pd.Series([50, 100, 112.5], index=sectors)
    
    def test_capital_intensity_calculation(self):
        """Test κ = k/x calculation."""
        kappa = calc_capital_intensity(self.output, self.capital)
        
        self.assertAlmostEqual(kappa['Fossil'], 0.5)
        self.assertAlmostEqual(kappa['Power'], 2.0)
        self.assertAlmostEqual(kappa['Mfg'], 1.5)
    
    def test_stranding_multipliers_shape(self):
        """Test S = κ̂ G^T has correct shape."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        self.assertEqual(S.shape, (3, 3))
        self.assertTrue(np.all(S.index == self.G.index))
        self.assertTrue(np.all(S.columns == self.G.columns))
    
    def test_stranding_multipliers_diagonal(self):
        """Test diagonal elements S_ii = κ_i (self-stranding equals capital intensity)."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        # For Ghosh G, if G_ii = 1 and κ̂ is diagonal, then S_ii = κ_i * G_ii = κ_i
        self.assertAlmostEqual(S.loc['Fossil', 'Fossil'], 0.5, places=2)
        self.assertAlmostEqual(S.loc['Power', 'Power'], 2.0, places=2)
        self.assertAlmostEqual(S.loc['Mfg', 'Mfg'], 1.5, places=2)
    
    def test_stranding_multipliers_positive(self):
        """Test all stranding multipliers are non-negative."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        self.assertTrue(np.all(S.values >= 0))
    
    def test_round_decomposition_basic(self):
        """Test basic round decomposition logic."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        rounds = decompose_stranding_rounds(
            S, self.G, 
            fossil_sector='Fossil', 
            q=2,  # Top 2 per round
            max_rounds=2
        )
        
        # Should have at least Round 1
        self.assertIn(1, rounds)
        self.assertIsInstance(rounds[1], dict)
        
        # Round 1 should not include 'Fossil' itself
        self.assertNotIn('Fossil', rounds[1])
    
    def test_aggregate_stranding_by_rounds(self):
        """Test aggregation of stranding across rounds."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        rounds = decompose_stranding_rounds(
            S, self.G,
            fossil_sector='Fossil',
            q=2,
            max_rounds=2
        )
        
        result = aggregate_stranding_by_rounds(S, 'Fossil', rounds)
        
        # Should have columns for rounds and total
        self.assertIn('Total', result.columns)
        self.assertTrue(any('Round' in col for col in result.columns))
        
        # Total should be sum of rounds
        for sector in result.index:
            round_cols = [col for col in result.columns if col != 'Total']
            total_check = result.loc[sector, round_cols].sum()
            self.assertAlmostEqual(result.loc[sector, 'Total'], total_check, places=5)
    
    def test_external_stranding_multiplier(self):
        """Test external stranding calculation (excluding self)."""
        S = calc_stranding_multipliers(self.G, self.kappa)
        
        external = calc_external_stranding_multiplier(S)
        
        # External should be column sum minus diagonal
        for sector in S.columns:
            expected = S[sector].sum() - S.loc[sector, sector]
            self.assertAlmostEqual(external[sector], expected, places=5)


class TestStrandingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_zero_capital_intensity(self):
        """Test handling of zero capital intensity."""
        G = pd.DataFrame([[1.0]], index=['A'], columns=['A'])
        kappa = pd.Series([0.0], index=['A'])
        
        S = calc_stranding_multipliers(G, kappa)
        
        self.assertEqual(S.loc['A', 'A'], 0.0)
    
    def test_mismatched_indices(self):
        """Test handling of mismatched indices between G and κ."""
        G = pd.DataFrame(
            [[1.0, 0.1], [0.2, 1.0]],
            index=['A', 'B'],
            columns=['A', 'B']
        )
        kappa = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
        
        # Should handle gracefully by reindexing
        S = calc_stranding_multipliers(G, kappa)
        
        self.assertEqual(S.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
