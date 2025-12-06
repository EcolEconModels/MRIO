import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import sys
import os

# Retrieve the load_mrio function from the package
from mrio_pyramids.data import load_mrio

class TestDataFlow(unittest.TestCase):
    
    @patch("mrio_pyramids.data.pymrio")
    @patch("mrio_pyramids.data.Path")
    def test_load_test_data(self, mock_path, mock_pymrio):
        """Test that test dataset is loaded without download checks."""
        # Setup mock return for load_test
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'c1': [1,2], 'c2': [3,4]})
        mock_pymrio.load_test.return_value = mock_system
        
        mrio, y = load_mrio("test", 2018)
        
        mock_pymrio.load_test.assert_called_once()
        mock_pymrio.download_oecd.assert_not_called()
        self.assertTrue(y is not None)

    @patch("mrio_pyramids.data.pymrio")
    @patch("mrio_pyramids.data.Path")
    def test_oecd_download_if_missing(self, mock_path, mock_pymrio):
        """Test OECD download triggered if file missing."""
        # Mock path behavior
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        # glob return empty list -> file missing
        mock_dir.glob.return_value = []
        
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'EXP': [1], 'row': [2]})
        mock_pymrio.parse_oecd.return_value = mock_system
        
        load_mrio("oecd", 2018)
        
        # Verify download called
        mock_pymrio.download_oecd.assert_called_once()
        mock_pymrio.parse_oecd.assert_called_once()

    @patch("mrio_pyramids.data.pymrio")
    @patch("mrio_pyramids.data.Path")
    def test_oecd_cached_skip_download(self, mock_path, mock_pymrio):
        """Test OECD download skipped if file exists."""
        # Mock path behavior
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        # glob returns match -> file exists
        mock_dir.glob.return_value = [Path("dummy/file_parameters.json")]
        
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'EXP': [1], 'row': [2]})
        mock_pymrio.parse_oecd.return_value = mock_system
        
        load_mrio("oecd", 2018)
        
        # Verify download NOT called
        mock_pymrio.download_oecd.assert_not_called()
        mock_pymrio.parse_oecd.assert_called_once()

if __name__ == "__main__":
    unittest.main()
