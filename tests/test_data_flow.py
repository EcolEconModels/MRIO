import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import sys
import os

from mrio_pyramids.data import load_mrio

class TestDataFlow(unittest.TestCase):
    
    @patch("mrio_pyramids.data.pymrio")
    def test_load_test_data(self, mock_pymrio):
        """Test that test dataset is loaded without download checks."""
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'c1': [1,2], 'c2': [3,4]})
        # Ensure G access return None initially to trigger calc_all
        mock_system.G = None
        mock_pymrio.load_test.return_value = mock_system
        
        mrio, y = load_mrio("test", 2018)
        
        mock_pymrio.load_test.assert_called_once()
        mock_pymrio.download_oecd.assert_not_called()
        self.assertTrue(y is not None)
        # Verify calc_all triggered
        mock_system.calc_all.assert_called_with(include_ghosh=True)

    @patch("mrio_pyramids.data.pymrio")
    @patch("mrio_pyramids.data.Path")
    def test_wiod_download_check(self, mock_path, mock_pymrio):
        """Test WIOD download behavior."""
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        
        # Case 1: metadata.json missing -> trigger download
        # (data_dir / "metadata.json").exists() -> False
        mock_path.return_value.__truediv__.return_value.exists.side_effect = [False, False, False] # 1st: metadata check, 2nd: pkl check
        
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'EXP': [1], 'row': [2]})
        mock_system.G = "Exists" # No recalc needed
        mock_pymrio.parse_wiod.return_value = mock_system
        
        load_mrio("wiod", 2010)
        
        mock_pymrio.download_wiod2013.assert_called()
        mock_pymrio.parse_wiod.assert_called()

    @patch("mrio_pyramids.data.pymrio")
    @patch("mrio_pyramids.data.Path")
    def test_wiod_cached(self, mock_path, mock_pymrio):
        """Test WIOD uses cache if pickle exists."""
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        
        # (data_dir / "metadata.json").exists() -> True (skip download)
        # (data_dir / "mrio_system_2010.pkl").exists() -> True (load pickle)
        # Note: logic: if not metadata: download... if pickle: load else parse.
        # So we want [True, True] for exists calls
        
        # Mocking Path composition logic is tricky.
        # simpler: 
        # Path(..).exists()
        
        # Check logic:
        # data_dir = Path(...)
        # if not (data_dir / "metadata.json").exists(): download
        # if (data_dir / ...pkl).exists(): load
        
        file_mock = MagicMock()
        file_mock.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = file_mock
        
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'x':[1]})
        mock_system.G = "Exists"
        mock_pymrio.load.return_value = mock_system
        
        load_mrio("wiod", 2010)
        
        mock_pymrio.download_wiod2013.assert_not_called()
        mock_pymrio.load.assert_called() # Should load from pickle
        mock_pymrio.parse_wiod.assert_not_called()

if __name__ == "__main__":
    unittest.main()
