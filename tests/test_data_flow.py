import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import pandas as pd

from ecolecon_mrio.data import load_mrio

class TestDataFlow(unittest.TestCase):

    @patch("ecolecon_mrio.data.pymrio")
    def test_load_test_data(self, mock_pymrio):
        """Test that test dataset is loaded without download checks."""
        mock_system = MagicMock()
        mock_system.Y = pd.DataFrame({'c1': [1,2], 'c2': [3,4]})
        # Ensure G access return None initially to trigger calc_all
        mock_system.G = None
        mock_pymrio.load_test.return_value = mock_system

        mrio = load_mrio("test", 2018)

        mock_pymrio.load_test.assert_called_once()
        mock_pymrio.download_oecd.assert_not_called()
        self.assertIs(mrio, mock_system)
        self.assertIsNotNone(mrio.Y)
        # Verify calc_all triggered
        mock_system.calc_all.assert_called_with(include_ghosh=True)

    @patch("ecolecon_mrio.data.pymrio")
    def test_wiod_download_check(self, mock_pymrio):
        """Test WIOD download behavior."""
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            # Case 1: metadata.json missing -> trigger download
            # Ensure no files exist.

            mock_system = MagicMock()
            mock_system.Y = pd.DataFrame({'EXP': [1], 'row': [2]})
            mock_system.G = "Exists" # No recalc needed
            mock_pymrio.parse_wiod.return_value = mock_system

            def _fake_download(storage_folder, years=None, **kwargs):
                # Create the expected WIOT file so the loader's preflight passes.
                yy = str(list(years)[0])[-2:]
                p = Path(storage_folder) / f"wiot{yy}_test.xlsx"
                p.write_bytes(b"stub")
                (Path(storage_folder) / "metadata.json").write_text("{}", encoding="utf-8")

            mock_pymrio.download_wiod2013.side_effect = _fake_download

            load_mrio("wiod", 2010, data_root=data_root)

            mock_pymrio.download_wiod2013.assert_called()
            mock_pymrio.parse_wiod.assert_called()

    @patch("ecolecon_mrio.data.pymrio")
    def test_wiod_cached(self, mock_pymrio):
        """Test WIOD uses cache if pickle exists."""
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            data_dir = data_root / "wiod13_2010"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (data_dir / "mrio_system_2010.pkl").write_text("stub", encoding="utf-8")

            mock_system = MagicMock()
            mock_system.Y = pd.DataFrame({'x':[1]})
            mock_system.G = "Exists"
            mock_pymrio.load.return_value = mock_system

            load_mrio("wiod", 2010, data_root=data_root)

            mock_pymrio.download_wiod2013.assert_not_called()
            mock_pymrio.load.assert_called() # Should load from pickle
            mock_pymrio.parse_wiod.assert_not_called()

    @patch("ecolecon_mrio.data.pymrio")
    def test_wiod_multiple_candidates_prefers_exact(self, mock_pymrio):
        """If multiple WIOT files match, prefer the exact wiotYY.xlsx when present."""
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            data_dir = data_root / "wiod13_2010"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Multiple candidates + an exact match.
            (data_dir / "wiot10.xlsx").write_bytes(b"stub")
            (data_dir / "wiot10_copy.xlsx").write_bytes(b"stub")

            mock_system = MagicMock()
            mock_system.Y = pd.DataFrame({"x": [1]})
            mock_system.G = "Exists"
            mock_pymrio.parse_wiod.return_value = mock_system

            load_mrio("wiod", 2010, data_root=data_root)

            mock_pymrio.download_wiod2013.assert_not_called()
            # parse_wiod should be called with the exact file path.
            called_path = Path(mock_pymrio.parse_wiod.call_args.kwargs["path"])
            self.assertEqual(called_path.name, "wiot10.xlsx")

if __name__ == "__main__":
    unittest.main()
