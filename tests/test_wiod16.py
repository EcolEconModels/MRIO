import unittest

import pandas as pd

from ecolecon_mrio.wiod16 import parse_wiod16_wiot_frame


class TestWIOD16Parser(unittest.TestCase):
    def test_parse_minimal_wiot_frame(self):
        # Minimal synthetic WIOD16-like table:
        # - 2 countries (AAA, BBB)
        # - 2 industries per country (n_industries inferred from max RNr)
        # - 2 final-demand columns per country (numbers > n_industries)
        cols = [
            "IndustryCode",
            "IndustryDescription",
            "Country",
            "RNr",
            "Year",
            "vAAA1",
            "vAAA2",
            "vAAA3",
            "vAAA4",
            "vBBB1",
            "vBBB2",
            "vBBB3",
            "vBBB4",
            "TOT",
        ]
        rows = [
            ["S1", "Sector 1", "AAA", 1, 2014, 1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 0.0],
            ["S2", "Sector 2", "AAA", 2, 2014, 5.0, 6.0, 50.0, 60.0, 7.0, 8.0, 70.0, 80.0, 0.0],
            ["S1", "Sector 1", "BBB", 1, 2014, 9.0, 10.0, 90.0, 100.0, 11.0, 12.0, 110.0, 120.0, 0.0],
            ["S2", "Sector 2", "BBB", 2, 2014, 13.0, 14.0, 130.0, 140.0, 15.0, 16.0, 150.0, 160.0, 0.0],
            ["GO", "Output at basic prices", "TOT", 99, 2014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        df = pd.DataFrame(rows, columns=cols)
        # Populate GO row for industry columns (x vector extraction).
        df.loc[df["IndustryCode"] == "GO", ["vAAA1", "vAAA2", "vBBB1", "vBBB2"]] = [100.0, 200.0, 300.0, 400.0]

        mrio = parse_wiod16_wiot_frame(df)

        self.assertEqual(mrio.Z.shape, (4, 4))
        self.assertEqual(mrio.Y.shape, (4, 4))
        self.assertEqual(mrio.x.shape[0], 4)

        # Spot-check one entry.
        self.assertAlmostEqual(mrio.Z.loc[("AAA", "S1"), ("BBB", "S2")], 4.0)
        self.assertAlmostEqual(mrio.Y.loc[("BBB", "S2"), ("AAA", "FD1")], 130.0)
        self.assertAlmostEqual(float(mrio.x.loc[("BBB", "S1")].iloc[0]), 300.0)


if __name__ == "__main__":
    unittest.main()

