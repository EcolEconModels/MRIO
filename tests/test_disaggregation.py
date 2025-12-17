import numpy as np
import pandas as pd


def _aggregate_back(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    idx = pd.Index([mapping.get(str(i), str(i)) for i in df.index], name="agg")
    cols = pd.Index([mapping.get(str(c), str(c)) for c in df.columns], name="agg")
    tmp = df.copy()
    tmp.index = idx
    tmp.columns = cols
    tmp = tmp.groupby(level=0).sum()
    return tmp.T.groupby(level=0).sum().T


def test_expand_is_clean():
    from ecolecon_mrio.disaggregation import SplitProfiles, expand_io_tables

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame([[5.0], [6.0]], index=sectors, columns=["FD"])

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.4, 0.6], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.2, 0.8], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.7, 0.3], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
    )

    Z0, Y0 = expand_io_tables(Z, Y, split_profiles={"C19": prof})
    mapping = {"C19::s1": "C19", "C19::s2": "C19"}

    Z_back = _aggregate_back(Z0, mapping)
    Y_back = Y0.copy()
    Y_back.index = pd.Index([mapping.get(i, i) for i in Y_back.index])
    Y_back = Y_back.groupby(level=0).sum()

    assert np.allclose(Z_back.loc[sectors, sectors].to_numpy(), Z.to_numpy())
    assert np.allclose(Y_back.loc[sectors, :].to_numpy(), Y.to_numpy())


def test_disaggregate_and_balance_ras():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame([[5.0], [6.0]], index=sectors, columns=["FD"])

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.4, 0.6], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.2, 0.8], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.7, 0.3], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
    )

    res = disaggregate_and_balance(Z, Y, split_profiles={"C19": prof}, max_iter=50_000, tol=1e-12)
    assert res.balancing_method == "RAS"
    assert res.balancing_stats.max_abs_row_gap <= 1e-9
    assert res.balancing_stats.max_abs_col_gap <= 1e-9


def test_disaggregate_and_balance_gras_with_negative_final_demand():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame([[5.0], [-1.0]], index=sectors, columns=["INV"])

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.4, 0.6], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.2, 0.8], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.7, 0.3], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
    )

    res = disaggregate_and_balance(Z, Y, split_profiles={"C19": prof}, max_iter=50_000, tol=1e-10)
    assert res.balancing_method == "GRAS"
    assert res.balancing_stats.max_abs_row_gap <= 2e-5
    assert res.balancing_stats.max_abs_col_gap <= 2e-5
