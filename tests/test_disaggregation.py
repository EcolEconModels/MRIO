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


def test_expand_is_clean_preserves_final_demand_columns_exactly():
    from ecolecon_mrio.disaggregation import SplitProfiles, expand_io_tables

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame(
        [[5.0, 1.0, 0.5], [6.0, 2.0, 1.5]],
        index=sectors,
        columns=["P3", "P5", "P6"],
    )

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.4, 0.6], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.2, 0.8], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.7, 0.3], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.25, 0.75], index=["C19::s1", "C19::s2"]),
    )

    _Z0, Y0 = expand_io_tables(Z, Y, split_profiles={"C19": prof})
    mapping = {"C19::s1": "C19", "C19::s2": "C19"}

    Y_back = Y0.copy()
    Y_back.index = pd.Index([mapping.get(i, i) for i in Y_back.index])
    Y_back = Y_back.groupby(level=0).sum()

    pd.testing.assert_frame_equal(Y_back.loc[sectors, Y.columns], Y)
    pd.testing.assert_series_equal(Y0.sum(axis=0).astype("float64"), Y.sum(axis=0).astype("float64"))


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

    res = disaggregate_and_balance(Z, Y, final_demand_columns=list(Y.columns), split_profiles={"C19": prof}, max_iter=50_000, tol=1e-12)
    assert res.balancing_method == "RAS"
    assert res.balancing_stats.max_abs_row_gap <= 1e-9
    assert res.balancing_stats.max_abs_col_gap <= 1e-9


def test_disaggregate_and_balance_with_negative_final_demand():
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

    res = disaggregate_and_balance(Z, Y, final_demand_columns=list(Y.columns), split_profiles={"C19": prof}, max_iter=50_000, tol=1e-10)
    assert res.balancing_method == "RAS"
    assert res.balancing_stats.max_abs_row_gap <= 2e-5
    assert res.balancing_stats.max_abs_col_gap <= 2e-5


def test_disaggregate_and_balance_local_keeps_untouched_blocks_fixed():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance

    sectors = ["A", "B", "C19"]
    Z = pd.DataFrame(
        [
            [1.0, 0.2, 0.3],
            [0.1, 2.0, 0.4],
            [0.5, 0.6, 3.0],
        ],
        index=sectors,
        columns=sectors,
    )
    Y = pd.DataFrame([[5.0], [6.0], [7.0]], index=sectors, columns=["FD"])

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.4, 0.6], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.2, 0.8], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.7, 0.3], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
    )

    res = disaggregate_and_balance(
        Z,
        Y,
        final_demand_columns=list(Y.columns),
        split_profiles={"C19": prof},
        balancing_scope="local",
        max_iter=50_000,
        tol=1e-12,
    )

    untouched = ["A", "B"]
    assert np.allclose(res.Z.loc[untouched, untouched].to_numpy(), Z.loc[untouched, untouched].to_numpy())
    assert np.allclose(res.Y.loc[untouched, :].to_numpy(), Y.loc[untouched, :].to_numpy())


def test_disaggregate_and_balance_local_preserves_final_demand_exactly():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance

    sectors = ["A", "C19", "D35"]
    Z = pd.DataFrame(
        [[10.0, 1.0, 2.0], [1.0, 5.0, 1.0], [2.0, 1.0, 6.0]],
        index=sectors,
        columns=sectors,
    )
    Y = pd.DataFrame({"FD": [3.0, 2.0, 1.0]}, index=sectors)

    split_profiles = {
        "C19": SplitProfiles(
            aggregate_sector="C19",
            subsectors=("C19::coke", "C19::refined"),
            output_share=pd.Series([0.3333333333, 0.6666666667], index=["C19::coke", "C19::refined"]),
            intermediate_sales_share=pd.Series([0.4054054054, 0.5945945946], index=["C19::coke", "C19::refined"]),
            intermediate_input_share=pd.Series([0.55, 0.45], index=["C19::coke", "C19::refined"]),
            final_demand_share=pd.Series([0.3333333333, 0.6666666667], index=["C19::coke", "C19::refined"]),
        ),
        "D35": SplitProfiles(
            aggregate_sector="D35",
            subsectors=("D35::elec", "D35::gas"),
            output_share=pd.Series([0.4857142857, 0.5142857143], index=["D35::elec", "D35::gas"]),
            intermediate_sales_share=pd.Series([0.6097560976, 0.3902439024], index=["D35::elec", "D35::gas"]),
            intermediate_input_share=pd.Series([0.5, 0.5], index=["D35::elec", "D35::gas"]),
            final_demand_share=pd.Series([0.4285714286, 0.5714285714], index=["D35::elec", "D35::gas"]),
        ),
    }

    res = disaggregate_and_balance(
        Z,
        Y,
        final_demand_columns=list(Y.columns),
        split_profiles=split_profiles,
        balancing_scope="local",
        max_iter=50_000,
        tol=1e-12,
        create_io_system=False,
    )

    mapping = {
        "C19::coke": "C19",
        "C19::refined": "C19",
        "D35::elec": "D35",
        "D35::gas": "D35",
    }
    Y_back = res.Y.copy()
    Y_back.index = pd.Index([mapping.get(str(i), str(i)) for i in Y_back.index])
    Y_back = Y_back.groupby(level=0).sum()

    pd.testing.assert_frame_equal(Y_back.loc[sectors, Y.columns].astype("float64"), Y.astype("float64"))


def test_disaggregate_and_balance_local_handles_output_share_fd_share_conflict():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance, expand_io_tables

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame({"FD": [5.0, 6.0]}, index=sectors)

    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.1, 0.9], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.8, 0.2], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.8, 0.2], index=["C19::s1", "C19::s2"]),
    )

    res = disaggregate_and_balance(
        Z,
        Y,
        final_demand_columns=list(Y.columns),
        split_profiles={"C19": prof},
        balancing_scope="local",
        max_iter=50_000,
        tol=1e-12,
        create_io_system=False,
    )

    Z0, Y0 = expand_io_tables(Z, Y, split_profiles={"C19": prof})
    expected_x = (res.Z.sum(axis=1) + res.Y.sum(axis=1)).astype("float64")
    clean_x = (Z0.sum(axis=1) + Y0.sum(axis=1)).astype("float64")

    mapping = {"C19::s1": "C19", "C19::s2": "C19"}
    Y_back = res.Y.copy()
    Y_back.index = pd.Index([mapping.get(str(i), str(i)) for i in Y_back.index])
    Y_back = Y_back.groupby(level=0).sum()

    pd.testing.assert_frame_equal(Y_back.loc[sectors, Y.columns].astype("float64"), Y.astype("float64"))
    pd.testing.assert_series_equal(expected_x, clean_x)

def test_kernel_uses_only_explicit_non_overlapping_final_demand_columns():
    import pytest

    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame(
        {
            "P3": [99.0, 99.0],
            "P3_S13": [1.0, 2.0],
            "P3_S14": [2.0, 3.0],
            "P3_S15": [3.0, 4.0],
        },
        index=sectors,
    )
    share = pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"], dtype="float64")
    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=share,
        intermediate_sales_share=share,
        intermediate_input_share=share,
        final_demand_share=share,
    )

    result = disaggregate_and_balance(
        Z,
        Y,
        final_demand_columns=["P3_S13", "P3_S14", "P3_S15"],
        split_profiles={"C19": prof},
        balancing_scope="local",
        create_io_system=False,
    )

    assert list(result.Y.columns) == ["P3_S13", "P3_S14", "P3_S15"]
    expected = result.Z.sum(axis=1) + result.Y.sum(axis=1)
    pd.testing.assert_series_equal(result.x, expected.astype("float64"))

    with pytest.raises(ValueError, match="contains duplicates"):
        disaggregate_and_balance(
            Z, Y, final_demand_columns=["P3", "P3"],
            split_profiles={"C19": prof}, create_io_system=False,
        )


def test_global_and_local_output_share_contracts_are_distinct():
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance, expand_io_tables

    sectors = ["A", "C19"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame({"FD": [5.0, 6.0]}, index=sectors)
    prof = SplitProfiles(
        aggregate_sector="C19",
        subsectors=("C19::s1", "C19::s2"),
        output_share=pd.Series([0.1, 0.9], index=["C19::s1", "C19::s2"]),
        intermediate_sales_share=pd.Series([0.5, 0.5], index=["C19::s1", "C19::s2"]),
        intermediate_input_share=pd.Series([0.8, 0.2], index=["C19::s1", "C19::s2"]),
        final_demand_share=pd.Series([0.8, 0.2], index=["C19::s1", "C19::s2"]),
    )
    profiles = {"C19": prof}

    local = disaggregate_and_balance(
        Z, Y, final_demand_columns=["FD"], split_profiles=profiles,
        balancing_scope="local", create_io_system=False,
    )
    Z0, Y0 = expand_io_tables(Z, Y, split_profiles=profiles)
    pd.testing.assert_series_equal(
        local.x, (Z0.sum(axis=1) + Y0.sum(axis=1)).astype("float64")
    )

    global_result = disaggregate_and_balance(
        Z, Y, final_demand_columns=["FD"], split_profiles=profiles,
        balancing_scope="global", create_io_system=False,
    )
    parent_output = float(Z.loc["C19"].sum() + Y.loc["C19"].sum())
    assert np.isclose(global_result.x.loc["C19::s1"], parent_output * 0.1)
    assert np.isclose(global_result.x.loc["C19::s2"], parent_output * 0.9)
