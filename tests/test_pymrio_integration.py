"""Tests for pymrio IOSystem integration in disaggregation."""
import numpy as np
import pandas as pd
import pytest


def test_disaggregation_creates_io_system():
    """Test that disaggregate_and_balance creates a pymrio IOSystem when requested."""
    from ecolecon_mrio.disaggregation import SplitProfiles, disaggregate_and_balance
    
    # Simple test data
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
    
    # Test with IOSystem creation enabled (default)
    res = disaggregate_and_balance(Z, Y, split_profiles={"C19": prof})
    
    # Check that IOSystem was created
    assert res.io_system is not None, "IOSystem should be created by default"
    
    # Verify IOSystem has the expected attributes
    assert hasattr(res.io_system, "Z"), "IOSystem should have Z attribute"
    assert hasattr(res.io_system, "Y"), "IOSystem should have Y attribute"
    assert hasattr(res.io_system, "x"), "IOSystem should have x attribute"
    assert hasattr(res.io_system, "A"), "IOSystem should have A attribute"
    
    # Verify matrices match
    assert res.io_system.Z.equals(res.Z), "IOSystem.Z should match result.Z"
    assert res.io_system.Y.equals(res.Y), "IOSystem.Y should match result.Y"
    assert res.io_system.x.equals(res.x), "IOSystem.x should match result.x"
    assert res.io_system.A.equals(res.A), "IOSystem.A should match result.A"


def test_disaggregation_without_io_system():
    """Test that IOSystem creation can be disabled."""
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
    
    # Test with IOSystem creation disabled
    res = disaggregate_and_balance(Z, Y, split_profiles={"C19": prof}, create_io_system=False)
    
    assert res.io_system is None, "IOSystem should be None when create_io_system=False"


def test_io_system_calc_system():
    """Test that pymrio IOSystem can calculate derived matrices."""
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
    
    res = disaggregate_and_balance(Z, Y, split_profiles={"C19": prof})
    
    # Calculate system matrices using pymrio
    res.io_system.calc_system()
    
    # Check that Leontief inverse was calculated
    assert hasattr(res.io_system, "L"), "IOSystem should have L (Leontief inverse) after calc_system()"
    assert res.io_system.L is not None, "Leontief inverse should be calculated"
    
    # Verify Leontief inverse has correct shape
    n = len(res.Z)
    assert res.io_system.L.shape == (n, n), f"Leontief inverse should be {n}x{n}"
    
    # Verify accounting identity: (I - A) * L = I
    I = np.eye(n)
    A_mat = res.A.to_numpy()
    L_mat = res.io_system.L.to_numpy()
    IminusA = I - A_mat
    product = IminusA @ L_mat
    
    assert np.allclose(product, I, atol=1e-10), "Leontief inverse should satisfy (I-A)*L = I"


def test_split_result_has_io_system():
    """Test that SplitResult also includes IOSystem."""
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.eurostat_energy_split import split_iot_using_exiobase_split_profiles
    
    # Minimal EXIO slice
    exio_sectors = ["coke oven products", "refined petroleum products"]
    Z_domestic = pd.DataFrame(
        [[1.0, 0.5], [0.2, 1.0]],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 2.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")
    
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=None,  # Not needed for this test
    )
    
    # Minimal Eurostat IOT
    euro_sectors = ["A", "C19"]
    euro_Z = pd.DataFrame(
        [[10.0, 1.0], [1.0, 5.0]],
        index=euro_sectors,
        columns=euro_sectors,
    )
    euro_Y = pd.DataFrame({"FD": [3.0, 2.0]}, index=euro_sectors)
    
    # Run split
    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_patterns={"C19": ("coke", "refined")},
    )
    
    # Verify IOSystem is present
    assert res.io_system is not None, "SplitResult should have IOSystem"
    assert hasattr(res.io_system, "Z"), "IOSystem should have Z"
    assert res.io_system.Z.equals(res.Z), "IOSystem.Z should match result.Z"


def test_outer_share_matrix_no_redundant_normalization():
    """Test that outer share matrix doesn't redundantly normalize."""
    from ecolecon_mrio.disaggregation import _outer_share_matrix
    
    # Create normalized shares (sum to 1)
    row_share = pd.Series([0.3, 0.7], index=["a", "b"])
    col_share = pd.Series([0.4, 0.6], index=["x", "y"])
    
    # Build outer product
    M = _outer_share_matrix(row_share, col_share)
    
    # Check that result sums to 1
    assert np.isclose(M.sum().sum(), 1.0), "Outer product should sum to 1"
    
    # Check individual elements
    assert np.isclose(M.loc["a", "x"], 0.3 * 0.4), "Element should be product of shares"
    assert np.isclose(M.loc["b", "y"], 0.7 * 0.6), "Element should be product of shares"
    
    # Check that assertions work for normalized inputs
    try:
        _outer_share_matrix(row_share, col_share)
    except AssertionError:
        pytest.fail("Should not raise assertion error for normalized shares")


def test_outer_share_matrix_asserts_normalization():
    """Test that outer share matrix asserts on non-normalized inputs."""
    from ecolecon_mrio.disaggregation import _outer_share_matrix
    
    # Create non-normalized shares
    row_share = pd.Series([0.3, 0.8], index=["a", "b"])  # Sums to 1.1
    col_share = pd.Series([0.4, 0.6], index=["x", "y"])  # Sums to 1.0
    
    # Should raise assertion error
    with pytest.raises(AssertionError, match="must sum to 1"):
        _outer_share_matrix(row_share, col_share)
