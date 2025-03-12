# tests/test_allele_utils.py

import pytest
import numpy as np
import anndata as ad
from ploidex.allele_utils import AlleleRatioCalculator, calculate_allelic_ratios


@pytest.fixture
def dummy_adata():
    """Create a dummy AnnData object for testing."""
    # Create a simple dataset with 12 observations and 2 variables
    X = np.array([
        [10, 5], [20, 10], [30, 15],  # Synt_id 1
        [5, 5], [15, 5], [20, 10],    # Synt_id 2
        [100, 50], [200, 100],        # Synt_id 3
        [1, 1], [2, 2], [3, 3], [4, 4]  # Synt_id 4
    ])
    
    adata = ad.AnnData(X)
    
    # Add a layer for unique counts
    adata.layers['unique_counts'] = np.copy(X)
    
    # Add a layer for ambiguous counts (just multiply by 2 for simplicity)
    adata.layers['ambigious_counts'] = X * 2
    
    # Create Synt_id in obsm
    adata.obsm['Synt_id'] = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4])
    
    # Add transcript_id to obs
    adata.obs['transcript_id'] = [f"transcript_{i}" for i in range(adata.n_obs)]
    
    return adata


def test_allele_ratio_calculator_initialization():
    """Test initialization of AlleleRatioCalculator."""
    # Test empty initialization
    calculator = AlleleRatioCalculator()
    assert calculator.adata is None
    
    # Test initialization with data
    dummy = ad.AnnData(np.random.rand(10, 5))
    calculator = AlleleRatioCalculator(dummy)
    assert calculator.adata is dummy


def test_set_data():
    """Test set_data method."""
    calculator = AlleleRatioCalculator()
    dummy = ad.AnnData(np.random.rand(10, 5))
    calculator.set_data(dummy)
    assert calculator.adata is dummy


def test_calculate_ratios(dummy_adata):
    """Test calculate_ratios method."""
    calculator = AlleleRatioCalculator(dummy_adata)
    result = calculator.calculate_ratios()
    
    # Check that the result is the original object
    assert result is dummy_adata
    
    # Check that the ratio layer was added
    assert 'allelic_ratio_unique_counts' in dummy_adata.layers
    
    # Verify ratios for Synt_id 1
    mask_synt_1 = dummy_adata.obsm['Synt_id'] == 1
    synt_1_counts = dummy_adata.layers['unique_counts'][mask_synt_1]
    synt_1_ratios = dummy_adata.layers['allelic_ratio_unique_counts'][mask_synt_1]
    
    # Calculate expected ratios manually
    total_counts_synt_1 = np.sum(synt_1_counts)
    expected_ratios_synt_1 = synt_1_counts / total_counts_synt_1
    
    # Check that calculated ratios match expected
    np.testing.assert_allclose(synt_1_ratios, expected_ratios_synt_1)
    
    # Check that ratios sum to 1 for each Synt_id
    for synt_id in np.unique(dummy_adata.obsm['Synt_id']):
        mask = dummy_adata.obsm['Synt_id'] == synt_id
        ratios = dummy_adata.layers['allelic_ratio_unique_counts'][mask]
        # Sum ratios and check they're close to 1.0
        assert np.isclose(np.sum(ratios), 1.0)


def test_calculate_ratios_custom_output(dummy_adata):
    """Test calculate_ratios with custom output suffix."""
    calculator = AlleleRatioCalculator(dummy_adata)
    calculator.calculate_ratios(output_suffix='test_suffix')
    
    # Check that the custom-named layer was added
    assert 'allelic_ratio_test_suffix' in dummy_adata.layers


def test_calculate_multiple_ratios(dummy_adata):
    """Test calculate_multiple_ratios method."""
    calculator = AlleleRatioCalculator(dummy_adata)
    result = calculator.calculate_multiple_ratios()
    
    # Check that ratios were calculated for both count layers
    assert 'allelic_ratio_unique_counts' in result.layers
    assert 'allelic_ratio_ambigious_counts' in result.layers


def test_get_ratios_for_synt_id(dummy_adata):
    """Test get_ratios_for_synt_id method."""
    calculator = AlleleRatioCalculator(dummy_adata)
    calculator.calculate_ratios()
    
    # Get ratios for Synt_id 1
    ratios = calculator.get_ratios_for_synt_id(1)
    
    # Check shape matches the number of observations with Synt_id 1
    mask_synt_1 = dummy_adata.obsm['Synt_id'] == 1
    assert ratios.shape == dummy_adata.layers['unique_counts'][mask_synt_1].shape
    
    # Check sum of ratios is 1
    assert np.isclose(np.sum(ratios), 1.0)


def test_error_no_adata():
    """Test error when no AnnData is set."""
    calculator = AlleleRatioCalculator()
    with pytest.raises(ValueError, match="No AnnData object has been set"):
        calculator.calculate_ratios()


def test_error_no_synt_id(dummy_adata):
    """Test error when Synt_id is missing."""
    calculator = AlleleRatioCalculator(dummy_adata.copy())
    del calculator.adata.obsm['Synt_id']
    
    with pytest.raises(ValueError, match="'Synt_id' not found in obsm"):
        calculator.calculate_ratios()


def test_error_no_layer(dummy_adata):
    """Test error when specified layer is missing."""
    calculator = AlleleRatioCalculator(dummy_adata)
    
    with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found"):
        calculator.calculate_ratios(counts_layer='nonexistent_layer')


def test_error_no_ratio_layer(dummy_adata):
    """Test error when getting ratios for nonexistent ratio layer."""
    calculator = AlleleRatioCalculator(dummy_adata)
    
    with pytest.raises(ValueError, match="Ratio layer 'nonexistent_layer' not found"):
        calculator.get_ratios_for_synt_id(1, ratio_layer='nonexistent_layer')


def test_function_calculate_allelic_ratios(dummy_adata):
    """Test the standalone calculate_allelic_ratios function."""
    result = calculate_allelic_ratios(dummy_adata)
    
    # Check that the result is the original object
    assert result is dummy_adata
    
    # Check that the ratio layer was added
    assert 'allelic_ratio_unique_counts' in result.layers