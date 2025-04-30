import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from polyase.allele_utils import AlleleRatioCalculator, calculate_allelic_ratios

@pytest.fixture
def create_test_adata():
    """Create a small test AnnData object with synthetic data."""
    # Create a simple test dataset
    n_obs = 10  # number of observations/cells
    n_vars = 5  # number of genes/features
    
    # Random count data
    X = np.random.randint(0, 10, size=(n_obs, n_vars))
    
    # Create AnnData object
    adata = AnnData(X=X)
    
    # Add a layer with unique counts
    adata.layers['unique_counts'] = X.copy()
    adata.layers['total_counts'] = X.copy() * 2  # Just another layer for testing
    
    # Add Synt_id in obsm
    # Create 3 synthetic transcripts: 0-3 (id=1), 4-6 (id=2), 7-9 (id=3)
    synt_ids = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    adata.obsm['Synt_id'] = synt_ids
    
    return adata

def test_init_and_set_data(create_test_adata):
    """Test initializing the calculator with and without data."""
    # Test initialization without data
    calc = AlleleRatioCalculator()
    assert calc.adata is None
    
    # Test initialization with data
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    assert calc.adata is adata
    
    # Test set_data method
    calc = AlleleRatioCalculator()
    calc.set_data(adata)
    assert calc.adata is adata

def test_calculate_ratios_basic(create_test_adata):
    """Test basic ratio calculation functionality."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios
    result = calc.calculate_ratios()
    
    # Check that the result is the same AnnData object
    assert result is adata
    
    # Check that the new layer was added
    assert 'allelic_ratio_unique_counts' in adata.layers
    
    # Check that the ratio matrix has the correct shape
    assert adata.layers['allelic_ratio_unique_counts'].shape == adata.shape

def test_calculate_ratios_custom_layer(create_test_adata):
    """Test ratio calculation with a custom layer."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios with custom layer
    result = calc.calculate_ratios(counts_layer='total_counts')
    
    # Check that the custom layer name was used
    assert 'allelic_ratio_total_counts' in adata.layers

def test_calculate_ratios_custom_suffix(create_test_adata):
    """Test ratio calculation with a custom output suffix."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios with custom suffix
    result = calc.calculate_ratios(output_suffix='custom')
    
    # Check that the custom suffix was used
    assert 'allelic_ratio_custom' in adata.layers

def test_calculate_ratios_values(create_test_adata):
    """Test that ratio values are calculated correctly."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Before we calculate ratios, let's set up a simple test case
    # Reset the counts for controlled testing
    test_counts = np.zeros((10, 5))
    # For Synt_id 1 (cells 0-3), set counts for gene 0
    test_counts[0, 0] = 5
    test_counts[1, 0] = 3
    test_counts[2, 0] = 2
    test_counts[3, 0] = 0
    # For Synt_id 2 (cells 4-6), set counts for gene 1
    test_counts[4, 1] = 4
    test_counts[5, 1] = 6
    test_counts[6, 1] = 0
    # For Synt_id 3 (cells 7-9), no counts for gene 2
    test_counts[7, 2] = 0
    test_counts[8, 2] = 0
    test_counts[9, 2] = 0
    
    adata.layers['unique_counts'] = test_counts
    
    # Calculate ratios
    calc.calculate_ratios()
    
    # Check ratios for Synt_id 1, gene 0 (total of 5+3+2=10 counts)
    assert adata.layers['allelic_ratio_unique_counts'][0, 0] == 0.5  # 5/10
    assert adata.layers['allelic_ratio_unique_counts'][1, 0] == 0.3  # 3/10
    assert adata.layers['allelic_ratio_unique_counts'][2, 0] == 0.2  # 2/10
    assert adata.layers['allelic_ratio_unique_counts'][3, 0] == 0.0  # 0/10
    
    # Check ratios for Synt_id 2, gene 1 (total of 4+6=10 counts)
    assert adata.layers['allelic_ratio_unique_counts'][4, 1] == 0.4  # 4/10
    assert adata.layers['allelic_ratio_unique_counts'][5, 1] == 0.6  # 6/10
    assert adata.layers['allelic_ratio_unique_counts'][6, 1] == 0.0  # 0/10
    
    # Check that gene 2 for Synt_id 3 has zero ratios (no counts)
    assert adata.layers['allelic_ratio_unique_counts'][7, 2] == 0.0
    assert adata.layers['allelic_ratio_unique_counts'][8, 2] == 0.0
    assert adata.layers['allelic_ratio_unique_counts'][9, 2] == 0.0

def test_calculate_ratios_sparse_matrix(create_test_adata):
    """Test ratio calculation with sparse matrix input."""
    adata = create_test_adata
    
    # Convert the counts to a sparse matrix
    adata.layers['sparse_counts'] = csr_matrix(adata.layers['unique_counts'])
    
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios with sparse matrix
    result = calc.calculate_ratios(counts_layer='sparse_counts')
    
    # Check that the result is correct
    assert 'allelic_ratio_sparse_counts' in adata.layers
    
    # The resulting ratio matrix should be dense
    assert not isinstance(adata.layers['allelic_ratio_sparse_counts'], csr_matrix)
    assert isinstance(adata.layers['allelic_ratio_sparse_counts'], np.ndarray)
    
def test_calculate_multiple_ratios(create_test_adata):
    """Test calculation of ratios for multiple layers."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios for multiple layers
    result = calc.calculate_multiple_ratios(['unique_counts', 'total_counts'])
    
    # Check that both layers were created
    assert 'allelic_ratio_unique_counts' in adata.layers
    assert 'allelic_ratio_total_counts' in adata.layers

def test_calculate_multiple_ratios_auto_detect(create_test_adata):
    """Test automatic detection of count layers."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios with auto-detection
    result = calc.calculate_multiple_ratios()
    
    # Check that both layers were created
    assert 'allelic_ratio_unique_counts' in adata.layers
    assert 'allelic_ratio_total_counts' in adata.layers

def test_get_ratios_for_synt_id(create_test_adata):
    """Test getting ratios for a specific Synt_id."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Calculate ratios first
    calc.calculate_ratios()
    
    # Get ratios for Synt_id 1
    ratios = calc.get_ratios_for_synt_id(1)
    
    # Check that we got the correct number of rows
    assert ratios.shape[0] == 4  # 4 cells have Synt_id 1
    assert ratios.shape[1] == 5  # 5 genes

def test_error_handling_no_adata(create_test_adata):
    """Test error handling when no AnnData object is set."""
    calc = AlleleRatioCalculator()
    
    # Should raise ValueError when no AnnData is set
    with pytest.raises(ValueError, match="No AnnData object has been set"):
        calc.calculate_ratios()

def test_error_handling_no_synt_id(create_test_adata):
    """Test error handling when Synt_id is not in obsm."""
    adata = create_test_adata
    # Remove Synt_id from obsm
    del adata.obsm['Synt_id']
    
    calc = AlleleRatioCalculator(adata)
    
    # Should raise ValueError when Synt_id is not found
    with pytest.raises(ValueError, match="'Synt_id' not found in obsm"):
        calc.calculate_ratios()

def test_error_handling_layer_not_found(create_test_adata):
    """Test error handling when specified layer is not found."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Should raise ValueError when layer is not found
    with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found"):
        calc.calculate_ratios(counts_layer='nonexistent_layer')

def test_error_handling_ratio_layer_not_found(create_test_adata):
    """Test error handling when ratio layer is not found."""
    adata = create_test_adata
    calc = AlleleRatioCalculator(adata)
    
    # Should raise ValueError when ratio layer is not found
    with pytest.raises(ValueError, match="Ratio layer 'nonexistent_layer' not found"):
        calc.get_ratios_for_synt_id(1, ratio_layer='nonexistent_layer')

def test_helper_function(create_test_adata):
    """Test the helper function calculate_allelic_ratios."""
    adata = create_test_adata
    
    # Use the helper function
    result = calculate_allelic_ratios(adata)
    
    # Check that the result is correct
    assert 'allelic_ratio_unique_counts' in adata.layers
    assert result is adata

def test_edge_case_zero_counts(create_test_adata):
    """Test case where all counts are zero."""
    adata = create_test_adata
    
    # Set all counts to zero
    adata.layers['unique_counts'] = np.zeros_like(adata.layers['unique_counts'])
    
    calc = AlleleRatioCalculator(adata)
    calc.calculate_ratios()
    
    # Check that all ratios are zero
    assert np.all(adata.layers['allelic_ratio_unique_counts'] == 0)

def test_edge_case_single_transcript(create_test_adata):
    """Test case where each Synt_id has only one transcript."""
    adata = create_test_adata
    
    # Modify Synt_id so each has only one transcript
    adata.obsm['Synt_id'] = np.arange(1, 11)
    
    calc = AlleleRatioCalculator(adata)
    calc.calculate_ratios()
    
    # Check that all ratios are either 0 or 1
    ratios = adata.layers['allelic_ratio_unique_counts']
    nonzero_mask = ratios > 0
    
    if np.any(nonzero_mask):
        assert np.all(ratios[nonzero_mask] == 1.0)