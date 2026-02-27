"""Tests for polyase.allele_utils module."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from anndata import AnnData

from polyase.allele_utils import AlleleRatioCalculator, calculate_allelic_ratios


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adata_two_groups():
    """
    AnnData with two syntelog groups (Synt_id 1 and 2), each with 2 transcripts.

    Counts (3 samples × 4 transcripts):
        t1   t2   t3   t4
    s1 [10,  30,  20,   0]
    s2 [ 5,  15,   0,  10]
    s3 [ 0,   0,   8,   2]

    Expected ratios for Synt_id=1 (t1, t2):
        s1: [0.25, 0.75]
        s2: [0.25, 0.75]
        s3: [0.0,  0.0]   (zero total → stays 0)

    Expected ratios for Synt_id=2 (t3, t4):
        s1: [1.0,  0.0]
        s2: [0.0,  1.0]
        s3: [0.8,  0.2]
    """
    counts = np.array([[10, 30, 20,  0],
                       [ 5, 15,  0, 10],
                       [ 0,  0,  8,  2]], dtype=float)

    adata = AnnData(X=counts.copy())
    adata.var_names = ['t1', 't2', 't3', 't4']
    adata.obs_names = ['s1', 's2', 's3']
    adata.var['Synt_id'] = [1, 1, 2, 2]
    adata.layers['unique_counts'] = counts.copy()
    return adata


@pytest.fixture
def adata_single_transcript_groups():
    """AnnData where each syntelog group contains only one transcript."""
    counts = np.array([[50, 0],
                       [20, 5]], dtype=float)
    adata = AnnData(X=counts.copy())
    adata.var_names = ['t1', 't2']
    adata.obs_names = ['s1', 's2']
    adata.var['Synt_id'] = [1, 2]
    adata.layers['unique_counts'] = counts.copy()
    return adata


@pytest.fixture
def adata_with_excluded_ids():
    """AnnData containing Synt_id values that should be excluded (0 and NaN)."""
    counts = np.array([[10, 30, 5],
                       [ 5, 15, 8]], dtype=float)
    adata = AnnData(X=counts.copy())
    adata.var_names = ['t1', 't2', 't_excluded']
    adata.obs_names = ['s1', 's2']
    adata.var['Synt_id'] = pd.array([1, 1, 0], dtype=object)
    adata.layers['unique_counts'] = counts.copy()
    return adata


@pytest.fixture
def adata_with_nan_synt_id():
    """AnnData containing NaN Synt_id that should be excluded."""
    counts = np.array([[10, 30, 5],
                       [ 5, 15, 8]], dtype=float)
    adata = AnnData(X=counts.copy())
    adata.var_names = ['t1', 't2', 't_nan']
    adata.obs_names = ['s1', 's2']
    adata.var['Synt_id'] = [1, 1, np.nan]
    adata.layers['unique_counts'] = counts.copy()
    return adata


# ---------------------------------------------------------------------------
# AlleleRatioCalculator — initialisation and set_data
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_without_adata(self):
        calc = AlleleRatioCalculator()
        assert calc.adata is None

    def test_init_with_adata(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        assert calc.adata is adata_two_groups

    def test_set_data(self, adata_two_groups):
        calc = AlleleRatioCalculator()
        calc.set_data(adata_two_groups)
        assert calc.adata is adata_two_groups

    def test_set_data_replaces_existing(self, adata_two_groups, adata_single_transcript_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.set_data(adata_single_transcript_groups)
        assert calc.adata is adata_single_transcript_groups


# ---------------------------------------------------------------------------
# AlleleRatioCalculator.calculate_ratios — error handling
# ---------------------------------------------------------------------------

class TestCalculateRatiosErrors:
    def test_raises_when_no_adata_set(self):
        calc = AlleleRatioCalculator()
        with pytest.raises(ValueError, match="No AnnData"):
            calc.calculate_ratios()

    def test_raises_when_synt_id_missing(self):
        adata = AnnData(X=np.ones((2, 2)))
        adata.layers['unique_counts'] = np.ones((2, 2))
        calc = AlleleRatioCalculator(adata)
        with pytest.raises(ValueError, match="Synt_id"):
            calc.calculate_ratios()

    def test_raises_when_layer_missing(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        with pytest.raises(ValueError, match="not found"):
            calc.calculate_ratios(counts_layer='nonexistent_layer')


# ---------------------------------------------------------------------------
# AlleleRatioCalculator.calculate_ratios — correct output
# ---------------------------------------------------------------------------

class TestCalculateRatiosOutput:
    def test_output_layer_is_added(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        assert 'allelic_ratio_unique_counts' in adata_two_groups.layers

    def test_output_layer_name_uses_custom_suffix(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts', output_suffix='custom')
        assert 'allelic_ratio_custom' in adata_two_groups.layers
        assert 'allelic_ratio_unique_counts' not in adata_two_groups.layers

    def test_returns_adata(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        result = calc.calculate_ratios('unique_counts')
        assert result is adata_two_groups

    def test_ratio_values_are_correct(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = adata_two_groups.layers['allelic_ratio_unique_counts']

        # Synt_id=1: t1 and t2 (columns 0 and 1)
        np.testing.assert_allclose(ratios[0, 0], 0.25)   # s1, t1
        np.testing.assert_allclose(ratios[0, 1], 0.75)   # s1, t2
        np.testing.assert_allclose(ratios[1, 0], 0.25)   # s2, t1
        np.testing.assert_allclose(ratios[1, 1], 0.75)   # s2, t2

        # Synt_id=2: t3 and t4 (columns 2 and 3)
        np.testing.assert_allclose(ratios[0, 2], 1.0)    # s1, t3
        np.testing.assert_allclose(ratios[0, 3], 0.0)    # s1, t4
        np.testing.assert_allclose(ratios[1, 2], 0.0)    # s2, t3
        np.testing.assert_allclose(ratios[1, 3], 1.0)    # s2, t4
        np.testing.assert_allclose(ratios[2, 2], 0.8)    # s3, t3
        np.testing.assert_allclose(ratios[2, 3], 0.2)    # s3, t4

    def test_ratios_within_group_sum_to_one_for_nonzero_samples(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = adata_two_groups.layers['allelic_ratio_unique_counts']

        for synt_id in [1, 2]:
            mask = adata_two_groups.var['Synt_id'] == synt_id
            group_ratios = ratios[:, mask]
            row_sums = group_ratios.sum(axis=1)

            # Rows with non-zero original counts must sum to 1
            original_counts = adata_two_groups.layers['unique_counts'][:, mask]
            nonzero_rows = original_counts.sum(axis=1) > 0
            np.testing.assert_allclose(row_sums[nonzero_rows], 1.0,
                                       err_msg=f"Ratios for Synt_id={synt_id} don't sum to 1")

    def test_zero_total_counts_produce_zero_ratios(self, adata_two_groups):
        """Sample s3 has all-zero counts for Synt_id=1 → ratios must be 0."""
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = adata_two_groups.layers['allelic_ratio_unique_counts']

        assert ratios[2, 0] == 0.0   # s3, t1
        assert ratios[2, 1] == 0.0   # s3, t2

    def test_output_shape_matches_adata(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = adata_two_groups.layers['allelic_ratio_unique_counts']
        assert ratios.shape == adata_two_groups.shape

    def test_single_transcript_per_group_ratio_is_one(self, adata_single_transcript_groups):
        """A lone transcript in its group gets ratio 1.0 wherever counts > 0."""
        calc = AlleleRatioCalculator(adata_single_transcript_groups)
        calc.calculate_ratios('unique_counts')
        ratios = adata_single_transcript_groups.layers['allelic_ratio_unique_counts']

        # t1 (Synt_id=1): s1 has count 50 → ratio 1.0; s2 has count 20 → ratio 1.0
        np.testing.assert_allclose(ratios[0, 0], 1.0)
        np.testing.assert_allclose(ratios[1, 0], 1.0)

        # t2 (Synt_id=2): s1 has count 0 → ratio 0.0; s2 has count 5 → ratio 1.0
        assert ratios[0, 1] == 0.0
        np.testing.assert_allclose(ratios[1, 1], 1.0)

    def test_synt_id_zero_is_excluded(self, adata_with_excluded_ids):
        """Transcripts with Synt_id=0 must remain 0 in the ratio matrix."""
        calc = AlleleRatioCalculator(adata_with_excluded_ids)
        calc.calculate_ratios('unique_counts')
        ratios = adata_with_excluded_ids.layers['allelic_ratio_unique_counts']

        # t_excluded (index 2) must stay 0 for all samples
        np.testing.assert_array_equal(ratios[:, 2], 0.0)

    def test_synt_id_nan_is_excluded(self, adata_with_nan_synt_id):
        """Transcripts with NaN Synt_id must remain 0 in the ratio matrix."""
        calc = AlleleRatioCalculator(adata_with_nan_synt_id)
        calc.calculate_ratios('unique_counts')
        ratios = adata_with_nan_synt_id.layers['allelic_ratio_unique_counts']

        # t_nan (index 2) must stay 0 for all samples
        np.testing.assert_array_equal(ratios[:, 2], 0.0)

    def test_sparse_and_dense_give_same_result(self, adata_two_groups):
        """Sparse input must produce identical ratios to dense input."""
        dense_adata = adata_two_groups.copy()
        sparse_adata = adata_two_groups.copy()
        sparse_adata.layers['unique_counts'] = scipy.sparse.csr_matrix(
            sparse_adata.layers['unique_counts']
        )

        AlleleRatioCalculator(dense_adata).calculate_ratios('unique_counts')
        AlleleRatioCalculator(sparse_adata).calculate_ratios('unique_counts')

        np.testing.assert_allclose(
            dense_adata.layers['allelic_ratio_unique_counts'],
            sparse_adata.layers['allelic_ratio_unique_counts'],
        )

    def test_original_layer_is_not_modified(self, adata_two_groups):
        """calculate_ratios must not alter the source counts layer."""
        original_counts = adata_two_groups.layers['unique_counts'].copy()
        AlleleRatioCalculator(adata_two_groups).calculate_ratios('unique_counts')
        np.testing.assert_array_equal(
            adata_two_groups.layers['unique_counts'], original_counts
        )

    def test_all_zero_counts_matrix(self):
        """All-zero input must produce all-zero ratios without errors."""
        counts = np.zeros((3, 4), dtype=float)
        adata = AnnData(X=counts.copy())
        adata.var_names = ['t1', 't2', 't3', 't4']
        adata.obs_names = ['s1', 's2', 's3']
        adata.var['Synt_id'] = [1, 1, 2, 2]
        adata.layers['unique_counts'] = counts.copy()

        AlleleRatioCalculator(adata).calculate_ratios('unique_counts')
        ratios = adata.layers['allelic_ratio_unique_counts']
        np.testing.assert_array_equal(ratios, 0.0)


# ---------------------------------------------------------------------------
# AlleleRatioCalculator.calculate_multiple_ratios
# ---------------------------------------------------------------------------

class TestCalculateMultipleRatios:
    def test_explicit_list_creates_all_layers(self, adata_two_groups):
        adata_two_groups.layers['em_counts'] = np.ones((3, 4))
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_multiple_ratios(['unique_counts', 'em_counts'])

        assert 'allelic_ratio_unique_counts' in adata_two_groups.layers
        assert 'allelic_ratio_em_counts' in adata_two_groups.layers

    def test_auto_detect_finds_all_count_layers(self, adata_two_groups):
        adata_two_groups.layers['em_counts'] = np.ones((3, 4))
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_multiple_ratios()   # None → auto-detect

        assert 'allelic_ratio_unique_counts' in adata_two_groups.layers
        assert 'allelic_ratio_em_counts' in adata_two_groups.layers

    def test_auto_detect_skips_non_count_layers(self, adata_two_groups):
        adata_two_groups.layers['some_metadata'] = np.ones((3, 4))
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_multiple_ratios()

        assert 'allelic_ratio_some_metadata' not in adata_two_groups.layers

    def test_returns_adata(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        result = calc.calculate_multiple_ratios(['unique_counts'])
        assert result is adata_two_groups


# ---------------------------------------------------------------------------
# AlleleRatioCalculator.get_ratios_for_synt_id
# ---------------------------------------------------------------------------

class TestGetRatiosForSyntId:
    def test_raises_when_ratio_layer_missing(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        with pytest.raises(ValueError, match="not found"):
            calc.get_ratios_for_synt_id(1)

    def test_returns_correct_values_for_synt_id(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = calc.get_ratios_for_synt_id(1)

        # Should contain only the two transcripts belonging to Synt_id=1
        expected = adata_two_groups.layers['allelic_ratio_unique_counts'][:, [0, 1]]
        np.testing.assert_array_equal(ratios, expected)

    def test_returns_correct_values_for_second_group(self, adata_two_groups):
        calc = AlleleRatioCalculator(adata_two_groups)
        calc.calculate_ratios('unique_counts')
        ratios = calc.get_ratios_for_synt_id(2)

        expected = adata_two_groups.layers['allelic_ratio_unique_counts'][:, [2, 3]]
        np.testing.assert_array_equal(ratios, expected)


# ---------------------------------------------------------------------------
# calculate_allelic_ratios convenience function
# ---------------------------------------------------------------------------

class TestCalculateAllelicRatiosFunction:
    def test_creates_ratio_layer(self, adata_two_groups):
        result = calculate_allelic_ratios(adata_two_groups)
        assert 'allelic_ratio_unique_counts' in result.layers

    def test_returns_same_adata_object(self, adata_two_groups):
        result = calculate_allelic_ratios(adata_two_groups)
        assert result is adata_two_groups

    def test_matches_class_based_result(self, adata_two_groups):
        adata_func = adata_two_groups.copy()
        adata_class = adata_two_groups.copy()

        calculate_allelic_ratios(adata_func)
        AlleleRatioCalculator(adata_class).calculate_ratios('unique_counts')

        np.testing.assert_array_equal(
            adata_func.layers['allelic_ratio_unique_counts'],
            adata_class.layers['allelic_ratio_unique_counts'],
        )

    def test_custom_layer_name(self, adata_two_groups):
        adata_two_groups.layers['em_counts'] = np.ones((3, 4))
        result = calculate_allelic_ratios(adata_two_groups, counts_layer='em_counts')
        assert 'allelic_ratio_em_counts' in result.layers
