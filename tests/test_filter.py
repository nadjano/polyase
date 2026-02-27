"""Tests for polyase.filter module."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from anndata import AnnData

from polyase.filter import filter_low_expressed_genes, _get_group_mapping


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adata():
    """
    3 samples × 4 transcripts in two Synt_id groups.

    Group A (t1, t2):  high expression
        s1: 100+200 = 300
        s2: 150+100 = 250
        s3: 200+ 50 = 250

    Group B (t3, t4):  low expression
        s1: 1+2 = 3
        s2: 2+1 = 3
        s3: 0+3 = 3
    """
    counts = np.array([[100, 200,  1,  2],
                       [150, 100,  2,  1],
                       [200,  50,  0,  3]], dtype=float)
    a = AnnData(X=counts.copy())
    a.var_names = ['t1', 't2', 't3', 't4']
    a.obs_names = ['s1', 's2', 's3']
    a.var['Synt_id'] = ['A', 'A', 'B', 'B']
    return a


@pytest.fixture
def adata_obs_groups():
    """
    4 samples × 2 genes; two obs-level groups for filter_axis=0 tests.

    Group 'high' (s1, s2): high expression across both genes
    Group 'low'  (s3, s4): low expression across both genes
    """
    counts = np.array([[100, 200],
                       [150, 100],
                       [  1,   2],
                       [  2,   1]], dtype=float)
    a = AnnData(X=counts.copy())
    a.var_names = ['gene1', 'gene2']
    a.obs_names = ['s1', 's2', 's3', 's4']
    a.obs['condition'] = ['high', 'high', 'low', 'low']
    return a


# ---------------------------------------------------------------------------
# _get_group_mapping — unit tests
# ---------------------------------------------------------------------------

class TestGetGroupMapping:
    def test_var_source_returns_correct_series(self, adata):
        mapping = _get_group_mapping(adata, 'Synt_id', 'var')
        pd.testing.assert_series_equal(mapping, adata.var['Synt_id'])

    def test_obs_source_returns_correct_series(self, adata_obs_groups):
        mapping = _get_group_mapping(adata_obs_groups, 'condition', 'obs')
        pd.testing.assert_series_equal(mapping, adata_obs_groups.obs['condition'])

    def test_missing_var_column_raises(self, adata):
        with pytest.raises(ValueError, match="not found in adata.var"):
            _get_group_mapping(adata, 'nonexistent', 'var')

    def test_missing_obs_column_raises(self, adata):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            _get_group_mapping(adata, 'nonexistent', 'obs')

    def test_invalid_group_source_raises(self, adata):
        with pytest.raises(ValueError, match="group_source must be one of"):
            _get_group_mapping(adata, 'Synt_id', 'invalid_source')

    def test_obsm_source_valid_tuple(self, adata):
        adata.obsm['X_pca'] = np.arange(adata.n_obs * 2).reshape(adata.n_obs, 2).astype(float)
        mapping = _get_group_mapping(adata, ('X_pca', 0), 'obsm')
        assert len(mapping) == adata.n_obs

    def test_obsm_source_invalid_tuple_raises(self, adata):
        adata.obsm['X_pca'] = np.ones((adata.n_obs, 2))
        with pytest.raises(ValueError, match="tuple"):
            _get_group_mapping(adata, 'X_pca', 'obsm')

    def test_obsm_source_missing_key_raises(self, adata):
        with pytest.raises(ValueError, match="not found"):
            _get_group_mapping(adata, ('missing_key', 0), 'obsm')

    def test_varm_source_valid_tuple(self, adata):
        adata.varm['loadings'] = np.ones((adata.n_vars, 2))
        mapping = _get_group_mapping(adata, ('loadings', 0), 'varm')
        assert len(mapping) == adata.n_vars

    def test_varm_source_invalid_tuple_raises(self, adata):
        adata.varm['loadings'] = np.ones((adata.n_vars, 2))
        with pytest.raises(ValueError, match="tuple"):
            _get_group_mapping(adata, 'loadings', 'varm')

    def test_varm_source_missing_key_raises(self, adata):
        with pytest.raises(ValueError, match="not found"):
            _get_group_mapping(adata, ('missing_key', 0), 'varm')


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — error handling
# ---------------------------------------------------------------------------

class TestFilterErrors:
    def test_invalid_mode_raises(self, adata):
        with pytest.raises(ValueError, match="mode must be"):
            filter_low_expressed_genes(adata, mode='invalid', verbose=False)

    def test_invalid_lib_size_normalization_raises(self, adata):
        with pytest.raises(ValueError, match="Unsupported normalization"):
            filter_low_expressed_genes(adata, lib_size_normalization='rpkm', verbose=False)

    def test_missing_layer_raises(self, adata):
        with pytest.raises(ValueError, match="not found"):
            filter_low_expressed_genes(adata, layer='nonexistent', verbose=False)


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — basic filtering behaviour
# ---------------------------------------------------------------------------

class TestFilterBasicBehavior:
    def test_low_group_is_removed(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        assert 't3' not in result.var_names
        assert 't4' not in result.var_names

    def test_high_group_is_kept(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        assert 't1' in result.var_names
        assert 't2' in result.var_names

    def test_n_vars_decreases(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 2

    def test_n_obs_unchanged(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        assert result.n_obs == adata.n_obs

    def test_threshold_zero_keeps_all(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=0, lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == adata.n_vars

    def test_very_high_threshold_drops_all(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=1e9, lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 0

    def test_expression_values_preserved(self, adata):
        """The returned AnnData must contain the original (non-normalised) counts."""
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        original_subset = adata[:, ['t1', 't2']].X
        np.testing.assert_array_equal(result.X, original_subset)

    def test_uses_x_by_default(self, adata):
        """Without a layer argument, .X is used for thresholding."""
        adata.layers['zero_layer'] = np.zeros_like(adata.X)
        # .X has high/low groups; zero_layer has no expression at all
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        # Group A should still be kept because .X (not zero_layer) is used
        assert result.n_vars == 2


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — modes
# ---------------------------------------------------------------------------

class TestFilterModes:
    """
    Group A sums (lib_size_normalization=None):
        s1: 300, s2: 250, s3: 250

    Group B sums:
        s1: 3, s2: 3, s3: 3
    """

    def test_mode_any_keeps_group_passing_one_sample(self, adata):
        """'any' keeps a group if it passes in at least one sample."""
        # Threshold between 250 and 300 → A passes only in s1, but 'any' keeps it
        result = filter_low_expressed_genes(
            adata, min_expression=280, mode='any',
            lib_size_normalization=None, verbose=False
        )
        assert 't1' in result.var_names

    def test_mode_any_drops_group_passing_no_sample(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, mode='any',
            lib_size_normalization=None, verbose=False
        )
        assert 't3' not in result.var_names

    def test_mode_all_drops_group_failing_one_sample(self, adata):
        """'all' drops a group if it fails in any single sample.
        Threshold=280: A has s1=300≥280 but s2=250<280 → dropped."""
        result = filter_low_expressed_genes(
            adata, min_expression=280, mode='all',
            lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 0

    def test_mode_all_keeps_group_passing_all_samples(self, adata):
        """'all' keeps a group when every sample exceeds the threshold."""
        result = filter_low_expressed_genes(
            adata, min_expression=200, mode='all',
            lib_size_normalization=None, verbose=False
        )
        assert 't1' in result.var_names

    def test_mode_mean_uses_average(self, adata):
        """'mean' keeps group A whose mean (266.7) exceeds 250 but drops at 300."""
        # Group A mean = (300+250+250)/3 ≈ 266.7
        result_kept = filter_low_expressed_genes(
            adata, min_expression=260, mode='mean',
            lib_size_normalization=None, verbose=False
        )
        assert 't1' in result_kept.var_names

        result_dropped = filter_low_expressed_genes(
            adata, min_expression=280, mode='mean',
            lib_size_normalization=None, verbose=False
        )
        assert 't1' not in result_dropped.var_names


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — copy behaviour
# ---------------------------------------------------------------------------

class TestFilterCopyBehavior:
    def test_copy_true_does_not_modify_original(self, adata):
        original_n_vars = adata.n_vars
        filter_low_expressed_genes(
            adata, min_expression=10, copy=True,
            lib_size_normalization=None, verbose=False
        )
        assert adata.n_vars == original_n_vars

    def test_copy_false_modifies_original(self, adata):
        filter_low_expressed_genes(
            adata, min_expression=10, copy=False,
            lib_size_normalization=None, verbose=False
        )
        assert adata.n_vars == 2

    def test_copy_true_returns_new_object(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, copy=True,
            lib_size_normalization=None, verbose=False
        )
        assert result is not adata

    def test_copy_false_returns_same_object(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, copy=False,
            lib_size_normalization=None, verbose=False
        )
        assert result is adata


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — return_dropped
# ---------------------------------------------------------------------------

class TestFilterReturnDropped:
    def test_return_dropped_false_returns_adata_only(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, return_dropped=False,
            lib_size_normalization=None, verbose=False
        )
        assert isinstance(result, AnnData)

    def test_return_dropped_true_returns_tuple(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10, return_dropped=True,
            lib_size_normalization=None, verbose=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_dropped_list_contains_correct_group(self, adata):
        _, dropped = filter_low_expressed_genes(
            adata, min_expression=10, return_dropped=True,
            lib_size_normalization=None, verbose=False
        )
        assert 'B' in dropped
        assert 'A' not in dropped

    def test_dropped_list_empty_when_nothing_filtered(self, adata):
        _, dropped = filter_low_expressed_genes(
            adata, min_expression=0, return_dropped=True,
            lib_size_normalization=None, verbose=False
        )
        assert dropped == []


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — layer parameter
# ---------------------------------------------------------------------------

class TestFilterLayer:
    def test_uses_specified_layer(self, adata):
        """When a layer is given it should be used instead of .X."""
        # Replace .X with zeros so every group would be dropped if .X were used
        adata.X = np.zeros_like(adata.X)
        adata.layers['real_counts'] = np.array(
            [[100, 200, 1, 2],
             [150, 100, 2, 1],
             [200,  50, 0, 3]], dtype=float
        )
        result = filter_low_expressed_genes(
            adata, min_expression=10, layer='real_counts',
            lib_size_normalization=None, verbose=False
        )
        # Group A should be kept because real_counts are high
        assert result.n_vars == 2

    def test_missing_layer_raises(self, adata):
        with pytest.raises(ValueError, match="not found"):
            filter_low_expressed_genes(
                adata, layer='missing', verbose=False
            )

    def test_sparse_layer_handled(self, adata):
        adata.layers['sparse_counts'] = scipy.sparse.csr_matrix(adata.X)
        result = filter_low_expressed_genes(
            adata, min_expression=10, layer='sparse_counts',
            lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 2


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — filter_axis
# ---------------------------------------------------------------------------

class TestFilterAxis:
    def test_filter_axis_1_filters_columns(self, adata):
        """Default axis=1 removes transcripts (vars), not samples (obs)."""
        result = filter_low_expressed_genes(
            adata, min_expression=10, filter_axis=1,
            lib_size_normalization=None, verbose=False
        )
        assert result.n_obs == adata.n_obs   # no samples removed
        assert result.n_vars < adata.n_vars  # transcripts removed

    def test_filter_axis_0_filters_rows(self, adata_obs_groups):
        """axis=0 removes samples (obs), not transcripts (vars)."""
        result = filter_low_expressed_genes(
            adata_obs_groups,
            min_expression=10,
            filter_axis=0,
            group_col='condition',
            group_source='obs',
            lib_size_normalization=None,
            verbose=False
        )
        assert result.n_vars == adata_obs_groups.n_vars  # no vars removed
        assert result.n_obs < adata_obs_groups.n_obs     # samples removed

    def test_filter_axis_0_keeps_high_group(self, adata_obs_groups):
        result = filter_low_expressed_genes(
            adata_obs_groups,
            min_expression=10,
            filter_axis=0,
            group_col='condition',
            group_source='obs',
            lib_size_normalization=None,
            verbose=False
        )
        assert 's1' in result.obs_names
        assert 's2' in result.obs_names

    def test_filter_axis_0_drops_low_group(self, adata_obs_groups):
        result = filter_low_expressed_genes(
            adata_obs_groups,
            min_expression=10,
            filter_axis=0,
            group_col='condition',
            group_source='obs',
            lib_size_normalization=None,
            verbose=False
        )
        assert 's3' not in result.obs_names
        assert 's4' not in result.obs_names


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — CPM normalisation
# ---------------------------------------------------------------------------

class TestFilterCPMNormalization:
    def test_cpm_normalization_scales_counts(self):
        """
        With a tiny library (10 reads total), raw counts of 8 are below
        a threshold of 10 but CPM is 800,000 — so the group is kept with
        lib_size_normalization='cpm' but dropped without normalization.
        """
        counts = np.array([[5, 3]], dtype=float)  # lib_size = 8
        a = AnnData(X=counts.copy())
        a.var_names = ['t1', 't2']
        a.obs_names = ['s1']
        a.var['Synt_id'] = ['A', 'A']

        # Without CPM: group A total = 8 < 10 → dropped
        result_raw = filter_low_expressed_genes(
            a, min_expression=10, lib_size_normalization=None, verbose=False
        )
        assert result_raw.n_vars == 0

        # With CPM: CPM total >> 10 → kept
        result_cpm = filter_low_expressed_genes(
            a.copy(), min_expression=10, lib_size_normalization='cpm', verbose=False
        )
        assert result_cpm.n_vars == 2

    def test_lib_size_normalization_none_uses_raw_counts(self, adata):
        """Explicitly passing None skips normalization."""
        result = filter_low_expressed_genes(
            adata, min_expression=10, lib_size_normalization=None, verbose=False
        )
        # Group A raw sum (250–300) is above 10 → kept
        assert 't1' in result.var_names


# ---------------------------------------------------------------------------
# filter_low_expressed_genes — min_expression types
# ---------------------------------------------------------------------------

class TestMinExpressionTypes:
    def test_float_threshold(self, adata):
        result = filter_low_expressed_genes(
            adata, min_expression=10.0, lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 2

    def test_dict_threshold_per_sample(self, adata):
        """
        Per-sample thresholds.
        Group A sums: s1=300, s2=250, s3=250.
        With thresholds {'s1':10, 's2':10, 's3':10} and mode='all': A passes all → kept.
        """
        thresholds = {'s1': 10, 's2': 10, 's3': 10}
        result = filter_low_expressed_genes(
            adata, min_expression=thresholds, mode='all',
            lib_size_normalization=None, verbose=False
        )
        assert 't1' in result.var_names

    def test_dict_threshold_drops_when_one_sample_fails(self, adata):
        """
        With s3 threshold=300 and mode='all', group A (s3=250<300) is dropped.
        """
        thresholds = {'s1': 10, 's2': 10, 's3': 300}
        result = filter_low_expressed_genes(
            adata, min_expression=thresholds, mode='all',
            lib_size_normalization=None, verbose=False
        )
        assert result.n_vars == 0

    def test_callable_threshold_with_library_size(self, adata):
        """
        Callable threshold: keep group if expression >= 10% of library size.
        Group A lib-dependent threshold: ~30 (10% of ~300 lib size).
        Group A sums: 300, 250, 250 → all above ~30 → kept.
        Group B sums: 3 → below threshold → dropped.
        """
        result = filter_low_expressed_genes(
            adata,
            min_expression=lambda lib_size: lib_size * 0.1,
            library_size_dependent=True,
            lib_size_normalization=None,
            verbose=False
        )
        assert 't1' in result.var_names
        assert 't3' not in result.var_names
