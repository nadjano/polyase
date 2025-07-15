import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from isotools._transcriptome_stats import betabinom_lr_test

class TestIsoformDIUBetweenConditions(unittest.TestCase):

    def test_input_validation_adata(self):
        with self.assertRaises(ValueError):
            test_isoform_DIU_between_conditions(None)

    def test_input_validation_layer(self):
        adata = AnnData(np.random.rand(10, 10))
        with self.assertRaises(ValueError):
            test_isoform_DIU_between_conditions(adata, layer='nonexistent_layer')

    def test_input_validation_group_key(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        with self.assertRaises(ValueError):
            test_isoform_DIU_between_conditions(adata, group_key='nonexistent_group_key')

    def test_input_validation_gene_id_key(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        with self.assertRaises(ValueError):
            test_isoform_DIU_between_conditions(adata, gene_id_key='nonexistent_gene_id_key')

    def test_isoform_ratios(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        test_isoform_DIU_between_conditions(adata)
        self.assertIsNotNone(adata.layers['isoform_ratios'])

    def test_beta_binomial_likelihood_ratio_test(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        test_isoform_DIU_between_conditions(adata)
        self.assertIsNotNone(adata.uns['isoform_usage_test'])

    def test_multiple_testing_correction(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        test_isoform_DIU_between_conditions(adata)
        self.assertIsNotNone(adata.uns['isoform_usage_test']['FDR'])

    def test_storage_of_results(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        test_isoform_DIU_between_conditions(adata)
        self.assertIsNotNone(adata.uns['isoform_usage_test'])
        self.assertIsNotNone(adata.var['isoform_usage_pval'])
        self.assertIsNotNone(adata.var['isoform_usage_FDR'])

    def test_return_value_inplace_false(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        result = test_isoform_DIU_between_conditions(adata, inplace=False)
        self.assertIsInstance(result, AnnData)

    def test_return_value_inplace_true(self):
        adata = AnnData(np.random.rand(10, 10))
        adata.obs['condition'] = ['A'] * 5 + ['B'] * 5
        adata.var['gene_id'] = ['gene1'] * 5 + ['gene2'] * 5
        result = test_isoform_DIU_between_conditions(adata)
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
