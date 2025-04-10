import unittest
import pandas as pd
import anndata as ad
import numpy as np
import os
import tempfile
from pathlib import Path
from ase_data_loader import load_ase_data

class TestAseDataLoader(unittest.TestCase):
    """Test cases for the load_ase_data function."""
    
    def setUp(self):
        """Set up temporary directory and create mock data files for testing."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.gene_counts_dir = Path(self.temp_dir.name) / "gene_counts"
        self.gene_counts_dir.mkdir(exist_ok=True)
        
        # Create mock var_obs file
        self.var_obs_file = Path(self.temp_dir.name) / "var_obs.tsv"
        var_obs_data = {
            "transcript_id": ["gene1", "gene2", "gene3", "gene4"],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [10, 20, 30, 40]
        }
        pd.DataFrame(var_obs_data).to_csv(self.var_obs_file, sep="\t", index=False)
        
        # Create mock counts file
        self.counts_file = Path(self.temp_dir.name) / "counts.tsv"
        counts_data = {
            "transcript_id": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "count1": [100, 200, 300, 400, 500],
            "count2": [10, 20, 30, 40, 50]
        }
        pd.DataFrame(counts_data).to_csv(self.counts_file, sep="\t", index=False)
        
        # Create mock gene counts files
        sample_info = {
            "sample1": "leaf",
            "sample2": "leaf",
            "sample3": "tuber"
        }
        
        for sample_id, condition in sample_info.items():
            file_path = self.gene_counts_dir / f"{sample_id}_{condition}.counts.tsv"
            # Create some random data
            counts_data = {
                "UniqueCount": [10, 20, 30, 40, 50],
                "AmbigCount": [1, 2, 3, 4, 5]
            }
            df = pd.DataFrame(counts_data, index=["gene1", "gene2", "gene3", "gene4", "gene5"])
            df.to_csv(file_path, sep="\t")
        
        self.sample_info = sample_info
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()
    
    def test_load_ase_data_basic(self):
        """Test basic functionality of load_ase_data."""
        adata = load_ase_data(
            var_obs_file=str(self.var_obs_file),
            gene_counts_dir=str(self.gene_counts_dir),
            sample_info=self.sample_info
        )
        
        # Check dimensions
        self.assertEqual(adata.X.shape, (5, 3))  # 5 genes, 3 samples
        
        # Check sample names
        self.assertListEqual(list(adata.var_names), ["sample1", "sample2", "sample3"])
        
        # Check conditions
        self.assertListEqual(list(adata.var["condition"]), ["leaf", "leaf", "tuber"])
        
        # Check layers exist
        self.assertIn("unique_counts", adata.layers)
        self.assertIn("ambiguous_counts", adata.layers)
        
        # Check obsm contains var_obs data
        self.assertIn("transcript_id", adata.obsm)
        self.assertIn("feature1", adata.obsm)
        self.assertIn("feature2", adata.obsm)
    
    def test_load_ase_data_with_counts_file(self):
        """Test load_ase_data with counts_file parameter."""
        adata = load_ase_data(
            var_obs_file=str(self.var_obs_file),
            gene_counts_dir=str(self.gene_counts_dir),
            sample_info=self.sample_info,
            counts_file=str(self.counts_file)
        )
        
        # Basic checks should still pass
        self.assertEqual(adata.X.shape, (5, 3))
        self.assertListEqual(list(adata.var_names), ["sample1", "sample2", "sample3"])
    
    def test_load_ase_data_auto_sample_info(self):
        """Test load_ase_data with automatic sample info detection."""
        adata = load_ase_data(
            var_obs_file=str(self.var_obs_file),
            gene_counts_dir=str(self.gene_counts_dir),
            sample_info=None  # Auto-detect samples and conditions
        )
        
        # Should find the same 3 samples
        self.assertEqual(adata.X.shape[1], 3)
        
        # All sample IDs should be in the detected samples
        for sample_id in self.sample_info:
            self.assertIn(sample_id, adata.var_names)
        
        # Conditions should be correctly assigned
        for i, sample_id in enumerate(adata.var_names):
            self.assertEqual(adata.var["condition"][i], self.sample_info[sample_id])
    
    def test_na_handling(self):
        """Test handling of NA values."""
        # Create a file with NaN values
        sample_id = "sample_with_na"
        condition = "leaf"
        file_path = self.gene_counts_dir / f"{sample_id}_{condition}.counts.tsv"
        counts_data = {
            "UniqueCount": [10, np.nan, 30, 40, 50],
            "AmbigCount": [1, 2, np.nan, 4, 5]
        }
        df = pd.DataFrame(counts_data, index=["gene1", "gene2", "gene3", "gene4", "gene5"])
        df.to_csv(file_path, sep="\t")
        
        # Update sample_info
        sample_info = self.sample_info.copy()
        sample_info[sample_id] = condition
        
        # Load data with custom NA filling
        fill_value = -1
        adata = load_ase_data(
            var_obs_file=str(self.var_obs_file),
            gene_counts_dir=str(self.gene_counts_dir),
            sample_info=sample_info,
            fillna=fill_value
        )
        
        # Check NA values were filled
        self.assertFalse(np.isnan(adata.X).any())
        self.assertFalse(np.isnan(adata.layers["unique_counts"]).any())
        self.assertFalse(np.isnan(adata.layers["ambiguous_counts"]).any())
        
        # Find the column index for the sample with NA
        col_idx = np.where(adata.var_names == sample_id)[0][0]
        
        # Verify that NaN at gene2 was replaced with fill_value
        self.assertEqual(adata.layers["unique_counts"][1, col_idx], fill_value)
        
        # Verify that NaN at gene3 was replaced with fill_value
        self.assertEqual(adata.layers["ambiguous_counts"][2, col_idx], fill_value)

    def test_file_not_found_handling(self):
        """Test handling of missing files."""
        # Create sample_info with a non-existent file
        sample_info = {
            "sample1": "leaf",
            "non_existent_sample": "tuber"
        }
        
        # This should print a warning but not fail
        adata = load_ase_data(
            var_obs_file=str(self.var_obs_file),
            gene_counts_dir=str(self.gene_counts_dir),
            sample_info=sample_info
        )
        
        # Should only have data for sample1
        self.assertEqual(adata.X.shape[1], 1)
        self.assertEqual(adata.var_names[0], "sample1")

if __name__ == "__main__":
    unittest.main()