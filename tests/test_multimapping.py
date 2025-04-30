import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from polyase.multimapping import MultimappingRatioCalculator, calculate_multi_ratios

class TestMultimappingRatioCalculator(unittest.TestCase):
    
    def setUp(self):
        """Set up a simple AnnData object for testing"""
        # Create a simple test AnnData object
        n_obs = 10
        n_vars = 5
        X = np.zeros((n_obs, n_vars))
        
        # Create the AnnData object
        self.adata = AnnData(X)
        
        # Add Synt_id to obsm
        self.adata.obsm['Synt_id'] = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])
        
        # Create unique counts layer
        unique_counts = np.array([
            [10, 5, 0, 0, 0],     # Synt_id 1
            [8, 6, 0, 0, 0],      # Synt_id 1
            [12, 4, 0, 0, 0],     # Synt_id 1
            [0, 0, 15, 5, 0],     # Synt_id 2
            [0, 0, 10, 10, 0],    # Synt_id 2
            [0, 0, 0, 0, 20],     # Synt_id 3
            [0, 0, 0, 0, 15],     # Synt_id 3
            [0, 0, 0, 0, 25],     # Synt_id 3
            [30, 0, 0, 0, 0],     # Synt_id 4
            [20, 0, 0, 0, 0]      # Synt_id 4
        ])
        self.adata.layers['unique_counts'] = unique_counts
        
        # Create multimapping counts layer
        multi_counts = np.array([
            [5, 3, 0, 0, 0],      # Synt_id 1
            [4, 2, 0, 0, 0],      # Synt_id 1
            [6, 3, 0, 0, 0],      # Synt_id 1
            [0, 0, 5, 1, 0],      # Synt_id 2
            [0, 0, 3, 3, 0],      # Synt_id 2
            [0, 0, 0, 0, 6],      # Synt_id 3
            [0, 0, 0, 0, 4],      # Synt_id 3
            [0, 0, 0, 0, 5],      # Synt_id 3
            [9, 0, 0, 0, 0],      # Synt_id 4
            [6, 0, 0, 0, 0]       # Synt_id 4
        ])
        self.adata.layers['ambiguous_counts'] = multi_counts
        
        # Create calculator
        self.calculator = MultimappingRatioCalculator(self.adata)
    
    def test_initialization(self):
        """Test that the calculator initializes correctly"""
        # Test with AnnData
        calc = MultimappingRatioCalculator(self.adata)
        self.assertIs(calc.adata, self.adata)
        
        # Test without AnnData
        calc2 = MultimappingRatioCalculator()
        self.assertIsNone(calc2.adata)
        
    def test_set_data(self):
        """Test setting data after initialization"""
        calc = MultimappingRatioCalculator()
        calc.set_data(self.adata)
        self.assertIs(calc.adata, self.adata)
    
    def test_calculate_ratios(self):
        """Test calculating ratios works correctly"""
        # Calculate the ratios
        result = self.calculator.calculate_ratios()
        
        # Check that the result contains the multimapping_ratio layer
        self.assertIn('multimapping_ratio', result.layers)
        
        # Check the calculated values
        ratios = result.layers['multimapping_ratio']
        
        # Calculate expected values manually
        # For Synt_id 1: (5+4+6)+(3+2+3) / (10+8+12)+(5+6+4) = 23/45 = 0.511...
        # For Synt_id 2: (5+3)+(1+3) / (15+10)+(5+10) = 12/40 = 0.3
        # For Synt_id 3: (6+4+5) / (20+15+25) = 15/60 = 0.25
        # For Synt_id 4: (9+6) / (30+20) = 15/50 = 0.3
        
        # Check Synt_id 1 values
        expected_synt1 = 23/45
        self.assertAlmostEqual(ratios[0, 0], expected_synt1)
        self.assertAlmostEqual(ratios[1, 0], expected_synt1)
        self.assertAlmostEqual(ratios[2, 0], expected_synt1)
        
        # Check Synt_id 2 values
        expected_synt2 = 12/40
        self.assertAlmostEqual(ratios[3, 2], expected_synt2)
        self.assertAlmostEqual(ratios[4, 2], expected_synt2)
        
        # Check Synt_id 3 values
        expected_synt3 = 15/60
        self.assertAlmostEqual(ratios[5, 4], expected_synt3)
        self.assertAlmostEqual(ratios[6, 4], expected_synt3)
        self.assertAlmostEqual(ratios[7, 4], expected_synt3)
        
        # Check Synt_id 4 values
        expected_synt4 = 15/50
        self.assertAlmostEqual(ratios[8, 0], expected_synt4)
        self.assertAlmostEqual(ratios[9, 0], expected_synt4)
    
    def test_get_ratios_for_synt_id(self):
        """Test retrieving ratios for a specific Synt_id"""
        # First calculate the ratios
        self.calculator.calculate_ratios()
        
        # Get ratios for Synt_id 1
        ratios = self.calculator.get_ratios_for_synt_id(1, multi_layer='multimapping_ratio')
        
        # Expected value for Synt_id 1
        expected_value = 23/45
        
        # Check length and values
        self.assertEqual(len(ratios), 3)  # 3 items with Synt_id 1
        for ratio in ratios:
            self.assertAlmostEqual(ratio[0], expected_value)  # First column
    
    def test_error_handling(self):
        """Test that appropriate errors are raised"""
        # Test with no AnnData
        empty_calc = MultimappingRatioCalculator()
        with self.assertRaises(ValueError):
            empty_calc.calculate_ratios()
        
        # Test with invalid layer
        with self.assertRaises(ValueError):
            self.calculator.calculate_ratios(unique_layer='nonexistent_layer')
        
        with self.assertRaises(ValueError):
            self.calculator.calculate_ratios(multi_layer='nonexistent_layer')
        
        # Test getting ratios before calculating
        with self.assertRaises(ValueError):
            self.calculator.get_ratios_for_synt_id(1, multi_layer='multimapping_ratio')
    
    def test_wrapper_function(self):
        """Test the wrapper function works correctly"""
        # Use the wrapper function
        result = calculate_multi_ratios(self.adata)
        
        # Check that the multimapping_ratio layer exists
        self.assertIn('multimapping_ratio', result.layers)
        
        # Check a value to ensure calculations were done correctly
        ratios = result.layers['multimapping_ratio']
        expected_synt1 = 23/45
        self.assertAlmostEqual(ratios[0, 0], expected_synt1)

if __name__ == '__main__':
    unittest.main()