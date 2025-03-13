"""Utilities for calculating and analyzing allelic ratios."""

import numpy as np
import pandas as pd
import scipy.sparse

class AlleleRatioCalculator:
    """
    Class for calculating and managing allelic ratios in AnnData objects.
    """
    
    def __init__(self, adata=None):
        """
        Initialize the calculator with an optional AnnData object.
        
        Parameters:
        -----------
        adata : AnnData, optional
            AnnData object containing transcript data
        """
        self.adata = adata
        
    def set_data(self, adata):
        """
        Set or update the AnnData object.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object containing transcript data
        """
        self.adata = adata
        
    def calculate_ratios(self, counts_layer='unique_counts', output_suffix=None):
        """
        Calculate allelic ratios for each transcript grouped by Synt_id,
        computing ratios independently for each column (gene/feature).
        
        Parameters:
        -----------
        counts_layer : str, optional (default: 'unique_counts')
            Layer containing counts to use for ratio calculations
        output_suffix : str, optional
            Custom suffix for output layer name. If None, uses counts_layer name
            
        Returns:
        --------
        adata : AnnData
            Updated AnnData object with allelic ratio layer added
        """
        if self.adata is None:
            raise ValueError("No AnnData object has been set")
            
        # Make sure Synt_id is in obsm
        if 'Synt_id' not in self.adata.obsm:
            raise ValueError("'Synt_id' not found in obsm")
        
        # Get counts from specified layer
        if counts_layer not in self.adata.layers:
            raise ValueError(f"Layer '{counts_layer}' not found")
        
        # Get counts array and check if it's sparse
        counts = self.adata.layers[counts_layer]
        is_sparse = scipy.sparse.issparse(counts)
        
        # Get unique Synt_ids, excluding 0 and None
        synt_ids = self.adata.obsm['Synt_id']
        valid_synt_ids = pd.unique(synt_ids[(synt_ids != 0) & (~pd.isna(synt_ids))])
        
        # Initialize ratio matrix with zeros - always use dense array for results
        ratio_matrix = np.zeros(self.adata.shape, dtype=float)
        
        # Calculate ratios for each Synt_id group
        for synt_id in valid_synt_ids:
            # Create mask for current Synt_id
            mask = synt_ids == synt_id
            mask_indices = np.where(mask)[0]
            
            # Get counts for this group
            if is_sparse:
                # For sparse matrices, extract the rows as a dense array
                group_counts = counts[mask_indices].toarray()
            else:
                group_counts = counts[mask_indices]
            
            # Calculate column sums for this Synt_id (sum for each gene/feature)
            column_totals = np.sum(group_counts, axis=0)
            
            # Only process columns with non-zero totals
            non_zero_cols = column_totals > 0
            
            # Vectorized calculation for all rows in this group
            if np.any(non_zero_cols):
                # Broadcasting division for all rows in the group at once
                group_ratios = np.zeros_like(group_counts, dtype=float)
                group_ratios[:, non_zero_cols] = group_counts[:, non_zero_cols] / column_totals[non_zero_cols]
                
                # Assign results back to the ratio matrix
                for i, idx in enumerate(mask_indices):
                    ratio_matrix[idx] = group_ratios[i]
        
        # Determine output layer name
        suffix = output_suffix or counts_layer
        layer_name = f'allelic_ratio_{suffix}'
        
        # Add the ratios as a new layer in the AnnData object
        self.adata.layers[layer_name] = ratio_matrix
        
        return self.adata
    
    def calculate_multiple_ratios(self, counts_layers=None):
        """
        Calculate allelic ratios for multiple count layers.
        
        Parameters:
        -----------
        counts_layers : list of str, optional
            List of layer names to calculate ratios for.
            If None, calculates for all layers with 'counts' in their name.
            
        Returns:
        --------
        adata : AnnData
            Updated AnnData object with allelic ratio layers added
        """
        if counts_layers is None:
            # Automatically find layers with 'counts' in their name
            counts_layers = [layer for layer in self.adata.layers.keys() if 'counts' in layer]
            
        for layer in counts_layers:
            self.calculate_ratios(counts_layer=layer)
            
        return self.adata
    
    def get_ratios_for_synt_id(self, synt_id, ratio_layer='allelic_ratio_unique_counts'):
        """
        Get allelic ratios for a specific Synt_id.
        
        Parameters:
        -----------
        synt_id : int or str
            The Synt_id to get ratios for
        ratio_layer : str, optional
            Name of the layer containing the ratio data
            
        Returns:
        --------
        ratios : numpy array
            Array of ratio values for the specified Synt_id
        """
        if ratio_layer not in self.adata.layers:
            raise ValueError(f"Ratio layer '{ratio_layer}' not found. Calculate ratios first.")
            
        mask = self.adata.obsm['Synt_id'] == synt_id
        return self.adata.layers[ratio_layer][mask]


def calculate_allelic_ratios(adata, counts_layer='unique_counts'):
    """
    Calculate allelic ratios for each transcript grouped by Synt_id.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing transcript data
    counts_layer : str, optional (default: 'unique_counts')
        Layer containing counts to use for ratio calculations
        
    Returns:
    --------
    adata : AnnData
        Updated AnnData object with 'allelic_ratio' layer added
    """
    calculator = AlleleRatioCalculator(adata)
    return calculator.calculate_ratios(counts_layer=counts_layer)