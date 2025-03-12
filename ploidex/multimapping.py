"""Utilities for calculating mulitmapping ratios per syntelog."""

import numpy as np
import pandas as pd

class MultimappingRatioCalculator:
    """
    Class for calculating multimapping ratios in AnnData objects.
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
        
    def calculate_ratios(self, unique_layer='unique_counts', multi_layer='ambiguous_counts'):
        """
        Calculate multimapping ratios for each transcript grouped by Synt_id.
        
        Parameters:
        -----------
        unique_layer : str, optional (default: 'unique_counts')
            Layer containing unique counts to use for ratio calculations
        multi_layer : str, optional (default: 'ambiguous_counts')
            Layer containing multimapping counts to use for ratio calculations

        Returns:
        --------
        adata : AnnData
            Updated AnnData object with multimapping ratio layer added
        """
        if self.adata is None:
            raise ValueError("No AnnData object has been set")
            
        # Make sure Synt_id is in obsm
        if 'Synt_id' not in self.adata.obsm:
            raise ValueError("'Synt_id' not found in obsm")
        
        # Get counts from specified layer
        if multi_layer not in self.adata.layers:
            raise ValueError(f"Layer '{multi_layer}' not found")

        if unique_layer not in self.adata.layers:
            raise ValueError(f"Layer '{unique_layer}' not found")
        
        # Get unique Synt_ids
        unique_synt_ids = pd.unique(self.adata.obsm['Synt_id'])
        
        # Initialize ratio array with zeros
        ratio_matrix = np.zeros_like(self.adata.layers[multi_layer], dtype=float)
        
        # Calculate ratios for each Synt_id group
        for synt_id in unique_synt_ids:
            # Skip groups with Synt_id = 0 or None if needed
            if synt_id == 0 or synt_id is None:
                continue
                
            # Create mask for current Synt_id
            mask = self.adata.obsm['Synt_id'] == synt_id
            
            # Get counts for this group
            multi_group_counts = self.adata.layers[multi_layer][mask]
            unique_group_counts = self.adata.layers[unique_layer][mask]
            
            # Calculate total unique and ambiguous counts for this Synt_id
            multi_total_counts = np.sum(multi_group_counts)
            unique_total_counts = np.sum(unique_group_counts)
            
            # Calculate ratio if total_counts > 0
            if unique_total_counts > 0:
                ratio_matrix[mask] = multi_total_counts / unique_total_counts
        
        layer_name = f'multimapping_ratio'
        
        # Add the ratios as a new layer in the AnnData object
        self.adata.layers[layer_name] = ratio_matrix
        
        return self.adata
    

    def get_ratios_for_synt_id(self, synt_id, multi_layer='multimapping_ratio'):
        """
        Get multimapping ratios for a specific Synt_id.
        
        Parameters:
        -----------
        synt_id : int or str
            The Synt_id to get ratios for
        multi_layer : str, optional
            Name of the layer containing the ratio data
            
        Returns:
        --------
        ratios : numpy array
            Array of mulitmapping values for the specified Synt_id
        """
        if multi_layer not in self.adata.layers:
            raise ValueError(f"Mulitmapping layer '{multi_layer}' not found. Calculate ratios first.")
            
        mask = self.adata.obsm['Synt_id'] == synt_id
        return self.adata.layers[multi_layer][mask]


def calculate_multi_ratios(adata, unique_layer='unique_counts', multi_layer='ambiguous_counts'):
    """
    Calculatemultimappingratios for each transcript grouped by Synt_id.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing transcript data
    unique_layer : str, optional (default: 'unique_counts')
        Layer containing counts to use for ratio calculations
    multi_layer : str, optional (default: 'ambiguous_counts')
        Layer containing counts to use for ratio calculations
        
    Returns:
    --------
    adata : AnnData
        Updated AnnData object with 'multimapping_ratio' layer added
    """
    calculator = MultimappingRatioCalculator(adata)
    return calculator.calculate_ratios(unique_layer=unique_layer, multi_layer=multi_layer)