"""Utilities for calculating and analyzing allelic ratios."""

import numpy as np
import pandas as pd

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
        Calculate allelic ratios for each transcript grouped by Synt_id.
        
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
        
        # Get unique Synt_ids
        unique_synt_ids = pd.unique(self.adata.obsm['Synt_id'])
        
        # Initialize ratio array with zeros
        ratio_matrix = np.zeros_like(self.adata.layers[counts_layer], dtype=float)
        
        # Calculate ratios for each Synt_id group
        for synt_id in unique_synt_ids:
            # Skip groups with Synt_id = 0 or None if needed
            if synt_id == 0 or synt_id is None:
                continue
                
            # Create mask for current Synt_id
            mask = self.adata.obsm['Synt_id'] == synt_id
            
            # Get counts for this group
            group_counts = self.adata.layers[counts_layer][mask]
            
            # Calculate total counts for this Synt_id
            total_counts = np.sum(group_counts)
            
            # Calculate ratio if total_counts > 0
            if total_counts > 0:
                ratio_matrix[mask] = self.adata.layers[counts_layer][mask] / total_counts
        
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