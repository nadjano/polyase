import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
from typing import Dict, List, Optional, Set, Tuple, Union, Literal

# Global variable to store last dropped IDs
_last_dropped_ids = set()

def _get_group_mapping(
    adata: ad.AnnData,
    group_col: str,
    group_source: str
) -> pd.Series:
    """
    Extract group mapping from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing group information
    group_col : str
        Column name containing group IDs
    group_source : str
        Location of the group column in AnnData ('obs', 'var', 'obsm', 'varm')
        
    Returns
    -------
    pandas.Series
        Series mapping indices to their group IDs
    """
    if group_source == 'obs':
        if group_col not in adata.obs:
            raise ValueError(f"Group column '{group_col}' not found in AnnData.obs")
        return adata.obs[group_col]
    
    elif group_source == 'var':
        if group_col not in adata.var:
            raise ValueError(f"Group column '{group_col}' not found in AnnData.var")
        return adata.var[group_col]
    
    elif group_source == 'obsm':
        if group_col not in adata.obsm:
            raise ValueError(f"Group column '{group_col}' not found in AnnData.obsm")
        return pd.Series(adata.obsm[group_col], index=adata.obs_names)
    
    elif group_source == 'varm':
        if group_col not in adata.varm:
            raise ValueError(f"Group column '{group_col}' not found in AnnData.varm")
        return pd.Series(adata.varm[group_col], index=adata.var_names)
    
    else:
        raise ValueError("group_source must be one of: 'obs', 'var', 'obsm', 'varm'")

def filter_by_group_expression(
    adata: ad.AnnData,
    min_expression: Union[float, Dict[str, float]] = 1.0,
    layer: Optional[str] = None,
    group_col: str = 'Synt_id',
    group_source: str = 'var',
    mode: str = 'any',
    return_dropped: bool = False,
    copy: bool = True,
    filter_axis: Literal[0, 1] = 0,  # 0 = filter rows, 1 = filter columns
    verbose: bool = True
) -> Union[ad.AnnData, Tuple[ad.AnnData, List[str]]]:
    """
    Filter an AnnData object to remove groups with low expression across samples.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with group IDs and expression data
    min_expression : float or dict, default=1.0
        Minimum summed expression threshold for groups
        If float: same threshold applied to all samples/features
        If dict: {name: threshold} for sample/feature-specific thresholds
    layer : str or None, default=None
        Layer to use for expression values. If None, use .X
    group_col : str, default='Synt_id'
        Column name containing group IDs
    group_source : str, default='var'
        Location of the group column in AnnData ('obs', 'var', 'obsm', 'varm')
    mode : str, default='any'
        'any': Keep groups that pass threshold in any sample/feature
        'all': Keep groups that pass threshold in all samples/features
        'mean': Keep groups that pass threshold on average across samples/features
    return_dropped : bool, default=False
        If True, also return list of dropped group IDs
    copy : bool, default=True
        If True, return a copy of the filtered AnnData object
        If False, filter the AnnData object in place
    filter_axis : int, default=0
        0: Filter rows (obs) based on group expression across columns (var)
        1: Filter columns (var) based on group expression across rows (obs)
    verbose : bool, default=True
        Whether to print additional information during filtering
        
    Returns
    -------
    AnnData or tuple
        Filtered AnnData object, and optionally a list of dropped group IDs
    """
    global _last_dropped_ids
    
    # Check parameters
    if mode not in ['any', 'all', 'mean']:
        raise ValueError("mode must be one of 'any', 'all', or 'mean'")
    
    # Get group mapping
    group_mapping = _get_group_mapping(adata, group_col, group_source)
    
    # Get expression matrix from layer or .X
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in AnnData object")
        expr_matrix = adata.layers[layer]
    else:
        expr_matrix = adata.X
    
    # Convert to dense if sparse
    if scipy.sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()
    
    # Create a dataframe with expression data
    if filter_axis == 0:  # Filter rows based on group expression
        expr_df = pd.DataFrame(
            expr_matrix,
            index=adata.obs_names,
            columns=adata.var_names
        )
        # Add group column
        expr_df['group'] = group_mapping
        
        # Determine which dimension to sum over for thresholds
        threshold_dim = adata.var_names
        
    elif filter_axis == 1:  # Filter columns based on group expression
        expr_df = pd.DataFrame(
            expr_matrix.T,  # Transpose for column filtering
            index=adata.var_names,
            columns=adata.obs_names
        )
        # Add group column
        expr_df['group'] = group_mapping
        
        # Determine which dimension to sum over for thresholds
        threshold_dim = adata.obs_names
        
    else:
        raise ValueError("filter_axis must be 0 (filter rows) or 1 (filter columns)")
    
    # Group by group ID and sum expression
    grouped_expr = expr_df.groupby('group').sum()
    
    # Convert threshold to dictionary if it's a scalar
    if isinstance(min_expression, (int, float)):
        min_expression = {sample: min_expression for sample in threshold_dim}
    
    # Determine which groups to keep based on mode
    if mode == 'any':
        # Keep groups that pass threshold in any sample/feature
        keep_groups = grouped_expr.apply(lambda row: any(row[item] >= min_expression[item] 
                                            for item in threshold_dim), axis=1)
    elif mode == 'all':
        # Keep groups that pass threshold in all samples/features
        keep_groups = grouped_expr.apply(lambda row: all(row[item] >= min_expression[item] 
                                            for item in threshold_dim), axis=1)
    elif mode == 'mean':
        # Keep groups that pass threshold on average
        keep_groups = grouped_expr.mean(axis=1) >= np.mean(list(min_expression.values()))
    
    # Get group IDs to keep
    keep_group_ids = keep_groups[keep_groups].index.tolist()
    
    # Store dropped IDs for reference
    dropped_group_ids = set(group_mapping.unique()) - set(keep_group_ids)
    _last_dropped_ids = dropped_group_ids
    
    # Filter the AnnData object
    if filter_axis == 0:  # Filter rows
        keep_indices = group_mapping.isin(keep_group_ids)
    else:  # Filter columns
        keep_indices = group_mapping.isin(keep_group_ids)
    
    if verbose:
        print(f"Filtered out {len(dropped_group_ids)} groups")
        print(f"Kept {keep_indices.sum()} / {len(keep_indices)} items")
    
    if copy:
        if filter_axis == 0:
            filtered_adata = adata[keep_indices].copy()
        else:
            filtered_adata = adata[:, keep_indices].copy()
    else:
        # Filter in place
        if filter_axis == 0:
            adata._inplace_subset_obs(keep_indices)
        else:
            adata._inplace_subset_var(keep_indices)
        filtered_adata = adata
    
    if return_dropped:
        return filtered_adata, list(dropped_group_ids)
    
    return filtered_adata

def get_last_dropped_ids() -> Set[str]:
    """
    Get the set of group IDs that were dropped in the last filtering operation.
    
    Returns
    -------
    Set[str]
        Set of dropped group IDs
    """
    global _last_dropped_ids
    return _last_dropped_ids

def get_group_expression(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    group_col: str = 'Synt_id',
    group_source: str = 'obsm',
    normalize: bool = False,
    axis: Literal[0, 1] = 0  # 0 = group rows, 1 = group columns
) -> pd.DataFrame:
    """
    Calculate the expression of each group across samples or features.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with group IDs
    layer : str or None, default=None
        Layer to use for expression values. If None, use .X
    group_col : str, default='Synt_id'
        Column name containing group IDs
    group_source : str, default='obsm'
        Location of the group column in AnnData ('obs', 'var', 'obsm', 'varm')
    normalize : bool, default=False
        If True, normalize expression values by number of elements per group
    axis : int, default=0
        0: Group rows (obs) and calculate expression across columns (var)
        1: Group columns (var) and calculate expression across rows (obs)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with groups as rows and samples/features as columns
    """
    # Get group mapping
    group_mapping = _get_group_mapping(adata, group_col, group_source)
    
    # Get expression matrix from layer or .X
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in AnnData object")
        expr_matrix = adata.layers[layer]
    else:
        expr_matrix = adata.X
    
    # Convert to dense if sparse
    if scipy.sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()
    
    # Create a dataframe with expression data
    if axis == 0:  # Group rows
        expr_df = pd.DataFrame(
            expr_matrix,
            index=adata.obs_names,
            columns=adata.var_names
        )
    else:  # Group columns
        expr_df = pd.DataFrame(
            expr_matrix.T,  # Transpose for column grouping
            index=adata.var_names,
            columns=adata.obs_names
        )
    
    # Add group column
    expr_df['group'] = group_mapping
    
    # Group by group ID and sum expression
    grouped_expr = expr_df.groupby('group').sum()
    
    if normalize:
        # Count elements per group
        counts = group_mapping.value_counts()
        # Normalize by dividing each row by the number of elements
        for group_id in grouped_expr.index:
            if group_id in counts:  # Check if the group exists in counts
                grouped_expr.loc[group_id] = grouped_expr.loc[group_id] / counts[group_id]
    
    return grouped_expr

def summarize_groups(
    adata: ad.AnnData,
    group_col: str = 'Synt_id',
    group_source: str = 'obsm',
) -> pd.DataFrame:
    """
    Summarize groups in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with group IDs
    group_col : str, default='Synt_id'
        Column name containing group IDs
    group_source : str, default='obsm'
        Location of the group column in AnnData ('obs', 'var', 'obsm', 'varm')
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with summary statistics for each group
    """
    # Get group mapping
    group_mapping = _get_group_mapping(adata, group_col, group_source)
    
    # Count elements per group
    counts = group_mapping.value_counts().sort_values(ascending=False)
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'count': counts,
        'percentage': (counts / len(group_mapping) * 100).round(2)
    })
    
    return summary

def filter_low_expression(adata, min_expression=1.0, counts_layer='unique_counts', group_col='Synt_id', mode='any'):
    """
    Filter transcripts with low expression based on synteny groups.
    
    Parameters
    -----------
    adata : AnnData
        AnnData object containing transcript data
    min_expression : float, optional (default: 1.0)
        Minimum expression threshold for synteny groups
    counts_layer : str, optional (default: 'unique_counts')
        Layer containing counts to use for filtering
    group_col : str, optional (default: 'Synt_id')
        Column name containing group IDs
    mode: str, optional (default: 'any')
        'any': Keep groups that pass threshold in any sample/feature
        'all': Keep groups that pass threshold in all samples/features
        'mean': Keep groups that pass threshold on average across samples/features
        
    Returns
    --------
    adata : AnnData
        Filtered AnnData object
    """
    return filter_by_group_expression(
        adata, 
        min_expression=min_expression, 
        layer=counts_layer,
        group_col=group_col, 
        group_source='obsm',
        mode=mode
    )