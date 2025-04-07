import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple

def plot_allelic_ratios(
    adata,
    synteny_category: str,
    sample: Union[str, List[str]] = "all",
    multimapping_threshold: float = 0.5,
    ratio_type: str = "both",
    bins: int = 30,
    figsize: Tuple[int, int] = (12, 6),
    kde: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot allelic ratios for transcripts in a specific synteny category.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing transcript data
    synteny_category : str
        Synteny category to filter for
    sample : str or List[str], default="all"
        Sample(s) to plot. If list, will plot each sample separately
    multimapping_threshold : float, default=0.5
        Threshold for high multimapping ratio
    ratio_type : str, default="both"
        Type of ratio to plot: "unique", "salmon", or "both"
    bins : int, default=30
        Number of bins for the histogram
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
    kde : bool, default=True
        Whether to show KDE curve on histogram
    save_path : str, optional
        Path to save the plot. If None, plot is shown but not saved
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot(s)
    """
    # Validate parameters
    valid_ratio_types = ["unique", "salmon", "both"]
    if ratio_type not in valid_ratio_types:
        raise ValueError(f"ratio_type must be one of {valid_ratio_types}")
    
    # Make sample always a list for consistent processing
    if isinstance(sample, str):
        samples = [sample]
    else:
        samples = sample
    
    # Ensure all samples exist in obsm
    for s in samples:
        if s not in adata.obsm and s != "all":
            raise ValueError(f"Sample '{s}' not found in adata.obsm")
    
    # Filter the data for the specific synteny category
    filtered_data = adata[adata.obsm['synteny_category'] == synteny_category].copy()
    
    if len(filtered_data) == 0:
        print(f"No data found for synteny category: {synteny_category}")
        return None
    
    # Add a tag for high multimapping ratio
    filtered_data.obs['high_multimapping'] = np.any(
        filtered_data.layers['multimapping_ratio'] > multimapping_threshold, 
        axis=1
    )
    
    # Create figure with appropriate number of subplots
    if ratio_type == "both":
        n_ratio_plots = 2
    else:
        n_ratio_plots = 1
    
    n_sample_plots = len(samples)
    total_plots = n_ratio_plots * n_sample_plots

    # Calculate grid dimensions
    if total_plots <= 2:
        n_rows, n_cols = 1, total_plots
    else:
        n_cols = min(3, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    plot_idx = 0

    # Create plots for each sample and ratio type
    for sample_idx, current_sample in enumerate(samples):
        # Extract data for current sample
        if current_sample != "all":
            sample_indices = np.where(filtered_data.var_names == current_sample)[0]
        else:
            sample_indices = np.arange(filtered_data.shape[1])  # Use all columns if sample not found
                 
        if ratio_type == "unique":
            layer_name = "allelic_ratio_unique_counts"
            color = "blue"
            title_suffix = "Unique Counts"
        else:
            layer_name = "allelic_ratio_salmon_counts"
            color = "green"
            title_suffix = "Salmon Counts"
            
        # Extract allelic ratios for the current sample and layer
        allelic_ratios = filtered_data.layers[layer_name][:, sample_indices]
            
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'allelic_ratio': allelic_ratios.flatten(),
            'high_multimapping': np.repeat(filtered_data.obs['high_multimapping'].values, len(sample_indices))
        })
            
        # Drop NaN values
        plot_data = plot_data.dropna()
            
        if len(plot_data) == 0:
            print(f"No valid data for sample {current_sample}, ratio type {ratio_type}")
            continue
            
        # Plot histogram
        sns.histplot(
            data=plot_data, 
            x='allelic_ratio', 
            hue='high_multimapping',
            kde=kde,
            bins=bins,
            palette=['#1f77b4', '#ff7f0e'],
            ax=axes[plot_idx]
        )
            
        axes[plot_idx].set_title(f"{title_suffix} Allelic Ratio ({sample})")
        axes[plot_idx].set_xlabel('Allelic Ratio')
        axes[plot_idx].set_ylabel('Count')
        axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
        axes[plot_idx].set_xticks([0, 0.25, 0.5, 0.75, 1])    
        
        plot_idx += 1
    
    # Remove any empty subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    #return fig

def plot_allelic_ratios_comparison(
    adata,
    synteny_categories: List[str],
    sample: str = "all",
    ratio_layer: str = "allelic_ratio_unique_counts",
    multimapping_threshold: float = 0.5,
    bins: int = 30,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Compare allelic ratios across different synteny categories.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing transcript data
    synteny_categories : List[str]
        List of synteny categories to compare
    sample : str, default="all"
        Sample to plot
    ratio_layer : str, default="allelic_ratio_unique_counts"
        Layer containing allelic ratios to use
    multimapping_threshold : float, default=0.5
        Threshold for high multimapping ratio
    bins : int, default=30
        Number of bins for the histogram
    figsize : tuple, default=(14, 8)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the plot. If None, plot is shown but not saved
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    if sample not in adata.obsm and sample != "all":
        raise ValueError(f"Sample '{sample}' not found in adata.obsm")
    
    if ratio_layer not in adata.layers:
        raise ValueError(f"Layer '{ratio_layer}' not found in adata.layers")
    
    # Determine sample index
    if sample != "all":
        sample_indices = np.where(adata.var_names == sample)[0]
    else:
        sample_indices = np.arange(adata.shape[1])  # Use all columns if sample not found
    
    
    # Create figure
    fig, axes = plt.subplots(len(synteny_categories), 1, figsize=figsize, sharex=True)
    
    # Handle case with only one category
    if len(synteny_categories) == 1:
        axes = [axes]
    
    # Plot for each synteny category
    for idx, category in enumerate(synteny_categories):
        # Filter data for this category
        filtered_data = adata[adata.obsm['synteny_category'] == category].copy()
        
        if len(filtered_data) == 0:
            print(f"No data found for synteny category: {category}")
            axes[idx].text(0.5, 0.5, f"No data for {category}", 
                           ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        # Add multimapping tag
        filtered_data.obs['high_multimapping'] = np.any(
            filtered_data.layers['multimapping_ratio'] > multimapping_threshold, 
            axis=1
        )
        
        # Extract allelic ratios
        allelic_ratios = filtered_data.layers[ratio_layer][:, sample_indices]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'allelic_ratio': allelic_ratios.flatten(),
            'high_multimapping': np.repeat(filtered_data.obs['high_multimapping'].values, len(sample_indices))
        })
        
        # Drop NaN values
        plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            print(f"No valid data for category {category}")
            continue
        
        # Plot histogram
        sns.histplot(
            data=plot_data, 
            x='allelic_ratio', 
            hue='high_multimapping',
            kde=True,
            bins=bins,
            palette=['#1f77b4', '#ff7f0e'],
            ax=axes[idx]
        )
        
        axes[idx].set_title(f"{category} - {ratio_layer} ({sample})")
        axes[idx].set_ylabel('Count')
        axes[idx].set_xlabel('')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_xticks([0, 0.25, 0.5, 0.75, 1])        
        # Add count information
        axes[idx].text(
            0.98, 0.92, 
            f"n = {len(filtered_data)}", 
            transform=axes[idx].transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )
    
    # Add shared x-label
    fig.text(0.75, 0, 'Allelic Ratio', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    #return fig