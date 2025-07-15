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

    # Ensure all samples exist in var
    for s in samples:
        if s not in adata.var and s != "all":
            raise ValueError(f"Sample '{s}' not found in adata.var")

    # Filter the data for the specific synteny category
    filtered_data = adata[:,adata.var['synteny_category'] == synteny_category].copy()

    if len(filtered_data) == 0:
        print(f"No data found for synteny category: {synteny_category}")
        return None


    # Add a tag for high multimapping ratio
    filtered_data.var['ambiguous_counts'] = np.where(
    filtered_data.var['multimapping_ratio'] > multimapping_threshold,
    'high',
    'low')

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
            sample_indices = np.where(filtered_data.obs_names == current_sample)[0]
        else:
            sample_indices = np.arange(filtered_data.shape[0])  # Use all columns if sample not found

        if ratio_type == "unique":
            layer_name = "allelic_ratio_unique_counts"
            color = "blue"
            title_suffix = "Unique Counts"
        else:
            layer_name = "allelic_ratio_salmon_counts"
            color = "green"
            title_suffix = "Salmon Counts"

        # Extract allelic ratios for the current sample and layer
        allelic_ratios = filtered_data.layers[layer_name][sample_indices]

        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'allelic_ratio': allelic_ratios.flatten(order='F'),
            'ambiguous_counts': np.repeat(filtered_data.var['ambiguous_counts'].values, len(sample_indices))
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
            hue='ambiguous_counts',
            kde=kde,
            bins=bins,
            palette={'low':'green', 'high':'grey'},
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
    if sample not in adata.obs_names and sample != "all":
        raise ValueError(f"Sample '{sample}' not found in adata.obs_names")

    if ratio_layer not in adata.layers:
        raise ValueError(f"Layer '{ratio_layer}' not found in adata.layers")

    # Determine sample index
    if sample != "all":
        sample_indices = np.where(adata.obs_names == sample)[0]
    else:
        sample_indices = np.arange(adata.shape[0])  # Use all columns if sample not found


    # Create figure
    fig, axes = plt.subplots(len(synteny_categories), 1, figsize=figsize, sharex=True)

    # Handle case with only one category
    if len(synteny_categories) == 1:
        axes = [axes]

    # Plot for each synteny category
    for idx, category in enumerate(synteny_categories):
        # Filter data for this category
        filtered_data = adata[: ,adata.var['synteny_category'] == category].copy()

        if len(filtered_data) == 0:
            print(f"No data found for synteny category: {category}")
            axes[idx].text(0.5, 0.5, f"No data for {category}",
                           ha='center', va='center', transform=axes[idx].transAxes)
            continue

        # Add multimapping tag
        filtered_data.var['ambiguous_counts'] = np.where(
            filtered_data.var['multimapping_ratio'] > multimapping_threshold,
            'high',
            'low')

        # Extract allelic ratios
        allelic_ratios = filtered_data.layers[ratio_layer][sample_indices]

        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'allelic_ratio': allelic_ratios.flatten(order='F'),
            'ambiguous_counts': np.repeat(filtered_data.var['ambiguous_counts'].values, len(sample_indices))
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
            hue='ambiguous_counts',
            kde=True,
            bins=bins,
            palette={'low':'green', 'high':'grey'},
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



def plot_top_differential_syntelogs(results_df, n=5, figsize=(16, 12), palette=None, jitter=0.2, alpha=0.7, ylim=(0, 1), sort_by='p_value', output_file=None, sig_threshold=0.05, difference_threshold=0.05, sig_color='red'):
    """
    Plot the top n syntelogs with differential allelic ratios in a grid layout (3 plots per row).
    Syntelogs with significant differences will have their titles highlighted in red.

    Parameters
    -----------
    results_df : pd.DataFrame
        Results dataframe from test_allelic_ratios function
    n : int, optional
        Number of top syntelogs to plot (default: 5)
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (16, 12))
    palette : dict or None, optional
        Color palette for conditions (default: None, uses seaborn defaults)
    jitter : float, optional
        Amount of jitter for strip plot (default: 0.2)
    alpha : float, optional
        Transparency of points (default: 0.7)
    ylim : tuple, optional
        Y-axis limits (default: (0, 1))
    sort_by : str, optional
        Column to sort results by ('p_value', 'FDR', or 'ratio_difference') (default: 'p_value')
    output_file : str, optional
        Path to save the figure (default: None, displays figure but doesn't save)
    sig_threshold : float, optional
        Significance threshold for p-value or FDR (default: 0.05)
    sig_color : str, optional
        Color for titles of syntelogs with significant differences (default: 'red')

    Returns
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import math

    if len(results_df) == 0:
        print("No results to plot")
        return None

    # Validate sort_by parameter
    if sort_by not in ['p_value', 'FDR', 'ratio_difference']:
        print(f"Invalid sort_by parameter '{sort_by}'. Using 'p_value' instead.")
        sort_by = 'p_value'

    # Ensure FDR column exists
    if 'FDR' not in results_df.columns and sort_by == 'FDR':
        print("FDR column not found. Using p_value for sorting.")
        sort_by = 'p_value'

    # Ensure ratio_difference column exists
    if 'ratio_difference' not in results_df.columns and sort_by == 'ratio_difference':
        print("ratio_difference column not found. Using p_value for sorting.")
        sort_by = 'p_value'

    if sort_by == 'ratio_difference':
        ascending_bool = False
    else:
        ascending_bool = True

    # Get the condition names
    condition_columns = [col for col in results_df.columns if col.startswith('ratios_rep_')]
    if not condition_columns:
        print("No ratio columns found in dataframe")
        return None

    conditions = [col.replace('ratios_rep_', '') for col in condition_columns]

    # Get top n syntelogs with lowest sort_by values
    top_syntelogs = results_df.sort_values(sort_by, ascending=ascending_bool).drop_duplicates('Synt_id').head(n)['Synt_id'].unique()

    # Filter results to include only these syntelogs
    top_results = results_df[results_df['Synt_id'].isin(top_syntelogs)]

    # Calculate grid dimensions - 3 plots per row
    cols = 6
    rows = math.ceil(len(top_syntelogs) / cols)

    # Create the figure with grid layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Convert axes to flattened array for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each syntelog
    for i, synt_id in enumerate(top_syntelogs):
        # Get data for this syntelog
        synt_data = top_results[top_results['Synt_id'] == synt_id].copy()

        # Sort by allele for better visualization
        synt_data = synt_data.sort_values('allele')

        # Get stats for this syntelog (take first row since they're the same for all replicates)
        p_value = synt_data['p_value'].min()
        fdr = synt_data['FDR'].min() if 'FDR' in synt_data.columns else np.nan
        ratio_difference = synt_data.loc[synt_data['FDR'] < sig_threshold, 'ratio_difference'].max() if ('ratio_difference' in synt_data.columns and 'FDR' in synt_data.columns) else np.nan
        n_alleles = synt_data['n_alleles'].iloc[0]

        # Explode the replicate ratio columns
        explode_cols = [col for col in synt_data.columns if col.startswith('ratios_rep_')]
        synt_data_exploded = synt_data.explode(explode_cols)

        # Reshape data for seaborn
        synt_data_melted = pd.melt(
            synt_data_exploded,
            id_vars=['Synt_id', 'allele', 'transcript_id', 'FDR'] if 'FDR' in synt_data.columns else ['Synt_id', 'allele', 'transcript_id'],
            value_vars=condition_columns,
            var_name='condition',
            value_name='ratio'
        )

        # Clean up condition names
        synt_data_melted['condition'] = synt_data_melted['condition'].str.replace('ratios_rep_', '')

        # Create the stripplot
        ax = axes[i]
        sns.stripplot(
            x='allele',
            y='ratio',
            hue='condition',
            data=synt_data_melted,
            jitter=jitter,
            alpha=alpha,
            palette=palette,
            ax=ax
        )
        # get the asterisks for the FDRs
        pvalue_asterisks = [convert_pvalue_to_asterisks(x) for x in synt_data['FDR']]

        # Add mean values as horizontal lines for each allele and condition
        i = 0
        for allele in synt_data['allele'].unique():
            allele_pos = list(synt_data['allele'].unique()).index(allele)
            #xax.text(x=allele_pos-0.1 , y=0.9, s=pvalue_asterisks[i])
            i = i+1
            for j, cond in enumerate(conditions):
                mean_col = f'ratios_{cond}_mean'
                if mean_col in synt_data.columns:
                    mean_val = synt_data[synt_data['allele'] == allele][mean_col].iloc[0]
                    allele_pos = list(synt_data['allele'].unique()).index(allele)
                    ax.hlines(
                        y=mean_val,
                        xmin=allele_pos-0.2,
                        xmax=allele_pos+0.2,
                        colors=ax.get_legend().get_lines()[j].get_color(),
                        linewidth=2
                    )

        # Set title and labels
        fdr_text = f", FDR = {fdr:.2e}" if not np.isnan(fdr) else ""
        p_value_text = f", p = {p_value:.2e}"

        # Determine if this syntelog has a significant difference
        is_significant = False
        if 'FDR' in synt_data.columns and not np.isnan(fdr):
            is_significant = (fdr <= sig_threshold) & (ratio_difference > difference_threshold)
        else:
            is_significant = p_value <= sig_threshold

        # Set title color based on significance
        title_color = sig_color if is_significant else 'black'

        # Add title with optional stats and color based on significance
        ax.set_title(f"{synt_id}{fdr_text}", color=title_color)
        ax.set_xlabel('Allele')
        ax.set_ylabel('Expression Ratio')

        # Set y-limits
        ax.set_ylim(ylim)

        # Adjust legend
        ax.legend(title='Condition', loc='best')

    # Hide unused subplots if any
    for j in range(len(top_syntelogs), rows * cols):
        axes[j].set_visible(False)

    plt.tight_layout()

    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def plot_top_differential_isoforms(results_df, n=5, figsize=(16, 12), palette=None, jitter=0.2, alpha=0.7, ylim=(0, 1), sort_by='p_value', output_file=None, sig_threshold=0.05, difference_threshold=0.05, sig_color='red'):
    """
    Plot the top n genes with differential isoform usage in a grid layout (3 plots per row).
    Genes with significant differences will have their titles highlighted in red.

    Parameters
    -----------
    results_df : pd.DataFrame
        Results dataframe from test_allelic_ratios function
    n : int, optional
        Number of top genes to plot (default: 5)
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (16, 12))
    palette : dict or None, optional
        Color palette for conditions (default: None, uses seaborn defaults)
    jitter : float, optional
        Amount of jitter for strip plot (default: 0.2)
    alpha : float, optional
        Transparency of points (default: 0.7)
    ylim : tuple, optional
        Y-axis limits (default: (0, 1))
    sort_by : str, optional
        Column to sort results by ('p_value', 'FDR', or 'ratio_difference') (default: 'p_value')
    output_file : str, optional
        Path to save the figure (default: None, displays figure but doesn't save)
    sig_threshold : float, optional
        Significance threshold for p-value or FDR (default: 0.05)
    sig_color : str, optional
        Color for titles of genes with significant differences (default: 'red')

    Returns
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import math

    if len(results_df) == 0:
        print("No results to plot")
        return None

    # Validate sort_by parameter
    if sort_by not in ['p_value', 'FDR', 'ratio_difference']:
        print(f"Invalid sort_by parameter '{sort_by}'. Using 'p_value' instead.")
        sort_by = 'p_value'

    # Ensure FDR column exists
    if 'FDR' not in results_df.columns and sort_by == 'FDR':
        print("FDR column not found. Using p_value for sorting.")
        sort_by = 'p_value'

    # Ensure ratio_difference column exists
    if 'ratio_difference' not in results_df.columns and sort_by == 'ratio_difference':
        print("ratio_difference column not found. Using p_value for sorting.")
        sort_by = 'p_value'

    if sort_by == 'ratio_difference':
        ascending_bool = False
    else:
        ascending_bool = True

    # Get the condition names
    condition_columns = [col for col in results_df.columns if col.startswith('ratios_rep_')]
    if not condition_columns:
        print("No ratio columns found in dataframe")
        return None

    conditions = [col.replace('ratios_rep_', '') for col in condition_columns]

    # Get top n genes with lowest sort_by values
    top_genes = results_df.sort_values(sort_by, ascending=ascending_bool).drop_duplicates('gene_id').head(n)['gene_id'].unique()

    # Filter results to include only these syntelogs
    top_results = results_df[results_df['gene_id'].isin(top_genes)]

    # Calculate grid dimensions - 3 plots per row
    cols = 6
    rows = math.ceil(len(top_genes) / cols)

    # Create the figure with grid layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Convert axes to flattened array for easier indexing
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each gene
    for i, gene_id in enumerate(top_genes):
        # Get data for this gene
        gene_data = top_results[top_results['gene_id'] == gene_id].copy()

        # Sort by isoform number for better visualization
        gene_data = gene_data.sort_values('isoform_number')

        # Get stats for this syntelog (take first row since they're the same for all replicates)
        p_value = gene_data['p_value'].min()
        fdr = gene_data['FDR'].min() if 'FDR' in gene_data.columns else np.nan
        ratio_difference = gene_data.loc[gene_data['FDR'] < sig_threshold, 'ratio_difference'].max() if ('ratio_difference' in gene_data.columns and 'FDR' in gene_data.columns) else np.nan
        n_isoforms = gene_data['n_isoforms'].iloc[0]

        # Explode the replicate ratio columns
        explode_cols = [col for col in gene_data.columns if col.startswith('ratios_rep_')]
        gene_data_exploded = gene_data.explode(explode_cols)

        # Reshape data for seaborn
        gene_data_melted = pd.melt(
            gene_data_exploded,
            id_vars=['gene_id', 'isoform_number', 'transcript_id', 'FDR'] if 'FDR' in gene_data.columns else ['gene_id', 'isoform_number', 'transcript_id'],
            value_vars=condition_columns,
            var_name='condition',
            value_name='ratio'
        )

        # Clean up condition names
        gene_data_melted['condition'] = gene_data_melted['condition'].str.replace('ratios_rep_', '')

        # Create the stripplot
        ax = axes[i]
        sns.stripplot(
            x='isoform_number',
            y='ratio',
            hue='condition',
            data=gene_data_melted,
            jitter=jitter,
            alpha=alpha,
            palette=palette,
            ax=ax
        )


        # Add mean values as horizontal lines for each isoform and condition
        i = 0
        for isoform in gene_data['isoform_number'].unique():
            isoform_pos = list(gene_data['isoform_number'].unique()).index(isoform)
            #xax.text(x=isoform_pos-0.1 , y=0.9, s=pvalue_asterisks[i])
            i = i+1
            for j, cond in enumerate(conditions):
                mean_col = f'ratios_{cond}_mean'
                if mean_col in gene_data.columns:
                    mean_val = gene_data[gene_data['isoform_number'] == isoform][mean_col].iloc[0]
                    isoform_pos = list(gene_data['isoform_number'].unique()).index(isoform)
                    ax.hlines(
                        y=mean_val,
                        xmin=isoform_pos-0.2,
                        xmax=isoform_pos+0.2,
                        colors=ax.get_legend().get_lines()[j].get_color(),
                        linewidth=2
                    )

        # Set title and labels
        fdr_text = f", FDR = {fdr:.2e}" if not np.isnan(fdr) else ""
        p_value_text = f", p = {p_value:.2e}"

        # Determine if this gebe has a significant difference
        is_significant = False
        if 'FDR' in gene_data.columns and not np.isnan(fdr):
            is_significant = (fdr <= sig_threshold) & (ratio_difference > difference_threshold)
        else:
            is_significant = p_value <= sig_threshold

        # Set title color based on significance
        title_color = sig_color if is_significant else 'black'

        # Add title with optional stats and color based on significance
        ax.set_title(f"{gene_id}{fdr_text}", color=title_color)
        ax.set_xlabel('isoform_number')
        ax.set_ylabel('Expression Ratio')

        # Set y-limits
        ax.set_ylim(ylim)

        # Adjust legend
        ax.legend(title='Condition', loc='best')

    # Hide unused subplots if any
    for j in range(len(top_genes), rows * cols):
        axes[j].set_visible(False)

    plt.tight_layout()

    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig

# Note: The convert_pvalue_to_asterisks function was referenced but not defined in the original code.
# If needed, you would define it like this:
def convert_pvalue_to_asterisks(pvalue):
    """Convert p-values to significance asterisks notation."""
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    else:
        return ""
