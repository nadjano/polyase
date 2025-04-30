import pandas as pd
import anndata as ad
import os
from pathlib import Path
import numpy as np

def load_ase_data(
    var_obs_file,
    gene_counts_dir,
    sample_info=None,
    counts_file=None,
    fillna=0
):
    """
    Load allele-specific expression data from long-read RNAseq.
    
    Parameters
    -----------
    var_obs_file : str
        Path to the variant observations file
    gene_counts_dir : str
        Directory containing the gene counts files
    sample_info : dict, optional
        Dictionary mapping sample IDs to their conditions (e.g., {'SRR14993892': 'leaf', 'SRR14993896': 'tuber'})
        If None, all TSV files in gene_counts_dir will be used and conditions will be extracted from filenames
    counts_file : str, optional
        Path to additional counts file (salmon merged transcript counts). Optional.
    fillna : int or float, optional
        Value to fill NA values with
        
    Returns
    --------
    anndata.AnnData
        AnnData object containing the processed data
    """
    # Load variant observations file
    var_obs = pd.read_csv(var_obs_file, delimiter="\t", index_col=0)

    # Load additional counts file if provided
    if counts_file:
        additional_counts = pd.read_csv(counts_file, delimiter="\t")
        # You can add code here to use the additional_counts data if needed
    
    # Find sample files and their conditions if sample_info not provided
    if sample_info is None:
        sample_info = {}
        gene_counts_files = list(Path(gene_counts_dir).glob("*.counts.tsv"))
        
        for file_path in gene_counts_files:
            # Extract sample ID and condition from filename (assuming format like "SRR14993892_leaf.counts.tsv")
            filename = file_path.stem  # Gets filename without extension
            parts = filename.split('_')
            sample_id = parts[0]
            condition = parts[1] if len(parts) > 1 else "unknown"
            sample_info[sample_id] = condition
    
    sample_ids = list(sample_info.keys())
    
    # Load ambiguous and unique counts for each sample
    ambig_counts_dfs = []
    unique_counts_dfs = []
    
    for sample_id in sample_ids:
        condition = sample_info[sample_id]
        file_path = os.path.join(gene_counts_dir, f"{sample_id}_{condition}.counts.tsv")
        
        # Check if the file exists, otherwise try without condition
        if not os.path.exists(file_path):
            alternate_files = list(Path(gene_counts_dir).glob(f"{sample_id}*.counts.tsv"))
            if alternate_files:
                file_path = str(alternate_files[0])
            else:
                print(f"Warning: No file found for sample {sample_id}")
                continue
                
        counts_df = pd.read_csv(file_path, delimiter="\t", index_col=0)
        
        ambig_counts_dfs.append(counts_df['AmbigCount'])
        unique_counts_dfs.append(counts_df['UniqueCount'])
    
    # Concatenate ambiguous counts across samples
    ambig_counts = pd.concat(ambig_counts_dfs, axis=1)
    ambig_counts = ambig_counts.loc[~ambig_counts.index.duplicated(keep='first')]
    
    # Concatenate unique counts across samples
    unique_counts = pd.concat(unique_counts_dfs, axis=1)
    unique_counts = unique_counts.loc[~unique_counts.index.duplicated(keep='first')]

    # Rename columns
    ambig_counts.columns = sample_ids
    unique_counts.columns = sample_ids
    
    var_obs =var_obs.reindex(ambig_counts.index, fill_value= np.nan)

    # Reidex to also have the transcrits that where not maped
    ambig_counts = ambig_counts.reindex(var_obs.index, fill_value=fillna)
    unique_counts = unique_counts.reindex(var_obs.index, fill_value=fillna)

    # Fill NA values
    ambig_counts.fillna(fillna, inplace=True)
    unique_counts.fillna(fillna, inplace=True)

    # Create AnnData object
    adata = ad.AnnData(
        X=unique_counts[sample_ids].T,
        var=var_obs,
        obs=pd.DataFrame(index=sample_ids),
    )
    
    # Add conditions to var
    conditions = [sample_info[sample_id] for sample_id in sample_ids]
    adata.obs["condition"] = conditions
    
    # Add sample IDs to var_names
    adata.obs_names = sample_ids
    
    # Add layers of unique and ambiguous counts
    adata.layers["unique_counts"] = unique_counts.T.to_numpy()
    adata.layers["ambiguous_counts"] = ambig_counts.T.to_numpy()
    
    return adata