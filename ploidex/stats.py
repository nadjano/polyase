def test_allelic_ratios_within_conditions(adata, layer="unique_counts", test_condition="control", inplace=True):
    """
    Test if alleles of a gene have unequal expression and store results in AnnData object.
    
    Parameters
    -----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer containing count data (default: "unique_counts")
    test_condition : str, optional
        Variable column name containing condition for testing within (default: "control")
    inplace : bool, optional
        Whether to modify the input AnnData object or return a copy (default: True)
        
    Returns
    --------
    AnnData or None
        If inplace=False, returns modified copy of AnnData; otherwise returns None
        Results are stored in:
        - adata.uns['allelic_ratio_test']: Complete test results as DataFrame
        - adata.var['allelic_ratio_pval']: P-values for each allele
        - adata.var['allelic_ratio_FDR']: FDR-corrected p-values for each allele
    pd.DataFrame
        Results of statistical tests for each syntelog
    """
    import pandas as pd
    import numpy as np
    import re
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData

    # Validate inputs
    if not isinstance(adata, AnnData):
        raise ValueError("Input adata must be an AnnData object")
    
    # Check if layer exists
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object")
    
    
    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()
        
    # Get counts and metadata
    counts = adata.layers[layer].copy()  # Create a copy to avoid modifying original
    
    # Ensure allelic ratio layer exists
    if "allelic_ratio_unique_counts" not in adata.layers:
        raise ValueError("Layer 'allelic_ratio_unique_counts' not found in AnnData object")
    allelic_ratio_counts = adata.layers["allelic_ratio_unique_counts"].copy()
    
    # Check for syntelog IDs
    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    synt_ids = adata.var["Synt_id"]
    
    # Check for transcript IDs
    if not adata.var_names.any():
        raise ValueError("'transcript_id' not found in adata.var_names")
    transcript_ids = adata.var_names
    
    # Check conditions
    if test_condition not in adata.obs['condition'].unique() and test_condition != "all":
        raise ValueError(f"Condition '{test_condition}' not found in adata.obs['condition']")
    
   
    
    unique_synt_ids = np.unique(synt_ids)
    
    # Prepare results dataframe
    results = []
    
    # Create empty arrays for storing p-values in adata.obsm
    pvals = np.full(adata.n_vars, np.nan)
    fdr_pvals = np.full(adata.n_vars, np.nan)
    ratio_diff = np.full(adata.n_vars, np.nan)
    
    # Create empty arrays for mean ratios per condition
    mean_ratio_cond1 = np.full(adata.n_vars, np.nan)
    mean_ratio_cond2 = np.full(adata.n_vars, np.nan)
    
    # Track progress
    total_syntelogs = len(unique_synt_ids)
    processed = 0
    
    # Process each syntelog
    for synt_id in unique_synt_ids:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing syntelog {processed}/{total_syntelogs}")
            
        # Find alleles (observations) belonging to this syntelog
        allele_indices = np.where(synt_ids == synt_id)[0]

        # Skip if fewer than 2 alleles found (need at least 2 for ratio testing)
        if len(allele_indices) < 2:
            continue
            
        for allele_idx, allele_pos in enumerate(allele_indices):
            allele_counts = []
            condition_total = []
            allelic_ratios = {}
            
            
            # Get samples for this condition
            if test_condition == "all":
                condition_indices = np.arange(counts.shape[0])
            else:
                # Get samples for this condition
                condition_indices = np.where(adata.obs['condition'] == test_condition)[0]
                    
            # Extract counts for these alleles and samples
            condition_counts = counts[np.ix_(condition_indices, allele_indices)]

            # Sum across samples to get total counts per allele for this condition
            total_counts = np.sum(condition_counts, axis=1)
                
            # Get allelic ratios for this condition
            condition_ratios = allelic_ratio_counts[np.ix_(condition_indices, allele_indices)]
        
            # Append arrays for total counts                
            condition_total.append(total_counts)

            # Append array for this specific allele's counts
            allele_counts.append(condition_counts[:,allele_idx])
      
            # Store ratios for this test condition
            allelic_ratios = condition_ratios[:,allele_idx]
            # generate balanced allele counts based on condition total counts
            # balanced counts need to be integers for the test
            balanced_counts = [np.round(x * 1/len(allele_indices)) for x in condition_total]
            allele_counts.append(balanced_counts[0])
            # add the total counts again for the balanced counts
  
            condition_total.append(total_counts)
            # Run the beta-binomial likelihood ratio test
            try:
                test_result = betabinom_lr_test(allele_counts, condition_total)
                p_value, ratio_stats = test_result[0], test_result[1]
                # if p_value is np.nan:
                #     print(allele_counts, condition_total)
                # Calculate absolute difference in mean ratios between conditions
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing syntelog {synt_id}, allele {allele_idx}: {str(e)}")
                continue
            
            # Get transcript ID and parse allele info
            transcript_id = transcript_ids[allele_pos]
            
            # Extract allele number from transcript ID
            try:
                allele_match = re.search(r'\dG', transcript_id)
                if allele_match:
                    allele_num = allele_match.group(0).split('G')[0]
                else:
                    allele_num = f"{allele_idx+1}"  # Fallback if regex fails
            except:
                allele_num = f"{allele_idx+1}"  # Fallback if regex fails
            
            # Store p-value in the arrays we created
            pvals[allele_pos] = p_value
            ratio_diff[allele_pos] = ratio_difference
            mean_ratio_cond1[allele_pos] = ratio_stats[0]
            mean_ratio_cond2[allele_pos] = ratio_stats[2]
            
            # Store results for each replicate
            #for replicate in range(len(allelic_ratios[unique_conditions[0]])):
            results.append({
                    'Synt_id': synt_id,
                    'allele': allele_num,
                    'transcript_id': transcript_id,
                    'p_value': p_value,
                    'ratio_difference': ratio_difference,
                    'n_alleles': len(allele_indices),
                    f'ratios_{test_condition}_mean': ratio_stats[0],
                    f'ratios_rep_{test_condition}': allelic_ratios
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction if we have results
    if len(results_df) > 0:
        # PROBLEM: p_vale is nan sometimes, replace with 1 for now
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')
        
        # Map FDR values back to the individual alleles
        # Group by transcript_id and take the first FDR value (they should be the same for all replicates)
        fdr_map = results_df.groupby('transcript_id')['FDR'].first().to_dict()
        
        # Update the FDR array
        for i, transcript_id in enumerate(transcript_ids):
            if transcript_id in fdr_map:
                fdr_pvals[i] = fdr_map[transcript_id]
    
    # Store results in the AnnData object
    adata.uns['allelic_ratio_test'] = results_df
    adata.var['allelic_ratio_pval'] = pvals
    adata.var['allelic_ratio_FDR'] = fdr_pvals
    adata.var['allelic_ratio_difference'] = ratio_diff
    adata.var[f'allelic_ratio_mean_{test_condition}'] = mean_ratio_cond1
    adata.var[f'allelic_ratio_mean_{test_condition}'] = mean_ratio_cond2
    
    # Group by Synt_id and take mininum FDR value and max ratio difference
    grouped_results = results_df.groupby('Synt_id').min("FDR")
    grouped_results= results_df.groupby('Synt_id').agg({
    'FDR': 'min',
    'ratio_difference': 'max'  # Assuming this is the correct column name
        })
    # Print summary
    significant_results = grouped_results[(grouped_results['FDR'] < 0.005) & (grouped_results['ratio_difference'] > 0.1)]
    print(f"Found {len(significant_results)} from {len(grouped_results)} syntelogs with at least one significantly different allele (FDR < 0.05 and ratio difference > 0.1)")
    
    # Return AnnData object if not inplace
    if not inplace:
        return adata
    else:
        return results_df
    


def test_allelic_ratios_between_conditions(adata, layer="unique_counts", group_key="condition", inplace=True):
    """
    Test if allelic ratios change between conditions and store results in AnnData object.
    
    Parameters
    -----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer containing count data (default: "unique_counts")
    group_key : str, optional
        Variable column name containing condition information (default: "condition")
    inplace : bool, optional
        Whether to modify the input AnnData object or return a copy (default: True)
        
    Returns
    --------
    AnnData or None
        If inplace=False, returns modified copy of AnnData; otherwise returns None
        Results are stored in:
        - adata.uns['allelic_ratio_test']: Complete test results as DataFrame
        - adata.var['allelic_ratio_pval']: P-values for each allele
        - adata.var['allelic_ratio_FDR']: FDR-corrected p-values for each allele
    pd.DataFrame
        Results of statistical tests for each syntelog
    """
    import pandas as pd
    import numpy as np
    import re
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData

    # Validate inputs
    if not isinstance(adata, AnnData):
        raise ValueError("Input adata must be an AnnData object")
    
    # Check if layer exists
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object")
    
    # Check if group_key exists in var
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")
    
    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()
        
    # Get counts and metadata
    counts = adata.layers[layer].copy()  # Create a copy to avoid modifying original
    
    # Ensure allelic ratio layer exists
    if "allelic_ratio_unique_counts" not in adata.layers:
        raise ValueError("Layer 'allelic_ratio_unique_counts' not found in AnnData object")
    allelic_ratio_counts = adata.layers["allelic_ratio_unique_counts"].copy()
    
    # Check for syntelog IDs
    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    synt_ids = adata.var["Synt_id"]
    
    # Check for transcript IDs
    if not adata.var_names.any():
        raise ValueError("'transcript_id' not found in adata.var_names")
    transcript_ids = adata.var_names
    
    # Check conditions
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")
    conditions = adata.obs[group_key].values
    
    # Get unique conditions and syntelog IDs
    unique_conditions = np.unique(conditions)
    if len(unique_conditions) != 2:
        raise ValueError(f"Need exactly 2 conditions, found {len(unique_conditions)}: {unique_conditions}")
    
    unique_synt_ids = np.unique(synt_ids)
    
    # Prepare results dataframe
    results = []
    
    # Create empty arrays for storing p-values in adata.obsm
    pvals = np.full(adata.n_vars, np.nan)
    fdr_pvals = np.full(adata.n_vars, np.nan)
    ratio_diff = np.full(adata.n_vars, np.nan)
    
    # Create empty arrays for mean ratios per condition
    mean_ratio_cond1 = np.full(adata.n_vars, np.nan)
    mean_ratio_cond2 = np.full(adata.n_vars, np.nan)
    
    # Track progress
    total_syntelogs = len(unique_synt_ids)
    processed = 0
    
    # Process each syntelog
    for synt_id in unique_synt_ids:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing syntelog {processed}/{total_syntelogs}")
            
        # Find alleles (observations) belonging to this syntelog
        allele_indices = np.where(synt_ids == synt_id)[0]

        # Skip if fewer than 2 alleles found (need at least 2 for ratio testing)
        if len(allele_indices) < 2:
            continue
            
        for allele_idx, allele_pos in enumerate(allele_indices):
            allele_counts = []
            condition_total = []
            allelic_ratios = {}
            
            for condition_idx, condition in enumerate(unique_conditions):
                # Get samples for this condition
                condition_indices = np.where(conditions == condition)[0]
                
                # Extract counts for these alleles and samples
                condition_counts = counts[np.ix_(condition_indices, allele_indices)]

                # Sum across samples to get total counts per allele for this condition
                total_counts = np.sum(condition_counts, axis=1)
                
                # Get allelic ratios for this condition
                condition_ratios = allelic_ratio_counts[np.ix_(condition_indices, allele_indices)]
        
                # Append arrays for total counts
                condition_total.append(total_counts)

                # Append array for this specific allele's counts
                allele_counts.append(condition_counts[:,allele_idx])
                #print(condition_ratios[allele_idx])
                # Store ratios for this condition
                allelic_ratios[condition] = condition_ratios[:,allele_idx]
            
            # Run the beta-binomial likelihood ratio test
            try:
                test_result = betabinom_lr_test(allele_counts, condition_total)
                p_value, ratio_stats = test_result[0], test_result[1]
                
                # Calculate absolute difference in mean ratios between conditions
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing syntelog {synt_id}, allele {allele_idx}: {str(e)}")
                continue
            
            # Get transcript ID and parse allele info
            transcript_id = transcript_ids[allele_pos]
            
            # Extract allele number from transcript ID
            try:
                allele_match = re.search(r'\dG', transcript_id)
                if allele_match:
                    allele_num = allele_match.group(0).split('G')[0]
                else:
                    allele_num = f"{allele_idx+1}"  # Fallback if regex fails
            except:
                allele_num = f"{allele_idx+1}"  # Fallback if regex fails
            
            # Store p-value in the arrays we created
            pvals[allele_pos] = p_value
            ratio_diff[allele_pos] = ratio_difference
            mean_ratio_cond1[allele_pos] = ratio_stats[0]
            mean_ratio_cond2[allele_pos] = ratio_stats[2]
            
            # Store results for each replicate
            #for replicate in range(len(allelic_ratios[unique_conditions[0]])):
            results.append({
                    'Synt_id': synt_id,
                    'allele': allele_num,
                    'transcript_id': transcript_id,
                    'p_value': p_value,
                    'ratio_difference': ratio_difference,
                    'n_alleles': len(allele_indices),
                    f'ratios_{unique_conditions[0]}_mean': ratio_stats[0],
                    f'ratios_rep_{unique_conditions[0]}': allelic_ratios[unique_conditions[0]],
                    f'ratios_{unique_conditions[1]}_mean': ratio_stats[2],
                    f'ratios_rep_{unique_conditions[1]}': allelic_ratios[unique_conditions[1]]
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction if we have results
    if len(results_df) > 0:
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')
        
        # Map FDR values back to the individual alleles
        # Group by transcript_id and take the first FDR value (they should be the same for all replicates)
        fdr_map = results_df.groupby('transcript_id')['FDR'].first().to_dict()
        
        # Update the FDR array
        for i, transcript_id in enumerate(transcript_ids):
            if transcript_id in fdr_map:
                fdr_pvals[i] = fdr_map[transcript_id]
    
    # Store results in the AnnData object
    adata.uns['allelic_ratio_test'] = results_df
    adata.var['allelic_ratio_pval'] = pvals
    adata.var['allelic_ratio_FDR'] = fdr_pvals
    adata.var['allelic_ratio_difference'] = ratio_diff
    adata.var[f'allelic_ratio_mean_{unique_conditions[0]}'] = mean_ratio_cond1
    adata.var[f'allelic_ratio_mean_{unique_conditions[1]}'] = mean_ratio_cond2
    
    # Group by Synt_id and take mininum FDR value
    grouped_results = results_df.groupby('Synt_id').min("FDR")
    # Print summary
    significant_results = grouped_results[(grouped_results['FDR'] < 0.05)]
    print(f"Found {len(significant_results)} from {len(grouped_results)} syntelogs with at least one significantly different allelic ratio (FDR < 0.05)")
    
    # Return AnnData object if not inplace
    if not inplace:
        return adata
    else:
        return results_df
    


def get_top_differential_syntelogs(results_df, n=5, sort_by='p_value', fdr_threshold=0.05, ratio_threshold=0.1):
    """
    Get the top n syntelogs with differential allelic ratios.
    
    Parameters
    -----------
    results_df : pd.DataFrame
        Results dataframe from test_allelic_ratios function
    n : int, optional
        Number of top syntelogs to return (default: 5)
    sort_by : str, optional
        Column to sort results by ('p_value', 'FDR', or 'ratio_difference') (default: 'p_value')
    fdr_threshold : float, optional
        Maximum FDR to consider a result significant (default: 0.05)
        
    Returns
    --------
    pd.DataFrame
        Filtered dataframe containing only the top n syntelogs
    """
    if len(results_df) == 0:
        print("No results to filter")
        return results_df
    
    # Validate sort_by parameter
    if sort_by not in ['p_value', 'FDR', 'ratio_difference']:
        print(f"Invalid sort_by parameter '{sort_by}'. Using 'p_value' instead.")
        sort_by = 'p_value'
    
    if sort_by == 'ratio_difference':
        sort_bool = False
    else:
        sort_bool = True
    
    # Apply FDR filter if column exists
    if 'FDR' in results_df.columns:
        sig_results = results_df[(results_df['FDR'] <= fdr_threshold) & (results_df['ratio_difference'] >= ratio_threshold)]
    
        if len(sig_results) == 0:
            print(f"No results with FDR <= {fdr_threshold} and ratio_difference >= {ratio_threshold}. Using all results.")
            sig_results = results_df
    else:
        sig_results = results_df
    
    # Get top n syntelogs
    top_syntelogs = sig_results.sort_values(sort_by, ascending=sort_bool).drop_duplicates('Synt_id').head(n)['Synt_id'].unique()
    
    # Return filtered dataframe
    return results_df[results_df['Synt_id'].isin(top_syntelogs)]



