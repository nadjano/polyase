"""
stats.py
====================================
The stats module of polyase project
"""
def test_allelic_ratios_within_conditions(adata, layer="unique_counts", test_condition="control", inplace=True, FDR_cutoff=0.05, ratio_diff_cutoff=0.1):
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
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.1)

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

    # Get allelic ratio counts
    allelic_ratio_layer_name = 'allelic_ratio_' + layer
    if allelic_ratio_layer_name not in adata.layers:
        raise ValueError(f"Layer '{allelic_ratio_layer_name}' not found in AnnData object")
    allelic_ratio_counts = adata.layers[allelic_ratio_layer_name].copy()

    # Get CPM data if available
    cpm_layer_name = layer.replace('counts', 'cpm')  # e.g., unique_counts -> unique_cpm
    cpm_counts = None
    if cpm_layer_name in adata.layers:
        cpm_counts = adata.layers[cpm_layer_name].copy()
        print(f"Using CPM data from layer: {cpm_layer_name}")
    else:
        print(f"CPM layer '{cpm_layer_name}' not found, CPM values will not be included")

    # Check for syntelog IDs
    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    synt_ids = adata.var["Synt_id"]

    if "functional_annotation" in adata.var:
        functional_annotations = adata.var["functional_annotation"]
    else:
        functional_annotations = "Missing annotation"
        print("No functional annotations found in adata.var, skipping functional annotation processing.")

    # Check for transcript IDs
    if not adata.var_names.any():
        raise ValueError("'transcript_id' not found in adata.var_names")
    gene_ids = adata.var_names
    transcript_ids = adata.var['transcript_id']

    # Check conditions
    if test_condition not in adata.obs['condition'].unique() and test_condition != "all":
        raise ValueError(f"Condition '{test_condition}' not found in adata.obs['condition']")

    unique_synt_ids = np.unique(synt_ids)

    # Prepare results dataframe
    results = []

    # Create empty arrays for storing p-values in adata.var
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

            total_counts = np.round(total_counts)  # Round to avoid floating point issues in the test

            # Get allelic ratios for this condition
            condition_ratios = allelic_ratio_counts[np.ix_(condition_indices, allele_indices)]

            # Get CPM values for this condition if available
            condition_cpm = None
            if cpm_counts is not None:
                condition_cpm = cpm_counts[np.ix_(condition_indices, allele_indices)]

            # Append arrays for total counts
            condition_total.append(total_counts)

            # Append array for this specific allele's counts
            allele_counts.append(np.round(condition_counts[:,allele_idx]))  # Round to avoid floating

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
                # Calculate absolute difference in mean ratios between conditions
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing syntelog {synt_id}, allele {allele_idx}: {str(e)}")
                continue

            # Get gene ID and parse allele info
            gene_id = gene_ids[allele_pos]
            # Get transcript ID
            transcript_id = transcript_ids.iloc[allele_pos]
            functional_annotation = functional_annotations.iloc[allele_pos]

            haplotype = adata.var['haplotype'].iloc[allele_indices[allele_idx]]
            allele_num = str(haplotype) if pd.notna(haplotype) else str(allele_idx + 1)

            # Store p-value in the arrays we created
            pvals[allele_pos] = p_value
            ratio_diff[allele_pos] = ratio_difference
            mean_ratio_cond1[allele_pos] = ratio_stats[0]
            mean_ratio_cond2[allele_pos] = ratio_stats[2]

            # Prepare result dictionary
            result_dict = {
                'Synt_id': synt_id,
                'allele': allele_num,
                'functional_annotation': functional_annotation,
                'gene_id': gene_id,
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                'n_alleles': len(allele_indices),
                f'ratios_{test_condition}_mean': np.mean(allelic_ratios[total_counts > 0]),
                f'ratios_rep_{test_condition}': allelic_ratios
            }

            # Add CPM values if available
            if condition_cpm is not None:
                # Calculate mean CPM for this allele in this condition
                allele_cpm_values = condition_cpm[:, allele_idx]
                mean_cpm = np.mean(allele_cpm_values)

                result_dict[f'cpm_{test_condition}_mean'] = mean_cpm
                result_dict[f'cpm_rep_{test_condition}'] = allele_cpm_values

            results.append(result_dict)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction if we have results
    if len(results_df) > 0:
        # PROBLEM: p_value is nan sometimes, replace with 1 for now
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')

        # Map FDR values back to the individual alleles
        # Group by transcript_id and take the first FDR value (they should be the same for all replicates)
        fdr_map = results_df.groupby('gene_id')['FDR'].first().to_dict()

        # Update the FDR array
        for i, gene_id in enumerate(gene_ids):
            if gene_id in fdr_map:
                fdr_pvals[i] = fdr_map[gene_id]

    # Store results in the AnnData object
    adata.uns['allelic_ratio_test'] = results_df
    adata.var['allelic_ratio_pval'] = pvals
    adata.var['allelic_ratio_FDR'] = fdr_pvals
    adata.var['allelic_ratio_difference'] = ratio_diff
    adata.var[f'allelic_ratio_mean_{test_condition}'] = mean_ratio_cond1

    # Filter for significant results and then group by synt_id
    if len(results_df) > 0:
        results_df_sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        results_df_sig_grouped = results_df_sig.groupby('Synt_id').agg({
            'FDR': list,
            'ratio_difference': list
        })
        grouped_results = results_df.groupby('Synt_id').agg({
            'FDR': 'min',
            'ratio_difference': 'max'
        })   
        print(f"Found {len(results_df_sig_grouped)} from {len(grouped_results)} syntelogs with at least one significantly different allele (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")

    # Return AnnData object if not inplace
    if not inplace:
        return adata
    else:
        return results_df


def test_allelic_ratios_pairwise(adata, layer="unique_counts", test_condition="control", inplace=True, FDR_cutoff=0.05, ratio_diff_cutoff=0.1):
    """
    Test all pairwise combinations of alleles within a syntelog for differential expression.

    For each syntelog, every pair of alleles is tested against each other using a
    beta-binomial likelihood ratio test. This is in contrast to
    test_allelic_ratios_within_conditions, which tests each allele against a balanced
    (equal) expectation.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer containing count data (default: "unique_counts")
    test_condition : str, optional
        Condition to subset samples for testing, or "all" to use all samples (default: "control")
    inplace : bool, optional
        Whether to modify the input AnnData object or return a copy (default: True)
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.1)

    Returns
    --------
    AnnData or pd.DataFrame
        If inplace=False, returns modified copy of AnnData; otherwise returns results DataFrame.
        Results are stored in:
        - adata.uns['allelic_ratio_pairwise_test']: Complete pairwise test results as DataFrame
        - adata.var['allelic_ratio_pairwise_min_pval']: Minimum p-value across all pairs per allele
        - adata.var['allelic_ratio_pairwise_min_FDR']: Corresponding FDR-corrected value
    """
    import pandas as pd
    import numpy as np
    import re
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData

    if not isinstance(adata, AnnData):
        raise ValueError("Input adata must be an AnnData object")

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object")

    if not inplace:
        adata = adata.copy()

    counts = adata.layers[layer].copy()

    allelic_ratio_layer_name = 'allelic_ratio_' + layer
    if allelic_ratio_layer_name not in adata.layers:
        raise ValueError(f"Layer '{allelic_ratio_layer_name}' not found in AnnData object")
    allelic_ratio_counts = adata.layers[allelic_ratio_layer_name].copy()

    cpm_layer_name = layer.replace('counts', 'cpm')
    cpm_counts = None
    if cpm_layer_name in adata.layers:
        cpm_counts = adata.layers[cpm_layer_name].copy()
        print(f"Using CPM data from layer: {cpm_layer_name}")
    else:
        print(f"CPM layer '{cpm_layer_name}' not found, CPM values will not be included")

    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    synt_ids = adata.var["Synt_id"]

    if "functional_annotation" in adata.var:
        functional_annotations = adata.var["functional_annotation"]
    else:
        functional_annotations = "Missing annotation"
        print("No functional annotations found in adata.var, skipping functional annotation processing.")

    if not adata.var_names.any():
        raise ValueError("'transcript_id' not found in adata.var_names")
    gene_ids = adata.var_names
    transcript_ids = adata.var['transcript_id']

    if test_condition != "all" and test_condition not in adata.obs['condition'].unique():
        raise ValueError(f"Condition '{test_condition}' not found in adata.obs['condition']")

    if test_condition == "all":
        condition_indices = np.arange(counts.shape[0])
    else:
        condition_indices = np.where(adata.obs['condition'] == test_condition)[0]

    unique_synt_ids = np.unique(synt_ids)
    results = []

    # Per-allele arrays to track the minimum p-value across all pairs
    min_pvals = np.full(adata.n_vars, np.nan)
    min_fdr_pvals = np.full(adata.n_vars, np.nan)

    total_syntelogs = len(unique_synt_ids)
    processed = 0

    for synt_id in unique_synt_ids:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing syntelog {processed}/{total_syntelogs}")

        allele_indices = np.where(synt_ids == synt_id)[0]

        if len(allele_indices) < 2:
            continue

        # Subset counts to condition samples and alleles in this syntelog
        condition_counts = counts[np.ix_(condition_indices, allele_indices)]
        condition_ratios = allelic_ratio_counts[np.ix_(condition_indices, allele_indices)]
        condition_cpm = cpm_counts[np.ix_(condition_indices, allele_indices)] if cpm_counts is not None else None

        # Total counts per sample across all alleles in this syntelog
        total_counts = np.round(np.sum(condition_counts, axis=1))

        def get_haplotype_num(allele_pos_local):
            haplotype = adata.var['haplotype'].iloc[allele_indices[allele_pos_local]]
            return str(haplotype) if pd.notna(haplotype) else str(allele_pos_local + 1)

        # Test all pairwise combinations of alleles
        for i, j in combinations(range(len(allele_indices)), 2):
            allele_pos_i = allele_indices[i]
            allele_pos_j = allele_indices[j]

            counts_i = np.round(condition_counts[:, i])
            counts_j = np.round(condition_counts[:, j])
            pair_total = np.round(counts_i + counts_j)

            try:
                test_result = betabinom_lr_test([counts_i, counts_j], [pair_total, pair_total])
                p_value, ratio_stats = test_result[0], test_result[1]
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing syntelog {synt_id}, alleles {i} vs {j}: {str(e)}")
                continue

            hap_i = get_haplotype_num(i)
            hap_j = get_haplotype_num(j)

            # Ensure allele_i always has the lower allele number
            try:
                if int(hap_i) > int(hap_j):
                    hap_i, hap_j = hap_j, hap_i
                    i, j = j, i
                    allele_pos_i, allele_pos_j = allele_pos_j, allele_pos_i
            except ValueError:
                if hap_i > hap_j:
                    hap_i, hap_j = hap_j, hap_i
                    i, j = j, i
                    allele_pos_i, allele_pos_j = allele_pos_j, allele_pos_i

            gene_id_i = gene_ids[allele_pos_i]
            gene_id_j = gene_ids[allele_pos_j]
            functional_annotation_i = functional_annotations.iloc[allele_pos_i]

            result_dict = {
                'Synt_id': synt_id,
                'allele_i': hap_i,
                'allele_j': hap_j,
                'gene_id_i': gene_id_i,
                'gene_id_j': gene_id_j,
                'functional_annotation': functional_annotation_i,
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                'n_alleles': len(allele_indices),
                f'ratio_allele_{hap_i}_mean': np.mean(condition_ratios[:, i][total_counts > 0]),
                f'ratio_allele_{hap_j}_mean': np.mean(condition_ratios[:, j][total_counts > 0]),
                f'ratios_rep_allele_{hap_i}': condition_ratios[:, i],
                f'ratios_rep_allele_{hap_j}': condition_ratios[:, j],
            }

            if condition_cpm is not None:
                result_dict[f'cpm_allele_{hap_i}_mean'] = np.mean(condition_cpm[:, i])
                result_dict[f'cpm_allele_{hap_j}_mean'] = np.mean(condition_cpm[:, j])

            results.append(result_dict)

            # Track minimum p-value per allele across all its pairs
            for allele_pos in [allele_pos_i, allele_pos_j]:
                if np.isnan(min_pvals[allele_pos]) or p_value < min_pvals[allele_pos]:
                    min_pvals[allele_pos] = p_value

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')

        # Map minimum FDR back per allele
        fdr_map = {}
        for _, row in results_df.iterrows():
            for gid in [row['gene_id_i'], row['gene_id_j']]:
                if gid not in fdr_map or row['FDR'] < fdr_map[gid]:
                    fdr_map[gid] = row['FDR']

        for i, gene_id in enumerate(gene_ids):
            if gene_id in fdr_map:
                min_fdr_pvals[i] = fdr_map[gene_id]

        results_df_sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        sig_syntelogs = results_df_sig['Synt_id'].nunique()
        total_syntelogs_tested = results_df['Synt_id'].nunique()
        print(f"Found {sig_syntelogs} from {total_syntelogs_tested} syntelogs with at least one significantly different allele pair (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")

    adata.uns['allelic_ratio_pairwise_test'] = results_df
    adata.var['allelic_ratio_pairwise_min_pval'] = min_pvals
    adata.var['allelic_ratio_pairwise_min_FDR'] = min_fdr_pvals

    if not inplace:
        return adata
    else:
        return results_df


def test_allelic_ratios_between_conditions(adata, layer="unique_counts", group_key="condition", inplace=True, FDR_cutoff=0.05, ratio_diff_cutoff=0.1):
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
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.1)

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
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData

    # Validate inputs
    if not isinstance(adata, AnnData):
        raise ValueError("Input adata must be an AnnData object")

    # Check if layer exists
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object")

    # Check if group_key exists in obs
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")

    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()

    # Get counts and metadata
    counts = adata.layers[layer].copy()  # Create a copy to avoid modifying original

    # Ensure allelic ratio layer exists
    allelic_ratio_layer_name = 'allelic_ratio_' + layer
    if allelic_ratio_layer_name not in adata.layers:
        raise ValueError(f"Layer '{allelic_ratio_layer_name}' not found in AnnData object")
    allelic_ratio_counts = adata.layers[allelic_ratio_layer_name].copy()

    # Get CPM data if available
    cpm_layer_name = layer.replace('counts', 'cpm')  # e.g., unique_counts -> unique_cpm
    cpm_counts = None
    if cpm_layer_name in adata.layers:
        cpm_counts = adata.layers[cpm_layer_name].copy()
        print(f"Using CPM data from layer: {cpm_layer_name}")
    else:
        print(f"CPM layer '{cpm_layer_name}' not found, CPM values will not be included")

    # Check for syntelog IDs
    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    synt_ids = adata.var["Synt_id"]

    if "functional_annotation" in adata.var:
        functional_annotations = adata.var["functional_annotation"]
    else:
        functional_annotations = "Missing annotation"
        print("No functional annotations found in adata.var, skipping functional annotation processing.")

    if "gene_id" in adata.var:
        gene_ids = adata.var['gene_id']

    # Check for transcript IDs
    if not adata.var_names.any():
        raise ValueError("'transcript_id' not found in adata.var_names")
    gene_ids = adata.var_names
    transcript_ids = adata.var['transcript_id']

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

    # Create empty arrays for storing p-values in adata.var
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
            cpm_values = {}

            for condition_idx, condition in enumerate(unique_conditions):
                # Get samples for this condition
                condition_indices = np.where(conditions == condition)[0]

                # Extract counts for these alleles and samples
                condition_counts = counts[np.ix_(condition_indices, allele_indices)]

                # Sum across samples to get total counts per allele for this condition
                total_counts = np.sum(condition_counts, axis=1)

                total_counts = np.round(total_counts)  # Round to avoid floating point issues in the test

                # Get allelic ratios for this condition
                condition_ratios = allelic_ratio_counts[np.ix_(condition_indices, allele_indices)]

                # Get CPM values for this condition if available
                if cpm_counts is not None:
                    condition_cpm = cpm_counts[np.ix_(condition_indices, allele_indices)]
                    cpm_values[condition] = condition_cpm[:, allele_idx]

                # Append arrays for total counts
                condition_total.append(total_counts)

                # Append array for this specific allele's counts
                allele_counts.append(np.round(condition_counts[:,allele_idx]))  # Round to avoid floating point issues in the test

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
            gene_id = gene_ids[allele_pos]
            transcript_id = transcript_ids.iloc[allele_pos]
            functional_annotation = functional_annotations.iloc[allele_pos]

            haplotype = adata.var['haplotype'].iloc[allele_indices[allele_idx]]
            allele_num = str(haplotype) if pd.notna(haplotype) else str(allele_idx + 1)

            # Store p-value in the arrays we created
            pvals[allele_pos] = p_value
            ratio_diff[allele_pos] = ratio_difference
            mean_ratio_cond1[allele_pos] = ratio_stats[0]
            mean_ratio_cond2[allele_pos] = ratio_stats[2]

            # Prepare result dictionary
            result_dict = {
                'Synt_id': synt_id,
                'gene_id': gene_id,
                'transcript_id': transcript_id,
                'functional_annotation': functional_annotation,
                'allele': allele_num,
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                'n_alleles': len(allele_indices),
                f'ratios_{unique_conditions[0]}_mean': np.mean(allelic_ratios[unique_conditions[0]][condition_total[0] > 0]),
                f'ratios_rep_{unique_conditions[0]}': allelic_ratios[unique_conditions[0]],
                f'ratios_{unique_conditions[1]}_mean': np.mean(allelic_ratios[unique_conditions[1]][condition_total[1] > 0]),
                f'ratios_rep_{unique_conditions[1]}': allelic_ratios[unique_conditions[1]]
            }

            # Add CPM values if available
            if cpm_counts is not None:
                for condition in unique_conditions:
                    if condition in cpm_values:
                        mean_cpm = np.mean(cpm_values[condition])
                        result_dict[f'cpm_{condition}_mean'] = mean_cpm
                        result_dict[f'cpm_rep_{condition}'] = cpm_values[condition]

            results.append(result_dict)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction if we have results
    if len(results_df) > 0:
        # PROBLEM: p_value is nan sometimes, replace with 1 for now
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')

        # Map FDR values back to the individual alleles
        # Group by gene_id and take the first FDR value (they should be the same for all replicates)
        fdr_map = results_df.groupby('gene_id')['FDR'].first().to_dict()

        # Update the FDR array
        for i, gene_id in enumerate(gene_ids):
            if gene_id in fdr_map:
                fdr_pvals[i] = fdr_map[gene_id]

    # Store results in the AnnData object
    adata.uns['allelic_ratio_test'] = results_df
    adata.var['allelic_ratio_pval'] = pvals
    adata.var['allelic_ratio_FDR'] = fdr_pvals
    adata.var['allelic_ratio_difference'] = ratio_diff
    adata.var[f'allelic_ratio_mean_{unique_conditions[0]}'] = mean_ratio_cond1
    adata.var[f'allelic_ratio_mean_{unique_conditions[1]}'] = mean_ratio_cond2

    # Find significant results and group by synt_id
    if len(results_df) > 0:
        results_df_sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        results_df_sig_grouped = results_df_sig.groupby('Synt_id').agg({
            'FDR': list,
            'ratio_difference': list
        })
        grouped_results = results_df.groupby('Synt_id').agg({
            'FDR': 'min',
            'ratio_difference': 'max'
        })   
        print(f"Found {len(results_df_sig_grouped)} from {len(grouped_results)} syntelogs with at least one significantly different allele (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")


    # Return AnnData object if not inplace
    if not inplace:
        return adata
    else:
        return results_df

def test_pairwise_allele_response_between_conditions(adata, layer="unique_counts", group_key="condition", inplace=True, FDR_cutoff=0.05, ratio_diff_cutoff=0.1):
    """
    Test if pairs of alleles within a syntelog respond differently between conditions.

    For each pair of alleles (i, j) within a syntelog, tests whether the ratio
    count_i / (count_i + count_j) changes between conditions using a beta-binomial
    likelihood ratio test. A significant result means the two alleles have divergent
    fold-changes between conditions (e.g., allele1 up-regulated, allele2 down-regulated).

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
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.1)

    Returns
    --------
    AnnData or None
        If inplace=False, returns modified copy of AnnData; otherwise returns None
        Results are stored in:
        - adata.uns['pairwise_allele_response_test']: Complete test results as DataFrame
        - adata.var['pairwise_response_min_pval']: Min p-value across all pairs per allele
        - adata.var['pairwise_response_min_fdr']: Min FDR across all pairs per allele
    pd.DataFrame
        Results of pairwise statistical tests
    """
    import pandas as pd
    import numpy as np
    import re
    from collections import defaultdict
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData

    if not isinstance(adata, AnnData):
        raise ValueError("Input adata must be an AnnData object")
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object")
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")
    if "Synt_id" not in adata.var:
        raise ValueError("'Synt_id' not found in adata.var")
    if "transcript_id" not in adata.var:
        raise ValueError("'transcript_id' not found in adata.var")
    if "haplotype" not in adata.var:
        raise ValueError("'haplotype' not found in adata.var")

    if not inplace:
        adata = adata.copy()

    counts = adata.layers[layer].copy()

    allelic_ratio_layer_name = 'allelic_ratio_' + layer  # e.g. unique_counts -> allelic_ratio_unique_counts
    allelic_ratio_counts = None
    if allelic_ratio_layer_name in adata.layers:
        allelic_ratio_counts = adata.layers[allelic_ratio_layer_name].copy()
    else:
        print(f"Allelic ratio layer '{allelic_ratio_layer_name}' not found, ratio replicates will not be included")

    cpm_layer_name = layer.replace('counts', 'cpm')
    cpm_counts = None
    if cpm_layer_name in adata.layers:
        cpm_counts = adata.layers[cpm_layer_name].copy()
        print(f"Using CPM data from layer: {cpm_layer_name}")
    else:
        print(f"CPM layer '{cpm_layer_name}' not found, log2FC values will not be included")

    synt_ids = adata.var["Synt_id"]
    gene_ids = adata.var_names
    transcript_ids = adata.var['transcript_id']
    haplotypes = adata.var['haplotype']
    functional_annotations = adata.var.get("functional_annotation", None)

    conditions = adata.obs[group_key].values
    unique_conditions = np.unique(conditions)
    if len(unique_conditions) != 2:
        raise ValueError(f"Need exactly 2 conditions, found {len(unique_conditions)}: {unique_conditions}")

    cond0, cond1 = unique_conditions
    cond0_idx = np.where(conditions == cond0)[0]
    cond1_idx = np.where(conditions == cond1)[0]

    unique_synt_ids = np.unique(synt_ids)
    results = []
    min_pvals = np.full(adata.n_vars, np.nan)

    def _parse_hap(hap_str, fallback):
        try:
            m = re.search(r'hap(\d+)', hap_str)
            return m.group(1) if m else fallback
        except Exception:
            return fallback

    total_syntelogs = len(unique_synt_ids)
    processed = 0

    for synt_id in unique_synt_ids:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing syntelog {processed}/{total_syntelogs}")

        allele_indices = np.where(synt_ids == synt_id)[0]
        if len(allele_indices) < 2:
            continue

        for idx_i, idx_j in combinations(range(len(allele_indices)), 2):
            pos_i = allele_indices[idx_i]
            pos_j = allele_indices[idx_j]

            # Build per-condition count arrays for allele i and total (i+j)
            allele_counts_i = []
            totals_ij = []
            for cond_indices in (cond0_idx, cond1_idx):
                ci = np.asarray(counts[np.ix_(cond_indices, [pos_i])]).flatten()
                cj = np.asarray(counts[np.ix_(cond_indices, [pos_j])]).flatten()
                allele_counts_i.append(np.round(ci))
                totals_ij.append(np.round(ci + cj))

            # Skip if either condition has no counts for this pair
            if any(np.all(t == 0) for t in totals_ij):
                continue

            try:
                test_result = betabinom_lr_test(allele_counts_i, totals_ij)
                p_value, ratio_stats = test_result[0], test_result[1]
                if len(ratio_stats) < 3:
                    continue
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing syntelog {synt_id}, pair ({idx_i}, {idx_j}): {str(e)}")
                continue

            result_dict = {
                'Synt_id': synt_id,
                'gene_id_i': gene_ids[pos_i],
                'gene_id_j': gene_ids[pos_j],
                'transcript_id_i': transcript_ids.iloc[pos_i],
                'transcript_id_j': transcript_ids.iloc[pos_j],
                'haplotype_i': _parse_hap(haplotypes.iloc[pos_i], str(idx_i + 1)),
                'haplotype_j': _parse_hap(haplotypes.iloc[pos_j], str(idx_j + 1)),
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                f'ratio_{cond0}': ratio_stats[0],
                f'ratio_{cond1}': ratio_stats[2],
            }

            if functional_annotations is not None:
                result_dict['functional_annotation_i'] = functional_annotations.iloc[pos_i]
                result_dict['functional_annotation_j'] = functional_annotations.iloc[pos_j]

            # Per-replicate allelic ratios for each allele in each condition
            if allelic_ratio_counts is not None:
                for pos, label in ((pos_i, 'i'), (pos_j, 'j')):
                    r0 = np.asarray(allelic_ratio_counts[np.ix_(cond0_idx, [pos])]).flatten()
                    r1 = np.asarray(allelic_ratio_counts[np.ix_(cond1_idx, [pos])]).flatten()
                    result_dict[f'ratios_rep_{label}_{cond0}'] = r0
                    result_dict[f'ratios_rep_{label}_{cond1}'] = r1
                    result_dict[f'ratios_{label}_{cond0}_mean'] = np.mean(r0)
                    result_dict[f'ratios_{label}_{cond1}_mean'] = np.mean(r1)

            # log2FC per allele from CPM; store per-replicate arrays for plotting
            if cpm_counts is not None:
                pseudocount = 1e-3
                for pos, label in ((pos_i, 'i'), (pos_j, 'j')):
                    vals0 = np.asarray(cpm_counts[np.ix_(cond0_idx, [pos])]).flatten()
                    vals1 = np.asarray(cpm_counts[np.ix_(cond1_idx, [pos])]).flatten()
                    result_dict[f'cpm_{label}_{cond0}_mean'] = np.mean(vals0)
                    result_dict[f'cpm_{label}_{cond1}_mean'] = np.mean(vals1)
                    result_dict[f'cpm_rep_{label}_{cond0}'] = vals0
                    result_dict[f'cpm_rep_{label}_{cond1}'] = vals1
                    result_dict[f'log2FC_{label}'] = np.log2((np.mean(vals1) + pseudocount) / (np.mean(vals0) + pseudocount))
                result_dict['log2FC_difference'] = result_dict['log2FC_i'] - result_dict['log2FC_j']

            results.append(result_dict)

            # Track minimum p-value per allele position
            for pos in (pos_i, pos_j):
                if np.isnan(min_pvals[pos]) or p_value < min_pvals[pos]:
                    min_pvals[pos] = p_value

    results_df = pd.DataFrame(results)
    min_fdr = np.full(adata.n_vars, np.nan)

    if len(results_df) > 0:
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')

        # Map minimum FDR back to each allele position
        gene_min_fdr = defaultdict(lambda: np.nan)
        for _, row in results_df.iterrows():
            for col in ('gene_id_i', 'gene_id_j'):
                gid = row[col]
                if np.isnan(gene_min_fdr[gid]) or row['FDR'] < gene_min_fdr[gid]:
                    gene_min_fdr[gid] = row['FDR']
        for i, gid in enumerate(gene_ids):
            if gid in gene_min_fdr:
                min_fdr[i] = gene_min_fdr[gid]

    adata.uns['pairwise_allele_response_test'] = results_df
    adata.var['pairwise_response_min_pval'] = min_pvals
    adata.var['pairwise_response_min_fdr'] = min_fdr

    if len(results_df) > 0:
        sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        print(f"Found {sig['Synt_id'].nunique()} from {results_df['Synt_id'].nunique()} syntelogs with at least one allele pair showing divergent response (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")

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

def test_isoform_DIU_between_conditions(adata, layer="unique_counts", group_key="condition", gene_id_key="gene_id", inplace=True, FDR_cutoff=0.05, ratio_diff_cutoff=0.2):
    """
    Test if isoform usage ratios change between conditions and store results in AnnData object.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer containing count data (default: "unique_counts")
    group_key : str, optional
        Variable column name containing condition information (default: "condition")
    gene_id_key : str, optional
        Variable column name containing gene ID information (default: "gene_id")
    inplace : bool, optional
        Whether to modify the input AnnData object or return a copy (default: True)
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.2)

    Returns
    --------
    AnnData or None
        If inplace=False, returns modified copy of AnnData; otherwise returns None
        Results are stored in:
        - adata.uns['isoform_usage_test']: Complete test results as DataFrame
        - adata.var['isoform_usage_pval']: P-values for each isoform
        - adata.var['isoform_usage_FDR']: FDR-corrected p-values for each isoform
    pd.DataFrame
        Results of statistical tests for each gene
    pd.DataFrame
        Plotting results table with one row per replicate, condition, isoform ratio, and transcript
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

    # Check if group_key exists in obs
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs")

    # Check if gene_id_key exists in var
    if gene_id_key not in adata.var:
        raise ValueError(f"Gene ID key '{gene_id_key}' not found in adata.var")

    if "functional_annotation" in adata.var:
        functional_annotations = adata.var["functional_annotation"]
    else:
        functional_annotations = "Missing annotation"
        print("No functional annotations found in adata.var, skipping functional annotation processing.")

    # Work on a copy if not inplace
    if not inplace:
        adata = adata.copy()

    no_counts_isoform = 0

    # Get counts and metadata
    counts = adata.layers[layer].copy()  # Create a copy to avoid modifying original
    gene_ids = adata.var[gene_id_key]
    transcript_ids = adata.var_names
    conditions = adata.obs[group_key].values

    # Calculate library sizes (total counts per sample) for CPM calculation
    library_sizes = np.sum(counts, axis=1)

    # Calculate CPM (Counts Per Million)
    cpm = np.zeros_like(counts, dtype=float)
    for i, lib_size in enumerate(library_sizes):
        if lib_size > 0:
            cpm[i, :] = (counts[i, :] / lib_size) * 1e6
        else:
            cpm[i, :] = 0

    # Get unique conditions and gene IDs
    unique_conditions = np.unique(conditions)
    if len(unique_conditions) != 2:
        raise ValueError(f"Need exactly 2 conditions, found {len(unique_conditions)}: {unique_conditions}")

    unique_gene_ids = np.unique(gene_ids)

    # Calculate isoform ratios for each gene
    print("Calculating isoform ratios...")
    isoform_ratios = np.zeros_like(counts, dtype=float)

    for gene_id in unique_gene_ids:
        # Find isoforms (variables) belonging to this gene
        isoform_indices = np.where(gene_ids == gene_id)[0]

        if len(isoform_indices) < 2:
            continue  # Skip genes with only one isoform

        # Calculate total gene expression for each sample
        gene_totals = np.sum(counts[:, isoform_indices], axis=1, keepdims=True)

        # Avoid division by zero
        gene_totals[gene_totals == 0] = 1

        # Calculate isoform ratios
        isoform_ratios[:, isoform_indices] = counts[:, isoform_indices] / gene_totals

    # Store isoform ratios in a new layer
    adata.layers['isoform_ratios'] = isoform_ratios

    # Prepare results dataframe
    results = []
    plotting_results = []  # New list for plotting table

    # Create empty arrays for storing p-values in adata.var
    pvals = np.full(adata.n_vars, np.nan)
    fdr_pvals = np.full(adata.n_vars, np.nan)
    ratio_diff = np.full(adata.n_vars, np.nan)

    # Create empty arrays for mean ratios per condition
    mean_ratio_cond1 = np.full(adata.n_vars, np.nan)
    mean_ratio_cond2 = np.full(adata.n_vars, np.nan)

    # Track progress
    total_genes = len(unique_gene_ids)
    processed = 0

    # Process each gene
    for gene_id in unique_gene_ids:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing gene {processed}/{total_genes}")

        # Find isoforms (variables) belonging to this gene
        isoform_indices = np.where(gene_ids == gene_id)[0]

        # Skip if fewer than 2 isoforms found (need at least 2 for ratio testing)
        if len(isoform_indices) < 2:
            continue

        # Test each isoform within this gene
        for isoform_idx, isoform_pos in enumerate(isoform_indices):
            isoform_counts = []
            gene_total_counts = []
            isoform_ratios_per_condition = {}

            for condition_idx, condition in enumerate(unique_conditions):
                # Get samples for this condition
                condition_indices = np.where(conditions == condition)[0]

                # Extract counts for all isoforms of this gene in this condition
                condition_gene_counts = counts[np.ix_(condition_indices, isoform_indices)]

                # Get total gene counts per sample (sum across all isoforms)
                condition_gene_totals = np.sum(condition_gene_counts, axis=1)

                # Get this specific isoform's counts
                condition_isoform_counts = counts[np.ix_(condition_indices, [isoform_pos])].flatten()

                # Store data for beta-binomial test
                isoform_counts.append(condition_isoform_counts)
                gene_total_counts.append(condition_gene_totals)

                # Calculate isoform ratios for this condition
                condition_ratios = np.divide(condition_isoform_counts, condition_gene_totals,
                                           out=np.zeros_like(condition_isoform_counts, dtype=float),
                                           where=condition_gene_totals!=0)
                isoform_ratios_per_condition[condition] = condition_ratios

            # Check for zero counts before running the test
            all_isoform_counts = np.concatenate(isoform_counts)
            all_gene_totals = np.concatenate(gene_total_counts)

            if np.any(all_isoform_counts == 0) or np.any(all_gene_totals == 0) or np.all(all_isoform_counts == 0) or np.all(all_gene_totals == 0):
                no_counts_isoform = no_counts_isoform + 1
                continue

            # Handle different replicate numbers by padding shorter arrays
            max_replicates = max(len(arr) for arr in isoform_counts)

            # Pad arrays to have the same length
            padded_isoform_counts = []
            padded_gene_total_counts = []

            for i, (iso_counts, total_counts) in enumerate(zip(isoform_counts, gene_total_counts)):
                if len(iso_counts) < max_replicates:
                    # Calculate the mean for padding (to maintain the same statistical properties)
                    iso_mean = np.mean(iso_counts) if len(iso_counts) > 0 else 0
                    total_mean = np.mean(total_counts) if len(total_counts) > 0 else 0

                    # Pad with mean values
                    padded_iso = np.concatenate([iso_counts,
                                               np.full(max_replicates - len(iso_counts), iso_mean)])
                    padded_total = np.concatenate([total_counts,
                                                 np.full(max_replicates - len(total_counts), total_mean)])
                else:
                    padded_iso = iso_counts
                    padded_total = total_counts

                padded_isoform_counts.append(padded_iso.astype(int)) # to integer for testing function
                padded_gene_total_counts.append(padded_total.astype(int))

            # Run the beta-binomial likelihood ratio test
            try:
                test_result = betabinom_lr_test(padded_isoform_counts, padded_gene_total_counts)
                p_value, ratio_stats = test_result[0], test_result[1]

                # Calculate absolute difference in mean ratios between conditions
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                print(f"Error testing gene {gene_id}, isoform {isoform_idx}: {str(e)}")
                continue

            # Get transcript ID
            transcript_id = transcript_ids[isoform_pos]
            functional_annotation = functional_annotations.iloc[isoform_pos]

            # Store p-value in the arrays we created
            pvals[isoform_pos] = p_value
            ratio_diff[isoform_pos] = ratio_difference
            mean_ratio_cond1[isoform_pos] = ratio_stats[0]
            mean_ratio_cond2[isoform_pos] = ratio_stats[2]

            # Store results
            results.append({
                'transcript_id': transcript_id,
                'isoform_number': isoform_idx + 1,
                'gene_id': gene_id,
                'functional_annotation': functional_annotation,
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                'n_isoforms': len(isoform_indices),
                f'ratios_{unique_conditions[0]}_mean': np.mean(isoform_ratios_per_condition[unique_conditions[0]][gene_total_counts[0] > 0]),
                f'ratios_rep_{unique_conditions[0]}': isoform_ratios_per_condition[unique_conditions[0]],
                f'ratios_{unique_conditions[1]}_mean': np.mean(isoform_ratios_per_condition[unique_conditions[1]][gene_total_counts[1] > 0]),
                f'ratios_rep_{unique_conditions[1]}': isoform_ratios_per_condition[unique_conditions[1]]
            })

            # Create plotting results table - one row per replicate, condition, ratio, transcript
            for condition in unique_conditions:
                condition_indices = np.where(conditions == condition)[0]
                sample_names = adata.obs_names[condition_indices]
                ratios = isoform_ratios_per_condition[condition]

                # Get CPM values for this isoform in this condition
                isoform_cpm_values = cpm[condition_indices, isoform_pos]

                # Get raw counts for this isoform in this condition
                isoform_count_values = counts[condition_indices, isoform_pos]

                for rep_idx, (sample_name, ratio_value, cpm_value, count_value) in enumerate(zip(sample_names, ratios, isoform_cpm_values, isoform_count_values)):
                    plotting_results.append({
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        'functional_annotation': functional_annotation,
                        'isoform_number': isoform_idx + 1,
                        'condition': condition,
                        'replicate': rep_idx + 1,
                        'sample_name': sample_name,
                        'isoform_ratio': ratio_value,
                        f'{layer}_raw': count_value,  # Raw counts
                        f'{layer}_cpm': cpm_value,
                        'p_value': p_value,
                        'ratio_difference': ratio_difference,
                        'n_isoforms': len(isoform_indices)
                    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    plotting_df = pd.DataFrame(plotting_results)

    # Multiple testing correction if we have results
    if len(results_df) > 0:
        # Handle NaN p-values
        results_df['p_value'] = results_df['p_value'].fillna(1)
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df = results_df.sort_values('p_value')

        # Map FDR values back to the individual isoforms
        fdr_map = results_df.groupby('transcript_id')['FDR'].first().to_dict()

        # Update the FDR array
        for i, transcript_id in enumerate(transcript_ids):
            if transcript_id in fdr_map:
                fdr_pvals[i] = fdr_map[transcript_id]

        # Add FDR to plotting dataframe
        plotting_df['FDR'] = plotting_df['transcript_id'].map(fdr_map).fillna(1.0)

    # Store results in the AnnData object
    adata.uns['isoform_usage_test'] = results_df
    adata.var['isoform_usage_pval'] = pvals
    adata.var['isoform_usage_FDR'] = fdr_pvals
    adata.var['isoform_usage_difference'] = ratio_diff
    adata.var[f'isoform_usage_mean_{unique_conditions[0]}'] = mean_ratio_cond1
    adata.var[f'isoform_usage_mean_{unique_conditions[1]}'] = mean_ratio_cond2

    # Find significant results and group by synt_id
    if len(results_df) > 0:
        results_df_sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        results_df_sig_grouped = results_df_sig.groupby('gene_id').agg({
            'FDR': list,
            'ratio_difference': list
        })
        grouped_results = results_df.groupby('gene_id').agg({
            'FDR': 'min',
            'ratio_difference': 'max'
        })   
        print(f"Found {len(results_df_sig_grouped)} from {len(grouped_results)} genes with at least one significantly different isoform (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")
    else:
        print("No results found.")

    print(f"Skipped {no_counts_isoform} isoforms due to zero counts")
    print(f"Created plotting table with {len(plotting_df)} rows (one per replicate, condition, isoform ratio, and transcript)")

    # Return AnnData object if not inplace
    if not inplace:
        return adata, results_df, plotting_df
    else:
        return results_df, plotting_df


def test_differential_isoform_structure(
    adata, layer="unique_counts", test_condition="control",
    min_similarity_for_matching=0.9,
    use_introns=True,
    exon_weight=0.6,
    intron_weight=0.4,
    inplace=True, 
    verbose=False,
    return_plotting_data=True,
    ratio_diff_cutoff=0.2,
    FDR_cutoff=0.05
):
    """
    Test for DIU between alleles with intelligent major/minor isoform fallback.
    
    - Includes isoforms in plotting data even if they have zero expression in reference haplotype
    - Matches zero-expressed reference isoforms with corresponding isoforms in other haplotypes
    
    Parameters
    -----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer containing count data (default: "unique_counts")
    test_condition : str, optional
        Condition to test for differential isoform usage (default: "control")
    min_similarity_for_matching : float, optional
        Minimum similarity score for matching isoforms (default: 0.9)
    use_introns : bool, optional
        Whether to include intron structures in similarity calculations (default: True)
    exon_weight : float, optional
        Weight for exon similarity in overall similarity score (default: 0.6)
    intron_weight : float, optional
        Weight for intron similarity in overall similarity score (default: 0.4)
    inplace : bool, optional
        Whether to modify the input AnnData object or return a copy (default: True)
    verbose : bool, optional
        Whether to print progress messages (default: False)
    return_plotting_data : bool, optional
        Whether to return plotting data along with results (default: True)
    ratio_diff_cutoff : float, optional
        Minimum ratio difference for significance (default: 0.2)        
    FDR_cutoff : float, optional
        False discovery rate cutoff for significance (default: 0.05)

    Returns
    -------
    tuple or pd.DataFrame
        If return_plotting_data=True: returns (results_df, plotting_df)
        If return_plotting_data=False: returns results_df only
    """
    import pandas as pd
    import numpy as np
    import re
    from statsmodels.stats.multitest import multipletests
    from isotools._transcriptome_stats import betabinom_lr_test
    from anndata import AnnData
    
    if not isinstance(adata, AnnData):
        raise ValueError("Input must be an AnnData object")
    
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found")
    
    required_cols = ['Synt_id', 'haplotype']
    for col in required_cols:
        if col not in adata.var.columns:
            raise ValueError(f"Required column '{col}' not found in adata.var")
    
    if 'exon_lengths' not in adata.uns:
        raise ValueError("'exon_lengths' not found in adata.uns. Run add_exon_structure() first")
    
    if use_introns and 'intron_lengths' not in adata.uns:
        print("Warning: use_introns=True but 'intron_lengths' not found. Using only exon structures")
        use_introns = False
    
    if "functional_annotation" in adata.var:
        functional_annotations = adata.var["functional_annotation"]
    else:
        functional_annotations = "Missing annotation"
        print("No functional annotations found in adata.var, skipping functional annotation processing.")
    
    if not inplace:
        adata = adata.copy()
    
    counts = adata.layers[layer]
    synt_ids = adata.var["Synt_id"]
    haplotypes = adata.var["haplotype"]
    transcript_ids = adata.var_names
    exon_lengths_dict = adata.uns['exon_lengths']
    intron_lengths_dict = adata.uns.get('intron_lengths', {}) if use_introns else {}
    gene_ids = adata.var.get("gene_id", pd.Series([None] * len(adata.var), index=adata.var_names))
    
    if test_condition == "all":
        condition_indices = np.arange(counts.shape[0])
        condition_values = adata.obs['condition'].values
    else:
        if 'condition' not in adata.obs.columns:
            raise ValueError("'condition' not found in adata.obs")
        condition_indices = np.where(adata.obs['condition'] == test_condition)[0]
        condition_values = [test_condition] * len(condition_indices)
    
    print(f"DEBUG: Found {len(condition_indices)} samples for condition '{test_condition}'")
    
    sample_names = adata.obs_names[condition_indices]
    unique_syntelogs = synt_ids.dropna().unique()
    unique_syntelogs = unique_syntelogs[unique_syntelogs != 0]
    
    print(f"DEBUG: Found {len(unique_syntelogs)} syntelogs to process")
    
    results = []
    plotting_data = []
    
    stats = {
        'total': len(unique_syntelogs),
        'tested': 0,
        'comparisons': 0,
        'major_used': 0,
        'minor_used': 0,
        'failed': 0,
        'fail_reasons': {
            'no_indices': 0,
            'single_haplotype': 0,
            'no_major': 0,
            'no_matches': 0,
            'major_not_on_all_no_minor': 0,
            'no_reference': 0,
            'zero_counts': 0,
            'test_error': 0
        }
    }
    
    if verbose:
        print(f"Processing {len(unique_syntelogs)} syntelogs with major/minor fallback logic...")
        print(f"Using {'exon+intron' if use_introns else 'exon-only'} structures")
    
    for synt_idx, synt_id in enumerate(unique_syntelogs):
        if verbose and synt_idx % 100 == 0:
            print(f"Progress: {synt_idx}/{len(unique_syntelogs)}")
        
        synt_mask = synt_ids == synt_id
        synt_indices = np.where(synt_mask)[0]
        
        if len(synt_indices) == 0:
            stats['fail_reasons']['no_indices'] += 1
            stats['failed'] += 1
            continue
        
        synt_haplotypes = haplotypes.iloc[synt_indices]
        unique_haplotypes = synt_haplotypes.dropna().unique()
        functional_annotation = functional_annotations.iloc[synt_indices].iloc[0]

        if len(unique_haplotypes) < 2:
            stats['fail_reasons']['single_haplotype'] += 1
            stats['failed'] += 1
            continue
        
        if synt_idx < 3:
            print(f"\nDEBUG Syntelog {synt_id}: {len(unique_haplotypes)} haplotypes, {len(synt_indices)} transcripts")
        
      
        reference_haplotype, major_isoform, minor_isoform, all_ref_isoforms = _identify_all_reference_isoforms(
            synt_indices, haplotypes, transcript_ids,
            exon_lengths_dict, intron_lengths_dict,
            counts, condition_indices,
            verbose=synt_idx < 3
        )
        
        if not major_isoform:
            stats['fail_reasons']['no_major'] += 1
            stats['failed'] += 1
            continue
        
        haplotype_matches = _match_isoforms_across_haplotypes(
            synt_indices, unique_haplotypes, synt_haplotypes, transcript_ids,
            exon_lengths_dict, intron_lengths_dict,
            reference_haplotype, major_isoform, minor_isoform,
            counts, condition_indices,
            min_similarity_for_matching=min_similarity_for_matching,
            verbose=synt_idx < 3
        )
        
        if synt_idx < 3:
            print(f"DEBUG: Found {len(haplotype_matches)} haplotype matches")
        
        if len(haplotype_matches) < 2:
            stats['fail_reasons']['no_matches'] += 1
            stats['failed'] += 1
            continue
        
        major_on_all = all(match['matched_isoform_type'] == 'major' for match in haplotype_matches.values())
        
        if synt_idx < 3:
            print(f"DEBUG: Major on all haplotypes: {major_on_all}")
        
        if major_on_all:
            isoform_type_for_testing = 'major'
            reference_structure_exon = major_isoform['exon_structure']
            reference_structure_intron = major_isoform['intron_structure']
            reference_transcript = major_isoform['transcript_id']
            stats['major_used'] += 1
        elif minor_isoform:
            isoform_type_for_testing = 'minor'
            reference_structure_exon = minor_isoform['exon_structure']
            reference_structure_intron = minor_isoform['intron_structure']
            reference_transcript = minor_isoform['transcript_id']
            stats['minor_used'] += 1
            
            filtered_matches = {}
            for hap, match in haplotype_matches.items():
                if hap == reference_haplotype:
                    filtered_matches[hap] = {
                        'matched_isoform_type': 'minor',
                        'transcript_id': minor_isoform['transcript_id'],
                        'transcript_idx': minor_isoform['transcript_idx'],
                        'exon_structure': minor_isoform['exon_structure'],
                        'intron_structure': minor_isoform['intron_structure'],
                        'similarity_to_major': match['similarity_to_major'],
                        'similarity_to_minor': 1.0,
                        'expression': minor_isoform.get('expression', 0),
                        'n_isoforms_in_haplotype': match['n_isoforms_in_haplotype']
                    }
                elif match['matched_isoform_type'] == 'minor':
                    filtered_matches[hap] = match
            
            haplotype_matches = filtered_matches
            
            if synt_idx < 3:
                print(f"DEBUG: After filtering for minor, {len(haplotype_matches)} haplotypes remain")
            
            if len(haplotype_matches) < 2:
                stats['fail_reasons']['no_matches'] += 1
                stats['failed'] += 1
                continue
        else:
            stats['fail_reasons']['major_not_on_all_no_minor'] += 1
            stats['failed'] += 1
            continue
        
        if synt_idx < 3:
            print(f"DEBUG: Testing using {isoform_type_for_testing.upper()} isoform")
        
        if reference_haplotype not in haplotype_matches:
            if synt_idx < 3:
                print(f"DEBUG ERROR: Reference haplotype {reference_haplotype} not in matches!")
            stats['fail_reasons']['no_reference'] += 1
            stats['failed'] += 1
            continue
        
        ref_match = haplotype_matches[reference_haplotype]
        ref_hap_mask = synt_haplotypes == reference_haplotype
        ref_hap_indices = synt_indices[np.where(ref_hap_mask)[0]]
        ref_isoform_counts = counts[np.ix_(condition_indices, [ref_match['transcript_idx']])].flatten()
        ref_hap_totals = np.sum(counts[np.ix_(condition_indices, ref_hap_indices)], axis=1)
        
        if synt_idx < 3:
            print(f"DEBUG: Reference counts: {ref_isoform_counts[:3]}, totals: {ref_hap_totals[:3]}")
        
        comparisons_made = 0
        for other_hap, other_match in haplotype_matches.items():
            if other_hap == reference_haplotype:
                continue
            
            other_hap_mask = synt_haplotypes == other_hap
            other_hap_indices = synt_indices[np.where(other_hap_mask)[0]]
            other_isoform_counts = counts[np.ix_(condition_indices, [other_match['transcript_idx']])].flatten()
            other_hap_totals = np.sum(counts[np.ix_(condition_indices, other_hap_indices)], axis=1)
            
            if np.all(ref_isoform_counts == 0) or np.all(other_isoform_counts == 0):
                stats['fail_reasons']['zero_counts'] += 1
                continue
            if np.all(ref_hap_totals == 0) or np.all(other_hap_totals == 0):
                stats['fail_reasons']['zero_counts'] += 1
                continue
            
            try:
                test_result = betabinom_lr_test(
                    [ref_isoform_counts.astype(int), other_isoform_counts.astype(int)],
                    [ref_hap_totals.astype(int), other_hap_totals.astype(int)]
                )
                p_value = test_result[0]
                ratio_stats = test_result[1]
                ratio_difference = abs(ratio_stats[0] - ratio_stats[2])
            except Exception as e:
                if synt_idx < 3:
                    print(f"DEBUG: Error in test: {str(e)}")
                stats['fail_reasons']['test_error'] += 1
                continue
            
            stats['comparisons'] += 1
            comparisons_made += 1
            
            results.append({
                'Synt_id': synt_id,
                'reference_haplotype': reference_haplotype,
                'comparison_haplotype': other_hap,
                'reference_transcript': reference_transcript,
                'comparison_transcript': other_match['transcript_id'],
                'isoform_type_tested': isoform_type_for_testing,
                'p_value': p_value,
                'ratio_difference': ratio_difference,
                'reference_ratio_mean': ratio_stats[0],
                'comparison_ratio_mean': ratio_stats[2],
            })
        
        if comparisons_made > 0:
            stats['tested'] += 1
        
        # Create plotting data that includes ALL isoforms from reference haplotype
        if return_plotting_data and comparisons_made > 0:
            gene_id = gene_ids.get(reference_transcript, None)
            library_sizes = np.sum(counts[condition_indices, :], axis=1)
            
            # Build comprehensive isoform matching for ALL reference isoforms
            all_haplotype_isoform_matches = _match_all_isoforms_for_plotting(
                all_ref_isoforms,
                unique_haplotypes,
                synt_haplotypes,
                synt_indices,
                transcript_ids,
                exon_lengths_dict,
                intron_lengths_dict,
                reference_haplotype,
                major_isoform,
                minor_isoform,
                min_similarity_for_matching=min_similarity_for_matching
            )
            
            # Now generate plotting data for ALL matched isoforms
            for hap in unique_haplotypes:
                hap_mask = synt_haplotypes == hap
                hap_indices = synt_indices[np.where(hap_mask)[0]]
                
                if len(hap_indices) == 0:
                    continue
                
                # For each reference isoform, add plotting data
                for ref_iso in all_ref_isoforms:
                    ref_iso_id = ref_iso['transcript_id']
                    ref_iso_rank = ref_iso['rank']  # 'major', 'minor', or 'other'
                    
                    # Find matched isoform in this haplotype
                    matched_info = all_haplotype_isoform_matches.get((hap, ref_iso_id), None)
                    
                    if matched_info is not None:
                        matched_transcript_idx = matched_info['transcript_idx']
                        matched_transcript_id = matched_info['transcript_id']
                        similarity = matched_info['similarity']
                    else:
                        # No match found - use zeros
                        matched_transcript_idx = None
                        matched_transcript_id = ref_iso_id
                        similarity = 0.0
                    
                    # Calculate metrics for each sample
                    for sample_idx, (cond_idx, samp_name, cond) in enumerate(zip(condition_indices, sample_names, condition_values)):
                        hap_total = np.sum(counts[cond_idx, hap_indices])
                        
                        if matched_transcript_idx is not None:
                            iso_counts = counts[cond_idx, matched_transcript_idx]
                            iso_ratio = iso_counts / hap_total if hap_total > 0 else 0.0
                            lib_size = library_sizes[sample_idx]
                            iso_cpm = (iso_counts / lib_size) * 1e6 if lib_size > 0 else 0.0
                        else:
                            iso_counts = 0
                            iso_ratio = 0.0
                            iso_cpm = 0.0
                        
                        plotting_data.append({
                            'Synt_id': synt_id,
                            'gene_id': gene_id,
                            'functional_annotation': functional_annotation,
                            'haplotype': hap,
                            'sample': samp_name,
                            'condition': cond,
                            'isoform_rank': ref_iso_rank,
                            'isoform_id': ref_iso_id,  # Reference isoform ID
                            'transcript_id': matched_transcript_id,  # Actual matched transcript
                            'isoform_ratio': float(iso_ratio),
                            'isoform_counts': int(iso_counts),
                            f'{layer}_isoform_counts': int(iso_counts),
                            'isoform_cpm': float(iso_cpm),
                            f'{layer}_cpm': float(iso_cpm),
                            f'{layer}_total_counts': int(hap_total),
                            'similarity_to_reference': float(similarity),
                            'reference_haplotype': reference_haplotype,
                            'is_reference_haplotype': hap == reference_haplotype,
                            'ratio_difference': 0.0,  # Will be filled in later
                            'has_expression': iso_counts > 0
                        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nDetailed failure reasons:")
    for reason, count in stats['fail_reasons'].items():
        if count > 0:
            print(f"  {reason}: {count}")
    
    if len(results_df) > 0:
        results_df['FDR'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df_sig = results_df[(results_df['FDR'] < FDR_cutoff) & (results_df['ratio_difference'] > ratio_diff_cutoff)]
        results_df_sig_grouped = results_df_sig.groupby('Synt_id').agg({
            'FDR': list,
            'ratio_difference': list
        })
        grouped_results = results_df.groupby('Synt_id').agg({
            'FDR': 'min',
            'ratio_difference': 'max'
        })   
        print(f"Found {len(results_df_sig_grouped)} from {len(grouped_results)} genes with at least one significantly different isoform (FDR < {FDR_cutoff} and ratio difference > {ratio_diff_cutoff})")

    
        print(f"  Total syntelogs: {stats['total']}")
        print(f"  Successfully tested: {stats['tested']}")
        print(f"  Failed to test: {stats['failed']}")

    else:
        print("\nNo results generated!")
    
    # Return based on return_plotting_data parameter
    if return_plotting_data:
        plotting_df = pd.DataFrame(plotting_data)
        
        if len(plotting_df) > 0 and len(results_df) > 0:
            # Map the actual ratio_difference from results to plotting data
            synt_stats = results_df.groupby('Synt_id').agg({
                'p_value': 'min',
                'FDR': 'min',
                'ratio_difference': 'max'
            }).reset_index()
            
            # Merge the statistics into plotting data
            plotting_df = plotting_df.merge(
                synt_stats[['Synt_id', 'p_value', 'FDR', 'ratio_difference']],
                on='Synt_id',
                how='left',
                suffixes=('_old', '')
            )
            
            # Drop the old ratio_difference column if it exists
            if 'ratio_difference_old' in plotting_df.columns:
                plotting_df = plotting_df.drop(columns=['ratio_difference_old'])
            
            if verbose:
                print(f"\nPlotting data created:")
                print(f"  Total rows: {len(plotting_df)}")
                print(f"  Syntelogs: {plotting_df['Synt_id'].nunique()}")
                print(f"  Unique isoforms: {plotting_df['isoform_id'].nunique()}")
                print(f"  Major isoform rows: {(plotting_df['isoform_rank'] == 'major').sum()}")
                print(f"  Minor isoform rows: {(plotting_df['isoform_rank'] == 'minor').sum()}")
                print(f"  Other isoform rows: {(plotting_df['isoform_rank'] == 'other').sum()}")
                print(f"  Rows with expression: {plotting_df['has_expression'].sum()}")
                print(f"  Rows without expression: {(~plotting_df['has_expression']).sum()}")
        
        return results_df, plotting_df
    else:
        return results_df


def _identify_all_reference_isoforms(
    synt_indices, haplotypes, transcript_ids,
    exon_lengths_dict, intron_lengths_dict,
    counts, condition_indices,
    verbose=False
):
    """
    Identify ALL isoforms in the reference haplotype, including those with zero expression.
    Reference haplotype is chosen as the one with highest total expression.
    
    Returns
    -------
    tuple
        (reference_haplotype, major_isoform_dict, minor_isoform_dict, all_isoforms_list)
    """
    import numpy as np
    
    # Get haplotypes for this syntelog
    synt_haplotypes = haplotypes.iloc[synt_indices]
    unique_haplotypes = synt_haplotypes.dropna().unique()
    
    # Choose reference haplotype as the one with highest total expression
    haplotype_expressions = {}
    for hap in unique_haplotypes:
        hap_mask = synt_haplotypes == hap
        hap_indices = synt_indices[np.where(hap_mask)[0]]
        total_expr = np.sum(counts[np.ix_(condition_indices, hap_indices)])
        haplotype_expressions[hap] = total_expr
    
    # Select haplotype with maximum expression
    reference_haplotype = max(haplotype_expressions, key=haplotype_expressions.get)
    
    if verbose:
        print(f"DEBUG: Haplotype expressions: {haplotype_expressions}")
        print(f"DEBUG: Selected reference haplotype: {reference_haplotype} (expression={haplotype_expressions[reference_haplotype]})")
    
    # Get all transcripts from reference haplotype
    ref_hap_mask = synt_haplotypes == reference_haplotype
    ref_hap_indices = synt_indices[np.where(ref_hap_mask)[0]]
    
    # Calculate expression for each transcript
    isoform_expressions = []
    for idx in ref_hap_indices:
        tid = transcript_ids[idx]
        total_expr = np.sum(counts[np.ix_(condition_indices, [idx])])
        
        isoform_expressions.append({
            'transcript_id': tid,
            'transcript_idx': idx,
            'expression': total_expr,
            'exon_structure': exon_lengths_dict.get(tid, []),
            'intron_structure': intron_lengths_dict.get(tid, [])
        })
    
    # Sort by expression
    isoform_expressions.sort(key=lambda x: x['expression'], reverse=True)
    
    if verbose:
        print(f"DEBUG: Reference haplotype {reference_haplotype} has {len(isoform_expressions)} isoforms")
        for iso in isoform_expressions[:3]:
            print(f"  {iso['transcript_id']}: expression={iso['expression']}")
    
    # Identify major and minor
    major_isoform = isoform_expressions[0] if len(isoform_expressions) > 0 else None
    minor_isoform = isoform_expressions[1] if len(isoform_expressions) > 1 else None
    
    # Assign ranks to all isoforms
    all_isoforms = []
    for i, iso in enumerate(isoform_expressions):
        iso_copy = iso.copy()
        if i == 0:
            iso_copy['rank'] = 'major'
        elif i == 1:
            iso_copy['rank'] = 'minor'
        else:
            iso_copy['rank'] = 'other'
        all_isoforms.append(iso_copy)
    
    return reference_haplotype, major_isoform, minor_isoform, all_isoforms


def _match_all_isoforms_for_plotting(
    all_ref_isoforms,
    unique_haplotypes,
    synt_haplotypes,
    synt_indices,
    transcript_ids,
    exon_lengths_dict,
    intron_lengths_dict,
    reference_haplotype,
    major_isoform,
    minor_isoform,
    min_similarity_for_matching=0.4
):
    import numpy as np
    
    matches = {}
    
    for hap in unique_haplotypes:
        hap_mask = synt_haplotypes == hap
        hap_indices = synt_indices[np.where(hap_mask)[0]]
        
        if len(hap_indices) == 0:
            continue
        
        if hap == reference_haplotype:
            for ref_iso in all_ref_isoforms:
                matches[(hap, ref_iso['transcript_id'])] = {
                    'transcript_idx': ref_iso['transcript_idx'],
                    'transcript_id': ref_iso['transcript_id'],
                    'similarity': 1.0
                }
            continue
        
        # Build full similarity matrix: ref_isoform x hap_transcript
        similarity_matrix = {}
        for ref_iso in all_ref_isoforms:
            ref_iso_id = ref_iso['transcript_id']
            for idx in hap_indices:
                tid = transcript_ids[idx]
                exon_struct = exon_lengths_dict.get(tid, [])
                intron_struct = intron_lengths_dict.get(tid, [])
                
                similarity = _calculate_combined_structure_similarity(
                    ref_iso['exon_structure'], exon_struct,
                    ref_iso['intron_structure'], intron_struct
                )
                similarity_matrix[(ref_iso_id, idx)] = similarity
        
        # Greedy assignment: best similarity pair first, no reuse of hap transcripts
        assigned_hap_transcripts = set()
        assigned_ref_isoforms = set()
        
        # Sort all pairs by similarity descending
        sorted_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)
        
        for (ref_iso_id, idx), similarity in sorted_pairs:
            if ref_iso_id in assigned_ref_isoforms:
                continue
            if idx in assigned_hap_transcripts:
                continue
            if similarity < min_similarity_for_matching:
                continue  # remaining pairs will only be worse
            
            matches[(hap, ref_iso_id)] = {
                'transcript_idx': idx,
                'transcript_id': transcript_ids[idx],
                'similarity': similarity
            }
            assigned_hap_transcripts.add(idx)
            assigned_ref_isoforms.add(ref_iso_id)
    
    return matches

def _identify_major_minor_isoforms(
    synt_indices, haplotypes, transcript_ids,
    exon_lengths_dict, intron_lengths_dict,
    counts, condition_indices, verbose=False
):
    """Identify the reference haplotype and its major/minor isoforms."""
    import numpy as np
    synt_haplotypes = haplotypes.iloc[synt_indices]
    unique_haplotypes = synt_haplotypes.dropna().unique()
    
    # Find the most expressed haplotype
    haplotype_expressions = {}
    for hap in unique_haplotypes:
        hap_mask = synt_haplotypes == hap
        hap_indices_local = np.where(hap_mask)[0]
        hap_indices_global = synt_indices[hap_indices_local]
        total_expr = np.sum(counts[np.ix_(condition_indices, hap_indices_global)])
        haplotype_expressions[hap] = total_expr
    
    reference_haplotype = max(haplotype_expressions, key=haplotype_expressions.get)
    
    if verbose:
        print(f"  Reference haplotype: {reference_haplotype} (expression: {haplotype_expressions[reference_haplotype]:.0f})")
    
    # Get all isoforms from reference haplotype
    ref_hap_mask = synt_haplotypes == reference_haplotype
    ref_hap_indices_local = np.where(ref_hap_mask)[0]
    ref_hap_indices_global = synt_indices[ref_hap_indices_local]
    
    ref_isoforms = []
    for idx in ref_hap_indices_global:
        transcript_id = transcript_ids[idx]
        total_expr = np.sum(counts[np.ix_(condition_indices, [idx])])
        exon_structure = exon_lengths_dict.get(transcript_id, [])
        intron_structure = intron_lengths_dict.get(transcript_id, [])
        
        ref_isoforms.append({
            'transcript_id': transcript_id,
            'transcript_idx': idx,
            'expression': total_expr,
            'exon_structure': exon_structure,
            'intron_structure': intron_structure
        })
    
    ref_isoforms.sort(key=lambda x: x['expression'], reverse=True)
    
    major_isoform = ref_isoforms[0] if ref_isoforms else None
    minor_isoform = ref_isoforms[1] if len(ref_isoforms) > 1 else None
    
    if verbose and major_isoform:
        print(f"  Major isoform: {major_isoform['transcript_id']} (expression: {major_isoform['expression']:.0f})")
        if minor_isoform:
            print(f"  Minor isoform: {minor_isoform['transcript_id']} (expression: {minor_isoform['expression']:.0f})")
    
    return reference_haplotype, major_isoform, minor_isoform


def _match_isoforms_across_haplotypes(
    synt_indices, unique_haplotypes, synt_haplotypes, transcript_ids,
    exon_lengths_dict, intron_lengths_dict,
    reference_haplotype, major_isoform, minor_isoform,
    counts, condition_indices,
    min_similarity_for_matching=0.4,
    verbose=False
):
    """Match major and minor isoforms across all haplotypes."""
    import numpy as np
    haplotype_matches = {}
    
    for hap in unique_haplotypes:
        hap_mask = synt_haplotypes == hap
        hap_indices_local = np.where(hap_mask)[0]
        hap_indices_global = synt_indices[hap_indices_local]
        
        if len(hap_indices_global) == 0:
            continue
        
        if hap == reference_haplotype:
            haplotype_matches[hap] = {
                'matched_isoform_type': 'major',
                'transcript_id': major_isoform['transcript_id'],
                'transcript_idx': major_isoform['transcript_idx'],
                'exon_structure': major_isoform['exon_structure'],
                'intron_structure': major_isoform['intron_structure'],
                'similarity_to_major': 1.0,
                'similarity_to_minor': 1.0 if minor_isoform else None,
                'n_isoforms_in_haplotype': len(hap_indices_global)
            }
            continue
        
        candidates = []
        for idx in hap_indices_global:
            transcript_id = transcript_ids[idx]
            total_expr = np.sum(counts[np.ix_(condition_indices, [idx])])
            exon_structure = exon_lengths_dict.get(transcript_id, [])
            intron_structure = intron_lengths_dict.get(transcript_id, [])
            
            similarity_to_major = _calculate_combined_structure_similarity(
                major_isoform['exon_structure'], exon_structure,
                major_isoform['intron_structure'], intron_structure
            )
            
            similarity_to_minor = None
            if minor_isoform:
                similarity_to_minor = _calculate_combined_structure_similarity(
                    minor_isoform['exon_structure'], exon_structure,
                    minor_isoform['intron_structure'], intron_structure
                )
            
            candidates.append({
                'transcript_id': transcript_id,
                'transcript_idx': idx,
                'expression': total_expr,
                'exon_structure': exon_structure,
                'intron_structure': intron_structure,
                'similarity_to_major': similarity_to_major,
                'similarity_to_minor': similarity_to_minor
            })
        
        if not candidates:
            continue
        
        major_matches = [c for c in candidates if c['similarity_to_major'] >= min_similarity_for_matching]
        
        if major_matches:
            best_match = max(major_matches, key=lambda x: x['expression'])
            matched_type = 'major'
            if verbose:
                print(f"  {hap}: Matched to MAJOR ({best_match['transcript_id']}, sim={best_match['similarity_to_major']:.3f})")
        elif minor_isoform:
            minor_matches = [c for c in candidates 
                           if c['similarity_to_minor'] and c['similarity_to_minor'] >= min_similarity_for_matching]
            
            if minor_matches:
                best_match = max(minor_matches, key=lambda x: x['expression'])
                matched_type = 'minor'
                if verbose:
                    print(f"  {hap}: Matched to MINOR ({best_match['transcript_id']}, sim={best_match['similarity_to_minor']:.3f})")
            else:
                if len(candidates) == 1:
                    candidate = candidates[0]
                    if candidate['similarity_to_minor'] and candidate['similarity_to_minor'] > candidate['similarity_to_major']:
                        best_match = candidate
                        matched_type = 'minor'
                        if verbose:
                            print(f"  {hap}: Single isoform closer to MINOR")
                    else:
                        if verbose:
                            print(f"  {hap}: Single isoform closer to major but below threshold, skipping")
                        continue
                else:
                    if verbose:
                        print(f"  {hap}: No suitable matches found")
                    continue
        else:
            if verbose:
                print(f"  {hap}: No suitable matches found")
            continue
        
        haplotype_matches[hap] = {
            'matched_isoform_type': matched_type,
            'transcript_id': best_match['transcript_id'],
            'transcript_idx': best_match['transcript_idx'],
            'exon_structure': best_match['exon_structure'],
            'intron_structure': best_match['intron_structure'],
            'similarity_to_major': best_match['similarity_to_major'],
            'similarity_to_minor': best_match['similarity_to_minor'],
            'expression': best_match['expression'],
            'n_isoforms_in_haplotype': len(candidates)
        }
    
    return haplotype_matches

def _calculate_combined_structure_similarity(
    exon_structure1, exon_structure2,
    intron_structure1=None, intron_structure2=None,
    exon_weight=0.6, intron_weight=0.4,
    length_tolerance=10  # base pairs tolerance for length comparison
):
    """
    Calculate combined similarity using both exon and intron structures.
    
    Similarity is based on:
    - Whether the number of exons is the same
    - Similarity of corresponding exon lengths (with tolerance)
    - Similarity of corresponding intron lengths (with tolerance)
    
    Parameters
    ----------
    exon_structure1, exon_structure2 : list of int
        Exon lengths as [length1, length2, ...]
    intron_structure1, intron_structure2 : list of int or int, optional
        Intron lengths as [length1, length2, ...] or single int
    exon_weight : float, default=0.6
        Weight for exon similarity in combined score
    intron_weight : float, default=0.4
        Weight for intron similarity in combined score
    length_tolerance : int, default=10
        Base pair tolerance for length similarity (bp difference that doesn't reduce similarity)
    
    Returns
    -------
    float
        Combined similarity score between 0 and 1
    """
    if not exon_structure1 or not exon_structure2:
        return 0.0
    
    # Calculate exon similarity based on count and lengths
    exon_sim = _calculate_length_based_similarity(
        exon_structure1, exon_structure2, length_tolerance
    )
    
    # Calculate intron similarity if available
    if intron_structure1 is not None and intron_structure2 is not None:
        # Convert single int to list for consistency
        if isinstance(intron_structure1, (int, float)):
            intron_structure1 = [intron_structure1]
        if isinstance(intron_structure2, (int, float)):
            intron_structure2 = [intron_structure2]
            
        intron_sim = _calculate_length_based_similarity(
            intron_structure1, intron_structure2, length_tolerance
        )
        combined_sim = exon_weight * exon_sim + intron_weight * intron_sim
    else:
        combined_sim = exon_sim
    
    # Ensure the result is bounded between 0 and 1
    return max(0.0, min(1.0, combined_sim))


def _calculate_length_based_similarity(lengths1, lengths2, tolerance=10):
    """
    Calculate similarity between two genomic structures based on element counts and lengths.
    
    Parameters
    ----------
    lengths1, lengths2 : list of int
        Lengths of genomic elements (exons or introns)
    tolerance : int, default=10
        Base pair tolerance for length comparison
    
    Returns
    -------
    float
        Similarity score between 0 and 1
    """
    if not lengths1 or not lengths2:
        return 0.0
    
    # Ensure we're working with lists
    if not isinstance(lengths1, list):
        lengths1 = [lengths1]
    if not isinstance(lengths2, list):
        lengths2 = [lengths2]
    
    # Component 1: Check if the number of elements is the same
    count1 = len(lengths1)
    count2 = len(lengths2)
    
    # Penalize heavily for different counts
    if count1 != count2:
        count_similarity = 1.0 - abs(count1 - count2) / max(count1, count2)
        # If counts differ significantly, return low similarity
        if count_similarity < 0.5:
            return count_similarity * 0.5  # Max 0.25 if counts differ a lot
    else:
        count_similarity = 1.0
    
    # Component 2: Calculate length similarity for corresponding elements
    # Sort lengths to compare corresponding elements by size
    lengths1_sorted = sorted(lengths1)
    lengths2_sorted = sorted(lengths2)
    
    # Compare corresponding lengths (pair shortest with shortest, etc.)
    n_comparisons = min(len(lengths1_sorted), len(lengths2_sorted))
    
    if n_comparisons == 0:
        return 0.0
    
    length_similarities = []
    for i in range(n_comparisons):
        len1 = lengths1_sorted[i]
        len2 = lengths2_sorted[i]
        
        # Calculate absolute difference
        diff = abs(len1 - len2)
        
        # Apply tolerance: differences within tolerance have no penalty
        if diff <= tolerance:
            similarity = 1.0
        else:
            # Calculate similarity based on relative difference beyond tolerance
            adjusted_diff = diff - tolerance
            avg_length = (len1 + len2) / 2
            # Use exponential decay for penalty to avoid harsh drops
            similarity = max(0.0, 1.0 - (adjusted_diff / avg_length))
        
        length_similarities.append(similarity)
    
    # Average length similarity
    avg_length_similarity = sum(length_similarities) / n_comparisons
    
    # Penalize for extra elements (if counts differ)
    if count1 != count2:
        extra_elements = abs(count1 - count2)
        penalty = extra_elements / max(count1, count2) * 0.2  # 20% penalty per extra element
        avg_length_similarity *= (1 - penalty)
    
    # Combine count and length similarity
    # Give more weight to length similarity if counts match
    if count1 == count2:
        final_similarity = 0.2 * count_similarity + 0.8 * avg_length_similarity
    else:
        final_similarity = 0.5 * count_similarity + 0.5 * avg_length_similarity
    
    # Ensure the result is bounded between 0 and 1
    return max(0.0, min(1.0, final_similarity))
