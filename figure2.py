import os
import sys

# Set the working directory to the script's location
working_dir = "/Users/linxy29/Documents/Data/CIVET/simulation"
os.chdir(working_dir)
print(f"Working directory set to: {os.getcwd()}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import warnings
import re
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def sort_mutation_rates_numerically(conditions):
    """
    Sort mutation rate and depth conditions numerically instead of alphabetically.
    For example: mutation_rate_1, mutation_rate_2, mutation_rate_4, mutation_rate_8, mutation_rate_16
    Or: depth_50, depth_100, depth_200, depth_500, depth_1000
    """
    def extract_rate(condition):
        # Extract the number from mutation_rate_X pattern
        match = re.search(r'mutation_rate_(\d+)', condition)
        if match:
            return int(match.group(1))
        # Extract the number from depth_X pattern
        match = re.search(r'depth_(\d+)', condition)
        if match:
            return int(match.group(1))
        return float('inf')  # Put non-matching conditions at the end

    return sorted(conditions, key=extract_rate)

def load_and_combine_data(files):
    """
    Load and combine mutation identification results from multiple methods
    """
    dataframes = {}
    
    # Load each file
    for method, filepath in files.items():
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {method}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns in {method}: {', '.join(df.columns)}")
            dataframes[method] = df
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            continue
    
    if not dataframes:
        raise ValueError("No data files could be loaded")
    
    # Start with the first available dataframe
    first_method = list(dataframes.keys())[0]
    combined_df = dataframes[first_method].copy()

    # Find common merge columns across all dataframes, case-insensitive
    expected_cols = ['scenario', 'condition', 'mutation_name']
    merge_cols = []

    for expected_col in expected_cols:
        # Check first dataframe for column match (case-insensitive)
        matches = [col for col in combined_df.columns if col.lower() == expected_col.lower()]
        if matches:
            merge_cols.append(matches[0])
        else:
            print(f"Warning: Could not find column matching '{expected_col}' in the first dataframe")

    if len(merge_cols) != 3:
        print(f"Only found {len(merge_cols)} merge columns: {merge_cols}")
        print("Looking for columns with similar names...")

        # Try to find columns with similar names
        for expected_col in expected_cols:
            if not any(col.lower() == expected_col.lower() for col in merge_cols):
                potential_matches = [col for col in combined_df.columns if expected_col.lower() in col.lower()]
                if potential_matches:
                    merge_cols.append(potential_matches[0])
                    print(f"Found potential match for '{expected_col}': '{potential_matches[0]}'")

    if len(merge_cols) != 3:
        raise ValueError(f"Could not identify all required merge columns. Found: {merge_cols}")

    print(f"Using merge columns: {merge_cols}")

    # Rename identification column
    if 'detected' in combined_df.columns:
        combined_df = combined_df.rename(columns={'detected': f'{first_method}_detected'})

    # Rename p-value column if it exists
    if 'pval' in combined_df.columns:
        combined_df = combined_df.rename(columns={'pval': f'{first_method}_pval'})
    elif 'p_value' in combined_df.columns:
        combined_df = combined_df.rename(columns={'p_value': f'{first_method}_pval'})
    elif 'pvalue' in combined_df.columns:
        combined_df = combined_df.rename(columns={'pvalue': f'{first_method}_pval'})

    # Keep metadata columns from civet files (prefer civet_LRT if available)
    metadata_cols = [col for col in combined_df.columns if col.startswith('metadata_')]
    if 'informative' in combined_df.columns:
        metadata_cols.append('informative')
    
    # Merge other dataframes
    for method, df in dataframes.items():
        if method == first_method:
            continue

        # Prepare dataframe for merging
        df_merge = df.copy()

        # Rename identification column
        if 'detected' in df_merge.columns:
            df_merge = df_merge.rename(columns={'detected': f'{method}_detected'})

        # Rename p-value column if it exists
        if 'pval' in df_merge.columns:
            df_merge = df_merge.rename(columns={'pval': f'{method}_pval'})
        elif 'p_value' in df_merge.columns:
            df_merge = df_merge.rename(columns={'p_value': f'{method}_pval'})
        elif 'pvalue' in df_merge.columns:
            df_merge = df_merge.rename(columns={'pvalue': f'{method}_pval'})

        # Find matching column names in this dataframe
        matching_cols = []
        for col in merge_cols:
            matches = [c for c in df_merge.columns if c.lower() == col.lower()]
            if matches:
                matching_cols.append((col, matches[0]))
            else:
                print(f"Warning: Could not find column matching '{col}' in {method} dataframe")
                return None, None, None

        # Map original merge columns to this dataframe's columns
        rename_dict = {target: source for source, target in matching_cols}
        if rename_dict:
            df_merge = df_merge.rename(columns=rename_dict)

        # Select columns to merge (identification column + p-value column + merge keys)
        cols_to_keep = list(merge_cols)  # Use a copy of merge_cols
        detected_col = f'{method}_detected'
        pval_col = f'{method}_pval'
        if detected_col in df_merge.columns:
            cols_to_keep.append(detected_col)
        if pval_col in df_merge.columns:
            cols_to_keep.append(pval_col)

        # Add metadata columns if this is a civet file and we don't have them yet
        if method.startswith('civet') and not metadata_cols:
            civet_metadata_cols = [col for col in df_merge.columns if col.startswith('metadata_')]
            if 'informative' in df_merge.columns:
                civet_metadata_cols.append('informative')
            cols_to_keep.extend(civet_metadata_cols)
            metadata_cols = civet_metadata_cols

        # Keep only columns that exist in the dataframe
        cols_to_merge = [col for col in cols_to_keep if col in df_merge.columns]
        df_to_merge = df_merge[cols_to_merge]

        # Merge with combined dataframe
        combined_df = pd.merge(combined_df, df_to_merge, on=merge_cols, how='outer', suffixes=('', f'_{method}'))
    
    return combined_df, merge_cols, metadata_cols

def define_true_mutations(df):
    """
    Define true mutations based on rest_mutation column:
    1) true_mutation is TRUE if rest_mutation is TRUE, otherwise FALSE
    2) informative is FALSE if rest_mutation is FALSE, otherwise keeps original value
    """
    # Check if rest_mutation column exists
    if 'rest_mutation' in df.columns:
        print("Setting true_mutation based on rest_mutation column")
        # 1) Set true_mutation to TRUE if rest_mutation is TRUE, otherwise FALSE
        df['true_mutation'] = df['rest_mutation'] == True
        
        # 2) Set informative to FALSE if rest_mutation is FALSE, otherwise keep original value
        if 'informative' in df.columns:
            print("Adjusting informative values based on rest_mutation")
            # Create a mask for rows where rest_mutation is FALSE
            mask = df['rest_mutation'] == False
            # Only modify informative for those rows
            df.loc[mask, 'informative'] = False
    else:
        print("Warning: Could not define true mutations - rest_mutation column missing")
        df['true_mutation'] = False
    
    # Print summary of true mutations and informative values
    if 'true_mutation' in df.columns:
        true_count = df['true_mutation'].sum()
        total_count = len(df)
        print(f"True mutations: {true_count} out of {total_count} ({true_count/total_count:.1%})")
    
    if 'informative' in df.columns:
        informative_count = df['informative'].sum()
        total_count = len(df)
        print(f"Informative mutations: {informative_count} out of {total_count} ({informative_count/total_count:.1%})")
    
    return df

def calculate_performance_metrics(df, methods):
    """
    Calculate performance metrics for each method
    """
    results = {}

    for method in methods:
        detected_col = f'{method}_detected'
        if detected_col not in df.columns:
            print(f"Warning: {detected_col} not found in dataframe")
            continue

        # Handle missing values
        mask = ~(df[detected_col].isna() | df['true_mutation'].isna())
        y_true = df.loc[mask, 'true_mutation']
        y_pred = df.loc[mask, detected_col]

        if len(y_true) == 0:
            print(f"Warning: No valid data for {method}")
            continue

        # Convert to boolean if needed
        if y_pred.dtype == 'object':
            y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Calculate AUROC using p-values if available
        pval_col = f'{method}_pval'
        auroc = 0

        if pval_col in df.columns:
            # Use p-values for AUROC calculation
            try:
                # Filter to rows with valid p-values and true mutation labels
                pval_mask = ~(df[pval_col].isna() | df['true_mutation'].isna())
                y_true_auroc = df.loc[pval_mask, 'true_mutation']
                y_pval = df.loc[pval_mask, pval_col]

                if len(y_true_auroc) > 0 and y_true_auroc.nunique() > 1:
                    # Convert p-values to numeric and handle edge cases
                    y_pval = pd.to_numeric(y_pval, errors='coerce')

                    # Remove any remaining NaN values
                    valid_idx = ~y_pval.isna()
                    y_true_auroc = y_true_auroc[valid_idx]
                    y_pval = y_pval[valid_idx]

                    if len(y_true_auroc) > 0 and y_true_auroc.nunique() > 1:
                        # Since lower p-values indicate stronger evidence,
                        # use 1 - pval as the score (so higher scores = more likely to be true)
                        # Clip p-values to avoid issues with 0 or 1
                        y_pval_clipped = np.clip(y_pval, 1e-10, 1 - 1e-10)
                        y_score = 1 - y_pval_clipped

                        fpr, tpr, _ = roc_curve(y_true_auroc, y_score)
                        auroc = auc(fpr, tpr)
                        print(f"  {method}: AUROC calculated using p-values = {auroc:.3f}")
            except Exception as e:
                print(f"  Warning: Could not calculate AUROC for {method} using p-values: {str(e)}")
                auroc = 0
        else:
            # Fallback to binary predictions if p-values not available
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auroc = auc(fpr, tpr)
                print(f"  {method}: AUROC calculated using binary predictions = {auroc:.3f}")
            except:
                auroc = 0

        results[method] = {
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'Specificity': specificity,
            'F1_Score': f1_score, 'Accuracy': accuracy, 'AUROC': auroc,
            'Total_Identified': tp + fp,
            'Total_Mutations': len(y_true),
            'True_Positives_Rate': tp / len(y_true) if len(y_true) > 0 else 0
        }

    return pd.DataFrame(results).T

def calculate_performance_per_condition(df, methods):
    """
    Calculate AUROC, AUPRC, and F1 score for each method in each scenario-condition combination

    Returns:
        DataFrame with columns: scenario, condition, method, auroc, auprc, f1_score
    """
    results = []

    # Find scenario and condition columns
    scenario_col = None
    condition_col = None

    for col in df.columns:
        if 'scenario' in col.lower():
            scenario_col = col
        if 'condition' in col.lower():
            condition_col = col

    if not scenario_col or not condition_col:
        print("Warning: Could not find scenario or condition columns")
        return pd.DataFrame()

    # Get unique scenario-condition combinations
    scenarios = df[scenario_col].unique()

    for scenario in scenarios:
        scenario_df = df[df[scenario_col] == scenario]
        conditions = scenario_df[condition_col].unique()

        for condition in conditions:
            condition_df = scenario_df[scenario_df[condition_col] == condition]

            for method in methods:
                detected_col = f'{method}_detected'
                pval_col = f'{method}_pval'

                if detected_col not in condition_df.columns:
                    continue

                # Filter valid data
                mask = ~(condition_df[detected_col].isna() | condition_df['true_mutation'].isna())

                if mask.sum() == 0:
                    continue

                y_true = condition_df.loc[mask, 'true_mutation']
                y_pred = condition_df.loc[mask, detected_col]

                # Convert to boolean if needed
                if y_pred.dtype == 'object':
                    y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})

                # Calculate F1 score
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                except:
                    f1_score = 0

                # Calculate AUROC using p-values if available
                auroc = 0
                if pval_col in condition_df.columns:
                    try:
                        pval_mask = ~(condition_df[pval_col].isna() | condition_df['true_mutation'].isna())
                        y_true_auroc = condition_df.loc[pval_mask, 'true_mutation']
                        y_pval = condition_df.loc[pval_mask, pval_col]

                        if len(y_true_auroc) > 0 and y_true_auroc.nunique() > 1:
                            y_pval = pd.to_numeric(y_pval, errors='coerce')
                            valid_idx = ~y_pval.isna()
                            y_true_auroc = y_true_auroc[valid_idx]
                            y_pval = y_pval[valid_idx]

                            if len(y_true_auroc) > 0 and y_true_auroc.nunique() > 1:
                                y_pval_clipped = np.clip(y_pval, 1e-10, 1 - 1e-10)
                                y_score = 1 - y_pval_clipped
                                fpr, tpr, _ = roc_curve(y_true_auroc, y_score)
                                auroc = auc(fpr, tpr)
                    except Exception as e:
                        print(f"  Warning: Could not calculate AUROC for {method} in {scenario}/{condition}: {str(e)}")
                        auroc = 0
                else:
                    # Fallback to binary predictions
                    try:
                        if y_true.nunique() > 1:
                            fpr, tpr, _ = roc_curve(y_true, y_pred)
                            auroc = auc(fpr, tpr)
                    except:
                        auroc = 0

                # Calculate AUPRC using p-values if available
                auprc = 0
                if pval_col in condition_df.columns:
                    try:
                        pval_mask = ~(condition_df[pval_col].isna() | condition_df['true_mutation'].isna())
                        y_true_auprc = condition_df.loc[pval_mask, 'true_mutation']
                        y_pval_pr = condition_df.loc[pval_mask, pval_col]

                        if len(y_true_auprc) > 0 and y_true_auprc.nunique() > 1:
                            y_pval_pr = pd.to_numeric(y_pval_pr, errors='coerce')
                            valid_idx = ~y_pval_pr.isna()
                            y_true_auprc = y_true_auprc[valid_idx]
                            y_pval_pr = y_pval_pr[valid_idx]

                            if len(y_true_auprc) > 0 and y_true_auprc.nunique() > 1:
                                y_pval_clipped_pr = np.clip(y_pval_pr, 1e-10, 1 - 1e-10)
                                y_score_pr = 1 - y_pval_clipped_pr
                                precision, recall, _ = precision_recall_curve(y_true_auprc, y_score_pr)
                                auprc = auc(recall, precision)
                    except Exception as e:
                        print(f"  Warning: Could not calculate AUPRC for {method} in {scenario}/{condition}: {str(e)}")
                        auprc = 0
                else:
                    # Fallback to binary predictions
                    try:
                        if y_true.nunique() > 1:
                            precision, recall, _ = precision_recall_curve(y_true, y_pred)
                            auprc = auc(recall, precision)
                    except:
                        auprc = 0

                results.append({
                    'scenario': scenario,
                    'condition': condition,
                    'method': method,
                    'auroc': auroc,
                    'auprc': auprc,
                    'f1_score': f1_score
                })

    return pd.DataFrame(results)

def analyze_identification_patterns(df, methods):
    """
    Analyze identification patterns across methods
    """
    # identification counts per method
    identification_counts = {}
    for method in methods:
        detected_col = f'{method}_detected'
        if detected_col in df.columns:
            detected = df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
            identification_counts[method] = detected.sum()
    
    # Scenario analysis
    scenario_analysis = {}
    if 'scenario' in df.columns or any('scenario' in col.lower() for col in df.columns):
        scenario_col = 'scenario' if 'scenario' in df.columns else [col for col in df.columns if 'scenario' in col.lower()][0]
        
        for scenario in df[scenario_col].unique():
            scenario_data = df[df[scenario_col] == scenario]
            scenario_stats = {}
            
            for method in methods:
                detected_col = f'{method}_detected'
                if detected_col in scenario_data.columns:
                    detected = scenario_data[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    scenario_stats[f'{method}_identified'] = detected.sum()
                    
                    if 'true_mutation' in scenario_data.columns:
                        true_pos = ((scenario_data['true_mutation'] == True) & (detected == True)).sum()
                        false_pos = ((scenario_data['true_mutation'] == False) & (detected == True)).sum()
                        scenario_stats[f'{method}_true_pos'] = true_pos
                        scenario_stats[f'{method}_false_pos'] = false_pos
            
            scenario_analysis[scenario] = scenario_stats
    
    return identification_counts, scenario_analysis

def create_visualizations(df, methods, performance_df, identification_counts, scenario_analysis, performance_per_condition_df):
    """
    Create comprehensive visualizations with six specific subplots:
    1) Number of mutations identified by method
    2) Precision and % effective SNPs
    3) AUPRC distribution (box plot)
    4) Proportion of true/false mutations (stacked bar)
    5) Method identification correlation
    6) F1 Score distribution (box plot)
    """
    # Set even larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })

    fig = plt.figure(figsize=(30, 20))

    # 1. Number of mutations identified by method
    plt.subplot(2, 3, 1)
    methods_with_data = [m for m in methods if m in identification_counts]
    counts = [identification_counts[m] for m in methods_with_data]

    # Add labels to the bars
    bars = plt.bar(methods_with_data, counts)
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add label if count > 0 (for log scale)
            plt.text(bar.get_x() + bar.get_width()/2., height * 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=16)

    plt.title('Total identifications per Method', fontsize=22)
    plt.ylabel('Number of identifications (log scale)', fontsize=24)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(fontsize=24)

    # 2. Precision and % of effective SNPs by method
    plt.subplot(2, 3, 2)
    
    # Set up data for the bar plot
    methods_with_data = [m for m in methods if m in performance_df.index]
    
    # Calculate percentage of effective SNPs for each method
    effective_percentages = {}
    for method in methods_with_data:
        detected_col = f'{method}_detected'
        if detected_col in df.columns and 'informative' in df.columns:
            detected_mask = df[detected_col] == True
            if detected_mask.sum() > 0:
                effective_percentage = df.loc[detected_mask, 'informative'].mean()
                effective_percentages[method] = effective_percentage
    
    # Set up the bar positions
    x = np.arange(len(methods_with_data))
    width = 0.35
    
    # Create the first set of bars (Precision)
    precision_values = [performance_df.loc[m, 'Precision'] if m in performance_df.index else 0 for m in methods_with_data]
    bars1 = plt.bar(x - width/2, precision_values, width, label='Precision')
    
    # Create the second set of bars (% Effective)
    effective_values = [effective_percentages.get(m, 0) for m in methods_with_data]
    bars2 = plt.bar(x + width/2, effective_values, width, label='% Effective SNPs')
    
    # Add labels and legend with larger fonts
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Score (0-1)', fontsize=24)
    plt.title('Precision and Effective SNP Percentage', fontsize=22)
    plt.xticks(x, methods_with_data, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(0, 1)
    plt.legend(fontsize=16)

    # 3. AUPRC distribution (box plot)
    ax3 = plt.subplot(2, 3, 3)

    # Prepare data for box plot
    auprc_data = []
    labels = []
    for method in methods:
        method_data = performance_per_condition_df[performance_per_condition_df['method'] == method]['auprc']
        # Filter out zeros (failed calculations)
        method_data = method_data[method_data > 0]
        if len(method_data) > 0:
            auprc_data.append(method_data.values)
            labels.append(method)

    if auprc_data:
        bp = ax3.boxplot(auprc_data, labels=labels, patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', alpha=0.7),
                         medianprops=dict(linewidth=0),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))

        ax3.set_ylabel('AUPRC', fontsize=24)
        ax3.set_xlabel('Methods', fontsize=20)
        ax3.set_title('AUPRC Distribution Across Conditions', fontsize=22)
        ax3.tick_params(axis='x', rotation=45, labelsize=18)
        ax3.tick_params(axis='y', labelsize=24)
        ax3.set_ylim(0, 1.05)
        ax3.grid(axis='y', alpha=0.3)

        # Add mean markers
        for i, data in enumerate(auprc_data):
            mean_val = np.mean(data)
            ax3.plot(i+1, mean_val, marker='D', color='green', markersize=6)
    else:
        ax3.text(0.5, 0.5, "No AUPRC data available", ha='center', va='center',
                transform=ax3.transAxes, fontsize=18)

    # 4. Stacked bar plot showing true/false mutation proportions
    plt.subplot(2, 3, 4)
    
    stacked_data = []
    for method in methods:
        detected_col = f'{method}_detected'
        if detected_col not in df.columns or 'true_mutation' not in df.columns:
            continue
            
        # Filter to only identified mutations
        detected = df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
        detected_df = df[detected == True].copy()
        
        if len(detected_df) == 0:
            continue
            
        # Calculate proportions
        true_proportion = detected_df['true_mutation'].mean()
        false_proportion = 1 - true_proportion
        
        # If we have mutation_type information, break down false positives by type
        if 'metadata_mutation_type' in detected_df.columns:
            # Filter to only false positives
            false_df = detected_df[detected_df['true_mutation'] == False]
            if len(false_df) > 0:
                # Group by mutation type
                type_counts = false_df['metadata_mutation_type'].value_counts(normalize=True)
                type_props = {t: c * false_proportion for t, c in type_counts.items()}
                
                # Add to stacked data
                data_row = {'Method': method, 'True Mutations': true_proportion}
                data_row.update(type_props)
                stacked_data.append(data_row)
            else:
                stacked_data.append({'Method': method, 'True Mutations': true_proportion, 'False Mutations': false_proportion})
        else:
            stacked_data.append({'Method': method, 'True Mutations': true_proportion, 'False Mutations': false_proportion})
    
    if stacked_data:
        stacked_df = pd.DataFrame(stacked_data)
        stacked_df = stacked_df.set_index('Method')
        
        # Fill NaN with zeros
        stacked_df = stacked_df.fillna(0)
        
        # Plot stacked bar
        stacked_df.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Composition of Identified Mutations', fontsize=22)
        plt.xlabel('Method', fontsize=20)
        plt.ylabel('Proportion', fontsize=24)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

    # 5. Method identification correlation
    ax5 = plt.subplot(2, 3, 5)
    
    # Prepare data for box plot
    f1_data = []
    labels = []
    for method in methods:
        method_data = performance_per_condition_df[performance_per_condition_df['method'] == method]['f1_score']
        # Filter out zeros
        method_data = method_data[method_data > 0]
        if len(method_data) > 0:
            f1_data.append(method_data.values)
            labels.append(method)

    if f1_data:
        bp = ax5.boxplot(f1_data, labels=labels, patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.7),
                         medianprops=dict(linewidth=0),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))

        ax5.set_ylabel('F1 Score', fontsize=24)
        ax5.set_xlabel('Methods', fontsize=20)
        ax5.set_title('F1 Score Distribution Across Conditions', fontsize=22)
        ax5.tick_params(axis='x', rotation=45, labelsize=18)
        ax5.tick_params(axis='y', labelsize=24)
        ax5.set_ylim(0, 0.6)
        ax5.grid(axis='y', alpha=0.3)

        # Add mean markers
        for i, data in enumerate(f1_data):
            mean_val = np.mean(data)
            ax5.plot(i+1, mean_val, marker='D', color='green', markersize=6)
    else:
        ax5.text(0.5, 0.5, "No F1 data available", ha='center', va='center',
                transform=ax5.transAxes, fontsize=18)

    # 6. F1 Score distribution (box plot)
    plt.subplot(2, 3, 6)
    # Create a binary matrix for method identifications
    identification_matrix = pd.DataFrame()
    for method in methods:
        detected_col = f'{method}_detected'
        if detected_col in df.columns:
            detected = df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
            identification_matrix[method] = detected.fillna(False)
    
    if not identification_matrix.empty:
        # Calculate correlation between methods
        correlation_matrix = identification_matrix.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt='.2f',
                   annot_kws={'size': 16})
        plt.title('Method identification Correlation', fontsize=22)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

    plt.tight_layout()
    
    # Save the figure to the results directory
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    vis_path = os.path.join(results_dir, 'mutation_identification_analysis.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{vis_path}'")
    
    plt.show()

def create_confusion_matrices(df, methods):
    """
    Create and save a figure with confusion matrices for all methods
    """
    # Set even larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })
    
    # Create a figure with 2x2 subplots for up to 4 methods
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    for i, method in enumerate(methods[:4]):  # Show up to 4 methods
        detected_col = f'{method}_detected'
        if detected_col not in df.columns or 'true_mutation' not in df.columns:
            continue
            
        # Handle missing values and convert to boolean
        mask = ~(df[detected_col].isna() | df['true_mutation'].isna())
        y_true = df.loc[mask, 'true_mutation']
        y_pred = df.loc[mask, detected_col]
        
        if len(y_true) == 0:
            continue
            
        if y_pred.dtype == 'object':
            y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics for title
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted False', 'Predicted True'],
                   yticklabels=['Actual False', 'Actual True'],
                   ax=axes[i], annot_kws={'size': 18})
        
        # Add metrics to title with larger font
        axes[i].set_title(f'{method} Confusion Matrix\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}', 
                         fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=24)
    
    plt.tight_layout()
    
    # Save the figure to the results directory
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    cm_path = os.path.join(results_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved as '{cm_path}'")
    
    plt.show()

def create_scenario_comparisons(df, methods, performance_df):
    """
    Create a single plot with nine subplots comparing precision, identification counts,
    and mutation proportions across conditions for scenarios 1, 2, and 4
    """
    # Set even larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })
    
    # Check if scenario and condition columns exist
    scenario_col = None
    condition_col = None
    mutation_type_col = None
    
    for col in df.columns:
        if 'scenario' in col.lower():
            scenario_col = col
        if 'condition' in col.lower():
            condition_col = col
        if 'mutation_type' in col.lower():
            mutation_type_col = col
            
    if not scenario_col:
        print("Warning: No scenario column found. Cannot create scenario comparisons.")
        return
        
    if not condition_col:
        print("Warning: No condition column found. Cannot compare conditions within scenarios.")
        return
    
    # Create directory for results
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get unique scenarios and conditions
    scenarios = df[scenario_col].unique()
    print(f"Found {len(scenarios)} scenarios: {scenarios}")

    conditions = df[condition_col].unique()
    print(f"Found {len(conditions)} conditions: {conditions}")

    # We'll focus on scenarios 1, 2, and 4
    # Sort scenarios to ensure consistent ordering
    scenarios_sorted = sorted(scenarios)
    target_scenarios = []
    for scenario in scenarios_sorted:
        if 'SCENARIO_1' in scenario or 'SCENARIO_2' in scenario or 'SCENARIO_4' in scenario:
            target_scenarios.append(scenario)

    if len(target_scenarios) < 1:
        print("Could not find scenarios 1, 2, or 4 to create comparison plot.")
        return

    print(f"Creating comparison plot for {len(target_scenarios)} scenarios: {target_scenarios}")

    # Dynamically determine the number of rows needed
    n_scenarios = len(target_scenarios)
    n_cols = 3  # Always 3 columns (identifications, precision, proportions)

    # Create a single figure with dynamic rows
    # Increase figure width to accommodate scenario labels on the left
    fig = plt.figure(figsize=(36, 9 * n_scenarios))
    plt.suptitle(f'Scenario and Condition Comparison', fontsize=26)

    # Define subplot positions dynamically
    # Each row has: Number of identifications, Precision, and Mean Mutation Proportions
    subplot_positions = {}
    for i in range(n_scenarios):
        subplot_positions[i] = {
            'identifications': i * 3 + 1,
            'precision': i * 3 + 2,
            'proportions': i * 3 + 3
        }
    
    # For each target scenario, create identification count and precision subplots
    for i, scenario in enumerate(target_scenarios):
        scenario_df = df[df[scenario_col] == scenario]
        scenario_conditions = scenario_df[condition_col].unique()
        
        # Sort conditions numerically if they are mutation rates
        scenario_conditions = sort_mutation_rates_numerically(scenario_conditions)
        
        # 1. Number of identifications by condition for this scenario
        plt.subplot(n_scenarios, 3, subplot_positions[i]['identifications'])
        identification_counts = {}
        
        for method in methods:
            detected_col = f'{method}_detected'
            if detected_col not in scenario_df.columns:
                continue
                
            counts_by_condition = []
            for condition in scenario_conditions:
                condition_df = scenario_df[scenario_df[condition_col] == condition]
                
                # Count identifications in this condition
                detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                count = detected.sum()
                counts_by_condition.append(count)
                
            identification_counts[method] = counts_by_condition
        
        # Create DataFrame for easier plotting
        counts_df = pd.DataFrame(identification_counts, index=scenario_conditions)
        counts_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'Number of identifications by Condition', fontsize=22)
        plt.xlabel('Condition', fontsize=20)
        plt.ylabel('Count', fontsize=24)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        
        # 2. Precision by condition for this scenario
        plt.subplot(n_scenarios, 3, subplot_positions[i]['precision'])
        condition_precision = {}
        
        for method in methods:
            detected_col = f'{method}_detected'
            if detected_col not in scenario_df.columns:
                continue
                
            precision_by_condition = []
            for condition in scenario_conditions:
                condition_df = scenario_df[scenario_df[condition_col] == condition]
                
                # Calculate precision for this method in this condition
                mask = ~(condition_df[detected_col].isna() | condition_df['true_mutation'].isna())
                if mask.sum() == 0:
                    precision_by_condition.append(0)
                    continue
                    
                y_true = condition_df.loc[mask, 'true_mutation']
                y_pred = condition_df.loc[mask, detected_col]
                
                if y_pred.dtype == 'object':
                    y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    
                # Handle edge case of no predictions
                if y_pred.sum() == 0:
                    precision_by_condition.append(0)
                    continue
                    
                # Calculate precision
                true_positives = ((y_true == True) & (y_pred == True)).sum()
                false_positives = ((y_true == False) & (y_pred == True)).sum()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                precision_by_condition.append(precision)
                
            condition_precision[method] = precision_by_condition
        
        # Create DataFrame for easier plotting
        precision_df = pd.DataFrame(condition_precision, index=scenario_conditions)
        precision_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'Precision by Condition', fontsize=22)
        plt.xlabel('Condition', fontsize=20)
        plt.ylabel('Precision', fontsize=24)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        
        # 3. Mutation proportions for this scenario (mean across all conditions)
        plt.subplot(n_scenarios, 3, subplot_positions[i]['proportions'])
        
        # Calculate mean proportions across all conditions for each method
        stacked_data = []
        
        for method in methods:
            detected_col = f'{method}_detected'
            if detected_col not in scenario_df.columns or 'true_mutation' not in scenario_df.columns:
                continue
                
            # Filter to only identified mutations
            detected = scenario_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
            detected_df = scenario_df[detected == True].copy()
            
            if len(detected_df) == 0:
                continue
                
            # Calculate proportions
            true_proportion = detected_df['true_mutation'].mean()
            false_proportion = 1 - true_proportion
            
            # If we have mutation_type information, break down false positives by type
            if mutation_type_col and mutation_type_col in detected_df.columns:
                # Filter to only false positives
                false_df = detected_df[detected_df['true_mutation'] == False]
                if len(false_df) > 0:
                    # Group by mutation type
                    type_counts = false_df[mutation_type_col].value_counts(normalize=True)
                    type_props = {t: c * false_proportion for t, c in type_counts.items()}
                    
                    # Add to stacked data
                    data_row = {'Method': method, 'True Mutations': true_proportion}
                    data_row.update(type_props)
                    stacked_data.append(data_row)
                else:
                    stacked_data.append({'Method': method, 'True Mutations': true_proportion, 'False Mutations': false_proportion})
            else:
                stacked_data.append({'Method': method, 'True Mutations': true_proportion, 'False Mutations': false_proportion})
        
        if stacked_data:
            stacked_df = pd.DataFrame(stacked_data)
            stacked_df = stacked_df.set_index('Method')
            
            # Fill NaN with zeros
            stacked_df = stacked_df.fillna(0)
            
            # Plot stacked bar
            stacked_df.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'Mean Mutation Proportions', fontsize=22)
            plt.xlabel('Method', fontsize=20)
            plt.ylabel('Proportion', fontsize=24)
            plt.xticks(rotation=45, fontsize=24)
            plt.yticks(fontsize=24)
            plt.ylim(0, 1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        else:
            plt.text(0.5, 0.5, "No mutation data available", ha='center', va='center', fontsize=18)
    
    # Add scenario labels vertically on the left side
    for i, scenario in enumerate(target_scenarios):
        # Position for scenario label (left side, centered vertically for each row)
        x_pos = -0.15  # Position to the left of the subplots
        # Calculate y position dynamically based on number of scenarios
        # Start at top and space evenly
        spacing = 1.0 / n_scenarios
        y_pos = 1.0 - spacing * 0.5 - i * spacing  # Start at top, center of first row

        # Add scenario label
        fig.text(x_pos, y_pos, scenario, fontsize=24, fontweight='bold',
                rotation=90, ha='center', va='center')

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Adjust for the suptitle and scenario labels

    # Save the figure
    comparison_path = os.path.join(results_dir, 'scenario_condition_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Scenario-condition comparison saved as '{comparison_path}'")

    # Now create a separate figure for scenario 3 if it exists
    scenario3_candidates = [s for s in scenarios if 'SCENARIO_3' in s]
    if len(scenario3_candidates) > 0:
        scenario3 = scenario3_candidates[0]
        plt.figure(figsize=(22, 10))
        plt.suptitle(f'Analysis: {scenario3}', fontsize=26)

        scenario_df = df[df[scenario_col] == scenario3]
        scenario_conditions = scenario_df[condition_col].unique()
        
        # Sort conditions numerically if they are mutation rates
        scenario_conditions = sort_mutation_rates_numerically(scenario_conditions)
        
        # 1. Number of identifications by condition for scenario 3
        plt.subplot(1, 2, 1)
        identification_counts = {}
        
        for method in methods:
            detected_col = f'{method}_detected'
            if detected_col not in scenario_df.columns:
                continue
                
            counts_by_condition = []
            for condition in scenario_conditions:
                condition_df = scenario_df[scenario_df[condition_col] == condition]
                
                # Count identifications in this condition
                detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                count = detected.sum()
                counts_by_condition.append(count)
                
            identification_counts[method] = counts_by_condition
        
        # Create DataFrame for easier plotting
        counts_df = pd.DataFrame(identification_counts, index=scenario_conditions)
        counts_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'Number of identifications by Condition', fontsize=22)
        plt.xlabel('Condition', fontsize=20)
        plt.ylabel('Count', fontsize=24)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        
        # 2. Precision by condition for scenario 3
        plt.subplot(1, 2, 2)
        condition_precision = {}
        
        for method in methods:
            detected_col = f'{method}_detected'
            if detected_col not in scenario_df.columns:
                continue
                
            precision_by_condition = []
            for condition in scenario_conditions:
                condition_df = scenario_df[scenario_df[condition_col] == condition]
                
                # Calculate precision for this method in this condition
                mask = ~(condition_df[detected_col].isna() | condition_df['true_mutation'].isna())
                if mask.sum() == 0:
                    precision_by_condition.append(0)
                    continue
                    
                y_true = condition_df.loc[mask, 'true_mutation']
                y_pred = condition_df.loc[mask, detected_col]
                
                if y_pred.dtype == 'object':
                    y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    
                # Handle edge case of no predictions
                if y_pred.sum() == 0:
                    precision_by_condition.append(0)
                    continue
                    
                # Calculate precision
                true_positives = ((y_true == True) & (y_pred == True)).sum()
                false_positives = ((y_true == False) & (y_pred == True)).sum()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                precision_by_condition.append(precision)
                
            condition_precision[method] = precision_by_condition
        
        # Create DataFrame for easier plotting
        precision_df = pd.DataFrame(condition_precision, index=scenario_conditions)
        precision_df.plot(kind='bar', ax=plt.gca())
        plt.title(f'Precision by Condition', fontsize=22)
        plt.xlabel('Condition', fontsize=20)
        plt.ylabel('Precision', fontsize=24)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure
        scenario3_path = os.path.join(results_dir, 'scenario3_analysis.png')
        plt.savefig(scenario3_path, dpi=300, bbox_inches='tight')
        print(f"Scenario 3 analysis saved as '{scenario3_path}'")
    else:
        print("Scenario 3 not available in the data")
    
    plt.show()

def create_boxplot_visualizations(performance_per_condition_df, methods):
    """
    Create box plots showing AUROC, AUPRC, and F1 score distributions across conditions

    Args:
        performance_per_condition_df: DataFrame with columns [scenario, condition, method, auroc, auprc, f1_score]
        methods: List of method names
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })

    # Create directory for results
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create a figure with 3 subplots (AUROC, AUPRC, and F1)
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # 1. AUROC Box Plot
    ax1 = axes[0]

    # Prepare data for box plot
    auroc_data = []
    labels = []
    for method in methods:
        method_data = performance_per_condition_df[performance_per_condition_df['method'] == method]['auroc']
        # Filter out zeros (failed calculations)
        method_data = method_data[method_data > 0]
        if len(method_data) > 0:
            auroc_data.append(method_data.values)
            labels.append(method)

    if auroc_data:
        bp1 = ax1.boxplot(auroc_data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(linewidth=0),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))

        ax1.set_ylabel('AUROC', fontsize=20)
        ax1.set_xlabel('Methods', fontsize=20)
        ax1.set_title('AUROC Distribution Across Conditions', fontsize=22)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='y', alpha=0.3)

        # Add mean markers
        for i, data in enumerate(auroc_data):
            mean_val = np.mean(data)
            ax1.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                    label='Mean' if i == 0 else '')

        ax1.legend(loc='lower right', fontsize=16)
    else:
        ax1.text(0.5, 0.5, "No AUROC data available", ha='center', va='center',
                transform=ax1.transAxes, fontsize=18)

    # 2. AUPRC Box Plot
    ax2 = axes[1]

    # Prepare data for box plot
    auprc_data = []
    labels = []
    for method in methods:
        method_data = performance_per_condition_df[performance_per_condition_df['method'] == method]['auprc']
        # Filter out zeros (failed calculations)
        method_data = method_data[method_data > 0]
        if len(method_data) > 0:
            auprc_data.append(method_data.values)
            labels.append(method)

    if auprc_data:
        bp2 = ax2.boxplot(auprc_data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7),
                          medianprops=dict(linewidth=0),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))

        ax2.set_ylabel('AUPRC', fontsize=20)
        ax2.set_xlabel('Methods', fontsize=20)
        ax2.set_title('AUPRC Distribution Across Conditions', fontsize=22)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3)

        # Add mean markers
        for i, data in enumerate(auprc_data):
            mean_val = np.mean(data)
            ax2.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                    label='Mean' if i == 0 else '')

        ax2.legend(loc='lower right', fontsize=16)
    else:
        ax2.text(0.5, 0.5, "No AUPRC data available", ha='center', va='center',
                transform=ax2.transAxes, fontsize=18)

    # 3. F1 Score Box Plot
    ax3 = axes[2]

    # Prepare data for box plot
    f1_data = []
    labels = []
    for method in methods:
        method_data = performance_per_condition_df[performance_per_condition_df['method'] == method]['f1_score']
        # Filter out zeros
        method_data = method_data[method_data > 0]
        if len(method_data) > 0:
            f1_data.append(method_data.values)
            labels.append(method)

    if f1_data:
        bp3 = ax3.boxplot(f1_data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightcoral', alpha=0.7),
                          medianprops=dict(linewidth=0),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))

        ax3.set_ylabel('F1 Score', fontsize=20)
        ax3.set_xlabel('Methods', fontsize=20)
        ax3.set_title('F1 Score Distribution Across Conditions', fontsize=22)
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 0.6)
        ax3.grid(axis='y', alpha=0.3)

        # Add mean markers
        for i, data in enumerate(f1_data):
            mean_val = np.mean(data)
            ax3.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                    label='Mean' if i == 0 else '')

        ax3.legend(loc='lower right', fontsize=16)
    else:
        ax3.text(0.5, 0.5, "No F1 data available", ha='center', va='center',
                transform=ax3.transAxes, fontsize=18)

    plt.tight_layout()

    # Save the figure
    boxplot_path = os.path.join(results_dir, 'performance_boxplots_overall.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"Performance box plots saved as '{boxplot_path}'")

    plt.show()

    # Print summary statistics
    print("\n=== AUROC Summary Statistics ===")
    for method in methods:
        method_auroc = performance_per_condition_df[performance_per_condition_df['method'] == method]['auroc']
        method_auroc = method_auroc[method_auroc > 0]
        if len(method_auroc) > 0:
            print(f"{method}:")
            print(f"  Mean: {method_auroc.mean():.3f}")
            print(f"  Median: {method_auroc.median():.3f}")
            print(f"  Std: {method_auroc.std():.3f}")
            print(f"  Min: {method_auroc.min():.3f}")
            print(f"  Max: {method_auroc.max():.3f}")

    print("\n=== AUPRC Summary Statistics ===")
    for method in methods:
        method_auprc = performance_per_condition_df[performance_per_condition_df['method'] == method]['auprc']
        method_auprc = method_auprc[method_auprc > 0]
        if len(method_auprc) > 0:
            print(f"{method}:")
            print(f"  Mean: {method_auprc.mean():.3f}")
            print(f"  Median: {method_auprc.median():.3f}")
            print(f"  Std: {method_auprc.std():.3f}")
            print(f"  Min: {method_auprc.min():.3f}")
            print(f"  Max: {method_auprc.max():.3f}")

    print("\n=== F1 Score Summary Statistics ===")
    for method in methods:
        method_f1 = performance_per_condition_df[performance_per_condition_df['method'] == method]['f1_score']
        method_f1 = method_f1[method_f1 > 0]
        if len(method_f1) > 0:
            print(f"{method}:")
            print(f"  Mean: {method_f1.mean():.3f}")
            print(f"  Median: {method_f1.median():.3f}")
            print(f"  Std: {method_f1.std():.3f}")
            print(f"  Min: {method_f1.min():.3f}")
            print(f"  Max: {method_f1.max():.3f}")

def create_auroc_scenarios_figure(performance_per_condition_df, methods):
    """
    Create a single figure with AUROC box plots for all scenarios

    Args:
        performance_per_condition_df: DataFrame with columns [scenario, condition, method, auroc, auprc, f1_score]
        methods: List of method names
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })

    # Create directory for results
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Get unique scenarios and sort them
    scenarios = sorted(performance_per_condition_df['scenario'].unique())
    n_scenarios = len(scenarios)

    # Determine subplot layout
    if n_scenarios <= 2:
        n_rows, n_cols = 1, n_scenarios
        figsize = (12 * n_scenarios, 10)
    elif n_scenarios <= 4:
        n_rows, n_cols = 2, 2
        figsize = (24, 20)
    else:
        n_rows = (n_scenarios + 2) // 3
        n_cols = 3
        figsize = (36, 10 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('AUROC Distribution Across Scenarios', fontsize=28, y=0.995)

    # Flatten axes array for easier iteration
    if n_scenarios == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()

    # Create box plot for each scenario
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_data = performance_per_condition_df[performance_per_condition_df['scenario'] == scenario]

        # Prepare AUROC data for each method
        auroc_data = []
        labels = []
        for method in methods:
            method_data = scenario_data[scenario_data['method'] == method]['auroc']
            method_data = method_data[method_data > 0]
            if len(method_data) > 0:
                auroc_data.append(method_data.values)
                labels.append(method)

        if auroc_data:
            bp = ax.boxplot(auroc_data, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(linewidth=0),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

            ax.set_ylabel('AUROC', fontsize=20)
            ax.set_xlabel('Methods', fontsize=20)
            ax.set_title(f'{scenario}', fontsize=22)
            ax.tick_params(axis='x', rotation=45, labelsize=16)
            ax.tick_params(axis='y', labelsize=18)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

            # Add mean markers
            for i, data in enumerate(auroc_data):
                mean_val = np.mean(data)
                ax.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                       label='Mean' if i == 0 else '')

            if idx == 0:  # Add legend only to first subplot
                ax.legend(loc='lower right', fontsize=16)
        else:
            ax.text(0.5, 0.5, "No AUROC data available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=18)

    # Hide any unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save the figure
    auroc_scenarios_path = os.path.join(results_dir, 'auroc_scenarios_boxplots.png')
    plt.savefig(auroc_scenarios_path, dpi=300, bbox_inches='tight')
    print(f"AUROC scenarios figure saved as '{auroc_scenarios_path}'")

    plt.show()

def create_boxplot_per_scenario(performance_per_condition_df, methods):
    """
    Create separate box plots for each scenario showing AUROC, AUPRC, and F1 distributions

    Args:
        performance_per_condition_df: DataFrame with columns [scenario, condition, method, auroc, auprc, f1_score]
        methods: List of method names
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })

    # Create directory for results
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Get unique scenarios
    scenarios = performance_per_condition_df['scenario'].unique()

    for scenario in scenarios:
        scenario_data = performance_per_condition_df[performance_per_condition_df['scenario'] == scenario]

        # Create a figure with 3 subplots (AUROC, AUPRC, and F1)
        fig, axes = plt.subplots(1, 3, figsize=(36, 10))
        fig.suptitle(f'Performance Distribution: {scenario}', fontsize=26)

        # 1. AUROC Box Plot
        ax1 = axes[0]
        auroc_data = []
        labels = []
        for method in methods:
            method_data = scenario_data[scenario_data['method'] == method]['auroc']
            method_data = method_data[method_data > 0]
            if len(method_data) > 0:
                auroc_data.append(method_data.values)
                labels.append(method)

        if auroc_data:
            bp1 = ax1.boxplot(auroc_data, labels=labels, patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(linewidth=0),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))

            ax1.set_ylabel('AUROC', fontsize=20)
            ax1.set_xlabel('Methods', fontsize=20)
            ax1.set_title('AUROC Distribution', fontsize=22)
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylim(0, 1.05)
            ax1.grid(axis='y', alpha=0.3)

            for i, data in enumerate(auroc_data):
                mean_val = np.mean(data)
                ax1.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                        label='Mean' if i == 0 else '')

            ax1.legend(loc='lower right', fontsize=16)

        # 2. AUPRC Box Plot
        ax2 = axes[1]
        auprc_data = []
        labels = []
        for method in methods:
            method_data = scenario_data[scenario_data['method'] == method]['auprc']
            method_data = method_data[method_data > 0]
            if len(method_data) > 0:
                auprc_data.append(method_data.values)
                labels.append(method)

        if auprc_data:
            bp2 = ax2.boxplot(auprc_data, labels=labels, patch_artist=True,
                              boxprops=dict(facecolor='lightgreen', alpha=0.7),
                              medianprops=dict(linewidth=0),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))

            ax2.set_ylabel('AUPRC', fontsize=20)
            ax2.set_xlabel('Methods', fontsize=20)
            ax2.set_title('AUPRC Distribution', fontsize=22)
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylim(0, 1.05)
            ax2.grid(axis='y', alpha=0.3)

            for i, data in enumerate(auprc_data):
                mean_val = np.mean(data)
                ax2.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                        label='Mean' if i == 0 else '')

            ax2.legend(loc='lower right', fontsize=16)

        # 3. F1 Score Box Plot
        ax3 = axes[2]
        f1_data = []
        labels = []
        for method in methods:
            method_data = scenario_data[scenario_data['method'] == method]['f1_score']
            method_data = method_data[method_data > 0]
            if len(method_data) > 0:
                f1_data.append(method_data.values)
                labels.append(method)

        if f1_data:
            bp3 = ax3.boxplot(f1_data, labels=labels, patch_artist=True,
                              boxprops=dict(facecolor='lightcoral', alpha=0.7),
                              medianprops=dict(linewidth=0),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))

            ax3.set_ylabel('F1 Score', fontsize=20)
            ax3.set_xlabel('Methods', fontsize=20)
            ax3.set_title('F1 Score Distribution', fontsize=22)
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_ylim(0, 0.6)
            ax3.grid(axis='y', alpha=0.3)

            for i, data in enumerate(f1_data):
                mean_val = np.mean(data)
                ax3.plot(i+1, mean_val, marker='D', color='green', markersize=8,
                        label='Mean' if i == 0 else '')

            ax3.legend(loc='lower right', fontsize=16)

        plt.tight_layout()

        # Save the figure
        boxplot_path = os.path.join(results_dir, f'performance_boxplots_{scenario}.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        print(f"Performance box plots for {scenario} saved as '{boxplot_path}'")

        plt.show()

def generate_summary_report(performance_df, identification_counts, scenario_analysis):
    """
    Generate a comprehensive summary report
    """
    report_text = []
    report_text.append("="*80)
    report_text.append("MUTATION identification METHODS ANALYSIS REPORT")
    report_text.append("="*80)
    
    report_text.append("\n1. OVERALL PERFORMANCE METRICS")
    report_text.append("-" * 40)
    report_text.append(performance_df.round(3).to_string())
    
    report_text.append("\n2. IDENTIFICATION COUNTS")
    report_text.append("-" * 40)
    for method, count in identification_counts.items():
        report_text.append(f"{method}: {count} mutations identified")
    
    report_text.append("\n3. BEST PERFORMING METHOD BY METRIC")
    report_text.append("-" * 40)
    for metric in ['Precision', 'Recall', 'F1_Score', 'Accuracy']:
        if metric in performance_df.columns:
            best_method = performance_df[metric].idxmax()
            best_score = performance_df.loc[best_method, metric]
            report_text.append(f"{metric}: {best_method} ({best_score:.3f})")
    
    report_text.append("\n4. SCENARIO-SPECIFIC ANALYSIS")
    report_text.append("-" * 40)
    for scenario, stats in scenario_analysis.items():
        report_text.append(f"\nScenario: {scenario}")
        for stat_name, value in stats.items():
            report_text.append(f"  {stat_name}: {value}")
    
    # Print to console
    for line in report_text:
        print(line)
    
    # Save to file in the results directory
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    report_path = os.path.join(results_dir, 'mutation_identification_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_text))
    
    print(f"\nReport saved as '{report_path}'")

def save_detailed_metrics_to_csv(df, methods):
    """
    Calculate and save precision, effective SNP percentage, and mutation type proportions
    for each method in each scenario and condition to a CSV file
    """
    # Find scenario and condition columns
    scenario_col = None
    condition_col = None
    mutation_type_col = None
    
    for col in df.columns:
        if 'scenario' in col.lower():
            scenario_col = col
        if 'condition' in col.lower():
            condition_col = col
        if 'mutation_type' in col.lower():
            mutation_type_col = col
    
    if not scenario_col:
        print("Warning: No scenario column found. Cannot create detailed metrics.")
        return
    
    if not condition_col:
        print("Warning: No condition column found. Cannot create detailed metrics.")
        return
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize list to store all metrics
    all_metrics = []
    
    # Get unique scenarios and conditions
    scenarios = df[scenario_col].unique()
    conditions = df[condition_col].unique()
    
    print(f"Calculating detailed metrics for {len(scenarios)} scenarios and {len(conditions)} conditions...")
    
    # Calculate metrics for each scenario, condition, and method
    for scenario in scenarios:
        scenario_df = df[df[scenario_col] == scenario]
        
        for condition in conditions:
            condition_df = scenario_df[scenario_df[condition_col] == condition]
            
            if len(condition_df) == 0:
                continue
            
            for method in methods:
                detected_col = f'{method}_detected'
                if detected_col not in condition_df.columns:
                    continue
                
                # Calculate precision
                mask = ~(condition_df[detected_col].isna() | condition_df['true_mutation'].isna())
                if mask.sum() == 0:
                    precision = 0
                else:
                    y_true = condition_df.loc[mask, 'true_mutation']
                    y_pred = condition_df.loc[mask, detected_col]
                    
                    if y_pred.dtype == 'object':
                        y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    
                    # Calculate precision
                    true_positives = ((y_true == True) & (y_pred == True)).sum()
                    false_positives = ((y_true == False) & (y_pred == True)).sum()
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                
                # Calculate effective SNP percentage
                effective_snp_percentage = 0
                if 'informative' in condition_df.columns:
                    detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    detected_mask = detected == True
                    if detected_mask.sum() > 0:
                        effective_snp_percentage = condition_df.loc[detected_mask, 'informative'].mean()
                
                # Calculate mutation type proportions
                mutation_type_proportions = {}
                if mutation_type_col and mutation_type_col in condition_df.columns:
                    detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    detected_df = condition_df[detected == True].copy()
                    
                    if len(detected_df) > 0:
                        # Calculate proportions of true vs false mutations
                        true_proportion = detected_df['true_mutation'].mean()
                        false_proportion = 1 - true_proportion
                        
                        # Break down false positives by mutation type
                        false_df = detected_df[detected_df['true_mutation'] == False]
                        if len(false_df) > 0:
                            type_counts = false_df[mutation_type_col].value_counts(normalize=True)
                            for mutation_type, proportion in type_counts.items():
                                mutation_type_proportions[f"proportion_{mutation_type}"] = proportion * false_proportion
                        
                        mutation_type_proportions['proportion_true_mutations'] = true_proportion
                        mutation_type_proportions['proportion_false_mutations'] = false_proportion
                    else:
                        mutation_type_proportions['proportion_true_mutations'] = 0
                        mutation_type_proportions['proportion_false_mutations'] = 0
                else:
                    # If no mutation type column, just calculate true/false proportions
                    detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    detected_df = condition_df[detected == True].copy()
                    
                    if len(detected_df) > 0:
                        true_proportion = detected_df['true_mutation'].mean()
                        false_proportion = 1 - true_proportion
                        mutation_type_proportions['proportion_true_mutations'] = true_proportion
                        mutation_type_proportions['proportion_false_mutations'] = false_proportion
                    else:
                        mutation_type_proportions['proportion_true_mutations'] = 0
                        mutation_type_proportions['proportion_false_mutations'] = 0
                
                # Create row for this combination
                row = {
                    'scenario': scenario,
                    'condition': condition,
                    'method': method,
                    'precision': precision,
                    'effective_snp_percentage': effective_snp_percentage,
                    'total_identifications': detected_mask.sum() if 'detected_mask' in locals() else 0,
                    'total_mutations': len(condition_df)
                }
                
                # Add mutation type proportions
                row.update(mutation_type_proportions)
                
                all_metrics.append(row)
    
    # Create DataFrame and save to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        csv_path = os.path.join(results_dir, 'detailed_metrics_by_scenario_condition.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Detailed metrics saved to '{csv_path}'")
        print(f"Shape of metrics dataframe: {metrics_df.shape}")
        
        # Print summary
        print(f"\nSummary of detailed metrics:")
        print(f"- Total combinations: {len(metrics_df)}")
        print(f"- Scenarios: {metrics_df['scenario'].nunique()}")
        print(f"- Conditions: {metrics_df['condition'].nunique()}")
        print(f"- Methods: {metrics_df['method'].nunique()}")
        
        # Show sample of the data
        print(f"\nSample of the metrics data:")
        print(metrics_df.head(10).to_string())
        
        return metrics_df
    else:
        print("No metrics calculated - check if data is available")
        return None

# Main execution
def main(working_dir=None):
    # Set custom working directory if provided
    if working_dir:
        os.chdir(working_dir)
        print(f"Working directory changed to: {os.getcwd()}")
    
    # Create overall_analysis folder if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    else:
        print(f"Using existing results directory: {results_dir}") 
        
    # Define file paths
    files = {
        "maesterpp": "maesterpp_mutation_combine.csv",
        "mquad": "mquad_mutation_combine.csv",
        "civet_LRT": "civet_mutation_combine_LRT_pvals_threshold_0.05_with_metadata_with_informative.csv",
        "civet_Wald": "civet_mutation_combine_Wald_pvals_threshold_0.05_with_metadata_with_informative.csv",
        "MitoTracer": "MitoTracer_mutation_combine.csv",
        "scMitoMut": "scMitoMut_mutation_combine.csv"
    }
    
    try:
        # Load and combine data
        print("Loading and combining data...")
        combined_df, merge_cols, metadata_cols = load_and_combine_data(files)
        print(f"Combined dataframe shape: {combined_df.shape}")
        
        # Remove the timestamp from the condition column and reorder
        condition_col = None
        for col in combined_df.columns:
            if 'condition' in col.lower():
                condition_col = col
                break
        
        if condition_col:
            print(f"Processing condition column: {condition_col}")
            
            # Extract condition names without timestamps
            def clean_condition(condition_str):
                # Remove timestamp pattern (like _20250507_121703)
                condition_clean = re.sub(r'_\d{8}_\d{6}$', '', str(condition_str))
                return condition_clean
            
            # Apply cleaning function
            combined_df[condition_col] = combined_df[condition_col].apply(clean_condition)
        
        # Define true mutations
        combined_df = define_true_mutations(combined_df)
        
        # Get list of methods that were successfully loaded
        methods = [method for method in files.keys() if f'{method}_detected' in combined_df.columns]
        print(f"Methods with identification data: {methods}")
        
        # Calculate performance metrics
        print("\nCalculating performance metrics...")
        performance_df = calculate_performance_metrics(combined_df, methods)

        # Calculate performance per condition
        print("\nCalculating performance per condition...")
        performance_per_condition_df = calculate_performance_per_condition(combined_df, methods)
        print(f"Calculated performance for {len(performance_per_condition_df)} scenario-condition-method combinations")

        # Save performance per condition to CSV
        perf_per_condition_path = os.path.join(results_dir, 'performance_per_condition.csv')
        performance_per_condition_df.to_csv(perf_per_condition_path, index=False)
        print(f"Performance per condition saved to '{perf_per_condition_path}'")

        # Analyze identification patterns
        print("Analyzing identification patterns...")
        identification_counts, scenario_analysis = analyze_identification_patterns(combined_df, methods)
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(combined_df, methods, performance_df, identification_counts, scenario_analysis, performance_per_condition_df)
        
        # Create and save confusion matrices
        print("Creating confusion matrices...")
        create_confusion_matrices(combined_df, methods)
        
        # Create scenario comparisons
        print("Creating scenario comparisons...")
        create_scenario_comparisons(combined_df, methods, performance_df)

        # Create box plot visualizations for AUROC and F1
        print("\nCreating box plot visualizations for AUROC and F1...")
        create_boxplot_visualizations(performance_per_condition_df, methods)

        # Create AUROC scenarios figure (single figure with all scenarios)
        print("\nCreating AUROC scenarios figure...")
        create_auroc_scenarios_figure(performance_per_condition_df, methods)

        # Create box plots per scenario
        print("\nCreating box plots per scenario...")
        create_boxplot_per_scenario(performance_per_condition_df, methods)

        # Generate summary report
        generate_summary_report(performance_df, identification_counts, scenario_analysis)
        
        # Save detailed metrics to CSV
        print("Saving detailed metrics to CSV...")
        detailed_metrics_df = save_detailed_metrics_to_csv(combined_df, methods)
        
        # Save combined data and performance results to CSV in the results directory
        combined_data_path = os.path.join(results_dir, 'combined_mutation_data.csv')
        performance_path = os.path.join(results_dir, 'performance_metrics.csv')
        
        combined_df.to_csv(combined_data_path, index=False)
        performance_df.to_csv(performance_path)
        print(f"Data saved to '{combined_data_path}' and '{performance_path}'")
        
        # Return the combined dataframe for further analysis
        return combined_df, performance_df
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Run the analysis
if __name__ == "__main__":
    combined_data, performance_results = main()