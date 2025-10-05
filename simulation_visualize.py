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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
import re
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def sort_mutation_rates_numerically(conditions):
    """
    Sort mutation rate conditions numerically instead of alphabetically.
    For example: mutation_rate_1, mutation_rate_2, mutation_rate_4, mutation_rate_8, mutation_rate_16
    """
    def extract_rate(condition):
        # Extract the number from mutation_rate_X pattern
        match = re.search(r'mutation_rate_(\d+)', condition)
        if match:
            return int(match.group(1))
        return float('inf')  # Put non-mutation_rate conditions at the end
    
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
        
        # Select columns to merge (identification column + merge keys)
        cols_to_keep = list(merge_cols)  # Use a copy of merge_cols
        detected_col = f'{method}_detected'
        if detected_col in df_merge.columns:
            cols_to_keep.append(detected_col)
        
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

        # Calculate AUROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auroc = auc(fpr, tpr)
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

def create_visualizations(df, methods, performance_df, identification_counts, scenario_analysis):
    """
    Create comprehensive visualizations with six specific subplots:
    1) Number of mutations identified by method
    2) Precision and % effective SNPs
    3) AUROC by method
    4) Proportion of true/false mutations (stacked bar)
    5) Method identification correlation
    6) F1 Score by method
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
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=16)

    plt.title('Total identifications per Method', fontsize=22)
    plt.ylabel('Number of identifications', fontsize=24)
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

    # 3. AUROC by method
    plt.subplot(2, 3, 3)

    methods_with_data = [m for m in methods if m in performance_df.index]
    auroc_values = [performance_df.loc[m, 'AUROC'] if m in performance_df.index and 'AUROC' in performance_df.columns else 0 for m in methods_with_data]

    bars = plt.bar(methods_with_data, auroc_values, color='steelblue')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16)

    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('AUROC', fontsize=24)
    plt.title('AUROC by Method', fontsize=22)
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(0, 1)

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
    plt.subplot(2, 3, 5)
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

    # 6. F1 Score by method
    plt.subplot(2, 3, 6)

    methods_with_data = [m for m in methods if m in performance_df.index]
    f1_values = [performance_df.loc[m, 'F1_Score'] if m in performance_df.index and 'F1_Score' in performance_df.columns else 0 for m in methods_with_data]

    bars = plt.bar(methods_with_data, f1_values, color='coral')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16)

    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('F1 Score', fontsize=24)
    plt.title('F1 Score by Method', fontsize=22)
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(0, 1)

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
    Create a single plot with six subplots comparing precision, identification counts,
    and mutation proportions across conditions for scenarios 1 and 2,
    plus a separate plot for scenario 3
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
    
    # We'll focus on the first two scenarios (or one if only one exists)
    target_scenarios = scenarios[:min(2, len(scenarios))]
    if len(target_scenarios) < 1:
        print("Not enough scenarios to create comparison plot.")
        return
        
    # Create a single figure with a 2×3 layout (two rows with three subplots each)
    # Increase figure width to accommodate scenario labels on the left
    fig = plt.figure(figsize=(36, 18))
    plt.suptitle(f'Scenario and Condition Comparison', fontsize=26)
    
    # Define subplot positions
    # Row 1: Number of identifications and Precision for Scenario 1, and Mean Mutation Proportions for Scenario 1
    # Row 2: Number of identifications and Precision for Scenario 2, and Mean Mutation Proportions for Scenario 2
    subplot_positions = {
        0: {'identifications': 1, 'precision': 2, 'proportions': 3},
        1: {'identifications': 4, 'precision': 5, 'proportions': 6}
    }
    
    # For each target scenario, create identification count and precision subplots
    for i, scenario in enumerate(target_scenarios):
        scenario_df = df[df[scenario_col] == scenario]
        scenario_conditions = scenario_df[condition_col].unique()
        
        # Sort conditions numerically if they are mutation rates
        scenario_conditions = sort_mutation_rates_numerically(scenario_conditions)
        
        # 1. Number of identifications by condition for this scenario
        plt.subplot(2, 3, subplot_positions[i]['identifications'])
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
        plt.subplot(2, 3, subplot_positions[i]['precision'])
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
        plt.subplot(2, 3, subplot_positions[i]['proportions'])
        
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
        y_pos = 0.75 - i * 0.5  # Center vertically for each row
        
        # Add scenario label
        fig.text(x_pos, y_pos, scenario, fontsize=24, fontweight='bold', 
                rotation=90, ha='center', va='center')
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Adjust for the suptitle and scenario labels
    
    # Save the figure
    comparison_path = os.path.join(results_dir, 'scenario_condition_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Scenario-condition comparison saved as '{comparison_path}'")
    
    # Now create a separate figure for scenario 3 if it exists
    if len(scenarios) >= 3:
        scenario3 = scenarios[2]
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

def create_mutation_proportion_plots(df, methods):
    """
    Create stacked bar plots showing proportions of true mutations and different types of false mutations
    for each condition within each scenario
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
        print("Warning: No scenario column found. Cannot create scenario-based plots.")
        return
        
    if not condition_col:
        print("Warning: No condition column found. Cannot create condition-based plots.")
        return
    
    # Get unique scenarios and conditions
    scenarios = df[scenario_col].unique()
    
    # Create directory for results
    results_dir = os.path.join(os.getcwd(), "overall_analysis")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # For each scenario, create a figure with subplots for each condition
    for scenario in scenarios:
        scenario_df = df[df[scenario_col] == scenario]
        conditions = scenario_df[condition_col].unique()
        
        # Sort conditions numerically if they are mutation rates
        conditions = sort_mutation_rates_numerically(conditions)
        
        # Calculate number of rows and columns for subplots
        n_conditions = len(conditions)
        n_cols = min(3, n_conditions)  # Maximum 3 columns
        n_rows = (n_conditions + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with larger size
        plt.figure(figsize=(10*n_cols, 8*n_rows))
        plt.suptitle(f'Mutation Proportions: {scenario}', fontsize=26)
        
        # For each condition, create a subplot
        for i, condition in enumerate(conditions):
            condition_df = scenario_df[scenario_df[condition_col] == condition]
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Calculate proportions for each method
            stacked_data = []
            
            for method in methods:
                detected_col = f'{method}_detected'
                if detected_col not in condition_df.columns or 'true_mutation' not in condition_df.columns:
                    continue
                    
                # Filter to only identified mutations
                detected = condition_df[detected_col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                detected_df = condition_df[detected == True].copy()
                
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
                stacked_df.plot(kind='bar', stacked=True, ax=ax)
                plt.title(f'Condition: {condition}', fontsize=22)
                plt.xlabel('Method', fontsize=20)
                plt.ylabel('Proportion', fontsize=24)
                plt.xticks(rotation=45, fontsize=24)
                plt.yticks(fontsize=24)
                plt.ylim(0, 1)
                
                # Add legend to the first subplot only
                if i == 0:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
                else:
                    plt.legend().set_visible(False)
            else:
                plt.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=18)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure
        proportion_path = os.path.join(results_dir, f'scenario_{scenario}_mutation_proportions.png')
        plt.savefig(proportion_path, dpi=300, bbox_inches='tight')
        print(f"Mutation proportions for scenario {scenario} saved as '{proportion_path}'")
    
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
        "civet_Wald": "civet_mutation_combine_Wald_pvals_threshold_0.05_with_metadata_with_informative.csv"
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
        
        # Analyze identification patterns
        print("Analyzing identification patterns...")
        identification_counts, scenario_analysis = analyze_identification_patterns(combined_df, methods)
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(combined_df, methods, performance_df, identification_counts, scenario_analysis)
        
        # Create and save confusion matrices
        print("Creating confusion matrices...")
        create_confusion_matrices(combined_df, methods)
        
        # Create scenario comparisons
        print("Creating scenario comparisons...")
        create_scenario_comparisons(combined_df, methods, performance_df)
        
        # Create mutation proportion plots
        print("Creating mutation proportion plots...")
        create_mutation_proportion_plots(combined_df, methods)
        
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