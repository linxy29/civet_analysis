#!/usr/bin/env python3
"""
Evaluate CIVET performance across different modes for SCENARIO_6_CellCycle

This script compares the performance of CIVET with different covariates:
1. Full model: generation + cell_cycle_potential
2. Generation only: baseline model (generation only)
3. Cell cycle only: cell_cycle_potential only
4. Permuted generation: negative control (shuffled generation labels)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)

def load_civet_results(subrun_dir, mode='full'):
    """
    Load CIVET results for a specific mode

    Parameters
    ----------
    subrun_dir : str
        Path to the simulation subrun directory
    mode : str
        One of 'full', 'generation_only', 'cell_cycle_only', 'permuted'

    Returns
    -------
    pd.DataFrame or None
        CIVET results dataframe
    """
    mode_mapping = {
        'full': 'civet_res',
        'generation_only': 'civet_res_generation_only',
        'cell_cycle_only': 'civet_res_cell_cycle_only',
        'permuted': 'civet_res_permuted'
    }

    result_dir = os.path.join(subrun_dir, mode_mapping[mode])
    result_file = os.path.join(result_dir, 'civet_results.csv')

    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None

    return pd.read_csv(result_file)

def load_ground_truth(subrun_dir):
    """
    Load ground truth mutation information

    Parameters
    ----------
    subrun_dir : str
        Path to the simulation subrun directory

    Returns
    -------
    pd.DataFrame
        Ground truth dataframe with mutation classifications
    """
    metadata_file = os.path.join(subrun_dir, 'metadata', 'simulation_mutation_info.csv')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    df = pd.read_csv(metadata_file)

    # Rename columns to match expected format
    df = df.rename(columns={
        'mutation_id': 'mutation_name',
        'mutation_type': 'mutation_category'
    })

    # Convert mutation_category to lowercase for consistency
    df['mutation_category'] = df['mutation_category'].str.lower()

    return df

def evaluate_mode(civet_df, ground_truth_df, pval_column='LRT_pvals', threshold=0.05, mode='full'):
    """
    Evaluate CIVET performance for a specific mode

    Parameters
    ----------
    civet_df : pd.DataFrame
        CIVET results
    ground_truth_df : pd.DataFrame
        Ground truth mutation information
    pval_column : str
        P-value column to use ('LRT_pvals', 'Wald_pvals', etc.)
    threshold : float
        P-value threshold for significance
    mode : str
        Mode being evaluated (for determining which coefficient to use)

    Returns
    -------
    dict
        Dictionary with performance metrics
    """
    # Filter to get only p-values
    pval_df = civet_df[civet_df['value'] == pval_column].copy()

    if len(pval_df) == 0:
        return None

    # Rename variant column
    pval_df = pval_df.rename(columns={'variant': 'mutation_name'})

    # Determine which column contains the p-values for the main effect
    # Available columns vary by mode
    available_cols = [col for col in pval_df.columns if col not in ['mutation_name', 'value']]

    # Use the first covariate column (usually 'generation' or 'cell_cycle_potential')
    pval_col = available_cols[0] if available_cols else None

    if pval_col is None:
        raise ValueError(f"No p-value column found in CIVET results for mode {mode}")

    merged_df = pd.merge(
        ground_truth_df,
        pval_df[['mutation_name', pval_col]],
        on='mutation_name',
        how='left'
    )
    merged_df = merged_df.rename(columns={pval_col: 'pval'})

    # Remove NA p-values
    merged_df = merged_df.dropna(subset=['pval'])

    # Check if we have enough data after removing NAs
    if len(merged_df) == 0:
        print(f"  Warning: No valid p-values for {pval_column} in mode {mode}")
        return None

    # Classify mutations
    # True positives should be de novo mutations (not baseline, not false)
    merged_df['is_true_positive_class'] = merged_df['mutation_category'] == 'de novo'
    merged_df['is_baseline'] = merged_df['mutation_category'] == 'baseline'
    merged_df['is_false'] = merged_df['mutation_category'] == 'false'

    # Detected = p-value < threshold
    merged_df['detected'] = merged_df['pval'] < threshold

    # Calculate metrics for de novo mutations vs others
    y_true = merged_df['is_true_positive_class'].astype(int)
    y_pred = merged_df['detected'].astype(int)
    y_score = 1 - merged_df['pval']  # Convert p-value to score (lower p-value = higher score)

    # Check if we have valid data
    if len(y_true) == 0 or y_true.nunique() < 2:
        print(f"  Warning: Insufficient data for {pval_column} in mode {mode}")
        return None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 0:
        print(f"  Warning: Empty confusion matrix for {pval_column} in mode {mode}")
        return None

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        if y_true.iloc[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        print(f"  Warning: Unexpected confusion matrix shape {cm.shape} for {pval_column} in mode {mode}")
        return None

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except:
        roc_auc = np.nan
        fpr, tpr = None, None

    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    except:
        pr_auc = np.nan
        precision_curve, recall_curve = None, None

    # Breakdown by mutation category
    category_stats = merged_df.groupby('mutation_category').agg({
        'detected': ['sum', 'count', 'mean']
    }).reset_index()

    return {
        'metrics': {
            'total_mutations': len(merged_df),
            'de_novo_mutations': int(y_true.sum()),
            'baseline_mutations': int(merged_df['is_baseline'].sum()),
            'false_mutations': int(merged_df['is_false'].sum()),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        },
        'category_stats': category_stats,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision_curve, recall_curve),
        'detailed_results': merged_df
    }

def compare_modes(subrun_dir, proliferation_rate, pval_columns=['LRT_pvals', 'Wald_pvals', 'ANOVA_pvals'], threshold=0.05):
    """
    Compare all CIVET modes for a single simulation run using all p-value types

    Parameters
    ----------
    subrun_dir : str
        Path to the simulation subrun directory
    proliferation_rate : float
        Proliferation rate for this run
    pval_columns : list
        List of p-value columns to use
    threshold : float
        P-value threshold for significance

    Returns
    -------
    dict
        Results for all modes and p-value types
    """
    # Load ground truth
    ground_truth = load_ground_truth(subrun_dir)

    results = {}
    modes = ['full', 'generation_only', 'cell_cycle_only', 'permuted']

    for mode in modes:
        civet_df = load_civet_results(subrun_dir, mode=mode)

        if civet_df is not None:
            results[mode] = {}

            for pval_column in pval_columns:
                print(f"Evaluating {mode} mode with {pval_column} for proliferation rate {proliferation_rate}...")

                eval_result = evaluate_mode(civet_df, ground_truth, pval_column, threshold, mode)

                if eval_result is not None:
                    results[mode][pval_column] = eval_result
                    results[mode][pval_column]['proliferation_rate'] = proliferation_rate
                    results[mode][pval_column]['mode'] = mode
                    results[mode][pval_column]['pval_type'] = pval_column
                else:
                    print(f"  Warning: {pval_column} not available for {mode} mode")
        else:
            print(f"Skipping {mode} mode (results not found)")

    return results

def aggregate_results_across_proliferation_rates(scenario_dir, pval_columns=['LRT_pvals', 'Wald_pvals', 'ANOVA_pvals'], threshold=0.05):
    """
    Aggregate results across all proliferation rates and p-value types

    Parameters
    ----------
    scenario_dir : str
        Path to SCENARIO_6_CellCycle directory
    pval_columns : list
        List of p-value columns to use
    threshold : float
        P-value threshold for significance

    Returns
    -------
    dict
        Aggregated results across all proliferation rates
    """
    all_results = {}

    # Find all proliferation subdirectories
    subdirs = [d for d in os.listdir(scenario_dir)
               if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith('proliferation_')]

    for subdir in sorted(subdirs):
        # Extract proliferation rate
        rate_match = subdir.split('_')[1]
        proliferation_rate = float(rate_match)

        subrun_dir = os.path.join(scenario_dir, subdir)

        print(f"\n{'='*80}")
        print(f"Processing proliferation rate: {proliferation_rate}")
        print(f"{'='*80}")

        results = compare_modes(subrun_dir, proliferation_rate, pval_columns, threshold)
        all_results[proliferation_rate] = results

    return all_results

def create_summary_table(all_results):
    """
    Create a summary table of performance metrics

    Parameters
    ----------
    all_results : dict
        Results from aggregate_results_across_proliferation_rates

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    rows = []

    for proliferation_rate, mode_results in all_results.items():
        for mode, pval_results in mode_results.items():
            for pval_type, result in pval_results.items():
                if 'metrics' in result:
                    row = {
                        'proliferation_rate': proliferation_rate,
                        'mode': mode,
                        'pval_type': pval_type,
                        **result['metrics']
                    }
                    rows.append(row)

    return pd.DataFrame(rows)

def plot_performance_comparison(summary_df, output_dir):
    """
    Create visualization comparing performance across modes and p-value types

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary table from create_summary_table
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Plot for each p-value type separately
    for pval_type in summary_df['pval_type'].unique():
        pval_data = summary_df[summary_df['pval_type'] == pval_type]

        # 1. Sensitivity comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['sensitivity', 'specificity', 'precision', 'f1_score']
        titles = ['Sensitivity (Recall)', 'Specificity', 'Precision', 'F1 Score']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            for mode in pval_data['mode'].unique():
                mode_data = pval_data[pval_data['mode'] == mode]
                ax.plot(mode_data['proliferation_rate'], mode_data[metric],
                       marker='o', label=mode, linewidth=2)

            ax.set_xlabel('Proliferation Rate', fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f'{title} ({pval_type})', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_metrics_{pval_type}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. ROC-AUC comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode in pval_data['mode'].unique():
            mode_data = pval_data[pval_data['mode'] == mode]
            ax.plot(mode_data['proliferation_rate'], mode_data['roc_auc'],
                   marker='o', label=mode, linewidth=2)

        ax.set_xlabel('Proliferation Rate', fontsize=12)
        ax.set_ylabel('ROC-AUC', fontsize=12)
        ax.set_title(f'ROC-AUC Comparison ({pval_type})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_auc_comparison_{pval_type}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Compare p-value types for best mode
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['sensitivity', 'specificity', 'precision', 'f1_score']
    titles = ['Sensitivity (Recall)', 'Specificity', 'Precision', 'F1 Score']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for pval_type in summary_df['pval_type'].unique():
            # Use full model for comparison
            full_data = summary_df[(summary_df['pval_type'] == pval_type) & (summary_df['mode'] == 'full')]
            if len(full_data) > 0:
                ax.plot(full_data['proliferation_rate'], full_data[metric],
                       marker='o', label=pval_type, linewidth=2)

        ax.set_xlabel('Proliferation Rate', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} (Full Model)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pval_type_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}")

def main():
    """Main execution function"""

    # Configuration
    scenario_dir = '/Users/linxy29/Documents/Data/CIVET/simulation/SCENARIO_6_CellCycle'
    output_dir = '/Users/linxy29/Documents/Code/civet_analysis/figures/scenario6_mode_comparison'
    pval_columns = ['LRT_pvals', 'Wald_pvals', 'ANOVA_pvals']
    threshold = 0.05

    print("="*80)
    print("CIVET Mode Comparison for SCENARIO_6_CellCycle")
    print("Using all p-value types:", ', '.join(pval_columns))
    print("="*80)

    # Aggregate results
    all_results = aggregate_results_across_proliferation_rates(
        scenario_dir, pval_columns, threshold
    )

    # Create summary table
    summary_df = create_summary_table(all_results)

    # Save summary table
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'mode_comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary table saved to {summary_file}")

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # Format and display the summary
    display_cols = ['proliferation_rate', 'mode', 'pval_type', 'sensitivity', 'specificity',
                   'precision', 'f1_score', 'roc_auc']
    print(summary_df[display_cols].to_string(index=False))

    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)

    plot_performance_comparison(summary_df, output_dir)

    # Additional analysis: Mode ranking by p-value type
    print("\n" + "="*80)
    print("MODE RANKING BY PROLIFERATION RATE AND P-VALUE TYPE")
    print("="*80)

    for pval_type in sorted(summary_df['pval_type'].unique()):
        print(f"\n{'='*80}")
        print(f"P-value Type: {pval_type}")
        print(f"{'='*80}")

        pval_data = summary_df[summary_df['pval_type'] == pval_type]

        for rate in sorted(pval_data['proliferation_rate'].unique()):
            rate_data = pval_data[pval_data['proliferation_rate'] == rate].sort_values(
                'f1_score', ascending=False
            )
            print(f"\nProliferation Rate: {rate}")
            print(rate_data[['mode', 'f1_score', 'roc_auc', 'sensitivity', 'precision']].to_string(index=False))

    # Best overall performance
    print("\n" + "="*80)
    print("BEST PERFORMING CONFIGURATIONS")
    print("="*80)

    best_configs = summary_df.nlargest(10, 'f1_score')[
        ['proliferation_rate', 'mode', 'pval_type', 'f1_score', 'roc_auc', 'sensitivity', 'precision']
    ]
    print(best_configs.to_string(index=False))

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
