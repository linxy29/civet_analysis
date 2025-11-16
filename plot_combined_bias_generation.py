#!/usr/bin/env python3
"""
Combined Plot: Bias Comparison Bar Plots + Generation Covariate Bar Plots

This script creates a combined visualization with:
- Column 1: Bar plots comparing Precision and F1 Score across methods (mean across all bias conditions) (Scenario 2)
- Columns 2-3: Bar plots showing performance metrics for generation covariate (Scenario 6)

Layout: 2 rows × 3 columns
The style follows the barplot_generation_covariate.png format from civet_comprehensive_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set working directories
SIMULATION_DIR = "/Users/linxy29/Documents/Data/CIVET/simulation"
ANALYSIS_RESULTS_DIR = os.path.join(SIMULATION_DIR, "civet_analysis_results")
BIAS_COMPARISON_DIR = os.path.join(SIMULATION_DIR, "scenario2_bias_comparison")

def load_bias_comparison_data():
    """Load combined metrics from bias comparison"""
    csv_path = os.path.join(BIAS_COMPARISON_DIR, "combined_metrics_all_bias.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Bias comparison data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded bias comparison data: {len(df)} rows")
    return df

def load_generation_covariate_data():
    """Load generation covariate data from civet_analysis_results"""
    csv_path = os.path.join(ANALYSIS_RESULTS_DIR, "data", "complete_results.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Generation covariate data not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter for generation covariate only
    df = df[df['covariate'] == 'generation']
    print(f"Loaded generation covariate data: {len(df)} rows")
    return df

def create_combined_plot(bias_df, gen_df, output_path):
    """
    Create combined plot with bias bar plots (column 1) and generation bar plots (columns 2-3)

    Column 1 shows mean performance across all bias conditions for each method.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Bias comparison data
    gen_df : pd.DataFrame
        Generation covariate data
    output_path : str
        Path to save the figure
    """
    # Set font sizes to match barplot_generation_covariate.png
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
        'figure.titlesize': 15
    })

    # Create figure with 2 rows × 3 columns
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.8)

    # Define colors for generation bar plots (6 configurations)
    gen_colors = {
        ('full', 'Wald_pvals'): '#2E86AB',              # Dark Blue
        ('full', 'LRT_pvals'): '#5AB1BB',               # Light Blue
        ('permuted', 'Wald_pvals'): '#F18F01',          # Dark Orange
        ('permuted', 'LRT_pvals'): '#FFA500',           # Light Orange
        ('generation_only', 'Wald_pvals'): '#9D4EDD',   # Dark Purple
        ('generation_only', 'LRT_pvals'): '#C77DFF',    # Light Purple
    }

    # ==========================================================================
    # COLUMN 1: BIAS COMPARISON BAR PLOTS
    # ==========================================================================

    # Get unique methods and bias values
    methods = sorted(bias_df['Method'].unique())
    bias_values = sorted(bias_df['bias'].unique())

    # Subplot 1: Precision bar plot (row 0, column 0)
    ax1 = fig.add_subplot(gs[0, 0])

    # Calculate mean and std across all bias conditions for each method
    x = np.arange(len(methods))
    means = []
    stds = []

    for method in methods:
        # Get all precision values for this method across all bias conditions
        data = bias_df[bias_df['Method'] == method]['Precision'].values
        if len(data) > 0:
            means.append(np.mean(data))
            stds.append(np.std(data))
        else:
            means.append(0)
            stds.append(0)

    ax1.bar(x, means, yerr=stds,
            color='#3498db',
            alpha=0.85,
            capsize=4,
            error_kw={'linewidth': 1.5, 'elinewidth': 1.5})

    ax1.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax1.set_title('Precision by Method', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: F1 Score bar plot (row 1, column 0)
    ax2 = fig.add_subplot(gs[1, 0])

    # Calculate mean and std across all bias conditions for each method
    means_f1 = []
    stds_f1 = []

    for method in methods:
        # Get all F1 Score values for this method across all bias conditions
        data = bias_df[bias_df['Method'] == method]['F1_Score'].values
        if len(data) > 0:
            means_f1.append(np.mean(data))
            stds_f1.append(np.std(data))
        else:
            means_f1.append(0)
            stds_f1.append(0)

    ax2.bar(x, means_f1, yerr=stds_f1,
            color='#e74c3c',
            alpha=0.85,
            capsize=4,
            error_kw={'linewidth': 1.5, 'elinewidth': 1.5})

    ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax2.set_title('F1 Score by Method', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.0, 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # ==========================================================================
    # COLUMNS 2-3: GENERATION COVARIATE BAR PLOTS
    # ==========================================================================

    # Define metrics and titles
    metrics = ['precision', 'f1_score', 'pr_auc', 'roc_auc']
    titles = ['Precision', 'F1 Score', 'AUPRC', 'AUROC']

    # Define configurations for generation plot (6 configurations)
    gen_configs = [
        ('full', 'Wald_pvals', 'Full + Gen + Wald'),
        ('full', 'LRT_pvals', 'Full + Gen + LRT'),
        ('permuted', 'Wald_pvals', 'Permuted + Gen + Wald'),
        ('permuted', 'LRT_pvals', 'Permuted + Gen + LRT'),
        ('generation_only', 'Wald_pvals', 'CIVET_res + Gen + Wald'),
        ('generation_only', 'LRT_pvals', 'CIVET_res + Gen + LRT'),
    ]

    prolif_rates = sorted(gen_df['proliferation_rate'].unique())
    x = np.arange(len(prolif_rates))
    width = 0.13  # Narrower bars to accommodate 6 configurations

    # Create 4 subplots for generation covariate (filling columns 2 and 3)
    subplot_positions = [gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]]

    for subplot_idx, (metric, title, subplot_pos) in enumerate(zip(metrics, titles, subplot_positions)):
        ax = fig.add_subplot(subplot_pos)

        # Plot bars for each configuration
        for config_idx, (mode, pval, label) in enumerate(gen_configs):
            subset = gen_df[(gen_df['mode'] == mode) &
                           (gen_df['pval_type'] == pval)]

            values = [subset[subset['proliferation_rate'] == rate][metric].values[0]
                     if len(subset[subset['proliferation_rate'] == rate]) > 0 else 0
                     for rate in prolif_rates]

            offset = (config_idx - len(gen_configs)/2 + 0.5) * width
            ax.bar(x + offset, values, width,
                  label=label,
                  color=gen_colors[(mode, pval)],
                  alpha=0.85)

        ax.set_xlabel('Proliferation Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(prolif_rates)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3, axis='y')

    # Add overall title
    fig.suptitle('Combined Analysis: Scenario 2 Bias Comparison & Scenario 6 Generation Covariate',
                fontsize=15, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_path}")
    plt.close()

def main():
    """Main execution function"""
    print("="*80)
    print("CREATING COMBINED PLOT")
    print("="*80)

    # Load data
    print("\nLoading data...")
    bias_df = load_bias_comparison_data()
    gen_df = load_generation_covariate_data()

    # Create output directory
    output_dir = os.path.join(SIMULATION_DIR, "combined_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Create combined plot
    output_path = os.path.join(output_dir, "combined_bias_generation_plot.png")
    print("\nCreating combined plot...")
    create_combined_plot(bias_df, gen_df, output_path)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated file: {output_path}")

if __name__ == "__main__":
    main()
