#!/usr/bin/env python3
"""
Generate box plots comparing Precision and F1 scores across different bias conditions
for Scenario 2 (Segregation).

This script creates box plots showing the distribution of metrics across all methods
for bias 0.5, 0.7, and 0.9 conditions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set the working directory
working_dir = "/Users/linxy29/Documents/Data/CIVET/simulation"
os.chdir(working_dir)
print(f"Working directory set to: {os.getcwd()}")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_metrics_from_bias_condition(bias_value):
    """
    Load metrics from a specific bias condition analysis folder.

    Args:
        bias_value: The bias value (e.g., '05', '07', '09')

    Returns:
        DataFrame with metrics including a 'bias' column
    """
    # Construct the analysis folder name
    analysis_folder = f"scenario2_bias{bias_value}_analysis"
    metrics_file = os.path.join(working_dir, analysis_folder, "metrics_summary_all.csv")

    if not os.path.exists(metrics_file):
        print(f"Warning: {metrics_file} not found")
        return None

    # Load the metrics
    df = pd.read_csv(metrics_file)

    # Add bias column (convert to float for proper sorting)
    # bias_value is like '05', '07', '09', we want 0.5, 0.7, 0.9
    df['bias'] = float(bias_value) / 10.0

    print(f"Loaded {len(df)} methods from bias {bias_value}")

    return df

def create_box_plot_comparison(combined_df, output_path):
    """
    Create box plots comparing precision and F1 scores across bias conditions.

    Args:
        combined_df: DataFrame with metrics from all bias conditions
        output_path: Path to save the figure
    """
    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 20
    })

    # Create figure with 2 subplots (same layout as precision_f1_separate_all.png)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for each bias condition
    bias_colors = {
        0.5: '#3498db',  # Blue
        0.7: '#e74c3c',  # Red
        0.9: '#2ecc71'   # Green
    }

    # Plot 1: Precision comparison
    ax1 = axes[0]

    # Prepare data for box plot
    bias_values = sorted(combined_df['bias'].unique())
    precision_data = [combined_df[combined_df['bias'] == b]['Precision'].values for b in bias_values]

    # Create box plot
    bp1 = ax1.boxplot(precision_data,
                      positions=range(len(bias_values)),
                      widths=0.6,
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                      medianprops=dict(color='black', linewidth=2),
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

    # Color the boxes
    for patch, bias in zip(bp1['boxes'], bias_values):
        patch.set_facecolor(bias_colors[bias])
        patch.set_alpha(0.7)

    ax1.set_ylabel('Precision', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Bias Condition', fontsize=16, fontweight='bold')
    ax1.set_title('Precision by Bias Condition', fontsize=18, fontweight='bold')
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xticks(range(len(bias_values)))
    ax1.set_xticklabels([f'Bias {b}' for b in bias_values])
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: F1 Score comparison
    ax2 = axes[1]

    # Prepare data for box plot
    f1_data = [combined_df[combined_df['bias'] == b]['F1_Score'].values for b in bias_values]

    # Create box plot
    bp2 = ax2.boxplot(f1_data,
                      positions=range(len(bias_values)),
                      widths=0.6,
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                      medianprops=dict(color='black', linewidth=2),
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

    # Color the boxes
    for patch, bias in zip(bp2['boxes'], bias_values):
        patch.set_facecolor(bias_colors[bias])
        patch.set_alpha(0.7)

    ax2.set_ylabel('F1 Score', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Bias Condition', fontsize=16, fontweight='bold')
    ax2.set_title('F1 Score by Bias Condition', fontsize=18, fontweight='bold')
    ax2.set_ylim(0.0, 0.5)
    ax2.set_xticks(range(len(bias_values)))
    ax2.set_xticklabels([f'Bias {b}' for b in bias_values])
    ax2.grid(axis='y', alpha=0.3)

    # Add legend explaining box plot elements
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                   markersize=8, label='Mean'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.suptitle('Precision and F1 Score Comparison Across Bias Conditions\nScenario 2: Segregation (All Methods)',
                fontsize=20, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBox plot comparison saved to: {output_path}")
    plt.close()

def create_method_comparison_plot(combined_df, output_path):
    """
    Create grouped box plots showing each method's performance across bias conditions.

    Args:
        combined_df: DataFrame with metrics from all bias conditions
        output_path: Path to save the figure
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Get unique methods and bias values
    methods = combined_df['Method'].unique()
    bias_values = sorted(combined_df['bias'].unique())

    # Define colors for each bias condition
    bias_colors = {
        0.5: '#3498db',  # Blue
        0.7: '#e74c3c',  # Red
        0.9: '#2ecc71'   # Green
    }

    # Plot 1: Precision comparison by method
    ax1 = axes[0]
    x_positions = np.arange(len(methods))
    width = 0.25

    for idx, bias in enumerate(bias_values):
        bias_data = combined_df[combined_df['bias'] == bias]
        precision_values = [bias_data[bias_data['Method'] == m]['Precision'].values[0]
                           if len(bias_data[bias_data['Method'] == m]) > 0 else 0
                           for m in methods]

        offset = (idx - len(bias_values)/2 + 0.5) * width
        ax1.bar(x_positions + offset, precision_values, width,
               label=f'Bias {bias}', color=bias_colors[bias], alpha=0.8)

    ax1.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax1.set_title('Precision by Method and Bias Condition', fontsize=16, fontweight='bold')
    ax1.set_ylim(0.0, 1.1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: F1 Score comparison by method
    ax2 = axes[1]

    for idx, bias in enumerate(bias_values):
        bias_data = combined_df[combined_df['bias'] == bias]
        f1_values = [bias_data[bias_data['Method'] == m]['F1_Score'].values[0]
                    if len(bias_data[bias_data['Method'] == m]) > 0 else 0
                    for m in methods]

        offset = (idx - len(bias_values)/2 + 0.5) * width
        ax2.bar(x_positions + offset, f1_values, width,
               label=f'Bias {bias}', color=bias_colors[bias], alpha=0.8)

    ax2.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax2.set_title('F1 Score by Method and Bias Condition', fontsize=16, fontweight='bold')
    ax2.set_ylim(0.0, 0.5)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Method Performance Across Bias Conditions\nScenario 2: Segregation',
                fontsize=18, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Method comparison plot saved to: {output_path}")
    plt.close()

def print_summary_statistics(combined_df):
    """Print summary statistics for each bias condition."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY BIAS CONDITION")
    print("="*80)

    for bias in sorted(combined_df['bias'].unique()):
        bias_data = combined_df[combined_df['bias'] == bias]
        print(f"\nBias {bias}:")
        print(f"  Number of methods: {len(bias_data)}")
        print(f"  Precision - Mean: {bias_data['Precision'].mean():.3f}, "
              f"Median: {bias_data['Precision'].median():.3f}, "
              f"Std: {bias_data['Precision'].std():.3f}")
        print(f"  F1 Score  - Mean: {bias_data['F1_Score'].mean():.3f}, "
              f"Median: {bias_data['F1_Score'].median():.3f}, "
              f"Std: {bias_data['F1_Score'].std():.3f}")
        print(f"  Recall    - Mean: {bias_data['Recall'].mean():.3f}, "
              f"Median: {bias_data['Recall'].median():.3f}, "
              f"Std: {bias_data['Recall'].std():.3f}")

def main():
    """Main function to generate comparison plots."""
    print("\n" + "="*80)
    print("GENERATING BIAS COMPARISON PLOTS FOR SCENARIO 2")
    print("="*80)

    # Load metrics from all bias conditions
    bias_conditions = ['05', '07', '09']
    all_dfs = []

    for bias in bias_conditions:
        df = load_metrics_from_bias_condition(bias)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("Error: No data could be loaded from any bias condition")
        return

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined data: {len(combined_df)} rows")
    print(f"Bias conditions: {sorted(combined_df['bias'].unique())}")
    print(f"Methods: {combined_df['Method'].unique().tolist()}")

    # Create output directory
    output_dir = os.path.join(working_dir, "scenario2_bias_comparison")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate box plot comparison
    box_plot_path = os.path.join(output_dir, "bias_comparison_boxplot.png")
    create_box_plot_comparison(combined_df, box_plot_path)

    # Generate method comparison plot
    method_plot_path = os.path.join(output_dir, "method_comparison_by_bias.png")
    create_method_comparison_plot(combined_df, method_plot_path)

    # Save combined metrics to CSV
    combined_csv_path = os.path.join(output_dir, "combined_metrics_all_bias.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined metrics saved to: {combined_csv_path}")

    # Print summary statistics
    print_summary_statistics(combined_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {box_plot_path}")
    print(f"  2. {method_plot_path}")
    print(f"  3. {combined_csv_path}")

if __name__ == "__main__":
    main()
