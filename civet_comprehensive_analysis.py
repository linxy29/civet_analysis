#!/usr/bin/env python3
"""
Comprehensive CIVET Performance Analysis - SCENARIO_6_CellCycle
All-in-one script for complete analysis with separate covariate evaluation

Author: Analysis Pipeline
Date: November 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
SCENARIO_DIR = '/Users/linxy29/Documents/Data/CIVET/simulation/SCENARIO_6_CellCycle'
OUTPUT_BASE_DIR = '/Users/linxy29/Documents/Data/CIVET/simulation/civet_analysis_results'

# Analysis parameters
PVAL_TYPES = ['LRT_pvals', 'Wald_pvals']
THRESHOLD = 0.05
MODES = {
    'full': 'civet_res',
    'generation_only': 'civet_res_generation_only',
    'cell_cycle_only': 'civet_res_cell_cycle_only',
    'permuted': 'civet_res_permuted'
}

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_output_directory(base_dir):
    """Create organized output directory structure"""
    dirs = {
        'base': base_dir,
        'tables': os.path.join(base_dir, 'tables'),
        'figures': os.path.join(base_dir, 'figures'),
        'reports': os.path.join(base_dir, 'reports'),
        'data': os.path.join(base_dir, 'data')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def load_civet_results(subrun_dir, mode):
    """Load CIVET results for a specific mode"""
    result_dir = os.path.join(subrun_dir, MODES[mode])
    result_file = os.path.join(result_dir, 'civet_results.csv')

    if not os.path.exists(result_file):
        return None

    return pd.read_csv(result_file)

def load_ground_truth(subrun_dir):
    """Load ground truth mutation information"""
    metadata_file = os.path.join(subrun_dir, 'metadata', 'simulation_mutation_info.csv')

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    df = pd.read_csv(metadata_file)
    df = df.rename(columns={
        'mutation_id': 'mutation_name',
        'mutation_type': 'mutation_category'
    })
    df['mutation_category'] = df['mutation_category'].str.lower()

    return df

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_covariate(civet_df, ground_truth_df, pval_column, covariate_name, threshold):
    """
    Evaluate CIVET performance for a specific covariate

    Parameters
    ----------
    civet_df : pd.DataFrame
        CIVET results
    ground_truth_df : pd.DataFrame
        Ground truth mutation information
    pval_column : str
        P-value column to use ('LRT_pvals', 'Wald_pvals')
    covariate_name : str
        Name of covariate column to evaluate ('generation', 'cell_cycle_potential')
    threshold : float
        P-value threshold for significance

    Returns
    -------
    dict or None
        Dictionary with performance metrics
    """
    # Filter to get only p-values
    pval_df = civet_df[civet_df['value'] == pval_column].copy()

    if len(pval_df) == 0:
        return None

    # Check if covariate exists
    if covariate_name not in pval_df.columns:
        return None

    # Prepare data
    pval_df = pval_df.rename(columns={'variant': 'mutation_name'})

    merged_df = pd.merge(
        ground_truth_df,
        pval_df[['mutation_name', covariate_name]],
        on='mutation_name',
        how='left'
    )
    merged_df = merged_df.rename(columns={covariate_name: 'pval'})

    # Remove NA p-values
    merged_df = merged_df.dropna(subset=['pval'])

    if len(merged_df) == 0:
        return None

    # Classify mutations
    merged_df['is_true_positive_class'] = merged_df['mutation_category'] == 'de novo'
    merged_df['is_baseline'] = merged_df['mutation_category'] == 'baseline'
    merged_df['is_false'] = merged_df['mutation_category'] == 'false'

    # Detected = p-value < threshold
    merged_df['detected'] = merged_df['pval'] < threshold

    # Calculate metrics
    y_true = merged_df['is_true_positive_class'].astype(int)
    y_pred = merged_df['detected'].astype(int)
    y_score = 1 - merged_df['pval']

    if len(y_true) == 0 or y_true.nunique() < 2:
        return None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 0:
        return None

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if y_true.iloc[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        return None

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except:
        roc_auc = np.nan

    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except:
        pr_auc = np.nan

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
        'detailed_results': merged_df
    }

def analyze_simulation_run(subrun_dir, proliferation_rate):
    """Analyze all modes and covariates for a single simulation run"""

    ground_truth = load_ground_truth(subrun_dir)
    results = {}

    for mode_name, mode_dir in MODES.items():
        civet_df = load_civet_results(subrun_dir, mode_name)

        if civet_df is None:
            print(f"  Skipping {mode_name} (results not found)")
            continue

        results[mode_name] = {}

        # Determine available covariates
        sample_row = civet_df[civet_df['value'] == PVAL_TYPES[0]].iloc[0] if len(civet_df) > 0 else None

        if sample_row is None:
            continue

        available_covariates = [col for col in sample_row.index
                               if col not in ['variant', 'value']]

        for pval_type in PVAL_TYPES:
            results[mode_name][pval_type] = {}

            for covariate in available_covariates:
                eval_result = evaluate_covariate(
                    civet_df, ground_truth, pval_type, covariate, THRESHOLD
                )

                if eval_result is not None:
                    results[mode_name][pval_type][covariate] = eval_result
                    results[mode_name][pval_type][covariate]['proliferation_rate'] = proliferation_rate
                    results[mode_name][pval_type][covariate]['mode'] = mode_name
                    results[mode_name][pval_type][covariate]['pval_type'] = pval_type
                    results[mode_name][pval_type][covariate]['covariate'] = covariate

    return results

def aggregate_all_results():
    """Aggregate results across all proliferation rates"""

    print("="*80)
    print("CIVET COMPREHENSIVE ANALYSIS - SCENARIO_6_CellCycle")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  P-value types: {', '.join(PVAL_TYPES)}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Modes: {', '.join(MODES.keys())}")
    print(f"  Output: {OUTPUT_BASE_DIR}")

    all_results = {}

    # Find all proliferation subdirectories
    subdirs = [d for d in os.listdir(SCENARIO_DIR)
               if os.path.isdir(os.path.join(SCENARIO_DIR, d)) and d.startswith('proliferation_')]

    for subdir in sorted(subdirs):
        rate_match = subdir.split('_')[1]
        proliferation_rate = float(rate_match)

        subrun_dir = os.path.join(SCENARIO_DIR, subdir)

        print(f"\n{'='*80}")
        print(f"Processing proliferation rate: {proliferation_rate}")
        print(f"{'='*80}")

        results = analyze_simulation_run(subrun_dir, proliferation_rate)
        all_results[proliferation_rate] = results

    return all_results

def create_summary_dataframe(all_results):
    """Convert nested results dictionary to DataFrame"""

    rows = []

    for proliferation_rate, mode_results in all_results.items():
        for mode, pval_results in mode_results.items():
            for pval_type, covariate_results in pval_results.items():
                for covariate, result in covariate_results.items():
                    if 'metrics' in result:
                        row = {
                            'proliferation_rate': proliferation_rate,
                            'mode': mode,
                            'pval_type': pval_type,
                            'covariate': covariate,
                            **result['metrics']
                        }
                        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def create_summary_tables(df, output_dirs):
    """Create comprehensive summary tables"""

    print("\n" + "="*80)
    print("Creating Summary Tables")
    print("="*80)

    tables_dir = output_dirs['tables']

    # 1. Overall summary
    overall_summary = df.groupby(['mode', 'pval_type', 'covariate']).agg({
        'f1_score': ['mean', 'std', 'min', 'max'],
        'roc_auc': ['mean', 'std', 'min', 'max'],
        'sensitivity': ['mean', 'std', 'min', 'max'],
        'specificity': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max']
    }).round(4)

    overall_summary.to_csv(os.path.join(tables_dir, 'overall_summary.csv'))
    print(f"✓ Overall summary: overall_summary.csv")

    # 2. By covariate
    for covariate in df['covariate'].unique():
        cov_data = df[df['covariate'] == covariate]

        cov_summary = cov_data.groupby(['mode', 'pval_type', 'proliferation_rate'])[
            ['f1_score', 'roc_auc', 'sensitivity', 'specificity', 'precision',
             'tp', 'tn', 'fp', 'fn']
        ].mean().round(4)

        filename = f'summary_{covariate}.csv'
        cov_summary.to_csv(os.path.join(tables_dir, filename))
        print(f"✓ Covariate summary: {filename}")

    # 3. Best configurations
    best_configs = []
    for metric in ['f1_score', 'roc_auc', 'sensitivity', 'precision']:
        best_row = df.loc[df[metric].idxmax()]
        best_configs.append({
            'metric': metric,
            'value': best_row[metric],
            'mode': best_row['mode'],
            'pval_type': best_row['pval_type'],
            'covariate': best_row['covariate'],
            'proliferation_rate': best_row['proliferation_rate']
        })

    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(os.path.join(tables_dir, 'best_configurations.csv'), index=False)
    print(f"✓ Best configurations: best_configurations.csv")

    # 4. Statistical tests
    stats_results = []

    for pval_type in df['pval_type'].unique():
        for covariate in df['covariate'].unique():
            subset = df[(df['pval_type'] == pval_type) & (df['covariate'] == covariate)]

            if 'full' not in subset['mode'].unique():
                continue

            full_f1 = subset[subset['mode'] == 'full']['f1_score'].values

            for mode in subset['mode'].unique():
                if mode == 'full':
                    continue

                mode_f1 = subset[subset['mode'] == mode]['f1_score'].values

                if len(full_f1) > 1 and len(mode_f1) > 1:
                    stat, pval = mannwhitneyu(full_f1, mode_f1, alternative='greater')

                    stats_results.append({
                        'pval_type': pval_type,
                        'covariate': covariate,
                        'comparison': f'full vs {mode}',
                        'full_mean_f1': full_f1.mean(),
                        'mode_mean_f1': mode_f1.mean(),
                        'difference': full_f1.mean() - mode_f1.mean(),
                        'test_statistic': stat,
                        'p_value': pval,
                        'significant': 'Yes' if pval < 0.05 else 'No'
                    })

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(tables_dir, 'statistical_tests.csv'), index=False)
    print(f"✓ Statistical tests: statistical_tests.csv")

    return overall_summary, best_configs_df, stats_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(df, output_dirs):
    """Create 2 comprehensive barplot visualizations - one for each covariate"""

    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)

    figures_dir = output_dirs['figures']

    # Define colors for each configuration
    # Generation plot colors (6 configurations)
    gen_colors = {
        ('full', 'Wald_pvals'): '#2E86AB',              # Dark Blue
        ('full', 'LRT_pvals'): '#5AB1BB',               # Light Blue
        ('permuted', 'Wald_pvals'): '#F18F01',          # Dark Orange
        ('permuted', 'LRT_pvals'): '#FFA500',           # Light Orange
        ('generation_only', 'Wald_pvals'): '#9D4EDD',   # Dark Purple
        ('generation_only', 'LRT_pvals'): '#C77DFF',    # Light Purple
    }

    # Cell cycle plot colors (4 configurations) - keeping full same color as generation plot
    cc_colors = {
        ('full', 'Wald_pvals'): '#2E86AB',           # Dark Blue (same as gen)
        ('full', 'LRT_pvals'): '#5AB1BB',            # Light Blue (same as gen)
        ('cell_cycle_only', 'Wald_pvals'): '#E63946', # Dark Red
        ('cell_cycle_only', 'LRT_pvals'): '#FF6B6B'   # Light Red
    }

    # ===========================================================================
    # PLOT 1: GENERATION COVARIATE
    # ===========================================================================
    print("\n  Creating Generation covariate plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = ['precision', 'f1_score', 'pr_auc', 'roc_auc']
    titles = ['Precision', 'F1 Score', 'AUPRC', 'AUROC']

    # Get generation covariate data
    gen_data = df[df['covariate'] == 'generation']

    # Define configurations for generation plot (6 configurations)
    gen_configs = [
        ('full', 'Wald_pvals', 'Full + Gen + Wald'),
        ('full', 'LRT_pvals', 'Full + Gen + LRT'),
        ('permuted', 'Wald_pvals', 'Permuted + Gen + Wald'),
        ('permuted', 'LRT_pvals', 'Permuted + Gen + LRT'),
        ('generation_only', 'Wald_pvals', 'CIVET_res + Gen + Wald'),
        ('generation_only', 'LRT_pvals', 'CIVET_res + Gen + LRT'),
    ]

    prolif_rates = sorted(gen_data['proliferation_rate'].unique())
    x = np.arange(len(prolif_rates))
    width = 0.13  # Narrower bars to accommodate 6 configurations

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]

        # Plot bars for each configuration
        for config_idx, (mode, pval, label) in enumerate(gen_configs):
            subset = gen_data[(gen_data['mode'] == mode) &
                             (gen_data['pval_type'] == pval)]

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
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Performance Metrics - Generation Covariate',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'barplot_generation_covariate.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ barplot_generation_covariate.png")

    # ===========================================================================
    # PLOT 2: CELL CYCLE COVARIATE
    # ===========================================================================
    print("  Creating Cell Cycle covariate plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Get cell cycle covariate data
    cc_data = df[df['covariate'] == 'cell_cycle_potential']

    # Define configurations for cell cycle plot (4 configurations)
    cc_configs = [
        ('full', 'Wald_pvals', 'Full + CC + Wald'),
        ('full', 'LRT_pvals', 'Full + CC + LRT'),
        ('cell_cycle_only', 'Wald_pvals', 'CC Only + CC + Wald'),
        ('cell_cycle_only', 'LRT_pvals', 'CC Only + CC + LRT'),
    ]

    cc_width = 0.18  # Wider bars for 4 configurations

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]

        # Plot bars for each configuration
        for config_idx, (mode, pval, label) in enumerate(cc_configs):
            subset = cc_data[(cc_data['mode'] == mode) &
                            (cc_data['pval_type'] == pval)]

            values = [subset[subset['proliferation_rate'] == rate][metric].values[0]
                     if len(subset[subset['proliferation_rate'] == rate]) > 0 else 0
                     for rate in prolif_rates]

            offset = (config_idx - len(cc_configs)/2 + 0.5) * cc_width
            ax.bar(x + offset, values, cc_width,
                  label=label,
                  color=cc_colors[(mode, pval)],
                  alpha=0.85)

        ax.set_xlabel('Proliferation Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(prolif_rates)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Performance Metrics - Cell Cycle Potential Covariate',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'barplot_cell_cycle_covariate.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ barplot_cell_cycle_covariate.png")

    print("\n  Visualization complete!")

# ============================================================================
# REPORT GENERATION
# ============================================================================

def create_comprehensive_report(df, overall_summary, best_configs, stats_df, output_dirs):
    """Generate comprehensive text report"""

    report_file = os.path.join(output_dirs['reports'], 'COMPREHENSIVE_ANALYSIS_REPORT.txt')

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIVET COMPREHENSIVE PERFORMANCE ANALYSIS\n")
        f.write("SCENARIO_6_CellCycle - Separate Covariate Evaluation\n")
        f.write("="*80 + "\n\n")

        # Configuration
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"P-value Types: {', '.join(PVAL_TYPES)}\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"Modes Analyzed: {', '.join(MODES.keys())}\n")
        f.write(f"Covariates: {', '.join(df['covariate'].unique())}\n")
        f.write(f"Proliferation Rates: {', '.join(map(str, sorted(df['proliferation_rate'].unique())))}\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n\n")

        best_overall = df.loc[df['f1_score'].idxmax()]
        f.write(f"Best Overall Configuration:\n")
        f.write(f"  Mode: {best_overall['mode']}\n")
        f.write(f"  P-value Type: {best_overall['pval_type']}\n")
        f.write(f"  Covariate: {best_overall['covariate']}\n")
        f.write(f"  Proliferation Rate: {best_overall['proliferation_rate']}\n")
        f.write(f"  F1 Score: {best_overall['f1_score']:.4f}\n")
        f.write(f"  ROC-AUC: {best_overall['roc_auc']:.4f}\n")
        f.write(f"  Sensitivity: {best_overall['sensitivity']:.4f}\n")
        f.write(f"  Specificity: {best_overall['specificity']:.4f}\n\n")

        # Performance by Covariate
        f.write("="*80 + "\n")
        f.write("PERFORMANCE BY COVARIATE\n")
        f.write("="*80 + "\n\n")

        for covariate in sorted(df['covariate'].unique()):
            f.write(f"\nCovariate: {covariate}\n")
            f.write("-"*80 + "\n")

            cov_data = df[df['covariate'] == covariate]

            for pval_type in sorted(cov_data['pval_type'].unique()):
                f.write(f"\n{pval_type}:\n")

                pval_cov_data = cov_data[cov_data['pval_type'] == pval_type]
                summary = pval_cov_data.groupby('mode').agg({
                    'f1_score': ['mean', 'std'],
                    'roc_auc': ['mean', 'std'],
                    'sensitivity': ['mean', 'std']
                }).round(4)

                f.write(summary.to_string())
                f.write("\n\n")

        # Best Configurations
        f.write("="*80 + "\n")
        f.write("BEST CONFIGURATIONS BY METRIC\n")
        f.write("="*80 + "\n\n")
        f.write(best_configs.to_string(index=False))
        f.write("\n\n")

        # Statistical Tests
        f.write("="*80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("="*80 + "\n\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")

        # Key Findings
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")

        for covariate in sorted(df['covariate'].unique()):
            cov_data = df[df['covariate'] == covariate]

            f.write(f"{covariate.upper()}:\n")

            best_cov = cov_data.loc[cov_data['f1_score'].idxmax()]
            f.write(f"  Best F1 Score: {best_cov['f1_score']:.4f}\n")
            f.write(f"  Mode: {best_cov['mode']}, P-value: {best_cov['pval_type']}\n")
            f.write(f"  Proliferation Rate: {best_cov['proliferation_rate']}\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\n✓ Comprehensive report: COMPREHENSIVE_ANALYSIS_REPORT.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Setup output directory
    output_dirs = setup_output_directory(OUTPUT_BASE_DIR)

    print(f"\nOutput directory created: {OUTPUT_BASE_DIR}")
    print(f"  - Tables: {output_dirs['tables']}")
    print(f"  - Figures: {output_dirs['figures']}")
    print(f"  - Reports: {output_dirs['reports']}")
    print(f"  - Data: {output_dirs['data']}")

    # Aggregate all results
    all_results = aggregate_all_results()

    # Create summary DataFrame
    print("\n" + "="*80)
    print("Creating Summary DataFrame")
    print("="*80)
    df = create_summary_dataframe(all_results)
    print(f"✓ Created summary with {len(df)} rows")
    print(f"  Covariates: {df['covariate'].unique().tolist()}")
    print(f"  Modes: {df['mode'].unique().tolist()}")

    # Save main results
    main_results_file = os.path.join(output_dirs['data'], 'complete_results.csv')
    df.to_csv(main_results_file, index=False)
    print(f"✓ Main results saved: complete_results.csv")

    # Create summary tables
    overall_summary, best_configs, stats_df = create_summary_tables(df, output_dirs)

    # Create visualizations
    create_visualizations(df, output_dirs)

    # Create comprehensive report
    create_comprehensive_report(df, overall_summary, best_configs, stats_df, output_dirs)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_BASE_DIR}")
    print(f"\nGenerated files:")
    print(f"  Data: 1 CSV file (complete_results.csv)")
    print(f"  Tables: {len(os.listdir(output_dirs['tables']))} CSV files")
    print(f"  Figures: {len(os.listdir(output_dirs['figures']))} PNG files")
    print(f"  Reports: {len(os.listdir(output_dirs['reports']))} TXT files")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
