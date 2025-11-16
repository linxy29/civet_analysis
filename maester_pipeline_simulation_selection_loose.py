#!/usr/bin/env python3
"""
MAESTER Pipeline Loose - Variant Selection with Variable Thresholds

This script explores how different filtering thresholds affect variant detection
in a single simulation. It tests multiple combinations of:
- Coverage threshold (mean DP)
- Zero VAF percentage threshold (fraction of cells with VAF=0)
- High VAF cell count threshold (number of cells with VAF>50%)

Usage:
    python maester_pipeline_simulation_selection_loose.py --sim-dir /path/to/simulation/condition
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
import argparse
import itertools

def load_cellsnp_data(cellsnp_dir):
    """Load cellSNP output files (DP, AD matrices, mutations, barcodes, VCF)"""
    # Load DP matrix (Total depth)
    dp_matrix = mmread(os.path.join(cellsnp_dir, "cellSNP.tag.DP.mtx")).tocsr()

    # Load AD matrix (Allele depth)
    ad_matrix = mmread(os.path.join(cellsnp_dir, "cellSNP.tag.AD.mtx")).tocsr()

    # Load mutations
    with open(os.path.join(cellsnp_dir, "cellSNP.tag.mutations.txt"), 'r') as f:
        mutations = [line.strip() for line in f.readlines()]

    # Load barcodes
    with open(os.path.join(cellsnp_dir, "cellSNP.tag.barcodes.txt"), 'r') as f:
        barcodes = [line.strip() for line in f.readlines()]

    # Calculate VAF matrix (element-wise division)
    # For sparse matrices, we need to use element-wise operations on the data
    vaf_matrix = ad_matrix.copy()
    vaf_matrix = vaf_matrix.astype(float)

    # Element-wise division: VAF = AD / DP
    # Only divide where DP > 0
    dp_data = dp_matrix.toarray()
    ad_data = ad_matrix.toarray()
    vaf_data = np.divide(ad_data, dp_data, out=np.zeros_like(ad_data, dtype=float), where=dp_data!=0)
    vaf_matrix = sp.csr_matrix(vaf_data)

    return dp_matrix, ad_matrix, vaf_matrix, mutations, barcodes

def filter_variants_with_threshold(dp_matrix, vaf_matrix, mutations,
                                   coverage_threshold=5,
                                   zero_vaf_pct_threshold=0.5,
                                   vaf_threshold=0.1,
                                   min_cells_with_vaf=10,
                                   return_stats=False):
    """
    Filter variants with specified thresholds.

    Parameters
    ----------
    coverage_threshold : float
        Minimum mean coverage
    zero_vaf_pct_threshold : float
        Minimum fraction of cells with VAF < 1%
    vaf_threshold : float
        VAF threshold for detecting variants (e.g., 0.1 for 10%)
    min_cells_with_vaf : int
        Minimum number of cells with VAF > vaf_threshold
    return_stats : bool
        If True, return detailed statistics about each filter step
    """
    n_variants, n_cells = vaf_matrix.shape

    # Calculate metrics for each variant
    mean_coverage = np.array(dp_matrix.mean(axis=1)).flatten()
    cells_with_zero_vaf = np.array((vaf_matrix < 0.01).sum(axis=1)).flatten()
    cells_with_vaf_above_threshold = np.array((vaf_matrix > vaf_threshold).sum(axis=1)).flatten()

    # Track filtering cascade
    filter_stats = {}
    filter_stats['step0_initial'] = n_variants

    # Step 1: Coverage filter
    step1_pass = [i for i in range(n_variants) if mean_coverage[i] > coverage_threshold]
    filter_stats['step1_after_coverage'] = len(step1_pass)
    filter_stats['step1_removed_by_coverage'] = n_variants - len(step1_pass)

    # Step 2: Zero VAF filter (on variants passing step 1)
    step2_pass = [i for i in step1_pass if cells_with_zero_vaf[i] >= zero_vaf_pct_threshold * n_cells]
    filter_stats['step2_after_zero_vaf'] = len(step2_pass)
    filter_stats['step2_removed_by_zero_vaf'] = len(step1_pass) - len(step2_pass)

    # Step 3: VAF threshold filter (on variants passing step 2)
    informative_variants = [i for i in step2_pass if cells_with_vaf_above_threshold[i] >= min_cells_with_vaf]
    filter_stats['step3_final'] = len(informative_variants)
    filter_stats['step3_removed_by_vaf_threshold'] = len(step2_pass) - len(informative_variants)

    if return_stats:
        return informative_variants, filter_stats
    return informative_variants

def categorize_mutations(mutations):
    """Categorize mutations as baseline, false, or rest"""
    baseline_mutations = []
    false_mutations = []
    rest_mutations = []

    for mutation in mutations:
        if "baseline" in mutation.lower():
            baseline_mutations.append(mutation)
        elif "false" in mutation.lower():
            false_mutations.append(mutation)
        else:
            rest_mutations.append(mutation)

    return baseline_mutations, false_mutations, rest_mutations

def process_with_thresholds(sim_dir, threshold_combinations, show_filter_stats=False):
    """Process simulation data with multiple threshold combinations."""
    print(f"\nProcessing: {sim_dir}")

    # Load cellSNP data
    cellsnp_dir = os.path.join(sim_dir, "cellSNP")
    if not os.path.exists(cellsnp_dir):
        print(f"Error: cellSNP directory not found: {cellsnp_dir}")
        return None

    dp_matrix, ad_matrix, vaf_matrix, mutations, barcodes = load_cellsnp_data(cellsnp_dir)
    baseline_mutations, false_mutations, rest_mutations = categorize_mutations(mutations)

    print(f"Loaded {len(mutations)} variants, {len(barcodes)} cells")
    print(f"  Baseline: {len(baseline_mutations)}, False: {len(false_mutations)}, Rest: {len(rest_mutations)}")

    # Test each threshold combination
    results = []
    for idx, params in enumerate(threshold_combinations):
        if show_filter_stats:
            informative_variants, filter_stats = filter_variants_with_threshold(
                dp_matrix, vaf_matrix, mutations,
                coverage_threshold=params['coverage'],
                zero_vaf_pct_threshold=params['zero_vaf_pct'],
                vaf_threshold=params['vaf_threshold'],
                min_cells_with_vaf=params['min_cells'],
                return_stats=True
            )

            # Print filter statistics for first combination
            if idx == 0:
                print(f"\n{'='*80}")
                print("FILTER CASCADE ANALYSIS (for first threshold combination)")
                print(f"Coverage > {params['coverage']}, Zero VAF ≥{params['zero_vaf_pct']*100}%, VAF > {params['vaf_threshold']*100}% in ≥{params['min_cells']} cells")
                print("="*80)
                print(f"Step 0 - Initial: {filter_stats['step0_initial']} variants")
                print(f"Step 1 - After coverage filter: {filter_stats['step1_after_coverage']} variants")
                print(f"         Removed by coverage: {filter_stats['step1_removed_by_coverage']} ({100*filter_stats['step1_removed_by_coverage']/filter_stats['step0_initial']:.1f}%)")
                print(f"Step 2 - After zero VAF filter: {filter_stats['step2_after_zero_vaf']} variants")
                print(f"         Removed by zero VAF: {filter_stats['step2_removed_by_zero_vaf']} ({100*filter_stats['step2_removed_by_zero_vaf']/filter_stats['step0_initial']:.1f}%)")
                print(f"Step 3 - After VAF threshold filter: {filter_stats['step3_final']} variants")
                print(f"         Removed by VAF threshold: {filter_stats['step3_removed_by_vaf_threshold']} ({100*filter_stats['step3_removed_by_vaf_threshold']/filter_stats['step0_initial']:.1f}%)")
                print(f"\nTotal removed: {filter_stats['step0_initial'] - filter_stats['step3_final']} ({100*(filter_stats['step0_initial'] - filter_stats['step3_final'])/filter_stats['step0_initial']:.1f}%)")
                print("="*80 + "\n")
        else:
            informative_variants = filter_variants_with_threshold(
                dp_matrix, vaf_matrix, mutations,
                coverage_threshold=params['coverage'],
                zero_vaf_pct_threshold=params['zero_vaf_pct'],
                vaf_threshold=params['vaf_threshold'],
                min_cells_with_vaf=params['min_cells']
            )

        # Count detected mutations by type
        detected_baseline = sum(1 for i in informative_variants if mutations[i] in baseline_mutations)
        detected_false = sum(1 for i in informative_variants if mutations[i] in false_mutations)
        detected_rest = sum(1 for i in informative_variants if mutations[i] in rest_mutations)

        results.append({
            'coverage_threshold': params['coverage'],
            'zero_vaf_pct_threshold': params['zero_vaf_pct'],
            'vaf_threshold': params['vaf_threshold'],
            'min_cells_with_vaf': params['min_cells'],
            'total_detected': len(informative_variants),
            'detected_baseline': detected_baseline,
            'detected_false': detected_false,
            'detected_rest': detected_rest,
            'total_baseline': len(baseline_mutations),
            'total_false': len(false_mutations),
            'total_rest': len(rest_mutations)
        })

    return pd.DataFrame(results)

def create_visualizations(results_df, output_dir, sim_name):
    """Create visualizations showing how variant detection changes with thresholds."""
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Heatmap: Total detected variants by coverage and zero_vaf thresholds
    print("\nCreating heatmap...")
    pivot_data = results_df.groupby(['coverage_threshold', 'zero_vaf_pct_threshold'])['total_detected'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='zero_vaf_pct_threshold',
                                   columns='coverage_threshold',
                                   values='total_detected')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Mean Detected Variants'})
    ax.set_xlabel('Coverage Threshold', fontsize=12)
    ax.set_ylabel('Zero VAF % Threshold', fontsize=12)
    ax.set_title(f'{sim_name}\nDetected Variants by Coverage and Zero VAF Thresholds', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Line plots for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Coverage threshold effect
    ax = axes[0]
    grouped = results_df.groupby('coverage_threshold')['total_detected'].mean()
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Coverage Threshold', fontsize=12)
    ax.set_ylabel('Mean Detected Variants', fontsize=12)
    ax.set_title('Effect of Coverage Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Zero VAF percentage effect
    ax = axes[1]
    grouped = results_df.groupby('zero_vaf_pct_threshold')['total_detected'].mean()
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Zero VAF % Threshold', fontsize=12)
    ax.set_ylabel('Mean Detected Variants', fontsize=12)
    ax.set_title('Effect of Zero VAF % Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)

    # VAF threshold effect
    ax = axes[2]
    grouped = results_df.groupby('vaf_threshold')['total_detected'].mean()
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('VAF Threshold', fontsize=12)
    ax.set_ylabel('Mean Detected Variants', fontsize=12)
    ax.set_title('Effect of VAF Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Mutation type breakdown for top 5 threshold combinations
    print("\nCreating mutation type breakdown...")
    top5 = results_df.nlargest(5, 'total_detected')

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top5))
    width = 0.6

    ax.bar(x, top5['detected_baseline'], width, label='Baseline', color='#2ecc71')
    ax.bar(x, top5['detected_false'], width, bottom=top5['detected_baseline'],
           label='False', color='#e74c3c')
    ax.bar(x, top5['detected_rest'], width,
           bottom=top5['detected_baseline'] + top5['detected_false'],
           label='Rest', color='#3498db')

    ax.set_xlabel('Threshold Combination', fontsize=12)
    ax.set_ylabel('Detected Variants', fontsize=12)
    ax.set_title(f'{sim_name}\nTop 5 Threshold Combinations by Detection Count', fontsize=14)
    ax.set_xticks(x)
    labels = [f"cov={row['coverage_threshold']}\nzero={row['zero_vaf_pct_threshold']}\nVAF={row['vaf_threshold']}"
              for _, row in top5.iterrows()]
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_thresholds_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {output_dir}")

def generate_mutation_combine_for_scenario2():
    """
    Generate maesterpp_loose_mutation_combine.csv for Scenario 2 (bias_0.5 and bias_0.7)
    using the best loose thresholds identified
    """
    base_dir = "/Users/linxy29/Documents/Data/CIVET/simulation"

    print("\n\n" + "="*80)
    print("GENERATING MAESTERPP_LOOSE MUTATION COMBINE FOR SCENARIO_2")
    print("Using LOOSE thresholds:")
    print("  - Coverage > 3 (vs standard 5)")
    print("  - Cells with VAF<1% >= 30% (vs standard 50%)")
    print("  - VAF threshold: 0.05 (5% vs standard 50%)")
    print("  - Min cells with VAF > threshold: 10")
    print("="*80)

    # Define conditions to process
    conditions = {
        'bias_0.5': 'bias_0.5_20250507_121703',
        'bias_0.7': 'bias_0.7_20250507_121758'
    }

    all_results = []

    for condition_name, condition_dir in conditions.items():
        sim_dir = os.path.join(base_dir, 'SCENARIO_2_Segregation', condition_dir)

        if not os.path.exists(sim_dir):
            print(f"\nWarning: Directory not found: {sim_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {condition_name}")
        print(f"Directory: {sim_dir}")
        print(f"{'='*80}")

        # Load cellSNP data
        cellsnp_dir = os.path.join(sim_dir, "cellSNP")
        if not os.path.exists(cellsnp_dir):
            print(f"Error: cellSNP directory not found: {cellsnp_dir}")
            continue

        dp_matrix, ad_matrix, vaf_matrix, mutations, barcodes = load_cellsnp_data(cellsnp_dir)
        baseline_mutations, false_mutations, rest_mutations = categorize_mutations(mutations)

        print(f"Loaded {len(mutations)} variants, {len(barcodes)} cells")
        print(f"  Baseline: {len(baseline_mutations)}, False: {len(false_mutations)}, Rest: {len(rest_mutations)}")

        # Apply loose filters with best thresholds
        informative_variants = filter_variants_with_threshold(
            dp_matrix, vaf_matrix, mutations,
            coverage_threshold=3,
            zero_vaf_pct_threshold=0.3,
            vaf_threshold=0.05,
            min_cells_with_vaf=10
        )

        print(f"Detected {len(informative_variants)} informative variants with loose thresholds")

        # Calculate metrics for all mutations
        mean_coverage = np.array(dp_matrix.mean(axis=1)).flatten()
        n_cells = vaf_matrix.shape[1]
        cells_with_zero_vaf = np.array((vaf_matrix < 0.01).sum(axis=1)).flatten()
        cells_with_vaf_above_threshold = np.array((vaf_matrix > 0.05).sum(axis=1)).flatten()

        # Create results for ALL mutations
        for i, mutation in enumerate(mutations):
            detected = i in informative_variants
            baseline_mutation = mutation in baseline_mutations
            false_mutation = mutation in false_mutations
            rest_mutation = mutation in rest_mutations

            all_results.append({
                'scenario': 'SCENARIO_2_Segregation',
                'condition': condition_dir,
                'mutation_name': mutation,
                'detected': detected,
                'baseline_mutation': baseline_mutation,
                'false_mutation': false_mutation,
                'rest_mutation': rest_mutation,
                'mean_coverage': mean_coverage[i],
                'pct_cells_with_zero_vaf': cells_with_zero_vaf[i] / n_cells,
                'cells_with_high_vaf': cells_with_vaf_above_threshold[i]
            })

        # Print summary for this condition
        detected_baseline = sum(1 for i in informative_variants if mutations[i] in baseline_mutations)
        detected_false = sum(1 for i in informative_variants if mutations[i] in false_mutations)
        detected_rest = sum(1 for i in informative_variants if mutations[i] in rest_mutations)

        print(f"\nSummary for {condition_name}:")
        print(f"  Detected baseline: {detected_baseline}/{len(baseline_mutations)}")
        print(f"  Detected false: {detected_false}/{len(false_mutations)}")
        print(f"  Detected rest: {detected_rest}/{len(rest_mutations)}")

    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_path = os.path.join(base_dir, 'maesterpp_loose_mutation_combine.csv')
        combined_df.to_csv(output_path, index=False)

        print(f"\n\n{'='*80}")
        print(f"COMBINED RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Output file: {output_path}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total detected: {combined_df['detected'].sum()}")

        # Print per-condition summary
        print(f"\nPer-condition summary:")
        for condition in combined_df['condition'].unique():
            condition_df = combined_df[combined_df['condition'] == condition]
            detected_count = condition_df['detected'].sum()
            detected_rest = condition_df[condition_df['detected']]['rest_mutation'].sum()
            total_mutations = len(condition_df)
            print(f"  {condition}:")
            print(f"    Total mutations: {total_mutations}")
            print(f"    Detected: {detected_count}")
            print(f"    Detected rest (true): {detected_rest}")
        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='MAESTER Pipeline with variable filtering thresholds for one simulation')
    parser.add_argument('--sim-dir', type=str,
                       help='Simulation directory containing cellSNP folder')
    parser.add_argument('--coverage', type=float, nargs='+', default=[3, 5, 10],
                       help='Coverage thresholds to test (default: 3 5 10)')
    parser.add_argument('--zero-vaf-pct', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                       help='Zero VAF percentage thresholds to test (default: 0.3 0.5 0.7)')
    parser.add_argument('--vaf-threshold', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5],
                       help='VAF thresholds to test (default: 0.05 0.1 0.2 0.5)')
    parser.add_argument('--min-cells', type=int, default=10,
                       help='Minimum number of cells with VAF above threshold (default: 10)')
    parser.add_argument('--show-filter-stats', action='store_true',
                       help='Show detailed statistics for each filter step')
    parser.add_argument('--generate-scenario2', action='store_true',
                       help='Generate maesterpp_loose_mutation_combine.csv for Scenario 2 (bias_0.5 and bias_0.7)')
    args = parser.parse_args()

    # Check if we should generate scenario2 results
    if args.generate_scenario2:
        generate_mutation_combine_for_scenario2()
        return

    # Original functionality: analyze one simulation directory
    if not args.sim_dir:
        parser.error("--sim-dir is required when not using --generate-scenario2")

    sim_dir = args.sim_dir
    sim_name = os.path.basename(sim_dir)

    # Create threshold combinations
    threshold_combinations = []
    for cov, zero_pct, vaf_thresh in itertools.product(args.coverage,
                                                        args.zero_vaf_pct,
                                                        args.vaf_threshold):
        threshold_combinations.append({
            'coverage': cov,
            'zero_vaf_pct': zero_pct,
            'vaf_threshold': vaf_thresh,
            'min_cells': args.min_cells
        })

    print(f"Testing {len(threshold_combinations)} threshold combinations:")
    print(f"  Coverage thresholds: {args.coverage}")
    print(f"  Zero VAF % thresholds: {args.zero_vaf_pct}")
    print(f"  VAF thresholds: {args.vaf_threshold}")
    print(f"  Min cells with VAF: {args.min_cells}")

    # Process simulation
    results_df = process_with_thresholds(sim_dir, threshold_combinations, show_filter_stats=args.show_filter_stats)

    if results_df is not None:
        # Save results
        output_dir = os.path.join(sim_dir, "maesterpp_loose_results")
        os.makedirs(output_dir, exist_ok=True)

        # Save all results
        output_csv_all = os.path.join(output_dir, 'threshold_results_all.csv')
        results_df.to_csv(output_csv_all, index=False)
        print(f"\nAll results saved to: {output_csv_all}")

        # Save best result only
        best_result = results_df.nlargest(1, 'total_detected')
        output_csv = os.path.join(output_dir, 'threshold_results.csv')
        best_result.to_csv(output_csv, index=False)
        print(f"Best result saved to: {output_csv}")

        # Create visualizations
        create_visualizations(results_df, output_dir, sim_name)

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nTop 5 threshold combinations by total detected variants:")
        top5 = results_df.nlargest(5, 'total_detected')
        for idx, row in top5.iterrows():
            print(f"\n  Coverage={row['coverage_threshold']}, Zero_VAF%={row['zero_vaf_pct_threshold']}, VAF_threshold={row['vaf_threshold']}, Min_cells={row['min_cells_with_vaf']}")
            print(f"    Total detected: {row['total_detected']}")
            print(f"    Baseline: {row['detected_baseline']}/{row['total_baseline']}, False: {row['detected_false']}/{row['total_false']}, Rest: {row['detected_rest']}/{row['total_rest']}")

if __name__ == "__main__":
    main()
