import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

def process_simulation_data(sim_dir):
    """
    Process simulation data to filter variants and cells based on clonal hematopoiesis criteria:
    - Mean coverage >5 per cell
    - Mean quality >30 (Note: quality data not available in this simulation, will be skipped)
    - VAF of 0% in at least 90% of cells
    - VAF >50% in at least 10 cells
    
    Args:
        sim_dir: Path to the simulation directory
        
    Returns:
        Dictionary containing filtered variants and cells
    """
    print(f"Processing simulation data from: {sim_dir}")
    
    # Load cell barcodes
    barcodes_file = os.path.join(sim_dir, 'cellSNP', 'cellSNP.tag.barcodes.txt')
    with open(barcodes_file, 'r') as f:
        cell_barcodes = [line.strip() for line in f]
    
    # Load mutation names
    mutations_file = os.path.join(sim_dir, 'cellSNP', 'cellSNP.tag.mutations.txt')
    with open(mutations_file, 'r') as f:
        mutations = [line.split()[0] for line in f]
    
    # Track initial number of variants
    filter_stats = {
        'initial_variants': len(mutations),
        'after_filters': {}
    }
    print(f"Initial number of variants: {filter_stats['initial_variants']}")
    
    # Load AD (Allele Depth) matrix
    ad_file = os.path.join(sim_dir, 'cellSNP', 'cellSNP.tag.AD.mtx')
    ad_matrix = sio.mmread(ad_file).tocsr()
    
    # Load DP (Depth) matrix
    dp_file = os.path.join(sim_dir, 'cellSNP', 'cellSNP.tag.DP.mtx')
    dp_matrix = sio.mmread(dp_file).tocsr()
    
    # Calculate VAF (Variant Allele Frequency) matrix
    # VAF = AD / DP (element-wise division)
    # Handle potential division by zero
    vaf_matrix = csr_matrix(ad_matrix.shape, dtype=np.float32)
    
    # Only calculate VAF where depth > 0
    nonzero_indices = dp_matrix.nonzero()
    rows, cols = nonzero_indices
    data = np.zeros(len(rows))
    
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        ad_val = ad_matrix[row, col]
        dp_val = dp_matrix[row, col]
        if dp_val > 0:
            data[i] = ad_val / dp_val
    
    vaf_matrix = csr_matrix((data, (rows, cols)), shape=ad_matrix.shape)
    
    # Calculate mean coverage per variant (across cells)
    mean_coverage = np.array(dp_matrix.mean(axis=1)).flatten()
    
    # Calculate VAF percentiles and counts for each variant
    n_cells = vaf_matrix.shape[1]
    cells_with_zero_vaf = []
    cells_with_high_vaf = []
    
    for i in range(vaf_matrix.shape[0]):
        # Extract VAF values for this variant across all cells
        variant_vaf = vaf_matrix[i].toarray().flatten()
        
        # Count cells with VAF = 0
        zero_vaf_count = np.sum(variant_vaf == 0)
        cells_with_zero_vaf.append(zero_vaf_count / n_cells * 100)  # Percentage
        
        # Count cells with VAF > 0.5 (50%)
        high_vaf_count = np.sum(variant_vaf > 0.5)
        cells_with_high_vaf.append(high_vaf_count)
    
    # Apply filters for informative variants
    # Track variants after each filter
    coverage_filter_passed = [i for i in range(len(mutations)) if mean_coverage[i] > 5]
    filter_stats['after_filters']['coverage_filter'] = len(coverage_filter_passed)
    print(f"Variants after coverage filter (>5): {len(coverage_filter_passed)}")
    
    zero_vaf_filter_passed = [i for i in range(len(mutations)) if cells_with_zero_vaf[i] >= 90]
    filter_stats['after_filters']['zero_vaf_filter'] = len(zero_vaf_filter_passed)
    print(f"Variants after zero VAF filter (≥90% cells): {len(zero_vaf_filter_passed)}")
    
    high_vaf_filter_passed = [i for i in range(len(mutations)) if cells_with_high_vaf[i] >= 10]
    filter_stats['after_filters']['high_vaf_filter'] = len(high_vaf_filter_passed)
    print(f"Variants after high VAF filter (≥10 cells): {len(high_vaf_filter_passed)}")
    
    # Combine all filters
    informative_variants = []
    for i in range(len(mutations)):
        # Filter criteria:
        # 1. Mean coverage > 5
        # 2. VAF = 0% in at least 90% of cells
        # 3. VAF > 50% in at least 10 cells
        if (mean_coverage[i] > 5 and 
            cells_with_zero_vaf[i] >= 90 and 
            cells_with_high_vaf[i] >= 10):
            informative_variants.append(i)
    
    filter_stats['after_filters']['all_filters'] = len(informative_variants)
    print(f"Found {len(informative_variants)} informative variants that passed ALL filters")
    
    # Select cells with VAF > 1% for any informative variant
    selected_cells = set()
    for variant_idx in informative_variants:
        # Get VAF for this variant across all cells
        variant_vaf = vaf_matrix[variant_idx].toarray().flatten()
        
        # Find cells with VAF > 1%
        positive_cells = np.where(variant_vaf > 0.01)[0]
        selected_cells.update(positive_cells)
    
    selected_cells = sorted(list(selected_cells))
    print(f"Selected {len(selected_cells)} cells with VAF > 1% for any informative variant")
    
    # Create a result dictionary
    result = {
        'informative_variants': [mutations[i] for i in informative_variants],
        'selected_cells': [cell_barcodes[i] for i in selected_cells],
        'variant_metrics': pd.DataFrame({
            'mutation': mutations,
            'mean_coverage': mean_coverage,
            'pct_cells_with_zero_vaf': cells_with_zero_vaf,
            'cells_with_high_vaf': cells_with_high_vaf,
            'is_informative': [i in informative_variants for i in range(len(mutations))]
        }),
        'filter_stats': filter_stats
    }
    
    # Create visualization of selected variants
    create_variant_visualization(result, sim_dir, vaf_matrix, informative_variants, selected_cells)
    
    return result

def create_variant_visualization(result, sim_dir, vaf_matrix, informative_variants, selected_cells):
    """
    Create visualizations to show the selected variants and cells
    """
    # Create output directory for results if it doesn't exist
    output_dir = os.path.join(sim_dir, 'variant_selection_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the variant metrics
    result['variant_metrics'].to_csv(os.path.join(output_dir, 'variant_metrics.csv'), index=False)
    
    # Save filter statistics
    filter_stats_df = pd.DataFrame([
        {"filter_name": "Initial", "variants_remaining": result['filter_stats']['initial_variants']},
        {"filter_name": "Coverage > 5", "variants_remaining": result['filter_stats']['after_filters']['coverage_filter']},
        {"filter_name": "VAF = 0 in ≥90% cells", "variants_remaining": result['filter_stats']['after_filters']['zero_vaf_filter']},
        {"filter_name": "VAF > 50% in ≥10 cells", "variants_remaining": result['filter_stats']['after_filters']['high_vaf_filter']},
        {"filter_name": "All filters", "variants_remaining": result['filter_stats']['after_filters']['all_filters']}
    ])
    filter_stats_df.to_csv(os.path.join(output_dir, 'filter_statistics.csv'), index=False)
    
    # Save lists of informative variants and selected cells
    with open(os.path.join(output_dir, 'informative_variants.txt'), 'w') as f:
        for variant in result['informative_variants']:
            f.write(f"{variant}\n")
    
    with open(os.path.join(output_dir, 'selected_cells.txt'), 'w') as f:
        for cell in result['selected_cells']:
            f.write(f"{cell}\n")
    
    # Create a heatmap of VAF for informative variants across selected cells
    if len(informative_variants) > 0 and len(selected_cells) > 0:
        # Extract the submatrix
        sub_vaf = vaf_matrix[informative_variants, :][:, selected_cells].toarray()
        
        # Create a heatmap (limit to first 100 variants and 100 cells if there are too many)
        max_display = 100
        display_variants = min(len(informative_variants), max_display)
        display_cells = min(len(selected_cells), max_display)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            sub_vaf[:display_variants, :display_cells], 
            cmap="YlOrRd", 
            vmin=0, 
            vmax=1
        )
        plt.xlabel(f"Selected Cells (showing first {display_cells} of {len(selected_cells)})")
        plt.ylabel(f"Informative Variants (showing first {display_variants} of {len(informative_variants)})")
        plt.title("VAF Heatmap of Informative Variants in Selected Cells")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vaf_heatmap.png'))
        plt.close()
        
        # Create distribution plots of variant metrics
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.histplot(result['variant_metrics']['mean_coverage'], bins=30)
        plt.axvline(x=5, color='r', linestyle='--', label='Threshold (5)')
        plt.title('Distribution of Mean Coverage')
        plt.xlabel('Mean Coverage')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        sns.histplot(result['variant_metrics']['pct_cells_with_zero_vaf'], bins=30)
        plt.axvline(x=90, color='r', linestyle='--', label='Threshold (90%)')
        plt.title('Distribution of % Cells with Zero VAF')
        plt.xlabel('% Cells with VAF = 0')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        sns.histplot(result['variant_metrics']['cells_with_high_vaf'], bins=30)
        plt.axvline(x=10, color='r', linestyle='--', label='Threshold (10)')
        plt.title('Distribution of Cells with High VAF')
        plt.xlabel('Number of Cells with VAF > 50%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'variant_metrics_distribution.png'))
        plt.close()
        
        # Create a bar chart showing filter statistics
        plt.figure(figsize=(10, 6))
        sns.barplot(x='filter_name', y='variants_remaining', data=filter_stats_df)
        plt.title('Number of Variants Remaining After Each Filter')
        plt.xlabel('Filter')
        plt.ylabel('Number of Variants')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'filter_statistics.png'))
        plt.close()

def find_simulation_folders(base_dir="."):
    """
    Find all simulation folders under directories starting with SCENARIO*
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        List of paths to simulation folders
    """
    simulation_folders = []
    
    # Find all directories starting with SCENARIO
    scenario_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("SCENARIO")]
    
    for scenario_dir in scenario_dirs:
        scenario_path = os.path.join(base_dir, scenario_dir)
        # Find all subdirectories in the scenario directory
        for subdir in os.listdir(scenario_path):
            subdir_path = os.path.join(scenario_path, subdir)
            if os.path.isdir(subdir_path):
                # Check if this is a simulation directory (has cellSNP folder)
                if os.path.exists(os.path.join(subdir_path, "cellSNP")):
                    simulation_folders.append(subdir_path)
    
    return simulation_folders

def main():
    """Main function to run the analysis"""
    # Base directory where SCENARIO* folders are located
    base_dir = "/home/linxy29/data/CIVET/simulation"
    
    # Find all simulation folders
    sim_folders = find_simulation_folders(base_dir)
    
    if not sim_folders:
        print("No simulation folders found under SCENARIO* directories.")
        return
    
    print(f"Found {len(sim_folders)} simulation folders to process:")
    for i, folder in enumerate(sim_folders):
        print(f"  {i+1}. {folder}")
    
    # Process each simulation folder
    all_results = {}
    for sim_dir in sim_folders:
        print(f"\nProcessing {sim_dir}...")
        
        # Process the simulation data
        try:
            result = process_simulation_data(sim_dir)
            all_results[sim_dir] = result
            
            # Print summary statistics
            print(f"\nSummary for {sim_dir}:")
            print(f"Total variants analyzed: {len(result['variant_metrics'])}")
            
            # Print filter statistics
            filter_stats = result['filter_stats']
            print("\nFilter statistics:")
            print(f"  Initial variants: {filter_stats['initial_variants']}")
            print(f"  After coverage filter (>5): {filter_stats['after_filters']['coverage_filter']} " + 
                  f"({round(filter_stats['after_filters']['coverage_filter']/filter_stats['initial_variants']*100, 1)}% retained)")
            print(f"  After zero VAF filter (≥90% cells): {filter_stats['after_filters']['zero_vaf_filter']} " + 
                  f"({round(filter_stats['after_filters']['zero_vaf_filter']/filter_stats['initial_variants']*100, 1)}% retained)")
            print(f"  After high VAF filter (≥10 cells): {filter_stats['after_filters']['high_vaf_filter']} " + 
                  f"({round(filter_stats['after_filters']['high_vaf_filter']/filter_stats['initial_variants']*100, 1)}% retained)")
            print(f"  After all filters: {filter_stats['after_filters']['all_filters']} " + 
                  f"({round(filter_stats['after_filters']['all_filters']/filter_stats['initial_variants']*100, 1)}% retained)")
            
            print(f"Informative variants selected: {len(result['informative_variants'])}")
            print(f"Cells selected: {len(result['selected_cells'])}")
            
            # Print first few informative variants
            print("\nSample of informative variants:")
            for i, variant in enumerate(result['informative_variants'][:5]):
                print(f"  {i+1}. {variant}")
            if len(result['informative_variants']) > 5:
                print(f"  ... and {len(result['informative_variants'])-5} more")
                
            print(f"Results saved to {os.path.join(sim_dir, 'variant_selection_results')}")
        except Exception as e:
            print(f"Error processing {sim_dir}: {str(e)}")
    
    # Create a summary of all results
    create_summary_report(all_results, base_dir)

def create_summary_report(all_results, base_dir):
    """
    Create a summary report of all processed simulation folders
    
    Args:
        all_results: Dictionary mapping simulation folders to their results
        base_dir: Base directory where the summary will be saved
    """
    if not all_results:
        print("No results to summarize.")
        return
    
    # Create a summary directory
    summary_dir = os.path.join(base_dir, "variant_selection_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create a summary CSV file
    summary_data = []
    filter_data = []
    for sim_dir, result in all_results.items():
        # Extract scenario and parameter from sim_dir
        parts = sim_dir.split(os.sep)
        scenario = parts[-2] if len(parts) > 1 else "unknown"
        parameter = parts[-1] if len(parts) > 0 else "unknown"
        
        # Basic summary data
        summary_data.append({
            'scenario': scenario,
            'parameter': parameter,
            'total_variants': len(result['variant_metrics']),
            'informative_variants': len(result['informative_variants']),
            'selected_cells': len(result['selected_cells']),
            'simulation_directory': sim_dir
        })
        
        # Filter statistics data
        filter_stats = result['filter_stats']
        filter_data.append({
            'scenario': scenario,
            'parameter': parameter,
            'initial_variants': filter_stats['initial_variants'],
            'after_coverage_filter': filter_stats['after_filters']['coverage_filter'],
            'after_zero_vaf_filter': filter_stats['after_filters']['zero_vaf_filter'],
            'after_high_vaf_filter': filter_stats['after_filters']['high_vaf_filter'],
            'after_all_filters': filter_stats['after_filters']['all_filters'],
            'pct_retained': round(filter_stats['after_filters']['all_filters'] / filter_stats['initial_variants'] * 100, 2)
        })
    
    # Create DataFrames and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(summary_dir, 'summary.csv'), index=False)
    
    filter_df = pd.DataFrame(filter_data)
    filter_df.to_csv(os.path.join(summary_dir, 'filter_summary.csv'), index=False)
    
    # Create a bar chart comparing informative variants across scenarios
    plt.figure(figsize=(12, 6))
    
    # Group by scenario and calculate mean informative variants
    scenario_groups = summary_df.groupby('scenario')
    
    # Plot
    scenario_names = []
    variant_counts = []
    for scenario, group in scenario_groups:
        scenario_names.append(scenario)
        variant_counts.append(group['informative_variants'].values)
    
    # Create a boxplot
    plt.boxplot(variant_counts, labels=scenario_names)
    plt.title('Informative Variants by Scenario')
    plt.ylabel('Number of Informative Variants')
    plt.xlabel('Scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'informative_variants_by_scenario.png'))
    plt.close()
    
    # Create a scatter plot of informative variants vs selected cells
    plt.figure(figsize=(10, 6))
    for scenario, group in summary_df.groupby('scenario'):
        plt.scatter(
            group['informative_variants'], 
            group['selected_cells'],
            label=scenario,
            alpha=0.7,
            s=80
        )
    
    plt.title('Informative Variants vs Selected Cells')
    plt.xlabel('Number of Informative Variants')
    plt.ylabel('Number of Selected Cells')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'variants_vs_cells.png'))
    plt.close()
    
    # Create a stacked bar chart showing filter effects across scenarios
    filter_summary = filter_df.groupby('scenario').mean().reset_index()
    
    # Create a figure for filter statistics comparison
    plt.figure(figsize=(14, 8))
    
    # Calculate the reduction at each step
    filter_reduction = pd.DataFrame()
    filter_reduction['scenario'] = filter_summary['scenario']
    filter_reduction['Initial'] = filter_summary['initial_variants']
    filter_reduction['After Coverage'] = filter_summary['initial_variants'] - filter_summary['after_coverage_filter']
    filter_reduction['After Zero VAF'] = filter_summary['after_coverage_filter'] - filter_summary['after_zero_vaf_filter']
    filter_reduction['After High VAF'] = filter_summary['after_zero_vaf_filter'] - filter_summary['after_high_vaf_filter']
    filter_reduction['Remaining'] = filter_summary['after_all_filters']
    
    # Plot stacked bar chart
    ax = filter_reduction.set_index('scenario').plot(
        kind='bar', 
        stacked=True, 
        figsize=(14, 8),
        colormap='viridis'
    )
    
    plt.title('Average Variant Filtering by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Number of Variants')
    plt.legend(title='Filter Stage')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage labels
    for i, scenario in enumerate(filter_summary['scenario']):
        initial = filter_summary.loc[i, 'initial_variants']
        final = filter_summary.loc[i, 'after_all_filters']
        pct = round(final / initial * 100, 1)
        plt.text(
            i, 
            initial + 5, 
            f'{pct}% retained',
            ha='center',
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'filter_effects_by_scenario.png'))
    plt.close()
    
    print(f"\nSummary report created at {summary_dir}")

if __name__ == "__main__":
    main()