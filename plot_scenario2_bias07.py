import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set the working directory to the script's location
working_dir = "/Users/linxy29/Documents/Data/CIVET/simulation"
os.chdir(working_dir)
print(f"Working directory set to: {os.getcwd()}")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration: Number of top variants to select for loose methods
TOP_N_VARIANTS = 180

def calculate_metrics_for_condition(df, methods, scenario_name, condition_name):
    """
    Calculate precision and F1 score for each method in a specific scenario and condition

    Args:
        df: Combined dataframe with all methods
        methods: List of method names
        scenario_name: Name of the scenario (e.g., 'SCENARIO_2')
        condition_name: Name of the condition (e.g., 'bias_0.7')

    Returns:
        DataFrame with metrics for each method
    """
    # Find scenario and condition columns
    scenario_col = None
    condition_col = None

    for col in df.columns:
        if 'scenario' in col.lower():
            scenario_col = col
        if 'condition' in col.lower():
            condition_col = col

    if not scenario_col or not condition_col:
        print(f"Error: Could not find scenario or condition columns")
        return None

    # Filter for specific scenario and condition
    filtered_df = df[(df[scenario_col] == scenario_name) & (df[condition_col] == condition_name)]

    if len(filtered_df) == 0:
        print(f"Warning: No data found for {scenario_name}, {condition_name}")
        print(f"Available scenarios: {df[scenario_col].unique()}")
        print(f"Available conditions: {df[condition_col].unique()}")
        return None

    print(f"\nFiltered data: {len(filtered_df)} rows for {scenario_name}, {condition_name}")

    # Calculate metrics for each method
    results = []

    for method in methods:
        detected_col = f'{method}_detected'
        if detected_col not in filtered_df.columns:
            print(f"Warning: {detected_col} not found in dataframe")
            continue

        # Handle missing values
        mask = ~(filtered_df[detected_col].isna() | filtered_df['true_mutation'].isna())

        if mask.sum() == 0:
            print(f"Warning: No valid data for {method}")
            continue

        y_true = filtered_df.loc[mask, 'true_mutation']
        y_pred = filtered_df.loc[mask, detected_col]

        # Convert to boolean if needed
        if y_pred.dtype == 'object':
            y_pred = y_pred.map({'TRUE': True, 'FALSE': False, True: True, False: False})

        # Calculate confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'Method': method,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1_score,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'Total_Identified': tp + fp,
                'Total_True_Mutations': tp + fn
            })

            print(f"{method}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}, Identified={tp+fp}")

        except Exception as e:
            print(f"Error calculating metrics for {method}: {str(e)}")
            continue

    if results:
        return pd.DataFrame(results)
    else:
        return None

def create_precision_f1_plot(metrics_df, scenario_name, condition_name, output_path):
    """
    Create a plot showing precision and F1 score for each method

    Args:
        metrics_df: DataFrame with metrics for each method
        scenario_name: Name of the scenario
        condition_name: Name of the condition
        output_path: Path to save the figure
    """
    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision comparison
    ax1 = axes[0]
    methods = metrics_df['Method'].tolist()
    precision_values = metrics_df['Precision'].tolist()

    bars1 = ax1.bar(methods, precision_values, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Precision', fontsize=16)
    ax1.set_xlabel('Method', fontsize=16)
    ax1.set_title(f'Precision by Method\n{scenario_name}, {condition_name}', fontsize=18)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)

    # Plot 2: F1 Score comparison
    ax2 = axes[1]
    f1_values = metrics_df['F1_Score'].tolist()

    bars2 = ax2.bar(methods, f1_values, color='coral', alpha=0.8)
    ax2.set_ylabel('F1 Score', fontsize=16)
    ax2.set_xlabel('Method', fontsize=16)
    ax2.set_title(f'F1 Score by Method\n{scenario_name}, {condition_name}', fontsize=18)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()

def create_combined_plot(metrics_df, scenario_name, condition_name, output_path):
    """
    Create a single plot with both precision and F1 score as grouped bars

    Args:
        metrics_df: DataFrame with metrics for each method
        scenario_name: Name of the scenario
        condition_name: Name of the condition
        output_path: Path to save the figure
    """
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = metrics_df['Method'].tolist()
    precision_values = metrics_df['Precision'].tolist()
    f1_values = metrics_df['F1_Score'].tolist()

    x = np.arange(len(methods))
    width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x - width/2, precision_values, width, label='Precision',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_values, width, label='F1 Score',
                   color='coral', alpha=0.8)

    # Add labels and title
    ax.set_ylabel('Score', fontsize=16)
    ax.set_xlabel('Method', fontsize=16)
    ax.set_title(f'Precision and F1 Score by Method\n{scenario_name}, {condition_name}',
                fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to: {output_path}")
    plt.close()

def create_loose_versions_for_scenario(scenario_name, condition_name):
    """
    Create loose versions of mquad, MitoTracer, scMitoMut, and maesterpp
    for a specific scenario and condition only

    Args:
        scenario_name: Name of the scenario (e.g., 'SCENARIO_2_Segregation')
        condition_name: Name of the condition (e.g., 'bias_0.7')

    Returns:
        Dictionary with loose method dataframes
    """
    print("\n" + "="*80)
    print(f"CREATING LOOSE VERSIONS FOR {scenario_name}, {condition_name}")
    print("="*80)

    loose_results = {}

    # 1. Load maesterpp_loose (generated by maester_pipeline_simulation_selection_loose.py)
    print("\n--- maesterpp_loose ---")
    maesterpp_loose_path = "maesterpp_loose_mutation_combine.csv"
    if os.path.exists(maesterpp_loose_path):
        maesterpp_loose_df = pd.read_csv(maesterpp_loose_path)
        # Filter for scenario and condition
        scenario_df = maesterpp_loose_df[(maesterpp_loose_df['scenario'] == scenario_name) &
                                         (maesterpp_loose_df['condition'].str.contains(condition_name))]

        # Use the 'detected' column from the file
        scenario_df['detected_loose'] = scenario_df['detected']

        loose_results['maesterpp_loose'] = scenario_df
        print(f"maesterpp_loose: {scenario_df['detected_loose'].sum()} detected out of {len(scenario_df)} total")
    else:
        print(f"Warning: {maesterpp_loose_path} not found. Please run:")
        print("  python maester_pipeline_simulation_selection_loose.py --generate-scenario2")

    # 2. Create mquad_loose
    print("\n--- mquad_loose ---")
    mquad_path = "mquad_mutation_combine.csv"
    if os.path.exists(mquad_path):
        mquad_df = pd.read_csv(mquad_path)
        # Filter for scenario and condition (note: mquad uses 'Scenario' with capital S)
        scenario_df = mquad_df[(mquad_df['Scenario'] == scenario_name) &
                               (mquad_df['condition'].str.contains(condition_name))]

        # Sort by pval and select top N variants
        scenario_df_sorted = scenario_df.sort_values('pval', ascending=True)
        top_n = scenario_df_sorted.head(TOP_N_VARIANTS)

        # Create detected column
        scenario_df['detected_loose'] = False
        scenario_df.loc[top_n.index, 'detected_loose'] = True

        loose_results['mquad_loose'] = scenario_df
        print(f"mquad_loose: {scenario_df['detected_loose'].sum()} detected out of {len(scenario_df)} total")

    # 3. Create MitoTracer_loose
    print("\n--- MitoTracer_loose ---")
    mitotracer_path = "MitoTracer_mutation_combine.csv"
    if os.path.exists(mitotracer_path):
        mitotracer_df = pd.read_csv(mitotracer_path)
        # Filter for scenario and condition
        scenario_df = mitotracer_df[(mitotracer_df['scenario'] == scenario_name) &
                                    (mitotracer_df['condition'].str.contains(condition_name))]

        # Sort by distance (descending - higher is better) and select top N variants
        scenario_df_sorted = scenario_df.sort_values('distance', ascending=False)
        top_n = scenario_df_sorted.head(TOP_N_VARIANTS)

        # Create detected column
        scenario_df['detected_loose'] = False
        scenario_df.loc[top_n.index, 'detected_loose'] = True

        loose_results['MitoTracer_loose'] = scenario_df
        print(f"MitoTracer_loose: {scenario_df['detected_loose'].sum()} detected out of {len(scenario_df)} total")

    # 4. Create scMitoMut_loose
    print("\n--- scMitoMut_loose ---")
    scmitomut_path = "scMitoMut_mutation_combine.csv"
    if os.path.exists(scmitomut_path):
        scmitomut_df = pd.read_csv(scmitomut_path)
        # Filter for scenario and condition
        scenario_df = scmitomut_df[(scmitomut_df['scenario'] == scenario_name) &
                                   (scmitomut_df['condition'].str.contains(condition_name))]

        # Sort by pval (FDR-adjusted) and select top N variants
        scenario_df_sorted = scenario_df.sort_values('pval', ascending=True)
        top_n = scenario_df_sorted.head(TOP_N_VARIANTS)

        # Create detected column
        scenario_df['detected_loose'] = False
        scenario_df.loc[top_n.index, 'detected_loose'] = True

        loose_results['scMitoMut_loose'] = scenario_df
        print(f"scMitoMut_loose: {scenario_df['detected_loose'].sum()} detected out of {len(scenario_df)} total")

    return loose_results

def calculate_metrics_for_loose_methods(loose_results, scenario_name, condition_name):
    """
    Calculate metrics for loose method versions

    Args:
        loose_results: Dictionary of loose method dataframes
        scenario_name: Name of the scenario
        condition_name: Name of the condition

    Returns:
        DataFrame with metrics for each loose method
    """
    print("\n" + "="*80)
    print(f"CALCULATING METRICS FOR LOOSE VERSIONS")
    print("="*80)

    results = []

    for method_name, df in loose_results.items():
        print(f"\n--- {method_name} ---")

        # Define true mutations based on rest_mutation
        if 'rest_mutation' in df.columns:
            df['true_mutation'] = df['rest_mutation'] == True
        else:
            print(f"Warning: rest_mutation column not found for {method_name}")
            continue

        # Use detected_loose column
        detected_col = 'detected_loose'

        # Handle missing values
        mask = ~(df[detected_col].isna() | df['true_mutation'].isna())

        if mask.sum() == 0:
            print(f"Warning: No valid data for {method_name}")
            continue

        y_true = df.loc[mask, 'true_mutation']
        y_pred = df.loc[mask, detected_col]

        # Calculate confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'Method': method_name,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1_score,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'Total_Identified': tp + fp,
                'Total_True_Mutations': tp + fn
            })

            print(f"{method_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}, Identified={tp+fp}")

        except Exception as e:
            print(f"Error calculating metrics for {method_name}: {str(e)}")
            continue

    if results:
        return pd.DataFrame(results)
    else:
        return None

def main():
    """
    Main function to load data and create plots for Scenario 2, bias 0.7 condition
    """
    # Check if combined data already exists
    combined_data_path = os.path.join(os.getcwd(), "overall_analysis", "combined_mutation_data.csv")

    if os.path.exists(combined_data_path):
        print(f"Loading combined data from: {combined_data_path}")
        combined_df = pd.read_csv(combined_data_path)
        print(f"Loaded combined data: {combined_df.shape}")
    else:
        print("Combined data not found. Please run simulation_visualize.py first to generate the combined data.")
        return

    # Define methods to analyze
    methods = ['civet_LRT', 'civet_Wald', 'maesterpp', 'mquad', 'MitoTracer', 'scMitoMut']

    # Filter methods that exist in the data
    available_methods = [m for m in methods if f'{m}_detected' in combined_df.columns]
    print(f"\nAvailable methods: {available_methods}")

    # Specify scenario and condition
    scenario_name = 'SCENARIO_2_Segregation'
    condition_name = 'bias_0.9'

    # Calculate metrics for original methods
    print(f"\nCalculating metrics for {scenario_name}, {condition_name}...")
    metrics_df = calculate_metrics_for_condition(combined_df, available_methods,
                                                  scenario_name, condition_name)

    if metrics_df is None:
        print("Failed to calculate metrics. Exiting.")
        return

    # Print summary for original methods
    print("\n" + "="*60)
    print(f"METRICS SUMMARY (ORIGINAL): {scenario_name}, {condition_name}")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)

    # Create loose versions
    loose_results = create_loose_versions_for_scenario(scenario_name, condition_name)

    # Calculate metrics for loose versions
    metrics_loose_df = calculate_metrics_for_loose_methods(loose_results, scenario_name, condition_name)

    if metrics_loose_df is not None:
        # Print summary for loose methods
        print("\n" + "="*60)
        print(f"METRICS SUMMARY (LOOSE): {scenario_name}, {condition_name}")
        print("="*60)
        print(metrics_loose_df.to_string(index=False))
        print("="*60)

        # Combine original and loose metrics
        combined_metrics_df = pd.concat([metrics_df, metrics_loose_df], ignore_index=True)
    else:
        combined_metrics_df = metrics_df

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "scenario2_bias09_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")

    # Create plots
    print("\nCreating visualizations...")

    # Separate plots for precision and F1 (original methods only)
    output_path_separate = os.path.join(output_dir, 'precision_f1_separate_original.png')
    create_precision_f1_plot(metrics_df, scenario_name, condition_name, output_path_separate)

    # Combined plot (original methods only)
    output_path_combined = os.path.join(output_dir, 'precision_f1_combined_original.png')
    create_combined_plot(metrics_df, scenario_name, condition_name, output_path_combined)

    # Separate plots for precision and F1 (all methods including loose)
    output_path_separate_all = os.path.join(output_dir, 'precision_f1_separate_all.png')
    create_precision_f1_plot(combined_metrics_df, scenario_name, condition_name + " (All)", output_path_separate_all)

    # Combined plot (all methods including loose)
    output_path_combined_all = os.path.join(output_dir, 'precision_f1_combined_all.png')
    create_combined_plot(combined_metrics_df, scenario_name, condition_name + " (All)", output_path_combined_all)

    # Save metrics to CSV
    csv_path_original = os.path.join(output_dir, 'metrics_summary_original.csv')
    metrics_df.to_csv(csv_path_original, index=False)
    print(f"Original metrics saved to: {csv_path_original}")

    if metrics_loose_df is not None:
        csv_path_loose = os.path.join(output_dir, 'metrics_summary_loose.csv')
        metrics_loose_df.to_csv(csv_path_loose, index=False)
        print(f"Loose metrics saved to: {csv_path_loose}")

    csv_path_combined = os.path.join(output_dir, 'metrics_summary_all.csv')
    combined_metrics_df.to_csv(csv_path_combined, index=False)
    print(f"Combined metrics saved to: {csv_path_combined}")

if __name__ == "__main__":
    main()
