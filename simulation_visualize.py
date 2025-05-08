import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Define file paths for the four methods
files = {
    "maesterpp": "maesterpp_mutation_combine.csv",
    "mquad": "mquad_mutation_combine.csv",
    "civet_LRT": "civet_mutation_combine_LRT_pvals_threshold_0.05.csv",
    "civet_Wald": "civet_mutation_combine_Wald_pvals_threshold_0.05.csv"
}

# Function to load and preprocess data
def load_data(file_path):
    try:
        if "maesterpp" in file_path:
            df = pd.read_csv(file_path)
            # Renaming the first column to ensure consistency
            if df.columns[0].lower() == "scenario":
                df.rename(columns={df.columns[0]: "Scenario"}, inplace=True)
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to calculate metrics for a given method's data
def calculate_metrics(df):
    # True mutations are in rest_mutation column with value True
    # detected column tells us if the method detected it
    
    results = []
    
    # Group by scenario and condition
    for (scenario, condition), group in df.groupby(['Scenario', 'condition']):
        # True positives: detected=True and rest_mutation=True
        tp = len(group[(group['detected'] == True) & (group['rest_mutation'] == True)])
        
        # False positives: detected=True but rest_mutation=False
        fp = len(group[(group['detected'] == True) & (group['rest_mutation'] == False)])
        
        # False negatives: detected=False but rest_mutation=True
        fn = len(group[(group['detected'] == False) & (group['rest_mutation'] == True)])
        
        # True negatives: detected=False and rest_mutation=False
        tn = len(group[(group['detected'] == False) & (group['rest_mutation'] == False)])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # For maesterpp, extract mean coverage information
        mean_coverage = None
        if 'mean_coverage' in group.columns:
            mean_coverage = group['mean_coverage'].mean()
        
        results.append({
            'Scenario': scenario,
            'Condition': condition,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Mean_Coverage': mean_coverage
        })
    
    return pd.DataFrame(results)

# Function to analyze all methods
def analyze_all_methods():
    all_results = {}
    
    for method, file_path in files.items():
        df = load_data(file_path)
        if df is not None:
            metrics = calculate_metrics(df)
            all_results[method] = metrics
    
    return all_results

# Function to visualize metrics for each scenario and condition
def visualize_metrics_by_scenario(all_results):
    scenarios = set()
    for method, results in all_results.items():
        scenarios.update(results['Scenario'].unique())
    
    metrics = ['Precision', 'Recall', 'F1']
    
    for scenario in scenarios:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Performance Metrics for {scenario}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            # Create a dataframe for this scenario and metric
            plot_data = []
            for method, results in all_results.items():
                scenario_data = results[results['Scenario'] == scenario]
                if not scenario_data.empty:
                    for _, row in scenario_data.iterrows():
                        plot_data.append({
                            'Method': method,
                            'Condition': row['Condition'],
                            'Value': row[metric]
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Extract the variable part of the condition (e.g., "cell_type_frac_0.8")
                plot_df['Condition_Short'] = plot_df['Condition'].apply(
                    lambda x: x.split('_20')[0] if '_20' in x else x
                )
                
                # Plot
                sns.barplot(x='Condition_Short', y='Value', hue='Method', ax=axes[i], data=plot_df)
                axes[i].set_title(metric)
                axes[i].set_ylim(0, 1)
                if i > 0:  # Only show legend on the first plot
                    axes[i].get_legend().remove()
                
                # Rotate x-axis labels if there are many conditions
                if len(plot_df['Condition_Short'].unique()) > 3:
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{scenario}_metrics.png")
        plt.close()

# Function to analyze performance vs mean coverage (for maesterpp)
def analyze_coverage_impact(maesterpp_results):
    # Only apply to maesterpp which has mean_coverage
    if maesterpp_results is None or maesterpp_results.empty:
        print("No maesterpp results available for coverage analysis")
        return
    
    if 'Mean_Coverage' not in maesterpp_results.columns:
        print("Mean_Coverage not found in maesterpp results")
        return
    
    # Filter out rows where Mean_Coverage is None
    maesterpp_results = maesterpp_results.dropna(subset=['Mean_Coverage'])
    
    if maesterpp_results.empty:
        print("No valid coverage data found for analysis")
        return
    
    # Create bins for coverage
    maesterpp_results['Coverage_Bin'] = pd.cut(
        maesterpp_results['Mean_Coverage'], 
        bins=5,
        labels=False
    )
    
    # Calculate mean metrics for each bin
    coverage_analysis = maesterpp_results.groupby('Coverage_Bin')[['Mean_Coverage', 'Precision', 'Recall', 'F1']].mean().reset_index()
    
    # Plot the relationship
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Impact of Mean Coverage on Performance (maesterpp)', fontsize=16)
    
    metrics = ['Precision', 'Recall', 'F1']
    for i, metric in enumerate(metrics):
        sns.regplot(x='Mean_Coverage', y=metric, data=maesterpp_results, ax=axes[i])
        axes[i].set_title(f'{metric} vs Mean Coverage')
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("coverage_impact.png")
    plt.close()
    
    return coverage_analysis

# Function to create a combined visualization of all methods
def visualize_all_methods(all_results):
    # Prepare combined dataframe for comparison
    combined_data = []
    
    for method, results in all_results.items():
        for _, row in results.iterrows():
            combined_data.append({
                'Method': method,
                'Scenario': row['Scenario'],
                'Condition': row['Condition'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1': row['F1']
            })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Create a summary visualization
    metrics = ['Precision', 'Recall', 'F1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Overall Performance Comparison of Methods', fontsize=16)
    
    for i, metric in enumerate(metrics):
        sns.boxplot(x='Method', y=metric, data=combined_df, ax=axes[i])
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("overall_comparison.png")
    plt.close()
    
    # Create a heatmap of average metrics by method and scenario
    avg_by_scenario = combined_df.groupby(['Method', 'Scenario'])[metrics].mean().reset_index()
    
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        pivot_data = avg_by_scenario.pivot(index='Method', columns='Scenario', values=metric)
        sns.heatmap(pivot_data, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f')
        plt.title(f'Average {metric} by Method and Scenario')
        plt.tight_layout()
        plt.savefig(f"heatmap_{metric}.png")
        plt.close()

# Main execution function
def main():
    print("Starting mutation analysis...")
    
    # Analyze all methods
    all_results = analyze_all_methods()
    
    if not all_results:
        print("No results to analyze. Check file paths.")
        return
    
    # Display summary statistics
    print("\nSummary Statistics:")
    for method, results in all_results.items():
        print(f"\n{method.upper()} Metrics:")
        avg_metrics = results[['Precision', 'Recall', 'F1']].mean()
        print(f"Average Precision: {avg_metrics['Precision']:.3f}")
        print(f"Average Recall: {avg_metrics['Recall']:.3f}")
        print(f"Average F1 Score: {avg_metrics['F1']:.3f}")
    
    # Visualize metrics by scenario
    print("\nCreating visualizations by scenario...")
    visualize_metrics_by_scenario(all_results)
    
    # Analyze coverage impact for maesterpp
    print("\nAnalyzing impact of coverage on performance...")
    if 'maesterpp' in all_results:
        coverage_analysis = analyze_coverage_impact(all_results['maesterpp'])
        if coverage_analysis is not None:
            print("\nCoverage Analysis Results:")
            print(coverage_analysis)
    
    # Create combined visualizations
    print("\nCreating overall comparison visualizations...")
    visualize_all_methods(all_results)
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()