# This file is used to analyze the statistics of selected variants for CIVET from the simulation data.

import os
import pandas as pd
import glob
import re

def process_civet_results(root_dir):
    """
    Process civet_results.csv files and generate civet_mutation_combine files
    
    Args:
        root_dir: Root directory containing scenario folders
    """
    # Dictionary to store results by value type and threshold
    results_by_type_threshold = {}
    thresholds = [0.01, 0.05, 1]
    
    # Find all civet_results.csv files
    for civet_file in glob.glob(os.path.join(root_dir, "**/civet_res/civet_results.csv"), recursive=True):
        print(f"Processing {civet_file}")
        
        # Get scenario information from directory path
        scenario_path = os.path.dirname(os.path.dirname(civet_file))
        scenario_name = os.path.basename(scenario_path)
        condition_name = os.path.basename(os.path.dirname(scenario_path))
        
        # Read civet_results.csv
        try:
            civet_df = pd.read_csv(civet_file)
        except Exception as e:
            print(f"Error reading {civet_file}: {e}")
            continue
        
        # Filter out rows where value = 'LR_vals'
        filtered_df = civet_df[civet_df['value'] != 'LR_vals']
        
        # Get remaining value types after filtering
        remaining_value_types = filtered_df['value'].unique()
        
        # Get all mutations from the variant column
        all_mutations = civet_df['variant'].unique()
        
        # Read mutation info file to identify baseline and false mutations
        mutation_info_file = os.path.join(scenario_path, "metadata/simulation_mutation_info.csv")
        
        baseline_mutations = []
        false_mutations = []
        rest_mutations = []
        
        if os.path.exists(mutation_info_file):
            try:
                mutation_info_df = pd.read_csv(mutation_info_file)
                baseline_mutations = mutation_info_df[mutation_info_df['mutation_type'] == 'baseline']['mutation_name'].tolist()
                false_mutations = mutation_info_df[mutation_info_df['mutation_type'] == 'false']['mutation_name'].tolist()
                # Rest mutations are those that are neither baseline nor false
                rest_mutations = [m for m in mutation_info_df['mutation_name'].tolist() 
                                 if m not in baseline_mutations and m not in false_mutations]
            except Exception as e:
                print(f"Error reading mutation info file {mutation_info_file}: {e}")
        
        # Process each value type with thresholds
        for value_type in remaining_value_types:
            value_type_df = filtered_df[filtered_df['value'] == value_type]
            
            # Process for each threshold
            for threshold in thresholds:
                key = f"{value_type}_{threshold}"
                
                # Initialize results for this value type and threshold if not already done
                if key not in results_by_type_threshold:
                    results_by_type_threshold[key] = []
                
                # Identify detected mutations based on threshold
                detected_mutations = value_type_df[value_type_df['generation'] < threshold]['variant'].unique()
                
                # Process each mutation
                for mutation in all_mutations:
                    detected = mutation in detected_mutations
                    is_baseline = mutation in baseline_mutations
                    is_false = mutation in false_mutations
                    is_rest = mutation in rest_mutations if rest_mutations else not (is_baseline or is_false)
                    
                    # Add to results
                    results_by_type_threshold[key].append({
                        'Scenario': condition_name,
                        'condition': scenario_name,
                        'mutation_name': mutation,
                        'detected': detected,
                        'baseline_mutation': is_baseline,
                        'false_mutation': is_false,
                        'rest_mutation': is_rest
                    })
    
    # Create DataFrames and save results to CSV files
    for key, results in results_by_type_threshold.items():
        if not results:
            continue
            
        results_df = pd.DataFrame(results)
        
        # Create output filename
        value_type, threshold = key.rsplit('_', 1)
        output_file = os.path.join(root_dir, f"civet_mutation_combine_{value_type}_threshold_{threshold}.csv")
        
        # Save results to CSV
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Calculate and print summary statistics
        print(f"\nSummary Statistics for {key}:")
        detected_counts = results_df[results_df['detected'] == True].groupby(['baseline_mutation', 'false_mutation', 'rest_mutation']).size()
        non_detected_counts = results_df[results_df['detected'] == False].groupby(['baseline_mutation', 'false_mutation', 'rest_mutation']).size()
        
        print("\nDetected Mutations:")
        print(detected_counts)
        
        print("\nNon-Detected Mutations:")
        print(non_detected_counts)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process civet_results.csv files and generate mutation analysis')
    parser.add_argument('root_dir', type=str, help='Root directory containing scenario folders')
    
    args = parser.parse_args()
    
    process_civet_results(args.root_dir)