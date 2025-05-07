# This file is used to analyze the statistics of selected variants for CIVET from the simulation data.

import os
import pandas as pd
import re
import glob

# Define thresholds for detection
thresholds = [0.01, 0.05, 1.0]

# Function to classify mutation types based on their names
def classify_mutation(variant_name):
    if variant_name.startswith("baseline_"):
        return "baseline"
    elif variant_name.startswith("false_"):
        return "false"
    else:
        return "other"

# Function to analyze a single civet_results.csv file
def analyze_civet_file(file_path, scenario_name, condition_name):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Remove rows where value column equals "LR_vals"
        lr_vals_df = df[df['value'] == "LR_vals"]
        df = df[df['value'] != "LR_vals"]
        
        # Convert generation column to numeric for comparison
        df['generation'] = pd.to_numeric(df['generation'], errors='coerce')
        
        # Add mutation type classification
        df['mutation_type'] = df['variant'].apply(classify_mutation)
        
        # Create results dictionary to store statistics
        results = {
            'scenario': scenario_name,
            'condition': condition_name,
            'total_mutations': len(df),
            'total_baseline': len(df[df['mutation_type'] == 'baseline']),
            'total_false': len(df[df['mutation_type'] == 'false']),
            'total_other': len(df[df['mutation_type'] == 'other'])
        }
        
        # Calculate statistics for each threshold
        for threshold in thresholds:
            detected = df[df['generation'] < threshold]
            non_detected = df[df['generation'] >= threshold]
            
            detected_baseline = len(detected[detected['mutation_type'] == 'baseline'])
            detected_false = len(detected[detected['mutation_type'] == 'false'])
            detected_other = len(detected[detected['mutation_type'] == 'other'])
            
            non_detected_baseline = len(non_detected[non_detected['mutation_type'] == 'baseline'])
            non_detected_false = len(non_detected[non_detected['mutation_type'] == 'false'])
            non_detected_other = len(non_detected[non_detected['mutation_type'] == 'other'])
            
            # Add to results dictionary
            results[f'detected_total_{threshold}'] = len(detected)
            results[f'detected_baseline_{threshold}'] = detected_baseline
            results[f'detected_false_{threshold}'] = detected_false
            results[f'detected_other_{threshold}'] = detected_other
            results[f'non_detected_total_{threshold}'] = len(non_detected)
            results[f'non_detected_baseline_{threshold}'] = non_detected_baseline
            results[f'non_detected_false_{threshold}'] = non_detected_false
            results[f'non_detected_other_{threshold}'] = non_detected_other
            
            # Calculate detection rates
            if results['total_baseline'] > 0:
                results[f'baseline_detection_rate_{threshold}'] = detected_baseline / results['total_baseline']
            else:
                results[f'baseline_detection_rate_{threshold}'] = 0
                
            if results['total_false'] > 0:
                results[f'false_detection_rate_{threshold}'] = detected_false / results['total_false']
            else:
                results[f'false_detection_rate_{threshold}'] = 0
                
            if results['total_other'] > 0:
                results[f'other_detection_rate_{threshold}'] = detected_other / results['total_other']
            else:
                results[f'other_detection_rate_{threshold}'] = 0
        
        return results
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to find all civet_results.csv files
def find_civet_files(base_dir):
    pattern = os.path.join(base_dir, "SCENARIO_*", "*", "civet_res", "civet_results.csv")
    return glob.glob(pattern)

# Main function to analyze all files and create summary
def main():
    base_dir = "/home/linxy29/data/CIVET/simulation"  # Assuming the script runs from the simulation directory
    civet_files = find_civet_files(base_dir)
    
    all_results = []
    
    for file_path in civet_files:
        # Extract scenario and condition from the file path
        path_parts = file_path.split(os.sep)
        scenario_name = path_parts[-4]
        condition_name = path_parts[-3]
        
        # Analyze the file
        results = analyze_civet_file(file_path, scenario_name, condition_name)
        if results:
            all_results.append(results)
    
    # Create dataframe from all results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Save summary to CSV
        summary_df.to_csv(os.path.join(base_dir, "civet_analysis_summary.csv"), index=False)
        print(f"Analysis complete. Results saved to civet_analysis_summary.csv")
        
        # Also create per-threshold summary tables
        for threshold in thresholds:
            threshold_cols = [col for col in summary_df.columns if str(threshold) in col or col in ['scenario', 'condition']]
            threshold_df = summary_df[threshold_cols]
            threshold_df.to_csv(os.path.join(base_dir, f"civet_analysis_threshold_{threshold}.csv"), index=False)
            print(f"Threshold {threshold} summary saved to civet_analysis_threshold_{threshold}.csv")
    else:
        print("No results were generated. Check file paths and data format.")

if __name__ == "__main__":
    main()