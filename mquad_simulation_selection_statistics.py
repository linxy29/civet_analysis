## Usage: python civet_simulation_selection_statistics.py /Users/linxy29/Documents/Data/CIVET/simulation                   

import os
import pandas as pd
import glob
import re

def process_bic_params(root_dir):
    """
    Process BIC_params.csv files and generate maesterPP_mutation_combine.csv
    
    Args:
        root_dir: Root directory containing scenario folders
    """
    # List to store results
    results = []
    
    # Find all BIC_params.csv files under mquad_out folders
    for bic_params_file in glob.glob(os.path.join(root_dir, "**/mquad_out/BIC_params.csv"), recursive=True):
        print(f"Processing {bic_params_file}")
        
        # Get scenario information from directory path
        scenario_path = os.path.dirname(os.path.dirname(bic_params_file))
        scenario_name = os.path.basename(scenario_path)
        
        # Read BIC_params.csv
        try:
            bic_df = pd.read_csv(bic_params_file)
        except Exception as e:
            print(f"Error reading {bic_params_file}: {e}")
            continue
        
        # Filter rows where both PASS_KP and PASS_MINCELLS are True
        passed_df = bic_df[(bic_df['PASS_KP'] == True) & (bic_df['PASS_MINCELLS'] == True)]
        
        if passed_df.empty:
            print(f"No rows with PASS_KP=True and PASS_MINCELLS=True in {bic_params_file}")
            continue
        
        # Extract variant_name values and remove SNP prefix
        detected_variants = passed_df['variant_name'].tolist()
        detected_indices = [int(re.sub(r'SNP', '', var)) for var in detected_variants]
        
        # Find cellSNP folder and read mutations file
        cellsnp_dir = os.path.join(os.path.dirname(os.path.dirname(bic_params_file)), "cellSNP")
        mutations_file = os.path.join(cellsnp_dir, "cellSNP.tag.mutations.txt")
        
        if not os.path.exists(mutations_file):
            print(f"Mutations file not found: {mutations_file}")
            continue
        
        # Read mutations file
        with open(mutations_file, 'r') as f:
            mutations = [line.strip() for line in f.readlines()]
        
        # Read mutation info file to identify baseline and false mutations
        mutation_info_file = os.path.join(os.path.dirname(os.path.dirname(bic_params_file)), 
                                         "metadata/simulation_mutation_info.csv")
        
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
        
        # Process both detected and non-detected mutations
        for i, mutation in enumerate(mutations):
            detected = f"SNP{i+1}" in detected_variants
            is_baseline = mutation in baseline_mutations
            is_false = mutation in false_mutations
            is_rest = mutation in rest_mutations if rest_mutations else not (is_baseline or is_false)
            
            # Add to results
            results.append({
                'Scenario': os.path.basename(os.path.dirname(scenario_path)),
                'condition': scenario_name,
                'mutation_name': mutation,
                'detected': detected,
                'baseline_mutation': is_baseline,
                'false_mutation': is_false,
                'rest_mutation': is_rest
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    output_file = os.path.join(root_dir, "mquad_mutation_combine.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    detected_counts = results_df[results_df['detected'] == True].groupby(['baseline_mutation', 'false_mutation', 'rest_mutation']).size()
    non_detected_counts = results_df[results_df['detected'] == False].groupby(['baseline_mutation', 'false_mutation', 'rest_mutation']).size()
    
    print("\nDetected Mutations:")
    print(detected_counts)
    
    print("\nNon-Detected Mutations:")
    print(non_detected_counts)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process BIC_params.csv files and generate mutation analysis')
    parser.add_argument('root_dir', type=str, help='Root directory containing scenario folders')
    
    args = parser.parse_args()
    
    process_bic_params(args.root_dir)