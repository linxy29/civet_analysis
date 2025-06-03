import os
import pandas as pd
import glob
from collections import defaultdict

# Set the working directory
os.chdir("/Users/linxy29/Documents/Data/CIVET/simulation")

# Define the generation thresholds
THRESHOLD_1 = 5
THRESHOLD_2 = 95

def find_cell_metadata_files():
    """Find all simulation_cell_metadata.csv files in SCENARIO_* folders"""
    return glob.glob("SCENARIO_*/*/metadata/simulation_cell_metadata.csv")

def load_cell_metadata(file_path):
    """Load cell metadata file and parse the data"""
    # Load the CSV file
    df = pd.read_csv(file_path, sep=',')
    
    # Group cells based on generation thresholds
    early_cells = set(df[df['generation'] < THRESHOLD_1]['cell_id'])
    middle_cells = set(df[(df['generation'] >= THRESHOLD_1) & (df['generation'] <= THRESHOLD_2)]['cell_id'])
    late_cells = set(df[df['generation'] > THRESHOLD_2]['cell_id'])
    
    # Get denovo mutations for each cell
    cell_mutations = {}
    for _, row in df.iterrows():
        cell_id = row['cell_id']
        denovo_mutations = []
        if isinstance(row['denovo_mutations'], str) and row['denovo_mutations']:
            denovo_mutations = row['denovo_mutations'].split(',')
        cell_mutations[cell_id] = denovo_mutations
    
    return early_cells, middle_cells, late_cells, cell_mutations

def find_non_informative_mutations(early_cells, middle_cells, late_cells, cell_mutations):
    """
    Find non-informative mutations:
    1. Mutations in early cells (generation < THRESHOLD_1)
    2. Mutations in late cells but not in middle cells
    """
    # Track mutations by cell group
    early_mutations = set()
    middle_mutations = set()
    late_mutations = set()
    
    # Collect mutations for each cell group
    for cell, mutations in cell_mutations.items():
        if cell in early_cells:
            early_mutations.update(mutations)
        elif cell in middle_cells:
            middle_mutations.update(mutations)
        elif cell in late_cells:
            late_mutations.update(mutations)
    
    # Find non-informative mutations
    non_informative_mutations = early_mutations.union(late_mutations - middle_mutations)
    return non_informative_mutations

def process_all_metadata_files():
    """Process all cell metadata files and return non-informative mutations by scenario"""
    scenario_non_informative = defaultdict(set)
    
    for metadata_file in find_cell_metadata_files():
        # Extract scenario and condition from file path
        parts = metadata_file.split(os.sep)
        scenario = parts[0]
        condition = parts[1]
        
        # Process the cell metadata
        early_cells, middle_cells, late_cells, cell_mutations = load_cell_metadata(metadata_file)
        
        # Find non-informative mutations
        non_informative = find_non_informative_mutations(early_cells, middle_cells, late_cells, cell_mutations)
        
        # Store by scenario and condition
        key = f"{scenario},{condition}"
        scenario_non_informative[key].update(non_informative)
        
        print(f"Processed {metadata_file}: Found {len(non_informative)} non-informative mutations")
    
    return scenario_non_informative

def update_metadata_files(scenario_non_informative):
    """Update all *_with_metadata files with informative column"""
    mutation_files = glob.glob("*_with_metadata.csv")
    results = []
    
    for file_path in mutation_files:
        print(f"Processing {file_path}")
        df = pd.read_csv(file_path)
        
        # Add informative column
        df['informative'] = True
        
        # Mark non-informative mutations
        for idx, row in df.iterrows():
            scenario_condition = f"{row['Scenario']},{row['condition']}"
            mutation_name = row['mutation_name']
            
            if mutation_name in scenario_non_informative.get(scenario_condition, set()):
                df.at[idx, 'informative'] = False
        
        # Save updated file
        output_file = file_path.replace('_with_metadata.csv', '_with_metadata_with_informative.csv')
        df.to_csv(output_file, index=False)
        
        # Calculate proportion for this file
        detected_informative_denovo = df[(df['detected'] == True) & 
                                        (df['metadata_is_denovo'] == True) & 
                                        (df['informative'] == True)].shape[0]
        
        all_informative_denovo = df[(df['metadata_is_denovo'] == True) & 
                                    (df['informative'] == True)].shape[0]
        
        proportion = detected_informative_denovo / all_informative_denovo if all_informative_denovo > 0 else 0
        
        results.append({
            'file': file_path,
            'detected_informative_denovo': detected_informative_denovo,
            'all_informative_denovo': all_informative_denovo,
            'proportion': proportion
        })
        
        print(f"  Saved to {output_file}")
        print(f"  Proportion: {proportion:.4f} ({detected_informative_denovo}/{all_informative_denovo})")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv("informative_mutations_summary.csv", index=False)
    print(f"Summary saved to informative_mutations_summary.csv")

def main():
    print(f"Processing cell metadata files with thresholds {THRESHOLD_1} and {THRESHOLD_2}")
    scenario_non_informative = process_all_metadata_files()
    update_metadata_files(scenario_non_informative)
    print("Processing complete!")

if __name__ == "__main__":
    main() 