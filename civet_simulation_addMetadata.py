# This script adds metadata to the CIVET files.
# It loads all the metadata from the simulation_mutation_info.csv files and adds them to the civet_mutation_combine_* files.
# It then saves the civet_mutation_combine_* files with the metadata as civet_mutation_combine_*_with_metadata.csv.

import pandas as pd
import os
import glob
from pathlib import Path

def load_mutation_metadata(simulation_folder):
    """
    Load all simulation_mutation_info.csv files from scenario folders
    and combine them into a single dataframe with scenario information.
    """
    metadata_dict = {}
    
    # Find all simulation_mutation_info.csv files
    scenario_folders = glob.glob(os.path.join(simulation_folder, "SCENARIO_*"))
    
    for scenario_folder in scenario_folders:
        scenario_name = os.path.basename(scenario_folder)
        
        # Find all condition subfolders within each scenario
        condition_folders = glob.glob(os.path.join(scenario_folder, "*_*"))
        
        for condition_folder in condition_folders:
            condition_name = os.path.basename(condition_folder)
            metadata_file = os.path.join(condition_folder, "metadata", "simulation_mutation_info.csv")
            
            if os.path.exists(metadata_file):
                try:
                    df = pd.read_csv(metadata_file)
                    # Add scenario and condition information
                    df['scenario'] = scenario_name
                    df['condition'] = condition_name
                    
                    # Store with a unique key
                    key = f"{scenario_name}_{condition_name}"
                    metadata_dict[key] = df
                    print(f"Loaded metadata from: {metadata_file}")
                    
                except Exception as e:
                    print(f"Error loading {metadata_file}: {e}")
    
    # Combine all metadata into one dataframe
    if metadata_dict:
        combined_metadata = pd.concat(metadata_dict.values(), ignore_index=True)
        return combined_metadata
    else:
        print("No metadata files found!")
        return pd.DataFrame()

def add_metadata_to_civet_files(simulation_folder):
    """
    Add mutation metadata to all civet_mutation_combine_* files.
    """
    # Load all mutation metadata
    print("Loading mutation metadata...")
    metadata_df = load_mutation_metadata(simulation_folder)
    
    if metadata_df.empty:
        print("No metadata loaded. Exiting.")
        return
    
    print(f"Loaded metadata for {len(metadata_df)} mutations across {len(metadata_df.groupby(['scenario', 'condition']))} scenario-condition combinations")
    
    # Find all civet_mutation_combine_* files
    civet_files = glob.glob(os.path.join(simulation_folder, "civet_mutation_combine_*.csv"))
    
    if not civet_files:
        print("No civet_mutation_combine_* files found!")
        return
    
    print(f"Found {len(civet_files)} CIVET files to process")
    
    for civet_file in civet_files:
        print(f"\nProcessing: {os.path.basename(civet_file)}")
        
        try:
            # Load CIVET file
            civet_df = pd.read_csv(civet_file)
            print(f"  Original shape: {civet_df.shape}")
            
            # Create a merge key for matching
            # We'll merge on mutation_name, Scenario, and condition
            merge_df = metadata_df.copy()
            merge_df = merge_df.rename(columns={
                'mutation_id': 'mutation_name',
                'scenario': 'Scenario'  # Match the CIVET file column name
            })
            
            # Print column names for debugging
            print(f"  CIVET columns: {list(civet_df.columns)}")
            print(f"  Metadata columns: {list(merge_df.columns)}")
            
            # Check if we have the required columns for merging
            required_civet_cols = ['mutation_name', 'Scenario', 'condition']
            missing_cols = [col for col in required_civet_cols if col not in civet_df.columns]
            
            if missing_cols:
                print(f"  Missing columns in CIVET file: {missing_cols}")
                print(f"  Available columns: {list(civet_df.columns)}")
                print("  Skipping this file due to missing columns")
                continue
            
            # Show some sample data for verification
            print(f"  Sample CIVET scenarios: {civet_df['Scenario'].unique()[:3]}")
            print(f"  Sample metadata scenarios: {merge_df['Scenario'].unique()[:3]}")
            
            # Merge the dataframes
            # Left join to keep all CIVET records, even if no metadata match
            merged_df = pd.merge(
                civet_df,
                merge_df[['mutation_name', 'Scenario', 'condition', 'is_baseline', 
                         'is_denovo', 'is_false', 'mutation_type', 'cells_with_mutation']],
                on=['mutation_name', 'Scenario', 'condition'],
                how='left'
            )
            
            # Rename columns to match your preferred naming convention
            merged_df = merged_df.rename(columns={
                'is_baseline': 'metadata_is_baseline',
                'is_denovo': 'metadata_is_denovo', 
                'is_false': 'metadata_is_false',
                'mutation_type': 'metadata_mutation_type',
                'cells_with_mutation': 'metadata_cells_with_mutation'
            })
            
            print(f"  Merged shape: {merged_df.shape}")
            print(f"  Mutations with metadata: {merged_df['metadata_mutation_type'].notna().sum()}")
            print(f"  Mutations without metadata: {merged_df['metadata_mutation_type'].isna().sum()}")
            
            # Save the enhanced file
            output_file = civet_file.replace('.csv', '_with_metadata.csv')
            merged_df.to_csv(output_file, index=False)
            print(f"  Saved to: {os.path.basename(output_file)}")
            
            # Show some statistics
            if 'metadata_mutation_type' in merged_df.columns:
                type_counts = merged_df['metadata_mutation_type'].value_counts(dropna=False)
                print(f"  Mutation type distribution:")
                for mut_type, count in type_counts.items():
                    print(f"    {mut_type}: {count}")
                    
        except Exception as e:
            print(f"  Error processing {civet_file}: {e}")

def main():
    """
    Main function to run the metadata addition process.
    """
    # Define the simulation folder path
    simulation_folder = "/Users/linxy29/Documents/Data/CIVET/simulation"  # Adjust this path as needed
    
    # Check if the folder exists
    if not os.path.exists(simulation_folder):
        print(f"Simulation folder '{simulation_folder}' not found!")
        print("Please make sure you're running this script from the correct directory")
        print("or update the simulation_folder path in the script.")
        return
    
    print(f"Processing simulation folder: {os.path.abspath(simulation_folder)}")
    
    # Add metadata to CIVET files
    add_metadata_to_civet_files(simulation_folder)
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()
