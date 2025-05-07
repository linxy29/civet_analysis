#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import yaml
import json
from datetime import datetime
import argparse

from .simulation_framework_af import (
    run_basic_simulation,
    load_config
)
from .simulation_save import save_simulation_data

#############################################
# 1) Helper function to modify configuration
#############################################

def run_sim_and_save(config, scenario_name, output_subdir):
    """
    Helper function that runs the simulation and saves the data to output_subdir.
    """
    # 1) Run the simulation
    cells, mutations, expr_df, gene_params = run_basic_simulation(config=config)
    
    # 2) Save the results
    os.makedirs(output_subdir, exist_ok=True)
    save_simulation_data(cells, mutations, expr_df, gene_params, output_subdir)

    print(f"[{scenario_name}] Finished. Results saved to: {output_subdir}")


#############################################
# 2) SCENARIO 1: Exploring Different Mutation Rate Regimes
#############################################

def scenario_1_mutation_rate(config, rates=[1, 2, 4, 8, 16]):
    """
    Scenario 1: Explore different mtDNA mutation rates per mitosis.
    
    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    rates : list
        Different mutation rates to try.
    """
    scenario_name = "SCENARIO_1_Mutation_Rate"
    print(f"=== Running {scenario_name} ===")
    
    for r in rates:
        # Copy the configuration to avoid mutating the base config
        sim_config = config.copy()
        
        # Update the mutation rate
        sim_config["MTDNA_MUTATION_RATE_PER_MITOSIS"] = r
        
        # Create an output subdirectory for each mutation rate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = f"/Users/linxy29/Documents/Data/CIVET/simulation/{scenario_name}/mutation_rate_{r}_{timestamp}"
        
        # Run and save
        run_sim_and_save(sim_config, scenario_name, output_subdir)


#############################################
# 3) SCENARIO 2: Varying mtDNA Segregation Models
#############################################

def scenario_2_segregation_models(config, biases=[0.5, 0.7, 0.9]):
    """
    Scenario 2: Test different mtDNA segregation bias values
    (e.g., 0.5 = completely symmetric, 0.9 = strongly asymmetric).
    
    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    biases : list
        Different 'bias' factors to try for mtDNA segregation.
    """
    scenario_name = "SCENARIO_2_Segregation"
    print(f"=== Running {scenario_name} ===")
    
    for b in biases:
        # Copy the configuration
        sim_config = config.copy() 
        sim_config["SEGREGATION_BIAS"] = b
        
        # Create an output subdirectory for each bias value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = f"/Users/linxy29/Documents/Data/CIVET/simulation/{scenario_name}/bias_{b}_{timestamp}"
        
        # Run and save
        run_sim_and_save(sim_config, scenario_name, output_subdir)


#############################################
# 4) SCENARIO 3: Linear vs. Bifurcated Differentiation Paths
#############################################

def scenario_3_differentiation(config, fractions=[0.2, 0.4, 0.8]):
    """
    Scenario 3: Compare linear vs. bifurcated differentiation paths
    by varying `CELL_TYPE_SPECIFIC_FRAC`.
    
    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    fractions : list
        Different fractions to test (e.g., 0.2 => more linear, 0.8 => stronger bifurcation).
    """
    scenario_name = "SCENARIO_3_Differentiation"
    print(f"=== Running {scenario_name} ===")
    
    for frac in fractions:
        sim_config = config.copy()
        
        # Adjust how many genes are unique to each cell type
        sim_config["CELL_TYPE_FRAC"] = frac
        
        # Create an output subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = f"/Users/linxy29/Documents/Data/CIVET/simulation/{scenario_name}/cell_type_frac_{frac}_{timestamp}"
        
        # Run and save
        run_sim_and_save(sim_config, scenario_name, output_subdir)


#############################################
# 5) Main Entry Point
#############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mtDNA simulation scenarios")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file (YAML or JSON).")
    parser.add_argument("--scenario", type=str, default="all",
                        help="Which scenario to run: [1, 2, 3, all].")
    args = parser.parse_args()
    
    # 1) Load base configuration
    base_config = load_config(args.config)
    base_config = load_config("simulation_config.yaml")
    
    # 2) Decide which scenario(s) to run
    scenario_choice = args.scenario.lower()
    
    if scenario_choice in ["1", "all"]:
        scenario_1_mutation_rate(base_config)
        
    if scenario_choice in ["2", "all"]:
        scenario_2_segregation_models(base_config)
        
    if scenario_choice in ["3", "all"]:
        scenario_3_differentiation(base_config)
