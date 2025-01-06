#!/usr/bin/env python3
"""
visualization_scenarios.py

Combines:
1) Scenario 1 simulation (varying mtDNA mutation rates), saving results to a specified directory.
2) A collection of visualization functions for analyzing simulation data.
3) Example usage in `main()` to call *all* the visualization functions.

Author: Your Name
Date: 2023-XX-XX
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import pandas as pd

# ------------------------------------------------------------------------------
# Import all functions and classes from the simulation framework
# ------------------------------------------------------------------------------
from simulation_framework import (
    simulate_stem_cell_growth,
    simulate_cell_differentiation,
    mtDNA_mutation_rate_per_mitosis,
)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def count_unique_mutations(cells):
    all_mutations = set()
    for cell in cells:
        for mt in cell.mtDNA_list:
            all_mutations.update(mt.mutations)
    return len(all_mutations)

# ------------------------------------------------------------------------------
# Create a directory for saving figures
# ------------------------------------------------------------------------------
FIGURES_DIR = "/home/linxy29/data/CIVET/simulation"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Scenario 1: Varying Mitochondrial Mutation Rates
# ------------------------------------------------------------------------------
def scenario_1_varying_mutation_rates():
    """
    Compare simulations with mtDNA mutation rates ranging from 0.01 to 0.30
    (in increments of 0.01), then save the bar chart result as an image in FIGURES_DIR.
    """
    mutation_rates = np.arange(0.01, 0.31, 0.01)  # Mutation rates from 0.01 to 0.30
    unique_counts = []
    
    for rate in mutation_rates:
        global mtDNA_mutation_rate_per_mitosis
        old_rate = mtDNA_mutation_rate_per_mitosis
        mtDNA_mutation_rate_per_mitosis = rate
        
        # Run a reduced simulation for demonstration
        stem_cells = simulate_stem_cell_growth()
        all_cells = simulate_cell_differentiation(stem_cells)
        
        # Count unique mutations
        num_mutations = count_unique_mutations(all_cells)
        unique_counts.append(num_mutations)
        
        # Restore old rate
        mtDNA_mutation_rate_per_mitosis = old_rate
    
    # Save the bar chart
    output_path = os.path.join(FIGURES_DIR, "scenario1_mutation_rates_series.png")
    plt.figure(figsize=(8, 4))
    plt.bar([f"{r:.2f}" for r in mutation_rates], unique_counts, color='skyblue')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Number of Unique Mutations')
    plt.title('Effect of mtDNA Mutation Rate (0.01 to 0.30)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Scenario 1 plot saved to: {output_path}")

# ------------------------------------------------------------------------------
# Additional Visualization Functions (General Purpose)
# ------------------------------------------------------------------------------
def plot_mutations_vs_generations(generations, mutation_counts, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(generations, mutation_counts, marker='o', color='navy')
    plt.xlabel("Generation")
    plt.ylabel("Number of Unique Mutations")
    plt.title("Accumulation of Mutations Over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Mutations vs. Generations plot saved to: {save_path}")

def plot_heteroplasmy_distribution(heteroplasmy_values, save_path, bins=20):
    plt.figure(figsize=(6, 4))
    plt.hist(heteroplasmy_values, bins=bins, alpha=0.7, color='steelblue')
    plt.xlabel("Heteroplasmy Level")
    plt.ylabel("Count")
    plt.title("Distribution of Heteroplasmy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heteroplasmy Distribution plot saved to: {save_path}")

def plot_cell_type_distribution(all_cells, save_path):
    cell_types = [cell.cell_type for cell in all_cells]
    counts = Counter(cell_types)
    labels, sizes = zip(*counts.items())
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Cell Type Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Cell Type Distribution plot saved to: {save_path}")

def plot_gene_expression_pca(expression_df, cell_types, save_path):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(expression_df.values)
    
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=cell_types, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cell Type / Cluster')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Simulated Gene Expression")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gene Expression PCA plot saved to: {save_path}")

def plot_dp_ad_scatter(dp_values, ad_values, save_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(dp_values, ad_values, alpha=0.5, color='crimson')
    plt.xlabel("Total Depth (DP)")
    plt.ylabel("Allele Depth (AD)")
    plt.title("DP vs. AD in Simulated Reads")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"DP vs. AD Scatter plot saved to: {save_path}")

# ------------------------------------------------------------------------------
# Main Function: Run Scenario 1 and Additional Plots
# ------------------------------------------------------------------------------
def main():
    # Run Scenario 1
    scenario_1_varying_mutation_rates()
    
    # Additional Plots with Example Data
    generations = list(range(1, 11))
    mutation_counts = [g * 5 + np.random.randint(-3, 4) for g in generations]
    plot_mutations_vs_generations(
        generations, mutation_counts, os.path.join(FIGURES_DIR, "mutations_vs_generations.png")
    )

    heteroplasmy_vals = np.random.beta(a=2, b=5, size=300)
    plot_heteroplasmy_distribution(
        heteroplasmy_vals, os.path.join(FIGURES_DIR, "heteroplasmy_distribution.png")
    )

    class MockCell:
        def __init__(self, ctype):
            self.cell_type = ctype
    all_cells = [MockCell(ctype) for ctype in ["StemCell"] * 40 + ["Progenitor"] * 30 + ["DifferentiatedType1"] * 20]
    plot_cell_type_distribution(
        all_cells, os.path.join(FIGURES_DIR, "cell_type_distribution.png")
    )

    expression_data = np.abs(np.random.normal(loc=5, scale=2, size=(100, 200)))
    expression_df = pd.DataFrame(expression_data)
    cell_type_labels = np.random.randint(0, 3, size=100)
    plot_gene_expression_pca(
        expression_df, cell_type_labels, os.path.join(FIGURES_DIR, "gene_expression_pca.png")
    )

    dp_vals = np.random.randint(50, 1000, size=100)
    ad_vals = (dp_vals * 0.3).astype(int) + np.random.randint(-10, 11, size=100)
    plot_dp_ad_scatter(
        dp_vals, ad_vals, os.path.join(FIGURES_DIR, "dp_vs_ad_scatter.png")
    )

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
