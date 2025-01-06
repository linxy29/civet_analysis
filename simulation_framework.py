"""
simulation_framework.py

Contains the main simulation framework for:
1. Stem cell growth simulation
2. Mitochondrial genome proliferation
3. Cell differentiation (basic linear model)
4. Single-cell transcriptomic data simulation
5. Sequencing read depth (DP) and allele depth (AD) for mtDNA mutations
"""

import numpy as np
import pandas as pd
import random
from collections import deque

# Set random seed for reproducibility
np.random.seed(42)

# ------------------------------------------------------------------------------
# Simulation Parameters
# ------------------------------------------------------------------------------

# Stem cell growth parameters
r = 1.0        # Maximum division rate
kappa = 0.1    # Growth acceleration rate
t0 = 50        # Inflection point in time

# mtDNA parameters
mtDNA_initial_count = 500
mtDNA_mutation_rate_per_mitosis = 0.4   # Mutations per cell division

# Sequencing read depth parameters
mean_dp = 100           # Mean total read depth per mutation site
dp_dispersion = 0.2     # Dispersion parameter for total read depth (overdispersion)
base_error_rate = 0.001 # Sequencing error rate

# Cell differentiation parameters
kappa_diff = 0.1  # Differentiation rate parameter
t0_diff = 50      # Differentiation inflection time

# Gene expression parameters
num_genes = 4000
cell_type_specific_genes = int(num_genes * 0.4)
shared_genes = num_genes - cell_type_specific_genes
alpha = 0.1                # Dispersion parameter for Negative Binomial
zero_inflation_prob = 0.1  # Probability of zero-inflation in gene expression

# Simulation settings
total_cells = 1000  # Total number of stem cells to simulate
max_generations = 10  # Max generations for stem cell divisions

# ------------------------------------------------------------------------------
# Class Definitions
# ------------------------------------------------------------------------------

class mtDNA:
    def __init__(self, id, mutations=None):
        self.id = id
        self.mutations = set(mutations) if mutations else set()

class Cell:
    def __init__(self, id, parent_id, generation, time, mtDNA_list=None, cell_type='StemCell'):
        self.id = id
        self.parent_id = parent_id
        self.generation = generation
        self.time = time
        self.mtDNA_list = mtDNA_list if mtDNA_list else []
        self.cell_type = cell_type
        self.children = []
        # To store DP and AD for mutations (filled after simulating read depth)
        self.mutation_profile = {}

# ------------------------------------------------------------------------------
# mtDNA Dynamics
# ------------------------------------------------------------------------------

def introduce_mutations(mtDNA_list, mutation_rate):
    """
    Introduce mutations into mtDNA molecules.
    """
    for mt in mtDNA_list:
        num_mutations = np.random.poisson(mutation_rate)
        for _ in range(num_mutations):
            mutation = f"m{np.random.randint(1e6)}"
            mt.mutations.add(mutation)
    return mtDNA_list

def replicate_and_segregate_mtDNA(mtDNA_list):
    """
    Replicate and segregate mtDNA during cell division (default, no bottleneck).
    """
    # Replicate mtDNA
    replicated_mtDNA = []
    for mt in mtDNA_list:
        replicated_mtDNA.append(mt)
        replicated_mtDNA.append(mtDNA(mt.id, mt.mutations.copy()))
    # Introduce mutations
    if len(replicated_mtDNA) > 0:
        mutation_rate = mtDNA_mutation_rate_per_mitosis / len(replicated_mtDNA)
        replicated_mtDNA = introduce_mutations(replicated_mtDNA, mutation_rate)
    
    # Segregate mtDNA to daughter cells
    total_mtDNA = len(replicated_mtDNA)
    if total_mtDNA > 0:
        segregation_counts = np.random.multinomial(total_mtDNA, [0.5, 0.5])
        indices = np.random.permutation(total_mtDNA)
        daughter1_indices = indices[:segregation_counts[0]]
        daughter2_indices = indices[segregation_counts[0]:]
        daughter1_mtDNA = [replicated_mtDNA[i] for i in daughter1_indices]
        daughter2_mtDNA = [replicated_mtDNA[i] for i in daughter2_indices]
    else:
        # Edge case: if no mtDNA present (unlikely)
        daughter1_mtDNA = []
        daughter2_mtDNA = []
    
    return [daughter1_mtDNA, daughter2_mtDNA]

# ------------------------------------------------------------------------------
# Stem Cell Growth and Differentiation
# ------------------------------------------------------------------------------

def division_rate(t, r, kappa, t0):
    """
    Calculate cell division rate at time t.
    """
    return r * (1 - 1 / (1 + np.exp(-kappa * (t - t0))))

def differentiation_probability(t, kappa_diff, t0_diff):
    """
    Calculate cell differentiation probability at time t.
    """
    return 1 - (1 / (1 + np.exp(-kappa_diff * (t - t0_diff))))

def simulate_stem_cell_growth():
    """
    Simulate stem cell growth using a Gillespie-like approach 
    until total_cells is reached.
    """
    cells = []
    cell_id_counter = 0
    time = 0.0
    cell_queue = deque()

    # Initialize the first cell
    mtDNA_population = [mtDNA(i) for i in range(mtDNA_initial_count)]
    initial_cell = Cell(cell_id_counter, None, 0, time, mtDNA_population.copy(), 'StemCell')
    cells.append(initial_cell)
    cell_queue.append(initial_cell)
    cell_id_counter += 1

    while len(cells) < total_cells:
        current_cell = cell_queue.popleft()
        # Calculate division time
        div_rate = division_rate(current_cell.time, r, kappa, t0)
        if div_rate <= 0:
            # No more divisions
            break
        division_t = np.random.exponential(scale=1.0 / div_rate)
        time += division_t

        # Replicate and segregate mtDNA
        daughter_mtDNAs = replicate_and_segregate_mtDNA(current_cell.mtDNA_list)

        # Create daughter cells
        for i in range(2):
            new_cell = Cell(
                cell_id_counter,
                current_cell.id,
                current_cell.generation + 1,
                time,
                daughter_mtDNAs[i],
                'StemCell'
            )
            cells.append(new_cell)
            cell_queue.append(new_cell)
            current_cell.children.append(new_cell.id)
            cell_id_counter += 1

    return cells

def progenitor_asymmetric_division(cell):
    """
    Default: linear differentiation — one daughter remains progenitor, 
    other becomes 'DifferentiatedType1'.
    """
    daughter_mtDNAs = replicate_and_segregate_mtDNA(cell.mtDNA_list)
    progenitor_daughter = Cell(
        cell.id * 2,
        cell.id,
        cell.generation + 1,
        cell.time + 1,
        daughter_mtDNAs[0],
        'Progenitor'
    )
    differentiated_daughter = Cell(
        cell.id * 2 + 1,
        cell.id,
        cell.generation + 1,
        cell.time + 1,
        daughter_mtDNAs[1],
        'DifferentiatedType1'
    )
    return [progenitor_daughter, differentiated_daughter]

def simulate_cell_differentiation(cells):
    """
    Simulate cell differentiation (linear model).
    Stem cells become progenitor with probability, 
    progenitor cells divide asymmetrically.
    """
    differentiated_cells = []
    
    for cell in cells:
        if cell.cell_type == 'StemCell':
            # Decide whether the stem cell differentiates
            diff_prob = differentiation_probability(cell.time, kappa_diff, t0_diff)
            if np.random.rand() < diff_prob:
                cell.cell_type = 'Progenitor'
        elif cell.cell_type == 'Progenitor':
            # Asymmetric division
            daughters = progenitor_asymmetric_division(cell)
            differentiated_cells.extend(daughters)

    all_cells = cells + differentiated_cells
    return all_cells

# ------------------------------------------------------------------------------
# Sequencing Read Depth (DP and AD)
# ------------------------------------------------------------------------------

def simulate_read_depth(cell, all_mutations, mean_dp=mean_dp, base_error_rate=base_error_rate):
    """
    Simulate DP and AD for mitochondrial mutations in a cell.
    """
    mtDNA_count = len(cell.mtDNA_list)
    mutation_counts = {}

    # Count how many mtDNA molecules have each mutation
    for mt in cell.mtDNA_list:
        for mutation in mt.mutations:
            mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1

    # Calculate heteroplasmy
    heteroplasmy = {}
    for mutation, count in mutation_counts.items():
        heteroplasmy[mutation] = count / mtDNA_count if mtDNA_count > 0 else 0.0

    cell.mutation_profile = {}

    # Simulate DP and AD for each mutation
    for mutation in all_mutations:
        # Negative Binomial for total coverage
        dp_nb = np.random.negative_binomial(
            n=1/dp_dispersion,
            p=1/(1 + mean_dp*dp_dispersion)
        )
        # Shift to center around mean_dp
        dp_sim = max(int(dp_nb + mean_dp), 1)

        # Effective heteroplasmy includes error
        h = heteroplasmy.get(mutation, 0.0)
        effective_h = h*(1-base_error_rate) + (1-h)*base_error_rate
        
        # Binomial sampling for AD
        ad_sim = np.random.binomial(dp_sim, effective_h)

        cell.mutation_profile[mutation] = {
            'DP': dp_sim,
            'AD': ad_sim
        }

# ------------------------------------------------------------------------------
# Single-Cell Transcriptomic Simulation
# ------------------------------------------------------------------------------

def generate_gene_params():
    """
    Generate parameters for gene expression simulation.
    """
    gene_ids = [f'Gene_{i}' for i in range(num_genes)]
    specific_gene_ids = random.sample(gene_ids, cell_type_specific_genes)
    
    gene_params = {}
    for gene_id in gene_ids:
        if gene_id in specific_gene_ids:
            gene_params[gene_id] = {
                'base_expression': np.random.uniform(1, 5),
                'expression_rate': np.random.uniform(0.1, 0.5),
                'sigma': 1.0
            }
        else:
            # Shared genes
            gene_params[gene_id] = {
                'base_expression': np.random.uniform(1, 5),
                'expression_rate': 0.0,
                'sigma': 1.0
            }
    return gene_params

def generate_latent_expression(cell, gene_params):
    """
    Generate latent expression values for each gene in a cell.
    """
    z_t = {}
    for gene_id, params in gene_params.items():
        loc = params['base_expression'] + params['expression_rate'] * cell.generation
        scale = params['sigma']
        z_t[gene_id] = np.random.normal(loc=loc, scale=scale)
    return z_t

def sample_expression(z_t, alpha, zero_inflation_prob):
    """
    Sample gene expression counts from a Zero-Inflated Negative Binomial (ZINB).
    """
    x_t = {}
    for gene_id, z in z_t.items():
        mu = np.exp(z)
        r = 1 / alpha
        p = r / (r + mu)
        if np.random.rand() < zero_inflation_prob:
            x_t[gene_id] = 0
        else:
            x_t[gene_id] = np.random.negative_binomial(n=r, p=p)
    return x_t

def simulate_gene_expression_for_cells(cells, gene_params):
    """
    Simulate gene expression for a list of cells.
    """
    expression_data = {}
    for cell in cells:
        z_t = generate_latent_expression(cell, gene_params)
        x_t = sample_expression(z_t, alpha, zero_inflation_prob)
        expression_data[cell.id] = x_t

    expression_df = pd.DataFrame.from_dict(expression_data, orient='index')
    return expression_df

# ------------------------------------------------------------------------------
# Main Functions to Run a Basic Simulation
# ------------------------------------------------------------------------------

def run_basic_simulation():
    """
    Runs a default simulation:
    1. Stem cell growth
    2. Cell differentiation (linear)
    3. Collecting mutations & simulating DP/AD
    4. Simulating gene expression
    """
    # Step 1: Stem cell growth
    stem_cells = simulate_stem_cell_growth()
    
    # Step 2: Differentiation
    all_cells = simulate_cell_differentiation(stem_cells)
    
    # Step 3: Collect all unique mutations
    all_mutations = set()
    for cell in all_cells:
        for mt in cell.mtDNA_list:
            all_mutations.update(mt.mutations)
    all_mutations = sorted(all_mutations)
    
    # Simulate sequencing read depth
    for cell in all_cells:
        simulate_read_depth(cell, all_mutations, mean_dp, base_error_rate)

    # Step 4: Gene expression
    gene_params = generate_gene_params()
    expression_df = simulate_gene_expression_for_cells(all_cells, gene_params)

    return all_cells, all_mutations, expression_df

if __name__ == "__main__":
    # Example: Run the basic simulation if called directly
    cells, mutations, expr_df = run_basic_simulation()
    print(f"Simulation complete. Number of cells: {len(cells)}")
    print(f"Number of unique mutations: {len(mutations)}")
    print("First 5 rows of gene expression data:")
    print(expr_df.head())
