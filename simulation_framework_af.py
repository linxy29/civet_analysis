"""
simulation_framework_optimized.py

Optimized simulation framework for:
1. Stem cell growth simulation (Gillespie-like)
2. Mitochondrial genome proliferation
3. Cell differentiation (basic linear model)
4. Single-cell transcriptomic data simulation
5. Sequencing read depth (DP) and allele depth (AD) for mtDNA mutations
"""

import numpy as np
import pandas as pd
import random
from collections import deque
from scipy.sparse import csc_matrix
from scipy.io import mmwrite
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import os
import scipy.io
import scipy.sparse

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------------------------
# Simulation Parameters (Global Defaults)
# ------------------------------------------------------------------------------
# You can keep these in a config file or pass them into functions as needed.

# Stem cell growth parameters
R = 1.0       # Maximum division rate
KAPPA_GROW = 0.1    # Growth acceleration rate
T0_GROW = 50        # Inflection point in time

# mtDNA parameters
MTDNA_INITIAL_COUNT = 200
MTDNA_MUTATION_RATE_PER_MITOSIS = 4   # Mutations per cell division

# Sequencing read depth parameters
MEAN_DP = 100            # Mean total read depth per mutation site
DP_DISPERSION = 0.2      # Dispersion parameter for total read depth (overdispersion)
BASE_ERROR_RATE = 0.001  # Sequencing error rate

# Cell differentiation parameters
KAPPA_DIFF = 0.1  # Differentiation rate parameter
T0_DIFF = 50      # Differentiation inflection time

# Gene expression parameters
NUM_GENES = 6000
CELL_TYPE_SPECIFIC_FRAC = 0.4
CELL_TYPE_SPECIFIC_GENES = int(NUM_GENES * CELL_TYPE_SPECIFIC_FRAC)
SHARED_GENES = NUM_GENES - CELL_TYPE_SPECIFIC_GENES
ALPHA = 0.1                # Dispersion parameter for Negative Binomial
ZERO_INFLATION_PROB = 0.1  # Probability of zero-inflation in gene expression

# Simulation settings
TOTAL_CELLS = 1000    # Total number of stem cells to simulate
MAX_GENERATIONS = 100  # Max generations for stem cell divisions

# ------------------------------------------------------------------------------
# Class Definitions
# ------------------------------------------------------------------------------

class Cell:
    """
    Represents a cell in the simulation.
    """
    __slots__ = [
        'id', 'parent_id', 'generation', 'time',
        'mutation_afs', 'cell_type', 'children', 'mutation_profile'
    ]
    
    def __init__(self, cid, parent_id, generation, time_point, mutation_afs=None, cell_type='StemCell'):
        self.id = f"cell_{cid}"
        self.parent_id = None if parent_id is None else f"cell_{parent_id}"
        self.generation = generation
        self.time = time_point
        self.mutation_afs = mutation_afs if mutation_afs else {}  # Dictionary of mutation -> AF
        self.cell_type = cell_type
        self.children = []
        self.mutation_profile = {}  # For storing DP & AD results

# ------------------------------------------------------------------------------
# mtDNA Dynamics
# ------------------------------------------------------------------------------

def introduce_new_mutations(mutation_rate):
    """
    Generate new mutations with initial low AF.
    Ensures AFs stay between 0.0001 and 0.01
    
    Parameters
    ----------
    mutation_rate : float
        Average number of new mutations to introduce
    
    Returns
    -------
    dict
        Dictionary of new mutation IDs -> initial AFs
    """
    num_mutations = np.random.poisson(mutation_rate)
    new_mutations = {}
    
    if num_mutations > 0:
        # Generate random positions between 1 and 16569
        positions = np.random.randint(1, 16570, size=num_mutations)
        # Initialize new mutations with low AF (between 0.01% and 1%)
        for pos in positions:
            new_mutations[f"m{pos}"] = np.random.uniform(0.0001, 0.01)
    
    return new_mutations

def segregate_allele_frequencies(afs):
    """
    Simulate segregation of allele frequencies during cell division.
    
    Parameters
    ----------
    afs : dict
        Dictionary of mutation IDs -> allele frequencies
    
    Returns
    -------
    tuple
        Two dictionaries representing AFs in daughter cells
    """
    daughter1_afs = {}
    daughter2_afs = {}
    
    for mut, af in afs.items():
        if af > 0:
            # Model segregation as beta distribution around parent AF
            # Higher AF -> lower variance, lower AF -> higher variance
            alpha = af * (1/af - 1)  # Shape parameter increases with AF
            beta = 1 - af            # Other shape parameter
            
            # Sample new AFs for daughters
            d1_af = np.clip(
                np.random.beta(max(alpha, 0.01), max(beta, 0.01)),
                0.0001, 1.0  # Clip to ensure valid range
            )
            d2_af = np.clip(
                np.random.beta(max(alpha, 0.01), max(beta, 0.01)),
                0.0001, 1.0  # Clip to ensure valid range
            )
            
            # Store if AF is above detection threshold
            if d1_af >= 0.0001:
                daughter1_afs[mut] = d1_af
            if d2_af >= 0.0001:
                daughter2_afs[mut] = d2_af
    
    return daughter1_afs, daughter2_afs

def initialize_baseline_mutations(num_baseline_mutations=5):
    """
    Initialize baseline mutations with random allele frequencies.
    Ensures AFs stay between 0.001 and 0.5
    
    Parameters
    ----------
    num_baseline_mutations : int
        Number of baseline mutations to introduce
    
    Returns
    -------
    dict
        Dictionary of baseline mutation IDs -> initial AFs
    """
    baseline_mutations = {}
    
    # Generate baseline mutations
    positions = np.random.randint(1, 16570, size=num_baseline_mutations)
    for pos in positions:
        mutation_id = f"baseline_m{pos}"
        # Initialize with AF between 0.1% and 50%
        baseline_mutations[mutation_id] = np.random.uniform(0.001, 0.5)
    
    return baseline_mutations

def segregate_allele_frequencies_biased(parent_afs, bias=0.7):
    """
    Segregate allele frequencies with bias.
    
    Parameters
    ----------
    parent_afs : dict
        Dictionary of mutation IDs to allele frequencies
    bias : float
        Bias factor (0.5 = symmetric, >0.5 = biased toward daughter1)
        
    Returns
    -------
    tuple
        (daughter1_afs, daughter2_afs)
    """
    daughter1_afs = {}
    daughter2_afs = {}
    
    for mutation, af in parent_afs.items():
        # Apply bias to segregation
        if np.random.random() < bias:
            # Daughter 1 gets higher AF
            d1_factor = np.random.uniform(1.0, 1.5)
            d2_factor = 2.0 - d1_factor
        else:
            # Daughter 2 gets higher AF
            d2_factor = np.random.uniform(1.0, 1.5)
            d1_factor = 2.0 - d2_factor
            
        # Calculate new AFs, ensuring they stay in [0,1]
        d1_af = np.clip(af * d1_factor, 0.0, 1.0)
        d2_af = np.clip(af * d2_factor, 0.0, 1.0)
        
        # Add to dictionaries if AF > 0
        if d1_af > 0:
            daughter1_afs[mutation] = d1_af
        if d2_af > 0:
            daughter2_afs[mutation] = d2_af
    
    return daughter1_afs, daughter2_afs

def simulate_stem_cell_growth_biased(
    total_cells=1000,
    mutation_rate=MTDNA_MUTATION_RATE_PER_MITOSIS,
    num_baseline_mutations=5,
    bias=0.7,
    r=R,                # Default parameter for division rate
    kappa=KAPPA_GROW,   # Default parameter for division rate
    t0=T0_GROW          # Default parameter for division rate
):
    """
    Simulate stem cell growth with biased segregation.
    
    Parameters
    ----------
    total_cells : int
        Number of cells to simulate
    mutation_rate : float
        Mutation rate per mitosis
    num_baseline_mutations : int
        Number of baseline mutations
    bias : float
        Bias factor for segregation (>0.5 means daughter1 gets higher AFs)
    r : float
        Maximum division rate
    kappa : float
        Growth acceleration rate
    t0 : float
        Inflection point in time
    
    Returns
    -------
    tuple
        (cells, mutations)
    """
    cells = []
    cell_queue = []
    cid_counter = 0
    current_time = 0.0

    # Initialize first cell with baseline mutations
    first_cell = Cell(
        cid_counter,
        None,
        0,
        current_time,
        initialize_baseline_mutations(num_baseline_mutations)
    )
    first_cell.cell_type = 'stem'  # Set cell type
    cells.append(first_cell)
    cell_queue.append(first_cell)
    cid_counter += 1

    while len(cells) < total_cells and cell_queue:
        parent_cell = cell_queue.pop(0)
        
        # Division rate - pass all required parameters
        div_rate = division_rate(parent_cell.time, r, kappa, t0)
        if div_rate <= 0:
            continue
        
        # Calculate wait time and update current time
        wait_time = np.random.exponential(1.0 / div_rate)
        current_time = parent_cell.time + wait_time

        # Introduce new mutations
        new_mutations = introduce_new_mutations(mutation_rate)
        parent_afs = {**parent_cell.mutation_afs, **new_mutations}
        
        # Segregate with bias
        d1_afs, d2_afs = segregate_allele_frequencies_biased(parent_afs, bias)

        # Create daughter cells
        for d_afs in [d1_afs, d2_afs]:
            new_cell = Cell(
                cid_counter,
                parent_cell.id,
                parent_cell.generation + 1,
                current_time,
                d_afs
            )
            new_cell.cell_type = parent_cell.cell_type  # Inherit cell type
            cells.append(new_cell)
            parent_cell.children.append(new_cell.id)
            cell_queue.append(new_cell)
            cid_counter += 1

            if len(cells) >= total_cells:
                break

    # Collect all mutations
    all_mutations = set()
    for cell in cells:
        all_mutations.update(cell.mutation_afs.keys())
    
    return cells, sorted(all_mutations)

# ------------------------------------------------------------------------------
# Stem Cell Growth and Differentiation
# ------------------------------------------------------------------------------

def division_rate(t, r, kappa, t0):
    """
    Calculates the division rate at time t for a logistic growth curve.
    """
    return r * (1.0 - 1.0 / (1.0 + np.exp(-kappa * (t - t0))))

def differentiation_probability(t, kappa_diff, t0_diff):
    """
    Probability of a stem cell differentiating into a progenitor at time t.
    """
    return 1.0 - (1.0 / (1.0 + np.exp(-kappa_diff * (t - t0_diff))))

def simulate_cell_differentiation(cells, kappa_diff=KAPPA_DIFF, t0_diff=T0_DIFF):
    """
    Enhanced differentiation with multiple progenitor types:
    - Stem cells become either Progenitor1 or Progenitor2 cells with probability diff_prob
    - Progenitor cells do one asymmetric division
    - Asymmetric division produces different cell types based on parent type
    
    Returns an updated list of all cells after differentiation.
    """
    new_cells = []
    for c in cells:
        if c.cell_type == 'StemCell':
            diff_prob = differentiation_probability(c.time, kappa_diff, t0_diff)
            if np.random.rand() < diff_prob:
                # Randomly choose between Progenitor1 and Progenitor2
                c.cell_type = 'Progenitor1' if np.random.rand() < 0.5 else 'Progenitor2'
        
        elif c.cell_type == 'Progenitor1' or c.cell_type == 'Progenitor2':
            # Asymmetric division with type-specific daughters
            daughters = progenitor_asymmetric_division(c)
            new_cells.extend(daughters)

    # Combine original cells + newly created daughters
    all_cells = cells + new_cells
    return all_cells

def progenitor_asymmetric_division(parent_cell):
    """
    Asymmetric division of progenitor cells.
    - One daughter remains the same progenitor type
    - The other differentiates to a terminal cell with a type based on parent
    """
    # Create two daughter cells
    next_id = int(parent_cell.id.replace('cell_', '')) + 1000  # Ensure unique IDs
    
    # First daughter: same type as parent (self-renewal)
    prog_daughter = Cell(
        cid=next_id,
        parent_id=parent_cell.id.replace('cell_', ''),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time,
        cell_type=parent_cell.cell_type  # Keep same progenitor type
    )
    
    # Second daughter: differentiated cell with type based on parent
    diff_daughter = Cell(
        cid=next_id + 1,
        parent_id=parent_cell.id.replace('cell_', ''),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time
    )
    
    # Set terminal cell type based on parent progenitor type
    if parent_cell.cell_type == 'Progenitor1':
        diff_daughter.cell_type = 'TerminalCell_A'
    else:  # Progenitor2
        diff_daughter.cell_type = 'TerminalCell_B'
    
    # Replicate and segregate mtDNA to daughters
    d1_afs, d2_afs = segregate_allele_frequencies(parent_cell.mutation_afs)
    prog_daughter.mutation_afs = d1_afs
    diff_daughter.mutation_afs = d2_afs
    
    # Add daughters to parent's children list
    parent_cell.children.extend([prog_daughter.id, diff_daughter.id])
    
    return [prog_daughter, diff_daughter]

# ------------------------------------------------------------------------------
# Sequencing Read Depth (DP & AD)
# ------------------------------------------------------------------------------

def simulate_read_depth(
    cell,
    all_mutations,
    mean_dp=MEAN_DP,
    dp_dispersion=DP_DISPERSION,
    base_error_rate=BASE_ERROR_RATE,
    false_mutation_rate=BASE_ERROR_RATE  # Rate at which sequencing errors create false mutations
):
    """
    Simulate sequencing read depth with potential false mutations from sequencing errors.
    
    Parameters
    ----------
    cell : Cell object
        Cell to simulate read depth for
    all_mutations : list
        List of true mutation IDs
    mean_dp : float
        Mean depth of coverage
    dp_dispersion : float
        Dispersion parameter for negative binomial
    base_error_rate : float
        Base sequencing error rate
    false_mutation_rate : float
        Rate at which sequencing errors create false mutations
    """
    # Generate read depth using negative binomial
    r_param = 1.0 / dp_dispersion
    p_param = r_param / (r_param + mean_dp)
    dp_values = np.random.negative_binomial(n=r_param, p=p_param, size=len(all_mutations))
    
    cell.mutation_profile = {}
    
    # Process true mutations
    for i, mutation in enumerate(all_mutations):
        dp_sim = max(1, int(dp_values[i] + mean_dp))
        
        # Get true AF or use 0 if mutation not present
        true_af = np.clip(cell.mutation_afs.get(mutation, 0.0), 0.0, 1.0)
        
        # Add sequencing error
        effective_af = true_af * (1 - base_error_rate) + (1 - true_af) * base_error_rate
        effective_af = np.clip(effective_af, 0.0, 1.0)
        
        # Simulate read counts
        ad_sim = np.random.binomial(dp_sim, effective_af)
        
        cell.mutation_profile[mutation] = {
            'DP': dp_sim,
            'AD': ad_sim,
            'is_true_mutation': True
        }
    
    # Add false mutations from sequencing errors
    num_false_mutations = np.random.poisson(false_mutation_rate * len(all_mutations))
    if num_false_mutations > 0:
        # Generate positions for false mutations (avoiding true mutation positions)
        true_positions = set(int(mut.replace('m', '')) for mut in all_mutations if mut.startswith('m'))
        false_positions = set()
        while len(false_positions) < num_false_mutations:
            pos = np.random.randint(1, 16570)
            if pos not in true_positions:
                false_positions.add(pos)
        
        # Add false mutations
        for pos in false_positions:
            mutation_id = f"false_m{pos}"
            dp_sim = max(1, int(np.random.negative_binomial(r_param, p_param) + mean_dp))
            
            # False mutations typically have low AFs
            false_af = np.random.beta(1, 20)  # This will generate mostly low AFs
            ad_sim = np.random.binomial(dp_sim, false_af)
            
            cell.mutation_profile[mutation_id] = {
                'DP': dp_sim,
                'AD': ad_sim,
                'is_true_mutation': False
            }

def save_mutation_info(cells, true_mutations, output_dir, prefix):
    """
    Save detailed mutation information including true and false mutations.
    
    Parameters
    ----------
    cells : list
        List of Cell objects
    true_mutations : list
        List of true mutation IDs
    output_dir : str
        Directory to save results
    prefix : str
        Prefix for output files
    """
    # Collect all mutations (true and false)
    all_mutations = set()
    mutation_data = []
    
    for cell in cells:
        for mut_id, profile in cell.mutation_profile.items():
            all_mutations.add(mut_id)
            
            # Calculate VAF
            vaf = profile['AD'] / profile['DP'] if profile['DP'] > 0 else 0
            
            mutation_data.append({
                'cell_id': cell.id,
                'mutation_id': mut_id,
                'DP': profile['DP'],
                'AD': profile['AD'],
                'VAF': vaf,
                'is_true_mutation': profile['is_true_mutation']
            })
    
    # Create mutation summary
    mutation_summary = pd.DataFrame(mutation_data)
    
    # Save detailed mutation data
    mutation_file = os.path.join(output_dir, f"{prefix}_mutation_profiles.csv")
    mutation_summary.to_csv(mutation_file, index=False)
    
    # Create and save mutation statistics
    mutation_stats = pd.DataFrame([{
        'mutation_id': mut,
        'is_true_mutation': mut in true_mutations,
        'num_cells_present': sum(1 for cell in cells if mut in cell.mutation_profile),
        'mean_vaf': mutation_summary[mutation_summary['mutation_id'] == mut]['VAF'].mean(),
        'max_vaf': mutation_summary[mutation_summary['mutation_id'] == mut]['VAF'].max()
    } for mut in all_mutations])
    
    stats_file = os.path.join(output_dir, f"{prefix}_mutation_statistics.csv")
    mutation_stats.to_csv(stats_file, index=False)
    
    # Save summary of true vs false mutations
    with open(os.path.join(output_dir, f"{prefix}_mutation_summary.txt"), 'w') as f:
        f.write(f"Total mutations: {len(all_mutations)}\n")
        f.write(f"True mutations: {len(true_mutations)}\n")
        f.write(f"False mutations: {len(all_mutations) - len(true_mutations)}\n")
        f.write("\nMutation type distribution:\n")
        f.write(mutation_stats['is_true_mutation'].value_counts().to_string())

# ------------------------------------------------------------------------------
# Single-Cell Transcriptomic Simulation
# ------------------------------------------------------------------------------

## This part still needs to be improved:
## 1) terminal cell types are not included; 
def generate_gene_params(
    num_genes=NUM_GENES,
    cell_type_specific_genes=CELL_TYPE_SPECIFIC_GENES
):
    """
    Generate base expression parameters for cell-type-specific and shared genes.
    Creates distinct gene sets for each cell type, with:
    - Progenitors sharing genes with StemCell
    - Terminal cells sharing genes with their progenitors
    """
    gene_ids = [f"Gene_{i}" for i in range(num_genes)]
    
    # Calculate genes per type (for stem and progenitor cells)
    genes_per_type = cell_type_specific_genes // 5  # Divide among all 5 cell types
    
    # Randomly select genes for stem cells
    all_genes = set(gene_ids)
    stem_specific_genes = set(random.sample(list(all_genes), genes_per_type))
    all_genes -= stem_specific_genes
    
    # For progenitors, take half genes from stem cells and half new genes
    prog_genes_from_stem = genes_per_type // 2
    
    # Progenitor1: half from StemCell, half new
    prog1_from_stem = set(random.sample(list(stem_specific_genes), prog_genes_from_stem))
    prog1_new = set(random.sample(list(all_genes), prog_genes_from_stem))
    all_genes -= prog1_new
    prog1_genes = prog1_from_stem | prog1_new
    
    # Progenitor2: half from StemCell, half new
    prog2_from_stem = set(random.sample(list(stem_specific_genes - prog1_from_stem), prog_genes_from_stem))
    prog2_new = set(random.sample(list(all_genes), prog_genes_from_stem))
    all_genes -= prog2_new
    prog2_genes = prog2_from_stem | prog2_new
    
    # For terminal cells, take half genes from progenitors and half new genes
    terminal_genes_per_type = genes_per_type // 2
    
    # TerminalCell_A: half from Progenitor1, half new
    term_a_from_prog = set(random.sample(list(prog1_genes), terminal_genes_per_type))
    term_a_new = set(random.sample(list(all_genes), terminal_genes_per_type))
    all_genes -= term_a_new
    term_a_genes = term_a_from_prog | term_a_new
    
    # TerminalCell_B: half from Progenitor2, half new
    term_b_from_prog = set(random.sample(list(prog2_genes), terminal_genes_per_type))
    term_b_new = set(random.sample(list(all_genes), terminal_genes_per_type))
    all_genes -= term_b_new
    term_b_genes = term_b_from_prog | term_b_new
    
    # Create the final gene sets for each cell type
    cell_type_genes = {
        'StemCell': stem_specific_genes,
        'Progenitor1': prog1_genes,
        'Progenitor2': prog2_genes,
        'TerminalCell_A': term_a_genes,
        'TerminalCell_B': term_b_genes
    }
    
    # All cell-type specific genes combined
    all_specific_genes = set()
    for gene_set in cell_type_genes.values():
        all_specific_genes.update(gene_set)
    
    # Generate parameters for each gene
    gene_params = {}
    for gene_id in gene_ids:
        base_expr = np.random.uniform(1, 5)
        if gene_id in all_specific_genes:
            expr_rate = np.random.uniform(1, 5)
        else:
            expr_rate = 0.0
        
        # Store which cell types this gene is specific for
        specific_for = [ct for ct, genes in cell_type_genes.items() if gene_id in genes]
        
        # Store additional metadata about gene inheritance
        inherited_by = []
        # Stem → Progenitor inheritance
        if gene_id in stem_specific_genes:
            if gene_id in prog1_from_stem:
                inherited_by.append(('StemCell', 'Progenitor1'))
            if gene_id in prog2_from_stem:
                inherited_by.append(('StemCell', 'Progenitor2'))
        
        # Progenitor → Terminal inheritance
        if gene_id in prog1_genes and gene_id in term_a_from_prog:
            inherited_by.append(('Progenitor1', 'TerminalCell_A'))
        if gene_id in prog2_genes and gene_id in term_b_from_prog:
            inherited_by.append(('Progenitor2', 'TerminalCell_B'))
        
        gene_params[gene_id] = {
            'base_expression': base_expr,
            'expression_rate': expr_rate,
            'sigma': 1.0,
            'cell_type_specific': gene_id in all_specific_genes,
            'specific_for': specific_for,
            'inherited_by': inherited_by
        }
    
    return gene_params, cell_type_genes

def generate_latent_expression(cell, gene_params):
    """
    Generate latent expression values for each gene in a cell.
    Expression depends on cell type-specific genes and generation.
    """
    z_t = {}
    gen = cell.generation
    cell_type = cell.cell_type
    
    for gene_id, params in gene_params.items():
        # Base expression
        base_expr = params['base_expression']
        expr_rate = params['expression_rate']
        
        # Initialize modified rate
        modified_rate = 0
        
        # Check if this is a cell type-specific gene
        if params['cell_type_specific']:
            # Direct cell type-specific expression
            if cell_type in params['specific_for']:
                modified_rate = expr_rate
            
            # Check for inherited expression in terminal cells
            for parent_type, terminal_type in params['inherited_by']:
                if cell_type == terminal_type and parent_type in params['specific_for']:
                    modified_rate = expr_rate
        
        # Calculate final expression level
        if modified_rate > 0:
            # Only apply generation effect to expressed genes
            loc = base_expr + modified_rate * (1 + gen * 0.05)
        else:
            # Base expression only for non-specific genes
            loc = base_expr
            
        scale = params['sigma']
        z_t[gene_id] = np.random.normal(loc=loc, scale=scale)
    
    return z_t

def sample_expression(z_t, alpha=ALPHA, zero_inflation_prob=ZERO_INFLATION_PROB):
    """
    Sample gene expression from a Zero-Inflated Negative Binomial (ZINB).
    """
    x_t = {}
    r = 1.0 / alpha  # NB parameter
    for gene_id, z in z_t.items():
        mu = np.exp(z)
        p = r / (r + mu)  # success probability in NB
        if np.random.rand() < zero_inflation_prob:
            count = 0
        else:
            count = np.random.negative_binomial(r, p)
        x_t[gene_id] = count
    return x_t

def simulate_gene_expression_for_cells(cells, gene_params):
    """
    Generate transcriptomic data for each cell.
    """
    # We collect expression in a dictionary-of-dictionaries first
    expr_data = {}
    for c in cells:
        z_vals = generate_latent_expression(c, gene_params)
        x_vals = sample_expression(z_vals)
        expr_data[c.id] = x_vals

    expression_df = pd.DataFrame.from_dict(expr_data, orient='index')
    return expression_df

# ------------------------------------------------------------------------------
# Main Function to Run the Simulation
# ------------------------------------------------------------------------------

def run_basic_simulation(
    total_cells=TOTAL_CELLS,
    max_generations=MAX_GENERATIONS,
    mutation_rate=MTDNA_MUTATION_RATE_PER_MITOSIS,
    num_baseline_mutations=5
):
    """
    Run a basic simulation of stem cell growth, differentiation, and gene expression.
    
    Parameters
    ----------
    total_cells : int
        Total number of cells to simulate
    max_generations : int
        Maximum number of generations for stem cell divisions
    mutation_rate : float
        Mutation rate per mitosis
    num_baseline_mutations : int
        Number of baseline mutations to introduce with random AFs
        
    Returns
    -------
    tuple
        (cells, mutations, expr_df)
    """
    # 1. Simulate stem cell growth
    cells, mutations = simulate_stem_cell_growth_biased(
        total_cells=total_cells,
        mutation_rate=mutation_rate,
        num_baseline_mutations=num_baseline_mutations
    )
    
    # 2. Simulate cell differentiation
    cells = simulate_cell_differentiation(cells)
    
    # 3. Simulate read depth for each mutation in each cell
    for cell in cells:
        simulate_read_depth(
            cell, 
            all_mutations=mutations,
            mean_dp=MEAN_DP,
            dp_dispersion=DP_DISPERSION,
            base_error_rate=BASE_ERROR_RATE
        )
    
    # 4. Generate gene expression parameters
    gene_params, cell_type_genes = generate_gene_params(
        num_genes=NUM_GENES,
        cell_type_specific_genes=CELL_TYPE_SPECIFIC_GENES
    )
    
    # 5. Simulate gene expression for all cells
    expr_df = simulate_gene_expression_for_cells(cells, gene_params)
    
    return cells, mutations, expr_df

# ------------------------------------------------------------------------------
# Function to save the simulation results
# ------------------------------------------------------------------------------

def get_dp_ad_matrices(cells, all_mutations):
    """
    Collect DP and AD data into CSC (Compressed Sparse Column) matrices.
    
    Parameters
    ----------
    cells : list of Cell objects
        Each cell has 'mutation_profile' dict with DP/AD values
    all_mutations : list
        Sorted list of mutation identifiers
    
    Returns
    -------
    dict
        Contains:
        - 'dp_matrix': scipy.sparse.csc_matrix for DP values
        - 'ad_matrix': scipy.sparse.csc_matrix for AD values
    """
    # Number cells from 0..N-1, number mutations from 0..M-1
    num_cells = len(cells)
    num_mutations = len(all_mutations)
    
    # Collect all data first
    rows = []
    cols = []
    dp_data = []
    ad_data = []
    
    for row_idx, cell in enumerate(cells):
        for col_idx, mutation in enumerate(all_mutations):
            profile = cell.mutation_profile.get(mutation, {})
            dp_val = profile.get('DP', 0)
            ad_val = profile.get('AD', 0)
            
            if dp_val > 0:
                rows.append(row_idx)
                cols.append(col_idx)
                dp_data.append(dp_val)
                ad_data.append(ad_val)  # Add AD data at the same time
    
    # Create matrices using the same row/col indices
    dp_matrix = csc_matrix((dp_data, (rows, cols)), 
                          shape=(num_cells, num_mutations))
    ad_matrix = csc_matrix((ad_data, (rows, cols)), 
                          shape=(num_cells, num_mutations))

    return {
        'dp_matrix': dp_matrix,
        'ad_matrix': ad_matrix
    }

def write_mtx_file(matrix, output_file):
    """
    Write CSC matrix in Matrix Market format (.mtx)
    
    Parameters
    ----------
    matrix : scipy.sparse.csc_matrix
        Matrix to write
    output_file : str
        Path to output file
    """
    mmwrite(output_file, matrix)

def export_mtx_for_dp_ad(cells, all_mutations, prefix="cellSNP.tag"):
    """
    Export DP and AD data to Matrix Market format (.mtx).
    
    Parameters
    ----------
    cells : list of Cell objects
        Each cell has 'mutation_profile' dict mapping mutation -> {'DP': dp_val, 'AD': ad_val}
    all_mutations : list
        Sorted list of mutation identifiers
    prefix : str
        Filename prefix for output files
    """
    # Get the CSC matrices
    matrices = get_dp_ad_matrices(cells, all_mutations)
    
    # Construct output paths
    dp_outfile = f"{prefix}.DP.mtx"
    ad_outfile = f"{prefix}.AD.mtx"
    
    # Write the files
    write_mtx_file(matrices['dp_matrix'], dp_outfile)
    write_mtx_file(matrices['ad_matrix'], ad_outfile)
    
    print(f"[export_mtx_for_dp_ad] Wrote DP to '{dp_outfile}' and AD to '{ad_outfile}'")
    
    return matrices

# ------------------------------------------------------------------------------
# Function to visualize the simulation results
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns

def visualize_mito_mutations(cells, mutations, output_file):
    """
    Visualize mitochondrial mutations as a heatmap.
    
    Parameters
    ----------
    cells : list of Cell objects
        Each cell has 'mutation_profile' containing {'DP': val, 'AD': val} for each mutation.
    mutations : list of str
        Sorted list of mutation identifiers.
    output_file : str
        Path to save the heatmap plot.
    """
    sorted_cells = sorted(cells, key=lambda c: c.id)
    cell_ids = [c.id for c in sorted_cells]
    af_matrix = np.zeros((len(sorted_cells), len(mutations)), dtype=float)

    for i, cell in enumerate(sorted_cells):
        for j, mut in enumerate(mutations):
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af_matrix[i, j] = ad / dp
            else:
                af_matrix[i, j] = 0.0

    af_df = pd.DataFrame(af_matrix, index=cell_ids, columns=mutations)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        af_df,
        cmap="viridis",
        cbar_kws={'label': 'Allele Frequency (AD/DP)'}
    )
    plt.title("Heatmap of Mitochondrial Mutation Allele Frequencies")
    plt.xlabel("Mutations")
    plt.ylabel("Cells")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mitochondrial mutation heatmap saved as '{output_file}'")

def visualize_gene_expression(cells, expr_df, output_prefix, color_by='cell_type'):
    """
    Visualize gene expression using scanpy for dimensionality reduction and plotting.
    
    Parameters
    ----------
    cells : list of Cell objects
        List of cells with attributes to color by.
    expr_df : pd.DataFrame
        Single-cell expression matrix (cells x genes). Index = cell IDs, columns = gene IDs.
    output_dir : str
        Path to save the output files. Will be used as base name for PCA and UMAP plots.
    color_by : str
        Cell attribute to color UMAP scatter. Typically "cell_type" or "generation".
    
    Returns
    -------
    AnnData
        The AnnData object with computed PCA and UMAP embeddings.
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import os
    
    # Create output filenames for PCA and UMAP plots
    pca_output = f"{output_prefix}_pca.png"
    umap_output = f"{output_prefix}_umap.png"
    
    # Convert cell IDs to strings in expr_df
    expr_df.index = expr_df.index.astype(str)
    
    # Filter out cells with zero total counts
    cell_sums = expr_df.sum(axis=1)
    expr_df = expr_df.loc[cell_sums > 0]
    
    # Filter out genes with zero expression across all cells
    gene_sums = expr_df.sum(axis=0)
    expr_df = expr_df.loc[:, gene_sums > 0]
    
    # If we have no data left after filtering, create a simple plot
    if expr_df.shape[0] == 0 or expr_df.shape[1] == 0:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No expression data available after filtering", 
                 ha='center', va='center', fontsize=14)
        plt.title("Gene Expression Visualization")
        plt.savefig(pca_output)  # Save as PCA output
        plt.close()
        print(f"Warning: No valid expression data. Empty plot saved to '{pca_output}'")
        return None
    
    # Create AnnData object
    adata = sc.AnnData(X=expr_df.values)
    adata.obs_names = expr_df.index
    adata.var_names = expr_df.columns
    
    # Add cell attributes to obs
    cell_dict = {cell.id: cell for cell in cells}
    
    # Safely get cell attributes with fallbacks
    def safe_get_cell_attr(cell_id, attr):
        cell = cell_dict.get(cell_id)
        if cell is None:
            cell = cell_dict.get(cell_id.replace('cell_', ''))
        if cell is None:
            return 'Unknown' if attr == 'cell_type' else 0
        return getattr(cell, attr)
    
    adata.obs['cell_type'] = [safe_get_cell_attr(cid, 'cell_type') for cid in adata.obs_names]
    adata.obs['generation'] = [safe_get_cell_attr(cid, 'generation') for cid in adata.obs_names]
    
    # Preprocess the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Try to find highly variable genes, but handle errors gracefully
    try:
        # For small datasets or datasets with issues, use simpler criteria
        if expr_df.shape[1] < 100 or expr_df.shape[0] < 10:
            # Just use all genes instead of trying to find highly variable ones
            adata.var['highly_variable'] = True
        else:
            # Use more robust parameters for highly variable gene selection
            sc.pp.highly_variable_genes(
                adata, 
                min_mean=0.01,  # Lower threshold to include more genes
                max_mean=10,    # Higher threshold to include more genes
                min_disp=0.1,   # Lower dispersion threshold
                n_top_genes=min(2000, adata.n_vars)  # Use top N genes or all genes if fewer
            )
    except Exception as e:
        print(f"Warning: Error in highly variable gene selection: {e}")
        # Just use all genes
        adata.var['highly_variable'] = True
    
    # Filter to highly variable genes
    adata = adata[:, adata.var.highly_variable]
    
    # Scale the data
    try:
        sc.pp.scale(adata, max_value=10)
    except Exception as e:
        print(f"Warning: Error in scaling: {e}")
        # Continue without scaling if it fails
    
    # Run PCA with appropriate number of components
    n_components = min(30, min(adata.n_obs - 1, adata.n_vars - 1))
    
    # Handle very small datasets
    if n_components <= 1:
        print("Warning: Dataset too small for PCA/UMAP, using raw data for visualization")
        # Create a simple plot without dimensionality reduction
        plt.figure(figsize=(8, 6))
        
        if color_by == 'cell_type':
            for ct in adata.obs['cell_type'].unique():
                mask = adata.obs['cell_type'] == ct
                plt.scatter([0] * sum(mask), [0] * sum(mask), label=ct)
            plt.legend()
            plt.title("Cell Types (No dimensionality reduction - dataset too small)")
        else:
            plt.scatter([0] * adata.n_obs, [0] * adata.n_obs)
            plt.title(f"Cells colored by {color_by} (No dimensionality reduction)")
            
        plt.savefig(pca_output)
        plt.close()
        return adata
    
    # Run PCA
    try:
        sc.tl.pca(adata, n_comps=n_components)
        
        # Plot and save PCA
        sc.settings.figdir = os.path.dirname(pca_output)
        sc.settings.set_figure_params(dpi=150, figsize=(8, 6))
        
        sc.pl.pca(adata, color=color_by, show=False, 
                title=f"PCA of Gene Expression (colored by {color_by})")
        plt.savefig(pca_output)
        plt.close()
        print(f"PCA visualization saved as '{pca_output}'")
        
    except Exception as e:
        print(f"Warning: Error in PCA: {e}")
        # Create a simple plot without PCA
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"PCA failed: {str(e)}", ha='center', va='center')
        plt.savefig(pca_output)
        plt.close()
        return adata
    
    # Compute neighbors before UMAP
    try:
        if adata.n_obs < 15:
            # For small datasets, adjust neighbors parameters
            n_neighbors = min(3, adata.n_obs-1)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        else:
            # Default parameters for larger datasets
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
        
        # Run UMAP
        sc.tl.umap(adata)
        
        # Plot UMAP
        sc.pl.umap(adata, color=color_by, show=False, 
                title=f"UMAP of Gene Expression (colored by {color_by})")
        plt.savefig(umap_output)
        plt.close()
        print(f"UMAP visualization saved as '{umap_output}'")
        
    except Exception as e:
        print(f"Warning: Error in UMAP: {e}")
        # Fall back to PCA plot if UMAP fails
        plt.figure(figsize=(8, 6))
        
        if hasattr(adata, 'obsm') and 'X_pca' in adata.obsm:
            if color_by == 'cell_type':
                for ct in adata.obs['cell_type'].unique():
                    mask = adata.obs['cell_type'] == ct
                    plt.scatter(adata.obsm['X_pca'][mask, 0], adata.obsm['X_pca'][mask, 1], label=ct)
                plt.legend()
            else:
                plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1])
            
            plt.title(f"PCA of Gene Expression (colored by {color_by})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        else:
            plt.text(0.5, 0.5, f"Visualization failed: {str(e)}", ha='center', va='center')
            
        plt.savefig(umap_output)  # Save as UMAP output since UMAP failed
        plt.close()
    
    return adata

def visualize_simulation_results(cells, mutations, expr_df, output_prefix, color_by='cell_type'):
    """
    Visualize all simulation results, including both mitochondrial mutations and gene expression.
    
    Parameters
    ----------
    cells : list of Cell objects
        Each cell has 'mutation_profile' and attributes for coloring.
    mutations : list of str
        Sorted list of mutation identifiers.
    expr_df : pd.DataFrame
        Single-cell expression matrix (cells x genes). Index = cell IDs, columns = gene IDs.
    output_prefix : str
        Path prefix for saving the output files.
    color_by : str
        Cell attribute to color UMAP scatter. Typically "cell_type" or "generation".
    """
    # Visualize mitochondrial mutations
    mito_output = f"{output_prefix}_heatmap.png"
    visualize_mito_mutations(cells, mutations, mito_output)
    
    # Visualize gene expression
    visualize_gene_expression(cells, expr_df, output_prefix, color_by)

def analyze_af_distribution(cells, mutations, output_prefix):
    """
    Analyze the distribution of allele frequencies in the simulated data,
    separately for baseline and de novo mutations.
    """
    # Extract allele frequencies from the mutation profiles
    all_afs = []
    cell_type_afs = {}
    mutation_afs = {mut: [] for mut in mutations}
    
    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in cell_type_afs:
            cell_type_afs[cell_type] = []
        
        for mut in mutations:
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af = ad / dp
                all_afs.append(af)
                cell_type_afs[cell_type].append(af)
                mutation_afs[mut].append(af)
    
    # Separate baseline and de novo mutations
    baseline_mutations = [mut for mut in mutations if mut.startswith('baseline_')]
    denovo_mutations = [mut for mut in mutations if not mut.startswith('baseline_')]
    
    # Create separate dictionaries for baseline and de novo mutations
    baseline_afs = []
    denovo_afs = []
    baseline_cell_type_afs = {}
    denovo_cell_type_afs = {}
    
    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in baseline_cell_type_afs:
            baseline_cell_type_afs[cell_type] = []
            denovo_cell_type_afs[cell_type] = []
        
        # Process baseline mutations
        for mut in baseline_mutations:
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af = ad / dp
                baseline_afs.append(af)
                baseline_cell_type_afs[cell_type].append(af)
        
        # Process de novo mutations
        for mut in denovo_mutations:
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af = ad / dp
                denovo_afs.append(af)
                denovo_cell_type_afs[cell_type].append(af)
    
    # 1. Compare baseline vs de novo AF distributions
    plt.figure(figsize=(12, 6))
    plt.hist(baseline_afs, bins=30, alpha=0.5, label='Baseline Mutations', color='blue')
    plt.hist(denovo_afs, bins=30, alpha=0.5, label='De Novo Mutations', color='red')
    plt.title('Distribution of Allele Frequencies: Baseline vs De Novo Mutations')
    plt.xlabel('Allele Frequency')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add summary statistics for both types
    stats_text = (
        f"Baseline Mutations:\n"
        f"Count: {len(baseline_afs)}\n"
        f"Mean: {np.mean(baseline_afs):.4f}\n"
        f"Median: {np.median(baseline_afs):.4f}\n"
        f"Std Dev: {np.std(baseline_afs):.4f}\n\n"
        f"De Novo Mutations:\n"
        f"Count: {len(denovo_afs)}\n"
        f"Mean: {np.mean(denovo_afs):.4f}\n"
        f"Median: {np.median(denovo_afs):.4f}\n"
        f"Std Dev: {np.std(denovo_afs):.4f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_baseline_vs_denovo_af.png", dpi=300)
    plt.close()
    
    # 2. Box plots comparing cell types for baseline and de novo mutations
    plt.figure(figsize=(15, 6))
    
    # Create DataFrames for both mutation types
    baseline_data = []
    denovo_data = []
    
    for cell_type in baseline_cell_type_afs.keys():
        for af in baseline_cell_type_afs[cell_type]:
            baseline_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'Baseline'})
        for af in denovo_cell_type_afs[cell_type]:
            denovo_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'De Novo'})
    
    combined_df = pd.DataFrame(baseline_data + denovo_data)
    
    # Create side-by-side box plots
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined_df[combined_df['Mutation Type'] == 'Baseline'],
                x='Cell Type', y='AF')
    plt.title('Baseline Mutations by Cell Type')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=combined_df[combined_df['Mutation Type'] == 'De Novo'],
                x='Cell Type', y='AF')
    plt.title('De Novo Mutations by Cell Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_celltype_comparison.png", dpi=300)
    plt.close()
    
    # Save the combined data
    combined_df.to_csv(f"{output_prefix}_mutation_type_data.csv", index=False)
    
    return {
        'baseline_summary': {
            'afs': baseline_afs,
            'mean': np.mean(baseline_afs) if baseline_afs else 0,
            'median': np.median(baseline_afs) if baseline_afs else 0,
            'std': np.std(baseline_afs) if baseline_afs else 0,
            'count': len(baseline_afs)
        },
        'denovo_summary': {
            'afs': denovo_afs,
            'mean': np.mean(denovo_afs) if denovo_afs else 0,
            'median': np.median(denovo_afs) if denovo_afs else 0,
            'std': np.std(denovo_afs) if denovo_afs else 0,
            'count': len(denovo_afs)
        },
        'cell_type_afs': {
            'baseline': baseline_cell_type_afs,
            'denovo': denovo_cell_type_afs
        }
    }

def generate_gene_expression(cells, num_genes=1000, cell_type_specific_ratio=0.3):
    """
    Generate gene expression data with cell type-specific genes.
    
    Parameters
    ----------
    cells : list
        List of Cell objects
    num_genes : int
        Total number of genes to simulate
    cell_type_specific_ratio : float
        Ratio of genes that are cell type-specific
    
    Returns
    -------
    tuple
        (expression DataFrame, gene info DataFrame, cell metadata DataFrame)
    """
    # Get cell types (assuming cells have cell_type attribute)
    cell_types = ['stem', 'progenitor', 'differentiated']
    
    # Generate gene info
    gene_info = []
    gene_ids = []
    
    # Common genes
    num_specific_genes = int(num_genes * cell_type_specific_ratio)
    num_common_genes = num_genes - num_specific_genes
    
    # Add common genes
    for i in range(num_common_genes):
        gene_id = f'gene_{i}'
        gene_ids.append(gene_id)
        gene_info.append({
            'gene_id': gene_id,
            'is_cell_type_specific': False,
            'specific_to_cell_type': 'common',
            'base_expression': np.random.gamma(2, 2),
            'expression_variability': np.random.uniform(0.1, 0.3)
        })
    
    # Add cell type-specific genes
    genes_per_type = num_specific_genes // len(cell_types)
    for cell_type in cell_types:
        for i in range(genes_per_type):
            gene_id = f'gene_{len(gene_ids)}'
            gene_ids.append(gene_id)
            gene_info.append({
                'gene_id': gene_id,
                'is_cell_type_specific': True,
                'specific_to_cell_type': cell_type,
                'base_expression': np.random.gamma(2, 2),
                'expression_variability': np.random.uniform(0.1, 0.3)
            })
    
    gene_info_df = pd.DataFrame(gene_info)
    
    # Generate expression data
    expr_data = []
    for cell in cells:
        for _, gene in gene_info_df.iterrows():
            base_expr = gene['base_expression']
            variability = gene['expression_variability']
            
            if gene['is_cell_type_specific']:
                if cell.cell_type == gene['specific_to_cell_type']:
                    expr = np.random.normal(base_expr * 2, base_expr * variability)
                else:
                    expr = np.random.normal(base_expr * 0.1, base_expr * variability * 0.5)
            else:
                expr = np.random.normal(base_expr, base_expr * variability)
            
            expr = max(0, expr)  # Ensure non-negative expression
            expr_data.append({
                'cell_id': cell.id,
                'gene_id': gene['gene_id'],
                'expression': expr
            })
    
    # Create expression matrix
    expr_df = pd.DataFrame(expr_data)
    expr_matrix = expr_df.pivot(
        index='cell_id',
        columns='gene_id',
        values='expression'
    )
    
    # Add cell metadata
    cell_metadata = pd.DataFrame([{
        'cell_id': cell.id,
        'generation': cell.generation,
        'cell_type': cell.cell_type,
        'parent_id': cell.parent_id if cell.parent_id is not None else 'root',
        'time': cell.time,
        'num_children': len(cell.children)
    } for cell in cells])
    cell_metadata.set_index('cell_id', inplace=True)
    
    # Join expression matrix with cell metadata
    expr_matrix = expr_matrix.join(cell_metadata)
    
    return expr_matrix, gene_info_df, cell_metadata

def save_simulation_data(cells, mutations, output_dir, prefix):
    """
    Save simulation data including expression, cell info, and gene info.
    
    Parameters
    ----------
    cells : list
        List of Cell objects
    mutations : list
        List of mutation IDs
    output_dir : str
        Directory to save results
    prefix : str
        Prefix for output files
    """
    # Generate expression data
    expr_matrix, gene_info_df, cell_metadata = generate_gene_expression(cells)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    expr_dir = os.path.join(output_dir, "expression")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(expr_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Save expression matrix
    expr_file = os.path.join(expr_dir, f"{prefix}_expression_matrix.mtx")
    scipy.io.mmwrite(expr_file, scipy.sparse.csr_matrix(expr_matrix.values))
    
    # Save cell IDs
    cell_ids_file = os.path.join(expr_dir, f"{prefix}_barcodes.tsv")
    with open(cell_ids_file, 'w') as f:
        for cell_id in expr_matrix.index:
            f.write(f"{cell_id}\n")
    
    # Save gene IDs
    gene_ids_file = os.path.join(expr_dir, f"{prefix}_features.tsv")
    with open(gene_ids_file, 'w') as f:
        for gene_id in expr_matrix.columns:
            if gene_id not in cell_metadata.columns:  # Skip metadata columns
                f.write(f"{gene_id}\n")
    
    # Save gene information
    gene_info_file = os.path.join(metadata_dir, f"{prefix}_gene_info.csv")
    gene_info_df.to_csv(gene_info_file, index=False)
    
    # Save cell metadata
    cell_meta_file = os.path.join(metadata_dir, f"{prefix}_cell_metadata.csv")
    cell_metadata.to_csv(cell_meta_file)
    
    # Save mutation information
    mutation_info = pd.DataFrame({
        'mutation_id': mutations,
        'is_baseline': ['baseline' in mut for mut in mutations]
    })
    mut_file = os.path.join(metadata_dir, f"{prefix}_mutation_info.csv")
    mutation_info.to_csv(mut_file, index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_cells': len(cells),
        'total_mutations': len(mutations),
        'total_genes': len(gene_info_df),
        'cell_type_specific_genes': sum(gene_info_df['is_cell_type_specific']),
        'max_generation': max(cell.generation for cell in cells),
        'cell_type_counts': cell_metadata['cell_type'].value_counts().to_dict()
    }
    with open(os.path.join(metadata_dir, f"{prefix}_summary.txt"), 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    return expr_matrix, gene_info_df, cell_metadata

if __name__ == "__main__":
    # 1) Run the simulation
    np.random.seed(123)
    random.seed(123)
    cells, mutations, expr_df = run_basic_simulation(num_baseline_mutations=5)

    # 2) Save the simulation results
    matrices = export_mtx_for_dp_ad(cells, mutations, prefix="/Users/linxy29/Documents/Data/CIVET/simulation/cellSNP.tag")
    expr_df.to_csv("/Users/linxy29/Documents/Data/CIVET/simulation/expression.csv", index=False)

    # 3) Visualize the simulation results
    visualize_simulation_results(cells, mutations, expr_df, output_prefix="/Users/linxy29/Documents/Data/CIVET/simulation/simulation")
    
    # 4) Analyze allele frequency distribution
    af_analysis = analyze_af_distribution(cells, mutations, 
                                         "/Users/linxy29/Documents/Data/CIVET/simulation/af_analysis/af")
    
    # Print summary statistics for both baseline and de novo mutations
    print("\nBaseline Mutations Summary:")
    print(f"Count: {af_analysis['baseline_summary']['count']}")
    print(f"Mean AF: {af_analysis['baseline_summary']['mean']:.4f}")
    print(f"Median AF: {af_analysis['baseline_summary']['median']:.4f}")
    print(f"Std Dev: {af_analysis['baseline_summary']['std']:.4f}")
    
    print("\nDe Novo Mutations Summary:")
    print(f"Count: {af_analysis['denovo_summary']['count']}")
    print(f"Mean AF: {af_analysis['denovo_summary']['mean']:.4f}")
    print(f"Median AF: {af_analysis['denovo_summary']['median']:.4f}")
    print(f"Std Dev: {af_analysis['denovo_summary']['std']:.4f}")