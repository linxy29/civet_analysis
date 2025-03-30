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

class Mitochondrion:
    """
    Represents a single mtDNA molecule containing a set of mutations.
    Using a small, lightweight class instead of a dictionary for speed.
    """
    __slots__ = ['mutations']

    def __init__(self, mutations=None):
        self.mutations = mutations if mutations else set()

class Cell:
    """
    Represents a cell in the simulation.
    """
    __slots__ = [
        'id', 'parent_id', 'generation', 'time',
        'mtDNA_list', 'cell_type', 'children', 'mutation_profile'
    ]
    
    def __init__(self, cid, parent_id, generation, time_point, mtDNA_list=None, cell_type='StemCell'):
        self.id = f"cell_{cid}"  # Convert to string format
        self.parent_id = None if parent_id is None else f"cell_{parent_id}"
        self.generation = generation
        self.time = time_point
        self.mtDNA_list = mtDNA_list if mtDNA_list else []
        self.cell_type = cell_type
        self.children = []
        # DP & AD results for each mutation (filled after read-depth simulation)
        self.mutation_profile = {}

# ------------------------------------------------------------------------------
# mtDNA Dynamics
# ------------------------------------------------------------------------------

def introduce_mutations(mtDNA_list, mutation_rate):
    """
    Introduce mutations into a list of mtDNA molecules in-place.
    Vectorized approach for random draws ensures fewer Python-level loops.
    """
    if not mtDNA_list:
        return

    # Draw number of mutations for each mtDNA molecule
    num_muts_array = np.random.poisson(mutation_rate, size=len(mtDNA_list))

    for mt, nm in zip(mtDNA_list, num_muts_array):
        # Avoid multiple Python calls for each mutation by generating them at once
        if nm > 0:
            new_mutations = {f"m{val}" for val in np.random.randint(16569, size=nm)}
            mt.mutations.update(new_mutations)

def replicate_and_segregate_mtDNA(mtDNA_list, mutation_rate_per_mitosis):
    """
    Replicate and randomly segregate mtDNA during cell division.
    Each mtDNA replicates once, then we introduce mutations based on
    the overall mutation_rate_per_mitosis / total_mtDNA_count.
    """
    count = len(mtDNA_list)
    if count == 0:
        # Edge case: no mtDNA present
        return [[] , []]

    # Replicate mtDNA (we will copy references first, then create new objects)
    replicated_list = []
    for mt in mtDNA_list:
        # Original
        replicated_list.append(mt)
        # New copy with copied mutations
        new_mt = Mitochondrion(mutations=set(mt.mutations))
        replicated_list.append(new_mt)

    # Introduce new mutations after replication
    # Distribute the total mutation rate across the new total
    total_new_count = len(replicated_list)
    effective_mut_rate = mutation_rate_per_mitosis / total_new_count
    introduce_mutations(replicated_list, effective_mut_rate)

    # Segregation
    # We randomly split 'replicated_list' into two sets
    indices = np.random.permutation(total_new_count)
    half = total_new_count // 2
    daughter1_indices = indices[:half]
    daughter2_indices = indices[half:]

    daughter1 = [replicated_list[i] for i in daughter1_indices]
    daughter2 = [replicated_list[i] for i in daughter2_indices]
    return daughter1, daughter2

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

def initialize_baseline_mutations(mtDNA_list, num_baseline_mutations=5):
    """
    Introduce baseline mutations with random allele frequencies between 0.001 and 0.8
    to a subset of mtDNA molecules.
    
    Parameters
    ----------
    mtDNA_list : list
        List of Mitochondrion objects
    num_baseline_mutations : int
        Number of baseline mutations to introduce
    
    Returns
    -------
    list
        List of baseline mutation IDs that were introduced
    """
    if not mtDNA_list:
        return []
    
    # Generate baseline mutation IDs
    baseline_mutations = [f"baseline_m{i}" for i in range(num_baseline_mutations)]
    
    # For each baseline mutation, determine its allele frequency
    for mutation in baseline_mutations:
        # Random AF between 0.001 and 0.8
        target_af = np.random.uniform(0.001, 0.5)
        
        # Calculate how many mtDNA molecules should have this mutation
        num_molecules = int(len(mtDNA_list) * target_af)
        
        # Randomly select mtDNA molecules to receive this mutation
        selected_indices = np.random.choice(
            len(mtDNA_list), 
            size=num_molecules, 
            replace=False
        )
        
        # Add the mutation to selected mtDNA molecules
        for idx in selected_indices:
            mtDNA_list[idx].mutations.add(mutation)
    
    print(f"Added {num_baseline_mutations} baseline mutations with random AFs")
    return baseline_mutations

def simulate_stem_cell_growth(
    total_cells=TOTAL_CELLS,
    r=R,
    kappa=KAPPA_GROW,
    t0=T0_GROW,
    mtDNA_initial_count=MTDNA_INITIAL_COUNT,
    mtDNA_mutation_rate=MTDNA_MUTATION_RATE_PER_MITOSIS,
    num_baseline_mutations=5
):
    """
    Simulate stem cell growth with baseline mutations in founder cell
    """
    cells = []
    cell_queue = deque()
    cid_counter = 0
    current_time = 0.0

    # 1. Initialize founder cell with empty mtDNA
    founder_mtDNA = [Mitochondrion() for _ in range(mtDNA_initial_count)]
    founder_cell = Cell(cid_counter, None, 0, current_time, founder_mtDNA, 'StemCell')
    cells.append(founder_cell)
    cid_counter += 1

    # 2. Add baseline mutations to founder cell before any divisions
    baseline_mutations = initialize_baseline_mutations(
        founder_cell.mtDNA_list, 
        num_baseline_mutations=num_baseline_mutations
    )
    
    # 3. Start simulation with founder cell in queue
    cell_queue.append(founder_cell)

    while len(cells) < total_cells and cell_queue:
        parent_cell = cell_queue.popleft()
        
        # Skip cells that are too old
        if parent_cell.generation >= MAX_GENERATIONS:
            continue

        # Calculate time-dependent division rate
        div_rate = division_rate(parent_cell.time, r, kappa, t0)
        if div_rate <= 0:
            continue

        # Gillespie step
        waiting_time = np.random.exponential(1.0 / div_rate)
        current_time = parent_cell.time + waiting_time

        # mtDNA replication with new mutations
        d1_mtDNA, d2_mtDNA = replicate_and_segregate_mtDNA(
            parent_cell.mtDNA_list,
            mtDNA_mutation_rate
        )

        # Create daughter cells
        for d_mtDNA in (d1_mtDNA, d2_mtDNA):
            new_cell = Cell(
                cid_counter,
                parent_cell.id,
                parent_cell.generation + 1,
                current_time,
                d_mtDNA.copy(),
                'StemCell'
            )
            cells.append(new_cell)
            parent_cell.children.append(new_cell.id)
            cell_queue.append(new_cell)
            cid_counter += 1

            if len(cells) >= total_cells:
                break

    # Collect all mutations (baseline + de novo)
    all_mutations = set(baseline_mutations)
    for cell in cells:
        for mt in cell.mtDNA_list:
            all_mutations.update(mt.mutations)
    
    return cells, sorted(all_mutations)

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
    d1_mtDNA, d2_mtDNA = replicate_and_segregate_mtDNA(
        parent_cell.mtDNA_list, 
        mutation_rate_per_mitosis=MTDNA_MUTATION_RATE_PER_MITOSIS
    )
    prog_daughter.mtDNA_list = d1_mtDNA
    diff_daughter.mtDNA_list = d2_mtDNA
    
    # Add daughters to parent's children list
    parent_cell.children.extend([prog_daughter.id, diff_daughter.id])
    
    return [prog_daughter, diff_daughter]

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

# ------------------------------------------------------------------------------
# Sequencing Read Depth (DP & AD)
# ------------------------------------------------------------------------------

def simulate_read_depth(
    cell,
    all_mutations,
    mean_dp=MEAN_DP,
    dp_dispersion=DP_DISPERSION,
    base_error_rate=BASE_ERROR_RATE
):
    """
    Simulate total depth (DP) & allele depth (AD) for each known mutation in a cell.
    """
    # Count occurrences of each mutation
    mtDNA_count = len(cell.mtDNA_list)
    mutation_counts = {}

    for mt in cell.mtDNA_list:
        for mut in mt.mutations:
            mutation_counts[mut] = mutation_counts.get(mut, 0) + 1

    # Pre-generate negative binomial random draws for total depth
    # Using (r, p) parameterization: r = 1 / dp_dispersion, p = r/(r+mean)
    r_param = 1.0 / dp_dispersion
    p_param = r_param / (r_param + mean_dp)

    dp_values = np.random.negative_binomial(
        n=r_param, p=p_param, size=len(all_mutations)
    ).astype(int)
    
    cell.mutation_profile = {}

    for i, mutation in enumerate(all_mutations):
        dp_sim = int(dp_values[i] + mean_dp)  # Shift around mean
        dp_sim = dp_sim if dp_sim > 0 else 1  # Ensure DP >= 1

        count = mutation_counts.get(mutation, 0)
        if mtDNA_count > 0:
            h = float(count) / mtDNA_count
        else:
            h = 0.0

        # Effective heteroplasmy includes error
        # Probability that a read is the "mutant" allele
        effective_h = h * (1 - base_error_rate) + (1 - h) * base_error_rate
        ad_sim = np.random.binomial(dp_sim, effective_h)

        cell.mutation_profile[mutation] = {'DP': dp_sim, 'AD': ad_sim}

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
    mtDNA_initial_count=MTDNA_INITIAL_COUNT,
    mtDNA_mutation_rate=MTDNA_MUTATION_RATE_PER_MITOSIS,
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
    mtDNA_initial_count : int
        Initial number of mtDNA molecules per cell
    mtDNA_mutation_rate : float
        Mutation rate per mitosis
    num_baseline_mutations : int
        Number of baseline mutations to introduce with random AFs
        
    Returns
    -------
    tuple
        (cells, mutations, expr_df)
    """
    # 1. Simulate stem cell growth
    cells, mutations = simulate_stem_cell_growth(
        total_cells=total_cells,
        mtDNA_initial_count=mtDNA_initial_count,
        mtDNA_mutation_rate=mtDNA_mutation_rate,
        num_baseline_mutations=num_baseline_mutations
    )
    
    # 2. Simulate cell differentiation
    cells = simulate_cell_differentiation(cells)
    
    # 3. Collect all unique mutations
    all_mutations = set(mutations)
    for cell in cells:
        for mt in cell.mtDNA_list:
            all_mutations.update(mt.mutations)
    mutations = sorted(list(all_mutations))
    
    # 4. Simulate read depth for each mutation in each cell
    for cell in cells:
        simulate_read_depth(
            cell, 
            all_mutations=mutations,
            mean_dp=MEAN_DP,
            dp_dispersion=DP_DISPERSION,
            base_error_rate=BASE_ERROR_RATE
        )
    
    # 5. Generate gene expression parameters
    gene_params, cell_type_genes = generate_gene_params(
        num_genes=NUM_GENES,
        cell_type_specific_genes=CELL_TYPE_SPECIFIC_GENES
    )
    
    # 6. Simulate gene expression for all cells
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