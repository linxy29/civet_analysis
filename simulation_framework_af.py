#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Simulation Framework

This script demonstrates an example of a more structured, readable, and maintainable
single-file simulation framework for:

1. Stem cell growth (Gillespie-like)
2. Mitochondrial genome proliferation
3. Cell differentiation
4. Single-cell transcriptomic data simulation
5. Sequencing read depth (DP) and allele depth (AD) for mtDNA mutations

Author: Your Name
Date: 2025-03-31
"""

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import yaml  # Add this import for YAML support
import json  # Add this import for JSON support
from datetime import datetime

from collections import deque
from scipy.sparse import csc_matrix, csr_matrix
from scipy.io import mmwrite
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional

# Import Cell class from simulation_classes module
from simulation_classes import Cell

# Import visualization and saving functions from simulation_save
from simulation_save import *

# ------------------------------------------------------------------------------
# 1) Global Configuration
# ------------------------------------------------------------------------------
# Load configuration from file

def load_config(config_path=None):
    """
    Load configuration from a YAML or JSON file.
    If no file is provided or file doesn't exist, use default configuration.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file (YAML or JSON)
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        # Stem cell growth parameters
        "MAX_DIVISION_RATE": 1.0,
        "GROWTH_ACCELERATION": 0.1,
        "GROWTH_INFLECTION_TIME": 50,

        # mtDNA parameters
        "MTDNA_INITIAL_COUNT": 200,
        "MTDNA_MUTATION_RATE_PER_MITOSIS": 4,   # Mutations per cell division
        "SEGREGATION_BIAS": 0.5,

        # Sequencing read depth parameters
        "MEAN_DP": 100,            # Mean total read depth
        "DP_DISPERSION": 0.2,      # Overdispersion
        "BASE_ERROR_RATE": 0.001,  # Sequencing error rate

        # Cell differentiation parameters
        "DIFF_RATE": 0.1,
        "DIFF_INFLECTION_TIME": 30,
        "CELL_TYPE_FRAC": 0.5,

        # Gene expression parameters
        "NUM_GENES": 6000,
        "SPECIFIC_GENE_FRAC": 0.4,
        "ALPHA": 0.1,                # NB dispersion
        "ZERO_INFLATION_PROB": 0.1,  # Probability of zero inflation

        # Simulation scale
        "TOTAL_CELLS": 1000,
        "MAX_GENERATIONS": 100
    }
    
    # If no config file specified or file doesn't exist, return default config
    if config_path is None or not os.path.exists(config_path):
        print(f"Using default configuration")
        return default_config
    
    try:
        # Load configuration from file based on extension
        file_ext = os.path.splitext(config_path)[1].lower()
        
        if file_ext == '.yaml' or file_ext == '.yml':
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"Unsupported config file format: {file_ext}. Using default configuration.")
            return default_config
            
        print(f"Loaded configuration from {config_path}")
        
        # Merge with default config to ensure all parameters exist
        # This ensures any missing parameters in the config file will use default values
        merged_config = default_config.copy()
        merged_config.update(config)
        
        return merged_config
        
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print("Using default configuration")
        return default_config

# Load configuration
CONFIG = load_config()

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------------------------
# 2) Data Classes
# ------------------------------------------------------------------------------
# Cell class has been moved to simulation_classes.py


# ------------------------------------------------------------------------------
# 3) Utility Functions for mtDNA
# ------------------------------------------------------------------------------

def initialize_baseline_mutations(num_baseline_mutations: int = 5) -> Dict[str, float]:
    """
    Initialize a few baseline mutations with random allele frequencies in [0.001, 0.5].
    """
    baseline_mutations = {}
    positions = np.random.randint(1, 10000, size=num_baseline_mutations)
    for pos in positions:
        mut_id = f"baseline_m{pos}"
        # Uniform AF between 0.001 and 0.5
        baseline_mutations[mut_id] = np.random.uniform(0.001, 0.5)
    return baseline_mutations

def introduce_new_mutations(mutation_rate: float) -> Dict[str, float]:
    """
    Introduce new de novo mutations per cell division with small initial allele frequency.
    """
    num_mutations = np.random.poisson(mutation_rate)
    new_mutations = {}
    
    if num_mutations > 0:
        positions = np.random.randint(10001, 15000, size=num_mutations)
        for pos in positions:
            new_mutations[f"m{pos}"] = np.random.uniform(0.0001, 0.01)  # Very small AFs
    return new_mutations

def segregate_allele_frequencies(
    parent_afs: Dict[str, float],
    bias: float = 0.5
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Segregate allele frequencies during cell division. If bias != 0.5, one daughter
    inherits slightly higher allele frequencies.

    Parameters
    ----------
    parent_afs : dict
        Mutation -> allele frequency for the parent cell.
    bias : float
        Bias factor in [0,1]. 0.5 => symmetric, >0.5 => biased to daughter1, <0.5 => daughter2.

    Returns
    -------
    daughter1_afs, daughter2_afs : tuple of dicts
        Allele frequencies for the two daughter cells.
    """
    d1_afs, d2_afs = {}, {}
    
    for mutation, af in parent_afs.items():
        if af <= 0:
            continue
        
        # Introduce random factor for each daughter
        if np.random.rand() < bias:
            # Daughter1 gets slightly higher AF
            factor1 = np.random.uniform(1, bias*2)
            factor2 = 2.0 - factor1
        else:
            # Daughter2 gets slightly higher AF
            factor2 = np.random.uniform(1, bias*2)
            factor1 = 2.0 - factor2
        
        d1_new_af = np.clip(af * factor1, 0, 1.0)
        d2_new_af = np.clip(af * factor2, 0, 1.0)
        
        d1_afs[mutation] = d1_new_af
        d2_afs[mutation] = d2_new_af
    
    return d1_afs, d2_afs

# ------------------------------------------------------------------------------
# 4) Stem Cell Growth and Differentiation
# ------------------------------------------------------------------------------

def division_rate(
    t: float,
    r: float,
    kappa: float,
    t0: float
) -> float:
    """
    Logistic-like function for time-dependent cell division rate.
    """
    return r * (1.0 - 1.0 / (1.0 + np.exp(-kappa * (t - t0))))

def differentiation_probability(
    t: float,
    kappa_diff: float,
    t0_diff: float
) -> float:
    """
    Probability of a stem cell differentiating at time t (logistic-like).
    """
    return 1.0 - (1.0 / (1.0 + np.exp(-kappa_diff * (t - t0_diff))))

def simulate_stem_cell_growth_and_differentiation(
    total_cells: int,
    mutation_rate: float,
    num_baseline_mutations: int,
    bias: float,
    r: float,
    kappa: float,
    t0: float,
    diff_kappa: float,
    diff_t0: float
) -> Tuple[List[Cell], List[str]]:
    """
    Simulate expansion of cells starting from a single stem cell, with
    asymmetric division (one stem cell, one progenitor) and biased mtDNA segregation.

    Parameters
    ----------
    total_cells : int
        Target number of cells to simulate
    mutation_rate : float
        Rate of new mutations per cell division
    num_baseline_mutations : int
        Number of baseline mutations to initialize in the first cell
    bias : float
        Bias factor for mtDNA segregation (0.5 = symmetric)
    r, kappa, t0 : float
        Parameters for the division rate function
    diff_kappa, diff_t0 : float
        Parameters for the differentiation probability function

    Returns
    -------
    cells : list
        All cells in the final population.
    all_mutations : list
        Sorted list of all mutation IDs observed.
    """
    cells = []
    stem_cell_queue = deque()
    cid_counter = 0
    current_time = 0.0

    # Initialize the root cell
    root_cell = Cell(
        cid=cid_counter,
        parent_id=None,
        generation=0,
        time_point=current_time,
        mutation_afs=initialize_baseline_mutations(num_baseline_mutations),
        cell_type='StemCell'
    )
    cells.append(root_cell)
    stem_cell_queue.append(root_cell)
    cid_counter += 1

    while len(cells) < total_cells and stem_cell_queue:
        parent_cell = stem_cell_queue.popleft()
        
        # Skip if not a stem cell (shouldn't happen, but just in case)
        if parent_cell.cell_type != 'StemCell':
            continue
            
        # Division rate
        dr = division_rate(
            t=parent_cell.time,
            r=r,
            kappa=kappa,
            t0=t0
        )
        if dr <= 0:
            continue

        # Time to next division
        wait_time = np.random.exponential(1.0 / dr)
        new_time = parent_cell.time + wait_time

        # Introduce new mutations
        new_muts = introduce_new_mutations(mutation_rate)
        combined_afs = {**parent_cell.mutation_afs, **new_muts}

        # Segregate allele frequencies
        d1_afs, d2_afs = segregate_allele_frequencies(combined_afs, bias=bias)

        # Determine if this stem cell will produce a progenitor
        diff_prob = differentiation_probability(parent_cell.time, diff_kappa, diff_t0)
        will_differentiate = np.random.rand() < diff_prob
        
        # Create two daughters with asymmetric division
        # First daughter: Always a stem cell
        if len(cells) < total_cells:
            stem_daughter = Cell(
                cid=cid_counter,
                parent_id=int(parent_cell.id.replace("cell_", "")),
                generation=parent_cell.generation + 1,
                time_point=new_time,
                mutation_afs=d1_afs,
                cell_type='StemCell'  # Always a stem cell
            )
            cells.append(stem_daughter)
            parent_cell.children.append(stem_daughter.id)
            stem_cell_queue.append(stem_daughter)  # Add to queue for future divisions
            cid_counter += 1
        
        # Second daughter: Either a stem cell or a progenitor
        if len(cells) < total_cells:
            if will_differentiate:
                # Randomly pick which progenitor type
                prog_type = 'Progenitor1' if np.random.rand() < 0.5 else 'Progenitor2'
                
                prog_daughter = Cell(
                    cid=cid_counter,
                    parent_id=int(parent_cell.id.replace("cell_", "")),
                    generation=parent_cell.generation + 1,
                    time_point=new_time,
                    mutation_afs=d2_afs,
                    cell_type=prog_type
                )
                cells.append(prog_daughter)
                parent_cell.children.append(prog_daughter.id)
            else:
                # Another stem cell
                stem_daughter2 = Cell(
                    cid=cid_counter,
                    parent_id=int(parent_cell.id.replace("cell_", "")),
                    generation=parent_cell.generation + 1,
                    time_point=new_time,
                    mutation_afs=d2_afs,
                    cell_type='StemCell'
                )
                cells.append(stem_daughter2)
                parent_cell.children.append(stem_daughter2.id)
                stem_cell_queue.append(stem_daughter2)  # Add to queue for future divisions
            
            cid_counter += 1

    # Gather all mutations
    all_mutations = set()
    for c in cells:
        all_mutations.update(c.mutation_afs.keys())
    return cells, sorted(all_mutations)

def simulate_stem_cell_growth(
    total_cells: int,
    mutation_rate: float,
    num_baseline_mutations: int,
    bias: float,
    r: float,
    kappa: float,
    t0: float
) -> Tuple[List[Cell], List[str]]:
    """
    Simulate expansion of cells starting from a single stem cell, with optional
    biased mtDNA segregation.

    Returns
    -------
    cells : list
        All cells in the final population.
    all_mutations : list
        Sorted list of all mutation IDs observed.
    """
    cells = []
    queue = deque()
    cid_counter = 0
    current_time = 0.0

    # Initialize the root cell
    root_cell = Cell(
        cid=cid_counter,
        parent_id=None,
        generation=0,
        time_point=current_time,
        mutation_afs=initialize_baseline_mutations(num_baseline_mutations),
        cell_type='StemCell'
    )
    cells.append(root_cell)
    queue.append(root_cell)
    cid_counter += 1

    while len(cells) < total_cells and queue:
        parent_cell = queue.popleft()
        
        # Division rate
        dr = division_rate(
            t=parent_cell.time,
            r=r,
            kappa=kappa,
            t0=t0
        )
        if dr <= 0:
            continue

        # Time to next division
        wait_time = np.random.exponential(1.0 / dr)
        new_time = parent_cell.time + wait_time

        # Introduce new mutations
        new_muts = introduce_new_mutations(mutation_rate)
        combined_afs = {**parent_cell.mutation_afs, **new_muts}

        # Segregate allele frequencies
        d1_afs, d2_afs = segregate_allele_frequencies(combined_afs, bias=bias)

        # Create two daughters
        for daughter_afs in (d1_afs, d2_afs):
            if len(cells) >= total_cells:
                break
            new_cell = Cell(
                cid=cid_counter,
                parent_id=int(parent_cell.id.replace("cell_", "")),
                generation=parent_cell.generation + 1,
                time_point=new_time,
                mutation_afs=daughter_afs,
                cell_type=parent_cell.cell_type  # Inherit same type initially
            )
            cells.append(new_cell)
            parent_cell.children.append(new_cell.id)
            queue.append(new_cell)
            cid_counter += 1

    # Gather all mutations
    all_mutations = set()
    for c in cells:
        all_mutations.update(c.mutation_afs.keys())
    return cells, sorted(all_mutations)

def simulate_cell_differentiation(
    cells: List[Cell],
    diff_kappa: float,
    diff_t0: float,
    cell_type_frac: float = 0.5
) -> List[Cell]:
    """
    Convert some StemCells into Progenitor cells based on differentiation probability.
    Then handle one round of asymmetric division for those Progenitors.
    """
    # Get the maximum generation
    max_gen = max(c.generation for c in cells)

    for c in cells:
        if c.cell_type == 'StemCell' and c.generation == max_gen:
            # Probability of differentiating
            diff_prob = differentiation_probability(c.time, diff_kappa, diff_t0)
            if np.random.rand() < diff_prob:
                # Randomly pick which progenitor type
                c.cell_type = 'Progenitor1' if np.random.rand() < cell_type_frac else 'Progenitor2'
    
    return cells

def progenitor_asymmetric_division(parent_cell: Cell) -> List[Cell]:
    """
    Progenitor cell divides into:
      - One daughter that remains the same progenitor type
      - One daughter that becomes a terminal cell
    """
    # Generate unique IDs
    start_id = int(parent_cell.id.replace('cell_', '')) + 100000  # offset to avoid collisions
    daughter1_id = start_id
    daughter2_id = start_id + 1
    
    # Perform mtDNA segregation
    d1_afs, d2_afs = segregate_allele_frequencies(parent_cell.mutation_afs, bias=0.5)

    # Daughter 1: same as parent
    daughter1 = Cell(
        cid=daughter1_id,
        parent_id=int(parent_cell.id.replace("cell_", "")),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time,  # same time, no waiting modeled here
        mutation_afs=d1_afs,
        cell_type=parent_cell.cell_type
    )

    # Daughter 2: new terminal type
    daughter2_type = 'TerminalCell_A' if parent_cell.cell_type == 'Progenitor1' else 'TerminalCell_B'
    daughter2 = Cell(
        cid=daughter2_id,
        parent_id=int(parent_cell.id.replace("cell_", "")),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time,
        mutation_afs=d2_afs,
        cell_type=daughter2_type
    )

    parent_cell.children.extend([daughter1.id, daughter2.id])
    return [daughter1, daughter2]

# ------------------------------------------------------------------------------
# 5) Sequencing Simulation (DP & AD)
# ------------------------------------------------------------------------------

def simulate_read_depth(
    cell: Cell,
    all_mutations: List[str],
    mean_dp: float,
    dp_dispersion: float,
    base_error_rate: float,
    false_mutation_rate: float = 0.0001
) -> None:
    """
    Simulate read depth for each mutation in a cell using a negative binomial model,
    and then the allele depth using binomial sampling. Also add some false mutations.
    """
    r_param = 1.0 / dp_dispersion
    p_param = r_param / (r_param + mean_dp)
    
    # Sample DP for each mutation
    dp_samples = np.random.negative_binomial(r_param, p_param, size=len(all_mutations))
    
    cell.mutation_profile = {}

    for i, mutation in enumerate(all_mutations):
        # Ensure DP >= 1
        dp_sim = max(1, int(dp_samples[i] + mean_dp))
        
        # True allele fraction
        true_af = cell.mutation_afs.get(mutation, 0.0)
        # Adjust AF for base error
        effective_af = true_af * (1 - base_error_rate) + (1 - true_af) * base_error_rate
        effective_af = np.clip(effective_af, 0.0, 1.0)
        
        # Sample allele depth
        ad_sim = np.random.binomial(dp_sim, effective_af)
        
        cell.mutation_profile[mutation] = {
            'DP': dp_sim,
            'AD': ad_sim,
            'is_true_mutation': True
        }
    
    # Simulate false mutations
    num_false = np.random.poisson(false_mutation_rate * 16569)
    false_positions = set()
    while len(false_positions) < num_false:
        pos = np.random.randint(15001, 16570)
        false_positions.add(pos)
    
    for pos in false_positions:
        mut_id = f"false_m{pos}"
        dp_sim = max(1, int(np.random.negative_binomial(r_param, p_param) + mean_dp))
        false_af = np.random.beta(1, 20)  # Typically quite low
        ad_sim = np.random.binomial(dp_sim, false_af)
        
        cell.mutation_profile[mut_id] = {
            'DP': dp_sim,
            'AD': ad_sim,
            'is_true_mutation': False
        }

def get_dp_ad_matrices(
    cells: List[Cell],
    all_mutations: List[str]
) -> Dict[str, csc_matrix]:
    """
    Collect DP and AD data from all cells and return them as CSC matrices.
    Rows = cells, Columns = mutations.
    """
    num_cells = len(cells)
    num_muts = len(all_mutations)

    row_idx, col_idx = [], []
    dp_data, ad_data = [], []
    
    # Map cell.id -> row index
    cell_index_map = {c.id: i for i, c in enumerate(cells)}

    for c in cells:
        r_i = cell_index_map[c.id]
        for j, mut_id in enumerate(all_mutations):
            if mut_id in c.mutation_profile:
                dp_val = c.mutation_profile[mut_id]['DP']
                ad_val = c.mutation_profile[mut_id]['AD']
                if dp_val > 0:
                    row_idx.append(r_i)
                    col_idx.append(j)
                    dp_data.append(dp_val)
                    ad_data.append(ad_val)
    
    dp_mat = csc_matrix((dp_data, (row_idx, col_idx)), shape=(num_cells, num_muts))
    ad_mat = csc_matrix((ad_data, (row_idx, col_idx)), shape=(num_cells, num_muts))
    return {'dp_matrix': dp_mat, 'ad_matrix': ad_mat}

def export_mtx_for_dp_ad(
    cells: List[Cell],
    all_mutations: List[str],
    prefix: str
) -> Dict[str, csc_matrix]:
    """
    Write DP and AD matrices in Matrix Market format (.mtx) and return the matrices.
    """
    mats = get_dp_ad_matrices(cells, all_mutations)
    dp_outfile = f"{prefix}.DP.mtx"
    ad_outfile = f"{prefix}.AD.mtx"
    
    mmwrite(dp_outfile, mats['dp_matrix'])
    mmwrite(ad_outfile, mats['ad_matrix'])
    
    print(f"[export_mtx_for_dp_ad] DP -> {dp_outfile}, AD -> {ad_outfile}")
    return mats

# ------------------------------------------------------------------------------
# 6) Single-Cell Transcriptomics
# ------------------------------------------------------------------------------

def generate_gene_params(
    num_genes: int,
    specific_gene_frac: float
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, set]]:
    """
    Example function to define which genes are cell-type-specific, etc.
    Adjust as needed for your own logic.
    """
    gene_ids = [f"Gene_{i}" for i in range(num_genes)]
    num_specific = int(num_genes * specific_gene_frac)
    # Calculate genes per type (for stem and progenitor cells)
    genes_per_type = num_specific // 5  # Divide among all 5 cell types
    
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

def generate_latent_expression(
    cell: Cell,
    gene_params: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute a latent (log-scale) expression for each gene in this cell,
    factoring in its generation or cell type.
    """
    z_vals = {}
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
            loc = base_expr + modified_rate * (1 + gen * 0.005)
        else:
            # Base expression only for non-specific genes
            loc = base_expr
            
        scale = params['sigma']
        z_vals[gene_id] = np.random.normal(loc=loc, scale=scale)
    return z_vals

def sample_expression(
    z_vals: Dict[str, float],
    alpha: float,
    zero_inflation_prob: float
) -> Dict[str, int]:
    """
    Sample from a zero-inflated negative binomial using the latent parameter z_vals.
    """
    counts = {}
    r_param = 1.0 / alpha
    for g, z in z_vals.items():
        mu = np.exp(z)
        p = r_param / (r_param + mu)
        # Zero-inflation
        if np.random.rand() < zero_inflation_prob:
            x = 0
        else:
            x = np.random.negative_binomial(r_param, p)
        counts[g] = max(0, x)
    return counts

def simulate_gene_expression_for_cells(
    cells: List[Cell],
    gene_params: Dict[str, Dict[str, float]],
    alpha: float,
    zero_inflation_prob: float
) -> pd.DataFrame:
    """
    Build an expression matrix (cells x genes).
    """
    expr_dict = {}
    for c in cells:
        latent = generate_latent_expression(c, gene_params)
        observed = sample_expression(latent, alpha, zero_inflation_prob)
        expr_dict[c.id] = observed
    
    expr_df = pd.DataFrame.from_dict(expr_dict, orient='index')
    return expr_df

# ------------------------------------------------------------------------------
# Function to visualize the simulation results
# ------------------------------------------------------------------------------
# All visualization and saving functions have been moved to simulation_save.py

# ------------------------------------------------------------------------------
# 8) High-Level Simulation Pipeline
# ------------------------------------------------------------------------------

def run_basic_simulation(
    config: dict = CONFIG
) -> Tuple[List[Cell], List[str], pd.DataFrame]:
    """
    Full simulation pipeline:
      1) Stem cell growth with optional bias
      2) Differentiation
      3) Sequencing read depth
      4) Gene expression

    Returns
    -------
    cells : list of Cell
    mutations : list of all mutation IDs
    expr_df : pd.DataFrame of expression (cells x genes)
    """
    total_cells = config["TOTAL_CELLS"]
    print(f"Total simulation cells: {total_cells}")
    mutation_rate = config["MTDNA_MUTATION_RATE_PER_MITOSIS"]
    num_baseline_mutations = 5

    # 1. Differentiation while growth
    cells, mutations = simulate_stem_cell_growth_and_differentiation(
        total_cells=total_cells,
        mutation_rate=mutation_rate,
        num_baseline_mutations=num_baseline_mutations,
        bias=config["SEGREGATION_BIAS"],
        r=config["MAX_DIVISION_RATE"],
        kappa=config["GROWTH_ACCELERATION"],
        t0=config["GROWTH_INFLECTION_TIME"],
        diff_kappa=config["DIFF_RATE"],
        diff_t0=config["DIFF_INFLECTION_TIME"]
    )
    
    '''
    # 2. Growth then differentiation
    cells, mutations = simulate_stem_cell_growth(
        total_cells=total_cells,
        mutation_rate=mutation_rate,
        num_baseline_mutations=num_baseline_mutations,
        bias=config["SEGREGATION_BIAS"],
        r = config["MAX_DIVISION_RATE"],
        kappa = config["GROWTH_ACCELERATION"],
        t0 = config["GROWTH_INFLECTION_TIME"]
    )

    cells = simulate_cell_differentiation(
        cells=cells,
        diff_kappa=config["DIFF_RATE"],
        diff_t0=config["DIFF_INFLECTION_TIME"],
        cell_type_frac=config["CELL_TYPE_FRAC"]
    )
    '''

    # 3. Sequencing read depth
    for cell in cells:
        simulate_read_depth(
            cell,
            all_mutations=mutations,
            mean_dp=config["MEAN_DP"],
            dp_dispersion=config["DP_DISPERSION"],
            base_error_rate=config["BASE_ERROR_RATE"]
        )

    # 4. Gene expression
    gene_params, _ = generate_gene_params(
        num_genes=config["NUM_GENES"],
        specific_gene_frac=config["SPECIFIC_GENE_FRAC"]
    )
    expr_df = simulate_gene_expression_for_cells(
        cells,
        gene_params=gene_params,
        alpha=config["ALPHA"],
        zero_inflation_prob=config["ZERO_INFLATION_PROB"]
    )

    return cells, mutations, expr_df, gene_params

if __name__ == "__main__":
    # Parse command line arguments to get config file path
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cell simulation with mtDNA mutations')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file (YAML or JSON)')
    args = parser.parse_args()
    
    # Load configuration
    #CONFIG = load_config(args.config)
    CONFIG = load_config("simulation_config.yaml")
    
    # 1) Run the simulation
    np.random.seed(123)
    random.seed(123)
    cells, mutations, expr_df, gene_params = run_basic_simulation(config = CONFIG)
    
    # 2) Choose a single output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/Users/linxy29/Documents/Data/CIVET/simulation/simulation_{timestamp}"

    # 3) Save the simulation results
    save_simulation_data(cells, mutations, expr_df, gene_params, output_dir)