# simulate_scenarios.py

import numpy as np
import pandas as pd
import random
from copy import deepcopy

# Import your optimized simulation code
from simulation_framework_mito import (
    Cell,
    Mitochondrion,
    simulate_stem_cell_growth,
    simulate_cell_differentiation,
    simulate_read_depth,
    generate_gene_params,
    simulate_gene_expression_for_cells,
    replicate_and_segregate_mtDNA,
    introduce_mutations,
    R, KAPPA_GROW, T0_GROW,
    MTDNA_INITIAL_COUNT, MTDNA_MUTATION_RATE_PER_MITOSIS,
    MEAN_DP, DP_DISPERSION, BASE_ERROR_RATE,
    KAPPA_DIFF, T0_DIFF,
    NUM_GENES, CELL_TYPE_SPECIFIC_GENES, ALPHA, ZERO_INFLATION_PROB
)

# ------------------------------------------------------------------------------
# SCENARIO 1: Exploring Different Mutation Rate Regimes
# ------------------------------------------------------------------------------

def run_scenario_mutation_rate_regimes(rates, total_cells=1000):
    """
    Iterate over a list of mtDNA mutation rates and simulate for each.
    
    :param rates: List of mutation rates (floats).
    :param total_cells: Number of cells to simulate.
    :return: Dictionary where each key is a mutation rate,
             and value is (cells_list, mutations_list, expression_df).
    """
    scenario_results = {}

    for rate in rates:
        print(f"\n[Scenario 1] Running simulation with mtDNA mutation rate = {rate} ...")

        # 1) Stem cell growth
        cells = simulate_stem_cell_growth(
            total_cells=total_cells,
            r=R,
            kappa=KAPPA_GROW,
            t0=T0_GROW,
            mtDNA_initial_count=MTDNA_INITIAL_COUNT,
            mtDNA_mutation_rate=rate
        )

        # 2) (Optional) Basic differentiation
        all_cells = simulate_cell_differentiation(cells)

        # 3) Collect all unique mutations
        all_mutations = set()
        for c in all_cells:
            for mt in c.mtDNA_list:
                all_mutations.update(mt.mutations)
        all_mutations = sorted(all_mutations)

        # Simulate read depth
        for c in all_cells:
            simulate_read_depth(c, all_mutations)

        # 4) Gene expression
        gene_params = generate_gene_params(
            num_genes=NUM_GENES,
            cell_type_specific_genes=CELL_TYPE_SPECIFIC_GENES
        )
        expr_df = simulate_gene_expression_for_cells(all_cells, gene_params)

        scenario_results[rate] = (all_cells, all_mutations, expr_df)

    return scenario_results

# ------------------------------------------------------------------------------
# SCENARIO 2: Varying mtDNA Segregation Models
# ------------------------------------------------------------------------------

def replicate_and_segregate_mtDNA_unequal(mtDNA_list, mutation_rate_per_mitosis, ratio=0.8):
    """
    Example of an *unequal* segregation approach:
      - ratio fraction of replicated mtDNA goes to Daughter1, the remainder to Daughter2.
    """
    count = len(mtDNA_list)
    if count == 0:
        return ([], [])

    # Replicate
    replicated_list = []
    for mt in mtDNA_list:
        replicated_list.append(mt)
        replicated_list.append(Mitochondrion(mutations=set(mt.mutations)))

    # Introduce new mutations
    total_new = len(replicated_list)
    effective_mut_rate = mutation_rate_per_mitosis / total_new
    introduce_mutations(replicated_list, effective_mut_rate)

    # Unequal segregation
    n_d1 = int(total_new * ratio)
    indices = np.random.permutation(total_new)
    d1_indices = indices[:n_d1]
    d2_indices = indices[n_d1:]
    
    daughter1 = [replicated_list[i] for i in d1_indices]
    daughter2 = [replicated_list[i] for i in d2_indices]

    return daughter1, daughter2

def simulate_stem_cell_growth_unequal(
    total_cells=1000,
    r=R,
    kappa=KAPPA_GROW,
    t0=T0_GROW,
    mtDNA_initial_count=MTDNA_INITIAL_COUNT,
    mtDNA_mutation_rate=MTDNA_MUTATION_RATE_PER_MITOSIS,
    ratio=0.8
):
    """
    Same as simulate_stem_cell_growth, but uses an unequal segregation function.
    """
    from collections import deque

    cells = []
    queue = deque()
    cid_counter = 0
    current_time = 0.0

    # Initialize first cell
    initial_mtDNA = [Mitochondrion() for _ in range(mtDNA_initial_count)]
    first_cell = Cell(cid_counter, None, 0, current_time, initial_mtDNA, 'StemCell')
    cells.append(first_cell)
    queue.append(first_cell)
    cid_counter += 1

    while len(cells) < total_cells and queue:
        parent_cell = queue.popleft()

        # Division rate
        div_rate = r * (1.0 - 1.0/(1.0 + np.exp(-kappa*(parent_cell.time - t0))))
        if div_rate <= 0:
            continue
        
        # Waiting time for division
        wait_time = np.random.exponential(1.0 / div_rate)
        current_time = parent_cell.time + wait_time

        # Replicate & segregate (unequal)
        d1, d2 = replicate_and_segregate_mtDNA_unequal(parent_cell.mtDNA_list, mtDNA_mutation_rate, ratio)

        for daughter_mtDNA in (d1, d2):
            new_cell = Cell(
                cid_counter,
                parent_cell.id,
                parent_cell.generation+1,
                current_time,
                daughter_mtDNA,
                'StemCell'
            )
            cells.append(new_cell)
            parent_cell.children.append(new_cell.id)
            queue.append(new_cell)
            cid_counter += 1

        if len(cells) >= total_cells:
            break
    
    return cells

def run_scenario_varying_segregation_models(ratios=[0.5, 0.8], total_cells=1000):
    """
    For each ratio, simulate the growth with that segregation model and gather results.
    
    :param ratios: List of floats in (0,1) that define how 'unequal' the segregation is.
    :param total_cells: number of cells to simulate
    :return: dictionary with ratio -> (cell_list, mutations, expr_df)
    """
    results = {}
    for rseg in ratios:
        print(f"\n[Scenario 2] Running simulation with segregation ratio = {rseg} ...")

        # Stem cell growth with custom segregation
        cells_unequal = simulate_stem_cell_growth_unequal(
            total_cells=total_cells,
            ratio=rseg
        )

        # Optional: differentiation
        all_cells = simulate_cell_differentiation(cells_unequal)

        # Collect mutations
        all_mutations = set()
        for c in all_cells:
            for mt in c.mtDNA_list:
                all_mutations.update(mt.mutations)
        all_mutations = sorted(all_mutations)

        # DP & AD
        for c in all_cells:
            simulate_read_depth(c, all_mutations)

        # Gene expression
        gene_params = generate_gene_params()
        expr_df = simulate_gene_expression_for_cells(all_cells, gene_params)

        results[rseg] = (all_cells, all_mutations, expr_df)

    return results

# ------------------------------------------------------------------------------
# SCENARIO 3: Linear vs. Bifurcated Differentiation Paths
# ------------------------------------------------------------------------------

def simulate_bifurcated_differentiation(cells, kappa_diff=KAPPA_DIFF, t0_diff=T0_DIFF):
    """
    Instead of a single linear path, progenitor cells can branch into multiple cell types.
    Example: 'DifferentiatedTypeA' or 'DifferentiatedTypeB'.
    
    We treat stem cells -> progenitor with some probability,
    then each progenitor divides asymmetrically into:
      - 1 new progenitor
      - 1 of two possible fates (randomly chosen)
    """
    new_cells = []
    for c in cells:
        if c.cell_type == 'StemCell':
            diff_prob = 1.0 - (1.0 / (1.0 + np.exp(-kappa_diff*(c.time - t0_diff))))
            if np.random.rand() < diff_prob:
                c.cell_type = 'Progenitor'
        elif c.cell_type == 'Progenitor':
            # Asymmetric division => 1 progenitor, 1 specialized
            d1, d2 = replicate_and_segregate_mtDNA(c.mtDNA_list, MTDNA_MUTATION_RATE_PER_MITOSIS)
            
            # Daughter 1: still a progenitor
            daughter_progenitor = Cell(
                c.id*2, c.id, c.generation+1, c.time+1, d1, 'Progenitor'
            )
            # Daughter 2: randomly pick Type A or B
            fate = 'DifferentiatedTypeA' if np.random.rand() < 0.5 else 'DifferentiatedTypeB'
            daughter_diff = Cell(
                c.id*2+1, c.id, c.generation+1, c.time+1, d2, fate
            )
            new_cells.extend([daughter_progenitor, daughter_diff])

    return cells + new_cells

def run_scenario_linear_vs_bifurcated(linear=True, total_cells=1000):
    """
    If linear=True, use simulate_cell_differentiation() from the main code.
    Otherwise, use the bifurcated model.
    """
    mode = "linear" if linear else "bifurcated"
    print(f"\n[Scenario 3] Running {mode} differentiation path ...")

    # 1) Grow
    cells_init = simulate_stem_cell_growth(total_cells=total_cells)

    # 2) Differentiate
    if linear:
        all_cells = simulate_cell_differentiation(cells_init)
    else:
        all_cells = simulate_bifurcated_differentiation(cells_init)

    # 3) Mutations
    all_muts = set()
    for c in all_cells:
        for mt in c.mtDNA_list:
            all_muts.update(mt.mutations)
    all_muts = sorted(all_muts)

    for c in all_cells:
        simulate_read_depth(c, all_muts)

    # 4) Gene expression
    gene_params = generate_gene_params()
    expr_df = simulate_gene_expression_for_cells(all_cells, gene_params)

    return all_cells, all_muts, expr_df

# ------------------------------------------------------------------------------
# SCENARIO 4: Gene Expression Variation with Cell Generation vs. Cell Type
# ------------------------------------------------------------------------------

def generate_gene_params_generation_dominant(num_genes=NUM_GENES):
    """
    Example: Expression primarily depends on the generation depth, with minimal cell-type specificity.
    """
    gene_ids = [f"Gene_{i}" for i in range(num_genes)]
    gene_params = {}
    for g_id in gene_ids:
        # Larger expression_rate to emphasize generational effect
        base_expr = np.random.uniform(1, 2)
        expr_rate = np.random.uniform(0.3, 0.8)  # bigger slope
        gene_params[g_id] = {
            'base_expression': base_expr,
            'expression_rate': expr_rate,
            'sigma': 1.0
        }
    return gene_params

def generate_gene_params_type_dominant(num_genes=NUM_GENES, n_types=3):
    """
    Example: Expression primarily depends on the final cell type.
    We'll simulate different base expressions for each type and minimal generation effect.
    """
    gene_ids = [f"Gene_{i}" for i in range(num_genes)]
    
    # Let's define a random "type factor" for each cell type
    type_factors = {}
    for t in range(n_types):
        # e.g., each type shifts expression in different ways
        type_factors[t] = np.random.uniform(0, 2)

    # We'll store base_expr as a dict {type_index -> base_expr}, ignoring generation mostly
    gene_params = {}
    for g_id in gene_ids:
        # Minimal generation effect, random type offset
        # We'll store just one "canonical" expression rate
        expr_rate = np.random.uniform(0.0, 0.1)  # small generation effect
        # We'll store type offsets separately
        # For simplicity, store them in the dictionary so we can incorporate them later
        base_expr_per_type = {}
        for t in range(n_types):
            base_expr_per_type[t] = np.random.uniform(1, 5) + type_factors[t]

        gene_params[g_id] = {
            'base_expr_per_type': base_expr_per_type,
            'expression_rate': expr_rate,
            'sigma': 1.0
        }
    return gene_params

def generate_latent_expression_generation_dominant(cell, gene_params):
    """
    For "generation-dominant" expression: 
    z = base_expression + expression_rate * cell.generation
    """
    z_t = {}
    gen = cell.generation
    for gene_id, params in gene_params.items():
        loc = params['base_expression'] + params['expression_rate'] * gen
        z_t[gene_id] = np.random.normal(loc=loc, scale=params['sigma'])
    return z_t

def generate_latent_expression_type_dominant(cell, gene_params):
    """
    For "type-dominant" expression:
      z = base_expr_per_type[cell_type_index] + small expression_rate * generation
    We'll map cell_type to an index if we have multiple types.
    """
    z_t = {}
    # Simple mapping: StemCell=0, Progenitor=1, DifferentiatedTypeA=2, DifferentiatedTypeB=2, etc.
    # Adjust logic to your actual cell type naming
    type_map = {
        'StemCell': 0,
        'Progenitor': 1,
        'DifferentiatedType1': 2,
        'DifferentiatedTypeA': 2,
        'DifferentiatedTypeB': 3,
    }
    t_idx = type_map.get(cell.cell_type, 0)
    for gene_id, params in gene_params.items():
        base_expr = params['base_expr_per_type'].get(t_idx, 1.0)
        loc = base_expr + params['expression_rate'] * cell.generation
        z_t[gene_id] = np.random.normal(loc=loc, scale=params['sigma'])
    return z_t

def run_scenario_gene_expression_variation(
    approach='generation_dominant',
    total_cells=1000
):
    """
    Demonstrate how to vary gene expression logic based on generation vs. cell type.
    :param approach: 'generation_dominant' or 'type_dominant'
    :param total_cells: number of cells
    :return: (cells, all_mutations, expr_df)
    """
    print(f"\n[Scenario 4] Running gene expression scenario: {approach}")
    
    # 1) Grow the cells
    cells_init = simulate_stem_cell_growth(total_cells=total_cells)
    # 2) (Optional) Basic differentiation
    all_cells = simulate_cell_differentiation(cells_init)

    # 3) Mutations
    all_muts = set()
    for c in all_cells:
        for mt in c.mtDNA_list:
            all_muts.add(tuple(mt.mutations))
    # Flatten
    real_set = set()
    for c in all_cells:
        for mt in c.mtDNA_list:
            real_set |= mt.mutations
    all_muts_sorted = sorted(real_set)

    # DP & AD
    for c in all_cells:
        simulate_read_depth(c, all_muts_sorted)

    # 4) Gene Expression
    if approach == 'generation_dominant':
        gene_params = generate_gene_params_generation_dominant()
        
        # We'll mimic the existing "simulate_gene_expression_for_cells" approach
        # but override how latent expression is generated:
        expr_data = {}
        for cell in all_cells:
            z_vals = generate_latent_expression_generation_dominant(cell, gene_params)
            expr_counts = sample_expression_zinb(z_vals, alpha=ALPHA, zero_inflation_prob=ZERO_INFLATION_PROB)
            expr_data[cell.id] = expr_counts

    else:  # 'type_dominant'
        # Bifurcate or just rename cell types as needed
        # We define multiple cell types for demonstration
        gene_params = generate_gene_params_type_dominant()

        expr_data = {}
        for cell in all_cells:
            z_vals = generate_latent_expression_type_dominant(cell, gene_params)
            expr_counts = sample_expression_zinb(z_vals, alpha=ALPHA, zero_inflation_prob=ZERO_INFLATION_PROB)
            expr_data[cell.id] = expr_counts

    expr_df = pd.DataFrame.from_dict(expr_data, orient='index')
    return all_cells, all_muts_sorted, expr_df

# We'll re-use the sample_expression from your main code if needed:
def sample_expression_zinb(z_t, alpha, zero_inflation_prob):
    """
    For convenience, replicate the same ZINB sampling approach 
    from simulation_framework_optimized but separated out.
    """
    x_t = {}
    r = 1.0 / alpha
    for gene_id, z in z_t.items():
        mu = np.exp(z)
        p = r / (r + mu)
        if np.random.rand() < zero_inflation_prob:
            x_t[gene_id] = 0
        else:
            x_t[gene_id] = np.random.negative_binomial(r, p)
    return x_t

# ------------------------------------------------------------------------------
# DEMO: If you want to run all scenarios at once from this script:
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # SCENARIO 1
    mutation_rates = [0.1, 0.4, 1.0]
    scenario1_results = run_scenario_mutation_rate_regimes(mutation_rates, total_cells=500)

    # SCENARIO 2
    segregation_ratios = [0.5, 0.7, 0.9]
    scenario2_results = run_scenario_varying_segregation_models(segregation_ratios, total_cells=500)

    # SCENARIO 3
    # Linear
    cells_linear, muts_linear, expr_linear = run_scenario_linear_vs_bifurcated(linear=True, total_cells=500)
    # Bifurcated
    cells_bif, muts_bif, expr_bif = run_scenario_linear_vs_bifurcated(linear=False, total_cells=500)

    # SCENARIO 4
    # Generation-dominant
    cells_gen_dom, muts_gen_dom, expr_gen_dom = run_scenario_gene_expression_variation(
        approach='generation_dominant',
        total_cells=500
    )
    # Type-dominant
    cells_type_dom, muts_type_dom, expr_type_dom = run_scenario_gene_expression_variation(
        approach='type_dominant',
        total_cells=500
    )

    print("\nAll scenarios completed!")
