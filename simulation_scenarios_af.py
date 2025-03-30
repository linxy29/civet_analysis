import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from simulation_framework_af import *
from datetime import datetime

# ------------------------------------------------------------------------------
# SCENARIO 1: Exploring Different Mutation Rate Regimes
# ------------------------------------------------------------------------------
def scenario1_mutation_rate_exploration(
    mutation_rates = [1, 2, 4, 8],
    total_cells=1000,
    max_generations=100,
    num_baseline_mutations=5,
    output_dir="./scenario1_results"
):
    """
    Explore how different mtDNA mutation rates affect the simulation outcome.
    We simply loop over a list of mutation rates and run the simulation for each.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for rate in mutation_rates:
        print(f"\n[Scenario 1] Running simulation with mutation_rate={rate}")
        
        # Run the simulation
        cells, mutations, expr_df = run_basic_simulation(
            total_cells=total_cells,
            max_generations=max_generations,
            mutation_rate=rate,
            num_baseline_mutations=num_baseline_mutations
        )
        
        # Save DP/AD (Matrix Market format) for mtDNA mutations
        prefix = os.path.join(output_dir, f"mutationRate_{rate}")
        export_mtx_for_dp_ad(cells, mutations, prefix=prefix)
        
        # Save expression data
        expr_csv = prefix + "_expression.csv"
        expr_df.to_csv(expr_csv, index=True)
        
        # Visualize and save plots
        visualize_simulation_results(
            cells, mutations, expr_df,
            output_prefix=prefix,
            color_by='cell_type'
        )
        
        # Analyze AF distributions
        af_analysis = analyze_af_distribution(cells, mutations, prefix + "_af_analysis")
        
        results[rate] = {
            'cells': cells,
            'mutations': mutations,
            'expr_df': expr_df,
            'af_analysis': af_analysis
        }
    
    return results


# ------------------------------------------------------------------------------
# SCENARIO 2: Varying mtDNA Segregation Models
# ------------------------------------------------------------------------------
def simulate_stem_cell_growth_unbiased(
    total_cells=1000,
    mutation_rate=4,
    num_baseline_mutations=5,
    r=R,
    kappa=KAPPA_GROW,
    t0=T0_GROW
):
    """
    Example variant of stem cell growth using UNBIASED segregation.
    This function parallels `simulate_stem_cell_growth_biased`, but uses
    the normal `segregate_allele_frequencies` instead of `segregate_allele_frequencies_biased`.
    """
    cells = []
    cell_queue = []
    cid_counter = 0
    current_time = 0.0

    # Initialize first cell
    first_cell = Cell(
        cid=cid_counter,
        parent_id=None,
        generation=0,
        time_point=current_time,
        mutation_afs=initialize_baseline_mutations(num_baseline_mutations)
    )
    first_cell.cell_type = 'StemCell'
    cells.append(first_cell)
    cell_queue.append(first_cell)
    cid_counter += 1

    while len(cells) < total_cells and cell_queue:
        parent_cell = cell_queue.pop(0)
        
        # Determine division rate
        div_rate = division_rate(parent_cell.time, r, kappa, t0)
        if div_rate <= 0:
            continue
        
        # Calculate next division time from exponential waiting
        wait_time = np.random.exponential(1.0 / div_rate)
        current_time = parent_cell.time + wait_time
        
        # Introduce new mutations
        new_muts = introduce_new_mutations(mutation_rate)
        
        # Combine parent’s AF with new mutations
        parent_afs = {**parent_cell.mutation_afs, **new_muts}
        
        # Use the UNBIASED segregation function
        d1_afs, d2_afs = segregate_allele_frequencies(parent_afs)
        
        # Create daughters
        for d_afs in [d1_afs, d2_afs]:
            new_cell = Cell(
                cid_counter,
                parent_cell.id,
                parent_cell.generation + 1,
                current_time,
                d_afs
            )
            new_cell.cell_type = parent_cell.cell_type
            cells.append(new_cell)
            parent_cell.children.append(new_cell.id)
            cell_queue.append(new_cell)
            cid_counter += 1

            if len(cells) >= total_cells:
                break

    # Collect mutations
    all_mutations = set()
    for c in cells:
        all_mutations.update(c.mutation_afs.keys())

    return cells, sorted(all_mutations)

def run_simulation_with_segregation_model(
    segregation_model="biased",
    total_cells=1000,
    mutation_rate=4,
    num_baseline_mutations=5
):
    """
    Helper to run the entire pipeline but switch the mtDNA segregation model.
    `segregation_model` can be "biased" or "unbiased".
    """
    if segregation_model == "biased":
        print("[Scenario 2] Using biased mtDNA segregation...")
        cells, mutations = simulate_stem_cell_growth_biased(
            total_cells=total_cells,
            mutation_rate=mutation_rate,
            num_baseline_mutations=num_baseline_mutations,
            bias=0.7  # adjust as you like
        )
    else:
        print("[Scenario 2] Using unbiased mtDNA segregation...")
        cells, mutations = simulate_stem_cell_growth_unbiased(
            total_cells=total_cells,
            mutation_rate=mutation_rate,
            num_baseline_mutations=num_baseline_mutations
        )
    
    # Differentiate
    cells = simulate_cell_differentiation(cells)

    # Simulate sequencing
    for cell in cells:
        simulate_read_depth(cell, all_mutations=mutations)

    # Simulate gene expression
    gene_params, _ = generate_gene_params()
    expr_df = simulate_gene_expression_for_cells(cells, gene_params)
    
    return cells, mutations, expr_df

def scenario2_vary_segregation(
    models = ["unbiased", "biased"],
    total_cells=1000,
    mutation_rate=4,
    output_dir="./scenario2_results"
):
    """
    Compare how different segregation models (unbiased vs. biased) affect results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for model in models:
        print(f"\n[Scenario 2] Running simulation with segregation_model={model}")
        
        cells, mutations, expr_df = run_simulation_with_segregation_model(
            segregation_model=model,
            total_cells=total_cells,
            mutation_rate=mutation_rate
        )
        
        # Save DP/AD
        prefix = os.path.join(output_dir, f"segregation_{model}")
        export_mtx_for_dp_ad(cells, mutations, prefix=prefix)
        
        # Save expression
        expr_csv = prefix + "_expression.csv"
        expr_df.to_csv(expr_csv, index=True)
        
        # Visualize
        visualize_simulation_results(
            cells, mutations, expr_df,
            output_prefix=prefix,
            color_by='cell_type'
        )
        
        # AF analysis
        af_analysis = analyze_af_distribution(cells, mutations, prefix + "_af_analysis")
        
        results[model] = {
            'cells': cells,
            'mutations': mutations,
            'expr_df': expr_df,
            'af_analysis': af_analysis
        }
    
    return results


# ------------------------------------------------------------------------------
# SCENARIO 3: Linear vs. Bifurcated Differentiation Paths
# ------------------------------------------------------------------------------
# We define a simpler "linear" differentiation:
def simulate_cell_differentiation_linear(cells, kappa_diff=KAPPA_DIFF, t0_diff=T0_DIFF):
    """
    Example linear differentiation path:
    - StemCell => Progenitor => TerminalCell
    - No branching.
    
    For simplicity, we do this in a single pass:
        1) Some fraction of StemCells become Progenitors
        2) Progenitors do 1 division: 
           one daughter remains Progenitor, one becomes Terminal.
    """
    new_cells = []
    for c in cells:
        if c.cell_type == 'StemCell':
            diff_prob = differentiation_probability(c.time, kappa_diff, t0_diff)
            if np.random.rand() < diff_prob:
                c.cell_type = 'Progenitor'
        
        elif c.cell_type == 'Progenitor':
            # Asymmetric division: one remains Progenitor, one => Terminal
            daughters = progenitor_asymmetric_division_linear(c)
            new_cells.extend(daughters)
    
    return cells + new_cells

def progenitor_asymmetric_division_linear(parent_cell):
    """
    Linear version: 
      - One daughter: Progenitor
      - One daughter: TerminalCell
    """
    next_id = int(parent_cell.id.replace('cell_', '')) + 10000
    
    d1 = Cell(
        cid=next_id,
        parent_id=parent_cell.id.replace('cell_', ''),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time,
        cell_type='Progenitor'
    )
    d2 = Cell(
        cid=next_id + 1,
        parent_id=parent_cell.id.replace('cell_', ''),
        generation=parent_cell.generation + 1,
        time_point=parent_cell.time,
        cell_type='TerminalCell'
    )
    
    # Segregate mtDNA (unbiased for example)
    d1_afs, d2_afs = segregate_allele_frequencies(parent_cell.mutation_afs)
    d1.mutation_afs = d1_afs
    d2.mutation_afs = d2_afs
    
    parent_cell.children.extend([d1.id, d2.id])
    return [d1, d2]

def run_simulation_linear_path(total_cells=500, mutation_rate=4, num_baseline_mutations=3):
    """
    One example of a single-lane (linear) differentiation pipeline:
    StemCell => Progenitor => Terminal.
    """
    # 1) Grow (you can use either biased or unbiased; here use unbiased as example)
    cells, mutations = simulate_stem_cell_growth_unbiased(
        total_cells=total_cells,
        mutation_rate=mutation_rate,
        num_baseline_mutations=num_baseline_mutations
    )
    
    # 2) Linear differentiation
    cells = simulate_cell_differentiation_linear(cells)
    
    # 3) Sequencing coverage
    for cell in cells:
        simulate_read_depth(cell, mutations)
    
    # 4) Gene expression
    gene_params, _ = generate_gene_params()
    expr_df = simulate_gene_expression_for_cells(cells, gene_params)
    
    return cells, mutations, expr_df


def run_simulation_bifurcated_path(total_cells=500, mutation_rate=4, num_baseline_mutations=3):
    """
    Uses the existing multi-branch differentiation:
      StemCell => Progenitor1 or Progenitor2 => TerminalCell_A or TerminalCell_B
    """
    # 1) Grow
    cells, mutations = simulate_stem_cell_growth_unbiased(
        total_cells=total_cells,
        mutation_rate=mutation_rate,
        num_baseline_mutations=num_baseline_mutations
    )
    
    # 2) Multi-branch differentiation (already in your code)
    cells = simulate_cell_differentiation(cells)
    
    # 3) Sequencing coverage
    for cell in cells:
        simulate_read_depth(cell, mutations)
    
    # 4) Gene expression
    gene_params, _ = generate_gene_params()
    expr_df = simulate_gene_expression_for_cells(cells, gene_params)
    
    return cells, mutations, expr_df


def scenario3_differentiation_comparison(
    total_cells=500,
    mutation_rate=4,
    output_dir="./scenario3_results"
):
    """
    Compare a simple 'linear' differentiation path vs. the more complex 'bifurcated' path.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Linear path
    print("\n[Scenario 3] Running simulation with LINEAR differentiation path...")
    cells_linear, muts_linear, expr_linear = run_simulation_linear_path(
        total_cells=total_cells,
        mutation_rate=mutation_rate
    )
    prefix_linear = os.path.join(output_dir, "linear")
    export_mtx_for_dp_ad(cells_linear, muts_linear, prefix=prefix_linear)
    expr_linear.to_csv(prefix_linear + "_expression.csv", index=True)
    visualize_simulation_results(cells_linear, muts_linear, expr_linear, prefix_linear)
    af_analysis_linear = analyze_af_distribution(cells_linear, muts_linear, prefix_linear + "_af_analysis")
    
    results["linear"] = {
        'cells': cells_linear,
        'mutations': muts_linear,
        'expr_df': expr_linear,
        'af_analysis': af_analysis_linear
    }
    
    # Bifurcated path
    print("\n[Scenario 3] Running simulation with BIFURCATED differentiation path...")
    cells_bif, muts_bif, expr_bif = run_simulation_bifurcated_path(
        total_cells=total_cells,
        mutation_rate=mutation_rate
    )
    prefix_bif = os.path.join(output_dir, "bifurcated")
    export_mtx_for_dp_ad(cells_bif, muts_bif, prefix_bif)
    expr_bif.to_csv(prefix_bif + "_expression.csv", index=True)
    visualize_simulation_results(cells_bif, muts_bif, expr_bif, prefix_bif)
    af_analysis_bif = analyze_af_distribution(cells_bif, muts_bif, prefix_bif + "_af_analysis")
    
    results["bifurcated"] = {
        'cells': cells_bif,
        'mutations': muts_bif,
        'expr_df': expr_bif,
        'af_analysis': af_analysis_bif
    }
    
    return results


# ------------------------------------------------------------------------------
# Example driver code to run all three scenarios
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "/Users/linxy29/Documents/Data/CIVET/simulation/simulation_results/results_" + timestamp

    # SCENARIO 1
    scenario1_results = scenario1_mutation_rate_exploration(
        mutation_rates=[1, 2, 4, 8],
        total_cells=500,
        max_generations=50,
        output_dir= base_output_dir + "/scenario1"
    )

    # SCENARIO 2
    scenario2_results = scenario2_vary_segregation(
        models=["unbiased", "biased"],
        total_cells=500,
        mutation_rate=4,
        output_dir= base_output_dir + "/scenario2"
    )

    # SCENARIO 3
    scenario3_results = scenario3_differentiation_comparison(
        total_cells=500,
        mutation_rate=4,
        output_dir= base_output_dir + "/scenario3"
    )

    print("\nAll scenarios have completed!")
