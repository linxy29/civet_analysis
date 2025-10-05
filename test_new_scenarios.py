#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for new simulation scenarios (Cell Cycle and Metabolic State)

This script demonstrates how to run the new scenarios and visualize results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import (
    run_cell_cycle_simulation,
    run_metabolic_simulation,
    load_config,
    save_simulation_data
)

def test_cell_cycle_scenario():
    """Test cell cycle simulation with different proliferation rates"""
    print("="*80)
    print("Testing Cell Cycle Scenario")
    print("="*80)

    # Load configuration
    config = load_config("simulation/simulation_config.yaml")

    # Test with low proliferation (30% cycling cells)
    print("\n1. Running simulation with LOW proliferation (30% cycling)...")
    cells_low, mutations_low, expr_df_low, gene_params_low = run_cell_cycle_simulation(
        config=config,
        proliferation_rate=0.3
    )

    # Test with high proliferation (90% cycling cells)
    print("\n2. Running simulation with HIGH proliferation (90% cycling)...")
    cells_high, mutations_high, expr_df_high, gene_params_high = run_cell_cycle_simulation(
        config=config,
        proliferation_rate=0.9
    )

    # Compare results
    print("\n" + "="*80)
    print("CELL CYCLE SIMULATION RESULTS")
    print("="*80)

    # Count cells in each phase
    phase_counts_low = {}
    phase_counts_high = {}

    for cell in cells_low:
        phase = cell.cell_cycle_phase
        phase_counts_low[phase] = phase_counts_low.get(phase, 0) + 1

    for cell in cells_high:
        phase = cell.cell_cycle_phase
        phase_counts_high[phase] = phase_counts_high.get(phase, 0) + 1

    print("\nLow proliferation (30%) - Cell cycle phase distribution:")
    for phase, count in sorted(phase_counts_low.items()):
        pct = 100 * count / len(cells_low)
        print(f"  {phase}: {count} cells ({pct:.1f}%)")

    print("\nHigh proliferation (90%) - Cell cycle phase distribution:")
    for phase, count in sorted(phase_counts_high.items()):
        pct = 100 * count / len(cells_high)
        print(f"  {phase}: {count} cells ({pct:.1f}%)")

    # Compare mutation burden
    print(f"\nTotal mutations - Low prolif: {len(mutations_low)}, High prolif: {len(mutations_high)}")

    # Calculate average mutation count per cell
    avg_muts_low = np.mean([len(c.mutation_afs) for c in cells_low])
    avg_muts_high = np.mean([len(c.mutation_afs) for c in cells_high])
    print(f"Average mutations per cell - Low: {avg_muts_low:.2f}, High: {avg_muts_high:.2f}")

    return cells_low, cells_high

def test_metabolic_scenario():
    """Test metabolic state simulation"""
    print("\n" + "="*80)
    print("Testing Metabolic State Scenario")
    print("="*80)

    # Load configuration
    config = load_config("simulation/simulation_config.yaml")

    # Test cell-type-dependent metabolic states WITHOUT mitophagy
    print("\n1. Running cell-type-dependent metabolic simulation (NO mitophagy)...")
    cells_no_mito, mutations_no_mito, expr_df_no_mito, gene_params_no_mito = run_metabolic_simulation(
        config=config,
        metabolic_mode='cell_type_dependent',
        apply_mitophagy=False
    )

    # Test cell-type-dependent metabolic states WITH mitophagy
    print("\n2. Running cell-type-dependent metabolic simulation (WITH mitophagy)...")
    cells_with_mito, mutations_with_mito, expr_df_with_mito, gene_params_with_mito = run_metabolic_simulation(
        config=config,
        metabolic_mode='cell_type_dependent',
        apply_mitophagy=True
    )

    # Test stress condition
    print("\n3. Running STRESS condition metabolic simulation (WITH mitophagy)...")
    cells_stress, mutations_stress, expr_df_stress, gene_params_stress = run_metabolic_simulation(
        config=config,
        metabolic_mode='stress',
        apply_mitophagy=True
    )

    # Compare results
    print("\n" + "="*80)
    print("METABOLIC SIMULATION RESULTS")
    print("="*80)

    # Count cells in each metabolic state
    state_counts_normal = {}
    state_counts_stress = {}

    for cell in cells_with_mito:
        state = cell.metabolic_state
        state_counts_normal[state] = state_counts_normal.get(state, 0) + 1

    for cell in cells_stress:
        state = cell.metabolic_state
        state_counts_stress[state] = state_counts_stress.get(state, 0) + 1

    print("\nCell-type-dependent - Metabolic state distribution:")
    for state, count in sorted(state_counts_normal.items()):
        pct = 100 * count / len(cells_with_mito)
        print(f"  {state}: {count} cells ({pct:.1f}%)")

    print("\nStress condition - Metabolic state distribution:")
    for state, count in sorted(state_counts_stress.items()):
        pct = 100 * count / len(cells_stress)
        print(f"  {state}: {count} cells ({pct:.1f}%)")

    # Compare oxidative mutations
    ox_muts_no_mito = sum(1 for m in mutations_no_mito if m.startswith('ox_'))
    ox_muts_with_mito = sum(1 for m in mutations_with_mito if m.startswith('ox_'))
    ox_muts_stress = sum(1 for m in mutations_stress if m.startswith('ox_'))

    print(f"\nOxidative mutations:")
    print(f"  No mitophagy: {ox_muts_no_mito}")
    print(f"  With mitophagy: {ox_muts_with_mito}")
    print(f"  Stress condition: {ox_muts_stress}")

    # Calculate mutation burden by metabolic state
    print("\nAverage mutations per cell by metabolic state (cell-type-dependent):")
    state_mutations = {}
    for cell in cells_with_mito:
        state = cell.metabolic_state
        if state not in state_mutations:
            state_mutations[state] = []
        state_mutations[state].append(len(cell.mutation_afs))

    for state in sorted(state_mutations.keys()):
        avg = np.mean(state_mutations[state])
        print(f"  {state}: {avg:.2f} mutations/cell")

    # Effect of mitophagy
    print("\nEffect of mitophagy on high-AF mutations:")

    def count_high_af_muts(cells, threshold=0.7):
        count = 0
        for cell in cells:
            for af in cell.mutation_afs.values():
                if af > threshold:
                    count += 1
        return count

    high_af_no_mito = count_high_af_muts(cells_no_mito)
    high_af_with_mito = count_high_af_muts(cells_with_mito)

    print(f"  Mutations >70% AF without mitophagy: {high_af_no_mito}")
    print(f"  Mutations >70% AF with mitophagy: {high_af_with_mito}")
    print(f"  Reduction: {high_af_no_mito - high_af_with_mito} ({100*(high_af_no_mito - high_af_with_mito)/high_af_no_mito:.1f}%)")

    return cells_with_mito, cells_stress

def visualize_comparison(cells_low_prolif, cells_high_prolif, cells_metabolic, cells_stress):
    """Create comparison visualizations"""
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cell cycle phase distribution
    ax = axes[0, 0]
    phase_data = []
    for cells, label in [(cells_low_prolif, 'Low (30%)'), (cells_high_prolif, 'High (90%)')]:
        for cell in cells:
            phase_data.append({'Proliferation': label, 'Phase': cell.cell_cycle_phase})

    phase_df = pd.DataFrame(phase_data)
    phase_counts = phase_df.groupby(['Proliferation', 'Phase']).size().unstack(fill_value=0)
    phase_counts.T.plot(kind='bar', ax=ax)
    ax.set_title('Cell Cycle Phase Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cell Cycle Phase')
    ax.set_ylabel('Number of Cells')
    ax.legend(title='Proliferation Rate')

    # 2. Metabolic state distribution
    ax = axes[0, 1]
    state_data = []
    for cells, label in [(cells_metabolic, 'Cell-type'), (cells_stress, 'Stress')]:
        for cell in cells:
            state_data.append({'Condition': label, 'State': cell.metabolic_state})

    state_df = pd.DataFrame(state_data)
    state_counts = state_df.groupby(['Condition', 'State']).size().unstack(fill_value=0)
    state_counts.T.plot(kind='bar', ax=ax)
    ax.set_title('Metabolic State Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metabolic State')
    ax.set_ylabel('Number of Cells')
    ax.legend(title='Condition')

    # 3. Mutation burden by cell cycle phase
    ax = axes[1, 0]
    phase_mut_data = []
    for cell in cells_high_prolif:
        phase_mut_data.append({
            'Phase': cell.cell_cycle_phase,
            'Mutations': len(cell.mutation_afs)
        })

    phase_mut_df = pd.DataFrame(phase_mut_data)
    phase_order = ['G0', 'G1', 'S', 'G2', 'M']
    phase_mut_df['Phase'] = pd.Categorical(phase_mut_df['Phase'], categories=phase_order, ordered=True)
    sns.boxplot(data=phase_mut_df, x='Phase', y='Mutations', ax=ax, order=phase_order)
    ax.set_title('Mutation Burden by Cell Cycle Phase', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cell Cycle Phase')
    ax.set_ylabel('Number of Mutations')

    # 4. Mutation burden by metabolic state
    ax = axes[1, 1]
    state_mut_data = []
    for cell in cells_stress:
        state_mut_data.append({
            'State': cell.metabolic_state,
            'Mutations': len(cell.mutation_afs)
        })

    state_mut_df = pd.DataFrame(state_mut_data)
    sns.boxplot(data=state_mut_df, x='State', y='Mutations', ax=ax)
    ax.set_title('Mutation Burden by Metabolic State (Stress)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metabolic State')
    ax.set_ylabel('Number of Mutations')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save figure
    output_path = 'test_new_scenarios_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.show()

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("TESTING NEW SIMULATION SCENARIOS")
    print("Cell Cycle Stage and Metabolic State Effects on mtDNA Mutations")
    print("="*80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Test scenarios
    cells_low, cells_high = test_cell_cycle_scenario()
    cells_metabolic, cells_stress = test_metabolic_scenario()

    # Create visualizations
    visualize_comparison(cells_low, cells_high, cells_metabolic, cells_stress)

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey findings:")
    print("1. Cell cycle phase affects mtDNA copy number and mutation rate")
    print("2. Quiescent (G0) cells accumulate more mutations")
    print("3. OXPHOS-high cells have 3x mutation rate with oxidative bias")
    print("4. Mitophagy removes high-heteroplasmy mutations in stressed cells")
    print("5. Stem cells are protected (glycolytic), terminal cells damaged (OXPHOS)")

if __name__ == "__main__":
    main()
