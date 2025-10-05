# Simulation Package Documentation

This document provides an overview of the simulation package for cell growth, differentiation, and mutation analysis.

## File Structure

### `__init__.py`
- Defines the simulation package and exports key functions
- Imports and exposes essential functions from other modules for external use
- Provides a clean API for users of this package

### `simulation_classes.py`
- Contains core class definitions for the simulation framework
- Defines the `Cell` class with attributes for tracking:
  - Cell ID and parent ID
  - Generation and time point
  - Mutation allele frequencies
  - Cell type and children
  - Mutation profile

### `simulation_config.yaml`
- Configuration file for simulation parameters
- Contains settings for:
  - Stem cell growth (division rate, acceleration)
  - mtDNA parameters (initial count, mutation rate)
  - Sequencing read depth parameters
  - Cell differentiation parameters
  - Gene expression parameters
  - Simulation scale settings

### `simulation_framework_af.py`
- Main simulation engine with allele frequency tracking
- Implements functions for:
  - Loading configurations
  - Stem cell growth and division
  - Mitochondrial genome dynamics
  - Cell differentiation
  - Mutation tracking and inheritance
  - Read depth simulation
  - Gene expression simulation
  - Running complete simulations

### `simulation_save.py`
- Functions for saving and visualizing simulation results
- Provides functionality to:
  - Export depth (DP) and allele depth (AD) matrices
  - Save gene expression data
  - Save cell metadata
  - Save gene metadata
  - Visualize mutations and gene expression
  - Analyze allele frequency distributions
  - Create comprehensive output directories

### `simulation_scenarios_af.py`
- Implements different simulation scenarios
- Contains predefined scenarios for:
  - **Scenario 1**: Exploring different mutation rate regimes
  - **Scenario 2**: Testing varying mtDNA segregation models
  - **Scenario 3**: Comparing linear vs. bifurcated differentiation paths
  - **Scenario 4**: Cell cycle stage effects on mtDNA mutations
  - **Scenario 5**: Metabolic state effects (OXPHOS vs glycolytic)
  - Command-line interface for running scenarios

### `simulation_checkcode_af.py`
- Unit tests for the simulation framework
- Tests various components:
  - Mitochondrial dynamics (mutation introduction, segregation)
  - Cell growth mechanisms
  - Read depth simulation
  - Analysis functions

### `simulation.ipynb`
- Jupyter notebook for interactive exploration
- Demonstrates usage of the simulation framework
- Contains examples and visualizations

## Usage

The simulation package can be used by importing functions from the package:

```python
from simulation import run_basic_simulation, load_config

# Load configuration
config = load_config("simulation_config.yaml")

# Run simulation
cells, mutations, expr_df, gene_params = run_basic_simulation(config=config)
```

To run predefined scenarios, use the command line interface:

```bash
python -m simulation.simulation_scenarios_af --scenario all
```

## Scenario Descriptions

### Scenario 1: Mutation Rate Regimes
Tests different mtDNA mutation rates per cell division (1, 2, 4, 8, 16). Explores how mutation burden accumulates with different mutagenesis rates.

### Scenario 2: mtDNA Segregation Bias
Tests symmetric (0.5) vs asymmetric (0.7, 0.9) segregation. Models biased inheritance of mutant mtDNA during cell division.

### Scenario 3: Differentiation Paths
Compares cell type specificity fractions (0.2, 0.4, 0.8). Tests linear vs bifurcated differentiation trajectories.

### Scenario 4: Cell Cycle Stage Effects
**Biology**: mtDNA copy number, mutation rate, and repair capacity vary with cell cycle phase.

**Model Parameters**:
- **Proliferation rates**: 0.3 (30% cycling), 0.5, 0.7, 0.9 (90% cycling)
- **Cell cycle phases**: G0 (quiescent), G1, S, G2, M
- **Phase-dependent effects**:
  - **G0 (quiescent)**: 1.5x mutation rate, 0.75x mtDNA copy (damage accumulation)
  - **G1**: 1.3x mutation rate, 1x mtDNA copy
  - **S**: 1.0x mutation rate, 1.5x mtDNA copy (replication)
  - **G2/M**: 0.7x mutation rate, 2x mtDNA copy (dilution effect)

**Expected outcomes**:
- Low proliferation → more G0 cells → higher mutation burden
- High proliferation → dilution of heteroplasmy in cycling cells

### Scenario 5: Metabolic State Effects
**Biology**: OXPHOS (oxidative phosphorylation) generates ROS, increasing mutation rate. Mitophagy removes damaged mitochondria.

**Model Parameters**:
- **Metabolic modes**:
  - `cell_type_dependent`: Stem=Glycolytic, Progenitor=OXPHOS_low, Terminal=OXPHOS_high
  - `stress`: 70% cells in OXPHOS_high (oxidative stress)
- **State-dependent effects**:
  - **OXPHOS_high**: 3x mutation rate, 1.5x mtDNA copy, oxidative mutation bias
  - **OXPHOS_low**: 1.5x mutation rate, 1x mtDNA copy
  - **Glycolytic**: 0.8x mutation rate, 0.75x mtDNA copy
- **Mitophagy**: ON/OFF toggle, removes mutations >70% AF in OXPHOS_high cells

**Expected outcomes**:
- OXPHOS_high cells → more mutations, enriched for oxidative lesions (C>T, A>G)
- Mitophagy ON → removes high-heteroplasmy mutations
- Cell-type pattern: stem cells protected (glycolytic), terminal cells damaged (OXPHOS)

## Running Specific Scenarios

```bash
# Run only scenario 4 (cell cycle)
python -m simulation.simulation_scenarios_af --scenario 4

# Run only scenario 5 (metabolic)
python -m simulation.simulation_scenarios_af --scenario 5

# Run all scenarios (1-5)
python -m simulation.simulation_scenarios_af --scenario all
```

## Using New Functions in Python

```python
from simulation import run_cell_cycle_simulation, run_metabolic_simulation, load_config

# Load configuration
config = load_config("simulation_config.yaml")

# Run cell cycle simulation with 30% proliferation
cells, mutations, expr_df, gene_params = run_cell_cycle_simulation(
    config=config,
    proliferation_rate=0.3
)

# Run metabolic simulation with stress condition and mitophagy
cells, mutations, expr_df, gene_params = run_metabolic_simulation(
    config=config,
    metabolic_mode='stress',
    apply_mitophagy=True
)
```

---

# Detailed Documentation: New Scenarios 4 & 5

## Overview

Two biologically-motivated scenarios model how **cell cycle stage** and **metabolic state** influence mtDNA mutation dynamics in single cells.

---

## Scenario 4: Cell Cycle Stage Effects - Detailed Design

### Biological Motivation

mtDNA dynamics vary dramatically across the cell cycle:
- **Copy number**: G1 (~200) → S (~300, replication) → G2 (~400, pre-division)
- **Mutation accumulation**: Quiescent (G0) cells accumulate damage due to reduced repair
- **Heteroplasmy dynamics**: Rapid division dilutes mutant mtDNA; arrested cells retain it

### Implementation Design

**Cell Attributes Added:**
- `cell_cycle_phase`: 'G0' (quiescent), 'G1', 'S', 'G2', 'M'

**Phase-Dependent Parameters:**

| Phase | Mutation Rate | mtDNA Copy Number | Biology |
|-------|--------------|-------------------|---------|
| G0 (quiescent) | 1.5× | 0.75× (150) | Damage accumulation, low repair |
| G1 | 1.3× | 1.0× (200) | Moderate damage, normal copy |
| S | 1.0× | 1.5× (300) | Active replication, baseline rate |
| G2 | 0.7× | 2.0× (400) | Dilution effect, high copy |
| M | 0.7× | 2.0× (400) | Division, dilution effect |

**Simulation Variants:**
- **Proliferation rate**: 0.3, 0.5, 0.7, 0.9 (fraction of cycling cells)
- Low proliferation → more G0 cells → higher mutation burden
- High proliferation → dilution of heteroplasmy

**Key Functions:**
```python
# In simulation_framework_af.py
assign_cell_cycle_phase(cells, proliferation_rate=0.7)
get_cell_cycle_params(phase, base_mutation_rate, base_mtdna_count)
run_cell_cycle_simulation(config, proliferation_rate=0.7)
```

### Expected Biological Outcomes

1. **Low proliferation (30% cycling)**:
   - 70% cells in G0 → accumulated mutations
   - Higher overall mutation burden
   - Less heteroplasmy dilution

2. **High proliferation (90% cycling)**:
   - Most cells in active cycle (G1/S/G2)
   - Lower mutation burden per cell
   - Strong dilution of heteroplasmy in G2/M cells

---

## Scenario 5: Metabolic State Effects - Detailed Design

### Biological Motivation

Metabolic activity drives ROS production and mtDNA damage:
- **OXPHOS (oxidative phosphorylation)**: High ROS → 3× mutation rate → oxidative lesions (C>T, A>G)
- **Glycolysis**: Low ROS → baseline mutation rate
- **Mitophagy**: Quality control removes mitochondria with >70% heteroplasmy

### Implementation Design

**Cell Attributes Added:**
- `metabolic_state`: 'OXPHOS_high', 'OXPHOS_low', 'Glycolytic'

**State-Dependent Parameters:**

| State | Mutation Rate | mtDNA Copy | Mutation Bias | Biology |
|-------|--------------|------------|---------------|---------|
| OXPHOS_high | 3.0× | 1.5× (300) | Oxidative (60% C>T/A>G) | High ROS, biogenesis |
| OXPHOS_low | 1.5× | 1.0× (200) | Oxidative (60% C>T/A>G) | Moderate ROS |
| Glycolytic | 0.8× | 0.75× (150) | Normal | Low ROS, low copy |

**Simulation Modes:**

1. **Cell-type-dependent** (physiological):
   - StemCell → Glycolytic (protected)
   - Progenitor1/2 → OXPHOS_low
   - TerminalCell_A/B → OXPHOS_high (damaged)

2. **Stress** (oxidative stress):
   - 70% cells → OXPHOS_high
   - 20% cells → OXPHOS_low
   - 10% cells → Glycolytic

3. **Mitophagy toggle**:
   - ON: Removes mutations >70% AF in OXPHOS_high cells (80% probability)
   - OFF: No quality control

**Oxidative Mutation Spectrum:**
- Tag oxidative mutations as `ox_m{position}`
- 60% of mutations in OXPHOS cells are oxidative type
- Models C>T and A>G transitions from ROS damage

**Key Functions:**
```python
# In simulation_framework_af.py
assign_metabolic_state(cells, state_mode='cell_type_dependent')
get_metabolic_params(state, base_mutation_rate, base_mtdna_count)
introduce_metabolic_mutations(mutation_rate, mutation_bias='oxidative')
simulate_mitophagy(cell, threshold=0.7)
run_metabolic_simulation(config, metabolic_mode='cell_type_dependent', apply_mitophagy=True)
```

### Expected Biological Outcomes

1. **Cell-type-dependent mode**:
   - Stem cells: Low mutation burden (glycolytic protection)
   - Terminal cells: High mutation burden (OXPHOS damage)
   - Oxidative mutation enrichment in differentiated cells

2. **Stress condition**:
   - Global increase in mutation burden
   - Strong oxidative mutation signature
   - Mitophagy effect: ~80% reduction in high-AF mutations

3. **Mitophagy effect**:
   - Removes ~80% of mutations >70% AF
   - Prevents clonal expansion of highly mutant genomes
   - Maintains mitochondrial quality control

---

## File Modifications

### 1. `simulation_classes.py`
- Added `cell_cycle_phase` and `metabolic_state` attributes to Cell class
- Updated `__slots__` and `__init__` method

### 2. `simulation_framework_af.py`
- Added 9 new functions (lines 1041-1444):
  - Cell cycle: `assign_cell_cycle_phase`, `get_cell_cycle_params`, `run_cell_cycle_simulation`
  - Metabolic: `assign_metabolic_state`, `get_metabolic_params`, `introduce_metabolic_mutations`, `simulate_mitophagy`, `run_metabolic_simulation`

### 3. `simulation_scenarios_af.py`
- Added `scenario_4_cell_cycle()` (lines 141-177)
- Added `scenario_5_metabolic_state()` (lines 184-227)
- Updated main entry point to handle scenarios 4 and 5

### 4. `simulation_save.py`
- Updated `save_cell_metadata()` to export `cell_cycle_phase` and `metabolic_state` columns

### 5. `simulation/__init__.py`
- Exported new functions: `run_cell_cycle_simulation`, `run_metabolic_simulation`

---

## Output Data

### Metadata Files

Cell metadata now includes:
- `cell_cycle_phase`: G0, G1, S, G2, or M
- `metabolic_state`: OXPHOS_high, OXPHOS_low, or Glycolytic

Example output directory:
```
SCENARIO_4_CellCycle/proliferation_0.3_20250105_120000/
├── metadata/
│   └── simulation_cell_metadata.csv  # Includes cell_cycle_phase
├── cellSNP/
│   ├── cellSNP.tag.AD.mtx
│   └── cellSNP.tag.DP.mtx
└── figures/
    └── simulation_heatmap.png

SCENARIO_5_Metabolic/cell_type_dependent_mitophagy_on_20250105_120000/
├── metadata/
│   └── simulation_cell_metadata.csv  # Includes metabolic_state
│   └── simulation_mutation_info.csv  # Flags oxidative mutations (ox_m*)
└── ...
```

### Mutation Tags

- **Baseline**: `baseline_m{pos}` (inherited from root)
- **De novo**: `m{pos}` (acquired during simulation)
- **Oxidative**: `ox_m{pos}` (acquired in OXPHOS state, 60% in OXPHOS cells)
- **False**: `false_m{pos}` (technical artifacts)

---

## Biological Insights Enabled

### 1. Cell Cycle Effects
- **Question**: Does quiescence protect or damage mtDNA?
- **Answer**: G0 cells accumulate 1.5× more mutations (reduced repair)
- **Application**: Stem cell biology, aging, dormant cancer cells

### 2. Metabolic Effects
- **Question**: How does OXPHOS activity affect mutation spectrum?
- **Answer**: 3× mutation rate + oxidative bias (C>T/A>G) in high OXPHOS
- **Application**: Cancer metabolism, mitochondrial disease, metabolic stress

### 3. Quality Control
- **Question**: Does mitophagy prevent mutant expansion?
- **Answer**: Removes ~80% of high-AF mutations in stressed cells
- **Application**: Aging, mitochondrial quality control, selective pressure

### 4. Cell Type Hierarchy
- **Question**: Are stem cells protected from mtDNA damage?
- **Answer**: Yes - glycolytic metabolism → 0.8× mutation rate vs 3× in terminal cells
- **Application**: Stem cell maintenance, lineage tracing, clonal dynamics

---

## Testing

Run the comprehensive test script:
```bash
python test_new_scenarios.py
```

This will:
1. Run cell cycle simulations (low vs high proliferation)
2. Run metabolic simulations (cell-type vs stress, +/- mitophagy)
3. Compare mutation burden across conditions
4. Generate visualization plots

---

## Validation Suggestions

1. **Cell cycle validation**:
   - Compare G0 vs S-phase mutation burden → expect 1.5× difference
   - Check mtDNA copy number correlates with DP values → G2 should have 2× DP vs G1

2. **Metabolic validation**:
   - Count oxidative mutations (`ox_m*`) in OXPHOS vs Glycolytic cells → expect 3× enrichment
   - Verify mitophagy effect → high-AF mutations should be depleted in OXPHOS_high cells

3. **Biological consistency**:
   - Stem cells should have lowest mutation burden (glycolytic + low proliferation)
   - Terminal cells should have highest burden (OXPHOS + accumulated damage) 