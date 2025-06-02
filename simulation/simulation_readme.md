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
  - Exploring different mutation rate regimes
  - Testing varying mtDNA segregation models
  - Comparing linear vs. bifurcated differentiation paths
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