"""
Unit tests for simulation_framework_af.py
"""

import unittest
import numpy as np
import os
import pandas as pd
from simulation_framework_af import (
    Cell, introduce_new_mutations, segregate_allele_frequencies,
    initialize_baseline_mutations, simulate_stem_cell_growth,
    simulate_read_depth, division_rate, analyze_af_distribution
)

class TestMitochondrialDynamics(unittest.TestCase):
    def test_introduce_new_mutations(self):
        """Test new mutation introduction with allele frequencies"""
        # Test with high mutation rate for better testing
        mutation_rate = 5.0
        new_mutations = introduce_new_mutations(mutation_rate)
        
        # Check results
        self.assertIsInstance(new_mutations, dict)
        if new_mutations:  # Some mutations should be introduced with high rate
            # Check mutation format and values
            for mut_id, af in new_mutations.items():
                # Check mutation ID format
                self.assertTrue(mut_id.startswith('m'))
                position = int(mut_id[1:])
                self.assertTrue(1 <= position <= 16569)
                
                # Check AF range (should be between 0.01% and 1%)
                self.assertTrue(0.0001 <= af <= 0.01)
    
    def test_segregate_allele_frequencies(self):
        """Test allele frequency segregation during cell division"""
        # Create test AFs
        test_afs = {
            'm1': 0.5,    # High AF
            'm2': 0.01,   # Low AF
            'm3': 0.25    # Medium AF
        }
        
        # Run segregation multiple times to account for randomness
        num_trials = 10
        avg_differences = []
        
        for _ in range(num_trials):
            d1_afs, d2_afs = segregate_allele_frequencies(test_afs)
            
            # Basic checks
            for afs in [d1_afs, d2_afs]:
                self.assertIsInstance(afs, dict)
                for mut, af in afs.items():
                    self.assertTrue(0 <= af <= 1)
                    self.assertTrue(af >= 0.0001)
            
            # Check mutations present in both daughters
            common_muts = set(d1_afs.keys()) & set(d2_afs.keys())
            for mut in common_muts:
                # AFs should be different
                self.assertNotEqual(d1_afs[mut], d2_afs[mut])
                
                # Calculate average AF
                avg_af = (d1_afs[mut] + d2_afs[mut]) / 2
                avg_differences.append(abs(avg_af - test_afs[mut]))
        
        # Check that on average, the segregation maintains AF within reasonable bounds
        avg_difference = np.mean(avg_differences)
        self.assertLess(avg_difference, 0.4, 
                        "Average AF difference too large across multiple segregations")

    def test_initialize_baseline_mutations(self):
        """Test baseline mutation initialization"""
        num_baseline = 5
        baseline_muts = initialize_baseline_mutations(num_baseline)
        
        # Check number of mutations
        self.assertEqual(len(baseline_muts), num_baseline)
        
        # Check mutation format and AF ranges
        for mut_id, af in baseline_muts.items():
            # Check mutation ID format
            self.assertTrue(mut_id.startswith('baseline_m'))
            position = int(mut_id[len('baseline_m'):])
            self.assertTrue(1 <= position <= 16569)
            
            # Check AF range (should be between 0.1% and 50%)
            self.assertTrue(0.001 <= af <= 0.5)

class TestCellGrowth(unittest.TestCase):
    def test_stem_cell_growth_with_mutations(self):
        """Test stem cell growth with AF-based mutation tracking"""
        total_cells = 10
        mutation_rate = 2.0  # High rate for testing
        num_baseline = 3
        
        # Run simulation
        cells, mutations = simulate_stem_cell_growth(
            total_cells=total_cells,
            mutation_rate=mutation_rate,
            num_baseline_mutations=num_baseline
        )
        
        # Check basic cell properties
        self.assertLessEqual(len(cells), total_cells + 1)
        self.assertTrue(all(isinstance(c, Cell) for c in cells))
        
        # Check mutation properties
        baseline_muts = [m for m in mutations if m.startswith('baseline_m')]
        denovo_muts = [m for m in mutations if not m.startswith('baseline_m')]
        
        # Check baseline mutations
        self.assertEqual(len(baseline_muts), num_baseline)
        
        # Check mutation inheritance and AF properties
        for cell in cells:
            # Check mutation_afs structure
            self.assertIsInstance(cell.mutation_afs, dict)
            
            # All cells should have some baseline mutations
            baseline_in_cell = [m for m in cell.mutation_afs if m.startswith('baseline_m')]
            self.assertTrue(len(baseline_in_cell) > 0)
            
            # Check AF ranges
            for mut, af in cell.mutation_afs.items():
                self.assertTrue(0 <= af <= 1)
                self.assertTrue(af >= 0.0001)  # Above detection threshold
                
                # Check appropriate AF ranges
                if mut.startswith('baseline_m'):
                    self.assertTrue(af <= 1.0)  # Allow full range for baseline mutations

    def test_read_depth_simulation(self):
        """Test read depth simulation with allele frequencies"""
        # Create a test cell with known mutations
        cell = Cell(cid=1, parent_id=None, generation=0, time_point=0)
        cell.mutation_afs = {
            'baseline_m1': 0.5,    # High AF
            'm2': 0.01             # Low AF
        }
        
        # Run read depth simulation
        mutations = ['baseline_m1', 'm2']
        simulate_read_depth(cell, mutations, mean_dp=100)
        
        # Check results
        for mut in mutations:
            self.assertIn(mut, cell.mutation_profile)
            profile = cell.mutation_profile[mut]
            
            # Check DP and AD values
            self.assertTrue(profile['DP'] > 0)
            self.assertTrue(0 <= profile['AD'] <= profile['DP'])
            
            # Calculate observed AF
            obs_af = profile['AD'] / profile['DP']
            true_af = cell.mutation_afs[mut]
            
            # Observed AF should be roughly similar to true AF
            self.assertAlmostEqual(obs_af, true_af, delta=0.2)

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_output_dir = "test_output"
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_af_distribution_analysis(self):
        """Test allele frequency distribution analysis"""
        # Create test cells with known mutations
        cells = []
        mutations = ['baseline_m1', 'baseline_m2', 'm3', 'm4']
        
        for i in range(5):
            cell = Cell(cid=i, parent_id=None, generation=0, time_point=0)
            # First set mutation_afs
            cell.mutation_afs = {
                'baseline_m1': 0.5,
                'baseline_m2': 0.3,
                'm3': 0.01,
                'm4': 0.005
            }
            # Then initialize mutation_profile
            cell.mutation_profile = {}
            for mut in mutations:
                # Simulate DP around 100
                dp = np.random.negative_binomial(n=5, p=0.05)
                af = cell.mutation_afs[mut]
                ad = int(dp * af)  # Calculate AD based on AF
                cell.mutation_profile[mut] = {'DP': dp, 'AD': ad}
            cells.append(cell)
        
        # Run analysis
        output_prefix = os.path.join(self.test_output_dir, "af_test")
        results = analyze_af_distribution(cells, mutations, output_prefix)
        
        # Check results structure
        self.assertIn('baseline_summary', results)
        self.assertIn('denovo_summary', results)
        self.assertIn('cell_type_afs', results)
        
        # Check baseline mutations
        self.assertTrue(results['baseline_summary']['mean'] > results['denovo_summary']['mean'])
        self.assertEqual(results['baseline_summary']['count'], 10)  # 2 mutations × 5 cells
        
        # Check output files
        self.assertTrue(os.path.exists(f"{output_prefix}_baseline_vs_denovo_af.png"))
        self.assertTrue(os.path.exists(f"{output_prefix}_celltype_comparison.png"))
        self.assertTrue(os.path.exists(f"{output_prefix}_mutation_type_data.csv"))

if __name__ == '__main__':
    unittest.main() 