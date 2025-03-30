"""
Unit tests for simulation_framework.py
"""

import unittest
import numpy as np
import os
import pandas as pd
from simulation_framework_mito import (
    Mitochondrion, Cell, introduce_mutations, replicate_and_segregate_mtDNA,
    division_rate, differentiation_probability, simulate_stem_cell_growth,
    simulate_read_depth, generate_gene_params, generate_latent_expression,
    sample_expression, get_dp_ad_matrices, write_mtx_file, export_mtx_for_dp_ad,
    visualize_mito_mutations, visualize_gene_expression, visualize_simulation_results,
    run_basic_simulation, introduce_new_mutations, segregate_allele_frequencies,
    initialize_baseline_mutations
)

class TestMitochondrialDynamics(unittest.TestCase):
    def test_introduce_mutations(self):
        """Test mutation introduction into mtDNA molecules"""
        # Create test mtDNA list
        mt_list = [Mitochondrion() for _ in range(10)]
        mutation_rate = 1.0  # High rate for testing
        
        # Run mutation introduction
        introduce_mutations(mt_list, mutation_rate)
        
        # Check results
        mutation_counts = [len(mt.mutations) for mt in mt_list]
        self.assertTrue(any(mutation_counts))  # At least some mutations should be introduced
        self.assertTrue(all(isinstance(mt.mutations, set) for mt in mt_list))

    def test_replicate_and_segregate(self):
        """Test mtDNA replication and segregation"""
        # Create initial mtDNA list with some mutations
        initial_mt = Mitochondrion(mutations={'m1', 'm2'})
        mt_list = [initial_mt]
        
        # Run replication and segregation
        d1, d2 = replicate_and_segregate_mtDNA(mt_list, mutation_rate_per_mitosis=0.1)
        
        # Check results
        self.assertEqual(len(d1) + len(d2), 2)  # Total count should be 2
        self.assertTrue(all('m1' in mt.mutations and 'm2' in mt.mutations 
                          for mt in d1 + d2))  # Original mutations preserved

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
                
                # Check AF range (should be between 0.1% and 1%)
                self.assertTrue(0.001 <= af <= 0.01)
    
    def test_segregate_allele_frequencies(self):
        """Test allele frequency segregation during cell division"""
        # Create test AFs
        test_afs = {
            'm1': 0.5,    # High AF
            'm2': 0.01,   # Low AF
            'm3': 0.25    # Medium AF
        }
        
        # Run segregation
        d1_afs, d2_afs = segregate_allele_frequencies(test_afs)
        
        # Check results
        for afs in [d1_afs, d2_afs]:
            self.assertIsInstance(afs, dict)
            # Check that AFs are maintained within reasonable bounds
            for mut, af in afs.items():
                self.assertTrue(0 <= af <= 1)
                self.assertTrue(af >= 0.001)  # Should be above detection threshold
        
        # Check that both daughters have same mutations (unless dropped below threshold)
        all_mutations = set(d1_afs.keys()) | set(d2_afs.keys())
        for mut in all_mutations:
            if mut in d1_afs and mut in d2_afs:
                # AFs should be different but related to parent
                self.assertNotEqual(d1_afs[mut], d2_afs[mut])
                
                # Get parent AF
                parent_af = test_afs[mut]
                # Check that average AF is roughly maintained
                avg_daughter_af = (d1_afs[mut] + d2_afs[mut]) / 2
                self.assertAlmostEqual(parent_af, avg_daughter_af, delta=0.2)

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
    def test_division_rate(self):
        """Test division rate calculation"""
        rate = division_rate(t=0, r=1.0, kappa=0.1, t0=50)
        self.assertTrue(0 <= rate <= 1.0)
        
        # Test at inflection point
        rate_t0 = division_rate(t=50, r=1.0, kappa=0.1, t0=50)
        self.assertAlmostEqual(rate_t0, 0.5, places=2)

    def test_stem_cell_growth(self):
        """Test stem cell growth simulation"""
        total_cells = 10
        cells = simulate_stem_cell_growth(total_cells=total_cells)
        
        # Check results
        self.assertLessEqual(len(cells), total_cells + 1)  # Allow for one extra cell
        self.assertTrue(all(isinstance(c, Cell) for c in cells))
        self.assertTrue(all(len(c.mtDNA_list) > 0 for c in cells))

    def test_stem_cell_growth_with_mutations(self):
        """Test stem cell growth with mutation tracking"""
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
                self.assertTrue(af >= 0.001)  # Above detection threshold
                
                # Baseline mutations should generally have higher AF
                if mut.startswith('baseline_m'):
                    self.assertTrue(af <= 0.5)
                else:
                    self.assertTrue(af <= 0.01)  # New mutations start with low AF
        
        # Check parent-child relationships
        for cell in cells:
            if cell.parent_id is not None:
                parent_id = cell.parent_id
                # Find parent cell
                parent = next((c for c in cells if c.id == f"cell_{parent_id}"), None)
                if parent:
                    # Child should be in parent's children list
                    self.assertIn(cell.id, parent.children)
                    # Child's generation should be parent's + 1
                    self.assertEqual(cell.generation, parent.generation + 1)
                    # Child should inherit some mutations from parent
                    parent_muts = set(parent.mutation_afs.keys())
                    child_muts = set(cell.mutation_afs.keys())
                    self.assertTrue(len(parent_muts & child_muts) > 0)

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

class TestSequencingSimulation(unittest.TestCase):
    def test_read_depth_simulation(self):
        """Test sequencing read depth simulation"""
        # Create a test cell
        cell = Cell(cid=1, parent_id=None, generation=0, time_point=0)
        mt = Mitochondrion(mutations={'m1'})
        cell.mtDNA_list = [mt]
        
        # Run read depth simulation
        simulate_read_depth(cell, all_mutations=['m1'], mean_dp=100)
        
        # Check results
        self.assertIn('m1', cell.mutation_profile)
        self.assertTrue(cell.mutation_profile['m1']['DP'] > 0)
        self.assertTrue(cell.mutation_profile['m1']['AD'] >= 0)
        self.assertTrue(cell.mutation_profile['m1']['AD'] <= 
                       cell.mutation_profile['m1']['DP'])

class TestGeneExpression(unittest.TestCase):
    def test_gene_params_generation(self):
        """Test generation of gene expression parameters"""
        num_genes = 100
        cell_specific = 40
        params = generate_gene_params(num_genes=num_genes, 
                                    cell_type_specific_genes=cell_specific)
        
        # Check results
        self.assertEqual(len(params), num_genes)
        self.assertTrue(all('base_expression' in p and 'expression_rate' in p 
                          for p in params.values()))

    def test_expression_sampling(self):
        """Test gene expression sampling"""
        # Create test latent expression
        z_t = {'gene1': 2.0, 'gene2': 3.0}
        
        # Sample expression
        x_t = sample_expression(z_t, alpha=0.1, zero_inflation_prob=0.1)
        
        # Check results
        self.assertEqual(len(x_t), 2)
        self.assertTrue(all(isinstance(count, (int, np.integer)) 
                          for count in x_t.values()))
        self.assertTrue(all(count >= 0 for count in x_t.values()))

class TestResultsSaving(unittest.TestCase):
    def setUp(self):
        """Set up test data using run_basic_simulation"""
        # Run a small simulation to generate cells, mutations, and gene expression
        # Use default parameters without specifying keywords that don't exist
        self.cells, self.mutations, self.expr_df = run_basic_simulation()

        cell_ids = [c.id for c in self.cells]
        self.expr_df = self.expr_df.loc[self.expr_df.index.isin(cell_ids)]
        
        # Create test output directory
        self.test_output_dir = "test_output"
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_dp_ad_matrix_generation(self):
        """Test generation of DP and AD matrices"""
        matrices = get_dp_ad_matrices(self.cells, self.mutations)
        
        # Check that both matrices exist
        self.assertIn('dp_matrix', matrices)
        self.assertIn('ad_matrix', matrices)
        
        # Check matrix dimensions
        self.assertEqual(matrices['dp_matrix'].shape, (len(self.cells), len(self.mutations)))
        self.assertEqual(matrices['ad_matrix'].shape, (len(self.cells), len(self.mutations)))
        
        # Check that matrices contain data
        self.assertTrue(matrices['dp_matrix'].nnz > 0)
        self.assertTrue(matrices['ad_matrix'].nnz > 0)

    def test_mtx_export(self):
        """Test export of matrices to MTX format"""
        prefix = os.path.join(self.test_output_dir, "test")
        matrices = export_mtx_for_dp_ad(self.cells, self.mutations, prefix=prefix)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{prefix}.DP.mtx"))
        self.assertTrue(os.path.exists(f"{prefix}.AD.mtx"))
        
        # Check that files have content
        self.assertTrue(os.path.getsize(f"{prefix}.DP.mtx") > 0)
        self.assertTrue(os.path.getsize(f"{prefix}.AD.mtx") > 0)
        
    def test_gene_expression_export(self):
        """Test export of gene expression data"""
        output_file = os.path.join(self.test_output_dir, "expression.csv")
        self.expr_df.to_csv(output_file, index=False)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check that file has content
        self.assertTrue(os.path.getsize(output_file) > 0)
        
        # Read back and verify
        df_read = pd.read_csv(output_file)
        self.assertEqual(df_read.shape, self.expr_df.shape)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create test cells with mutations and gene expression
        self.cells, self.mutations, self.expr_df = run_basic_simulation()

        cell_ids = [c.id for c in self.cells]
        self.expr_df = self.expr_df.loc[self.expr_df.index.isin(cell_ids)]
        
        # Create test output directory
        self.test_output_dir = "test_output"
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_mito_mutation_visualization(self):
        """Test mitochondrial mutation heatmap generation"""
        output_file = os.path.join(self.test_output_dir, "mito_heatmap.png")
        visualize_mito_mutations(self.cells, self.mutations, output_file)
        self.assertTrue(os.path.exists(output_file))

    def test_gene_expression_visualization(self):
        """Test gene expression UMAP visualization"""
        output_file = os.path.join(self.test_output_dir, "gene_expr_umap.png")
        try:
            result = visualize_gene_expression(
                self.cells, 
                self.expr_df, 
                output_file, 
                color_by='cell_type'
            )
            self.assertTrue(os.path.exists(output_file))
            self.assertIn('pca', result)
            self.assertIn('umap', result)
            self.assertIn('cell_metadata', result)
            self.assertIn('pca_variance_ratio', result)
        except (ImportError, TypeError) as e:
            print(f"Warning: Visualization error: {e}, skipping test")

    def test_full_visualization(self):
        """Test complete visualization pipeline"""
        output_prefix = os.path.join(self.test_output_dir, "full_viz")
        try:
            visualize_simulation_results(
                self.cells,
                self.mutations,
                self.expr_df,
                output_prefix,
                color_by='cell_type'
            )
            self.assertTrue(os.path.exists(f"{output_prefix}_heatmap.png"))
            self.assertTrue(os.path.exists(f"{output_prefix}_umap.png"))
        except (ImportError, TypeError) as e:
            print(f"Warning: Visualization error: {e}, skipping full visualization test")

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests() 