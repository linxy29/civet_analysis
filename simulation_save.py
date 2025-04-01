#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation Visualization and Saving Functions

This module contains functions for visualizing and saving simulation results.

Author: Your Name
Date: 2025-03-31
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import scipy.sparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
            af_matrix[i, j] = ad / dp if dp > 0 else 0.0

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
    
    logging.info(f"Mitochondrial mutation heatmap saved as '{output_file}'")

def visualize_gene_expression(cells, expr_df, output_prefix, color_by='cell_type'):
    """
    Visualize gene expression using scanpy for dimensionality reduction and plotting.
    
    Parameters
    ----------
    cells : list of Cell objects
        List of cells with attributes to color by.
    expr_df : pd.DataFrame
        Single-cell expression matrix (cells x genes). Index = cell IDs, columns = gene IDs.
    output_prefix : str
        Path prefix for saving the output files. Will be used as base name for PCA and UMAP plots.
    color_by : str
        Cell attribute to color UMAP scatter. Typically "cell_type" or "generation".
    
    Returns
    -------
    AnnData or None
        The AnnData object with computed PCA and UMAP embeddings, or None if insufficient data.
    """
    import scanpy as sc
    import os
    
    pca_output = f"{output_prefix}_pca.png"
    umap_output = f"{output_prefix}_umap.png"
    
    expr_df.index = expr_df.index.astype(str)
    
    cell_sums = expr_df.sum(axis=1)
    expr_df = expr_df.loc[cell_sums > 0]
    
    gene_sums = expr_df.sum(axis=0)
    expr_df = expr_df.loc[:, gene_sums > 0]
    
    if expr_df.shape[0] == 0 or expr_df.shape[1] == 0:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No expression data available after filtering", 
                 ha='center', va='center', fontsize=14)
        plt.title("Gene Expression Visualization")
        plt.savefig(pca_output)
        plt.close()
        logging.warning(f"No valid expression data. Empty plot saved to '{pca_output}'")
        return None
    
    adata = sc.AnnData(X=expr_df.values)
    adata.obs_names = expr_df.index
    adata.var_names = expr_df.columns
    
    cell_dict = {cell.id: cell for cell in cells}
    
    def safe_get_cell_attr(cell_id, attr):
        cell = cell_dict.get(cell_id)
        if cell is None:
            cell = cell_dict.get(cell_id.replace('cell_', ''))
        if cell is None:
            return 'Unknown' if attr == 'cell_type' else 0
        return getattr(cell, attr)
    
    adata.obs['cell_type'] = [safe_get_cell_attr(cid, 'cell_type') for cid in adata.obs_names]
    adata.obs['generation'] = [safe_get_cell_attr(cid, 'generation') for cid in adata.obs_names]
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    try:
        if expr_df.shape[1] < 100 or expr_df.shape[0] < 10:
            adata.var['highly_variable'] = True
        else:
            sc.pp.highly_variable_genes(
                adata, 
                min_mean=0.01,
                max_mean=10,
                min_disp=0.1,
                n_top_genes=min(2000, adata.n_vars)
            )
    except Exception as e:
        logging.warning(f"Error in highly variable gene selection: {e}")
        adata.var['highly_variable'] = True
    
    adata = adata[:, adata.var.highly_variable]
    
    try:
        sc.pp.scale(adata, max_value=10)
    except Exception as e:
        logging.warning(f"Error in scaling: {e}")
    
    n_components = min(30, min(adata.n_obs - 1, adata.n_vars - 1))
    
    if n_components <= 1:
        logging.warning("Dataset too small for PCA/UMAP, using raw data for visualization")
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
    
    try:
        sc.tl.pca(adata, n_comps=n_components)
        sc.settings.figdir = os.path.dirname(pca_output)
        sc.settings.set_figure_params(dpi=150, figsize=(8, 6))
        
        sc.pl.pca(adata, color=color_by, show=False, 
                  title=f"PCA of Gene Expression (colored by {color_by})")
        plt.savefig(pca_output)
        plt.close()
        logging.info(f"PCA visualization saved as '{pca_output}'")
    except Exception as e:
        logging.warning(f"Error in PCA: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"PCA failed: {str(e)}", ha='center', va='center')
        plt.savefig(pca_output)
        plt.close()
        return adata
    
    try:
        if adata.n_obs < 15:
            n_neighbors = min(3, adata.n_obs - 1)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
        else:
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
        
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=color_by, show=False, 
                   title=f"UMAP of Gene Expression (colored by {color_by})")
        plt.savefig(umap_output)
        plt.close()
        logging.info(f"UMAP visualization saved as '{umap_output}'")
    except Exception as e:
        logging.warning(f"Error in UMAP: {e}")
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
        plt.savefig(umap_output)
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
    mito_output = f"{output_prefix}_heatmap.png"
    visualize_mito_mutations(cells, mutations, mito_output)
    visualize_gene_expression(cells, expr_df, output_prefix, color_by)

def analyze_af_distribution(cells, mutations, output_prefix):
    """
    Analyze the distribution of allele frequencies in the simulated data,
    separately for baseline and de novo mutations.
    """
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
    
    baseline_mutations = [mut for mut in mutations if mut.startswith('baseline_')]
    denovo_mutations = [mut for mut in mutations if not mut.startswith('baseline_')]
    
    baseline_afs = []
    denovo_afs = []
    baseline_cell_type_afs = {}
    denovo_cell_type_afs = {}
    
    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in baseline_cell_type_afs:
            baseline_cell_type_afs[cell_type] = []
            denovo_cell_type_afs[cell_type] = []
        
        for mut in baseline_mutations:
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af = ad / dp
                baseline_afs.append(af)
                baseline_cell_type_afs[cell_type].append(af)
        
        for mut in denovo_mutations:
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            if dp > 0:
                af = ad / dp
                denovo_afs.append(af)
                denovo_cell_type_afs[cell_type].append(af)
    
    plt.figure(figsize=(12, 6))
    plt.hist(baseline_afs, bins=30, alpha=0.5, label='Baseline Mutations', color='blue')
    plt.hist(denovo_afs, bins=30, alpha=0.5, label='De Novo Mutations', color='red')
    plt.title('Distribution of Allele Frequencies: Baseline vs De Novo Mutations')
    plt.xlabel('Allele Frequency')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)
    
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
    
    plt.figure(figsize=(15, 6))
    baseline_data = []
    denovo_data = []
    
    for cell_type in baseline_cell_type_afs.keys():
        for af in baseline_cell_type_afs[cell_type]:
            baseline_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'Baseline'})
        for af in denovo_cell_type_afs[cell_type]:
            denovo_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'De Novo'})
    
    combined_df = pd.DataFrame(baseline_data + denovo_data)
    
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
        List of Cell objects.
    num_genes : int
        Total number of genes to simulate.
    cell_type_specific_ratio : float
        Ratio of genes that are cell type-specific.
    
    Returns
    -------
    tuple
        (expression DataFrame, gene info DataFrame, cell metadata DataFrame)
    """
    # Define cell types (can be parameterized if needed)
    cell_types_list = ['stem', 'progenitor', 'differentiated']
    
    gene_info = []
    gene_ids = []
    
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
    
    # Add cell type-specific genes evenly across cell types
    genes_per_type = num_specific_genes // len(cell_types_list)
    for cell_type in cell_types_list:
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
    n_genes_total = gene_info_df.shape[0]
    
    # Prepare cell metadata
    n_cells = len(cells)
    cell_ids = [cell.id for cell in cells]
    cell_types = np.array([cell.cell_type for cell in cells])
    generations = np.array([cell.generation for cell in cells])
    parent_ids = [cell.parent_id if cell.parent_id is not None else 'root' for cell in cells]
    times = np.array([cell.time for cell in cells])
    num_children = np.array([len(cell.children) for cell in cells])
    
    cell_metadata = pd.DataFrame({
        'cell_id': cell_ids,
        'generation': generations,
        'cell_type': cell_types,
        'parent_id': parent_ids,
        'time': times,
        'num_children': num_children
    }).set_index('cell_id')
    
    # Vectorized simulation of expression matrix
    expression_matrix = np.empty((n_cells, n_genes_total))
    
    # Process common genes in a vectorized manner
    common_mask = ~gene_info_df['is_cell_type_specific']
    common_genes = gene_info_df[common_mask]
    for col_idx, (_, gene) in enumerate(common_genes.iterrows()):
        base_expr = gene['base_expression']
        variability = gene['expression_variability']
        # Sample for all cells at once
        expression_matrix[:, col_idx] = np.random.normal(
            loc=base_expr, scale=base_expr * variability, size=n_cells
        )
    
    # Process cell type-specific genes vectorized over cells for each gene
    specific_mask = gene_info_df['is_cell_type_specific']
    specific_genes = gene_info_df[specific_mask]
    for col_idx, (_, gene) in enumerate(specific_genes.iterrows(), start=common_genes.shape[0]):
        base_expr = gene['base_expression']
        variability = gene['expression_variability']
        target_type = gene['specific_to_cell_type']
        # Determine means and stds based on cell type matching
        means = np.where(cell_types == target_type, base_expr * 2, base_expr * 0.1)
        stds = np.where(cell_types == target_type, base_expr * variability, base_expr * variability * 0.5)
        expression_matrix[:, col_idx] = np.random.normal(loc=means, scale=stds)
    
    # Ensure non-negative expression values
    expression_matrix = np.maximum(expression_matrix, 0)
    
    # Create expression DataFrame
    expr_df = pd.DataFrame(expression_matrix, index=cell_ids, columns=gene_info_df['gene_id'])
    
    # Join expression matrix with cell metadata
    expr_df = expr_df.join(cell_metadata)
    
    return expr_df, gene_info_df, cell_metadata

def save_simulation_data(cells, mutations, output_dir, prefix):
    """
    Save simulation data including expression, cell info, and gene info.
    
    Parameters
    ----------
    cells : list
        List of Cell objects.
    mutations : list
        List of mutation IDs.
    output_dir : str
        Directory to save results.
    prefix : str
        Prefix for output files.
    """
    expr_matrix, gene_info_df, cell_metadata = generate_gene_expression(cells)
    
    os.makedirs(output_dir, exist_ok=True)
    expr_dir = os.path.join(output_dir, "expression")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(expr_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    expr_file = os.path.join(expr_dir, f"{prefix}_expression_matrix.mtx")
    scipy.io.mmwrite(expr_file, scipy.sparse.csr_matrix(expr_matrix.drop(cell_metadata.columns, axis=1).values))
    
    cell_ids_file = os.path.join(expr_dir, f"{prefix}_barcodes.tsv")
    with open(cell_ids_file, 'w') as f:
        for cell_id in expr_matrix.index:
            f.write(f"{cell_id}\n")
    
    gene_ids_file = os.path.join(expr_dir, f"{prefix}_features.tsv")
    with open(gene_ids_file, 'w') as f:
        for gene_id in expr_matrix.columns:
            if gene_id not in cell_metadata.columns:
                f.write(f"{gene_id}\n")
    
    gene_info_file = os.path.join(metadata_dir, f"{prefix}_gene_info.csv")
    gene_info_df.to_csv(gene_info_file, index=False)
    
    cell_meta_file = os.path.join(metadata_dir, f"{prefix}_cell_metadata.csv")
    cell_metadata.to_csv(cell_meta_file)
    
    mutation_info = pd.DataFrame({
        'mutation_id': mutations,
        'is_baseline': ['baseline' in mut for mut in mutations]
    })
    mut_file = os.path.join(metadata_dir, f"{prefix}_mutation_info.csv")
    mutation_info.to_csv(mut_file, index=False)
    
    summary_stats = {
        'total_cells': len(cells),
        'total_mutations': len(mutations),
        'total_genes': len(gene_info_df),
        'cell_type_specific_genes': int(gene_info_df['is_cell_type_specific'].sum()),
        'max_generation': max(cell.generation for cell in cells),
        'cell_type_counts': cell_metadata['cell_type'].value_counts().to_dict()
    }
    with open(os.path.join(metadata_dir, f"{prefix}_summary.txt"), 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    logging.info(f"Simulation data saved to directory '{output_dir}'")
    return expr_matrix, gene_info_df, cell_metadata
