#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation Visualization and Saving Functions

This module contains functions for visualizing and saving simulation results
to a structured folder.

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
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def export_mtx_for_dp_ad(cells, mutations, output_dir):
    """
    Export DP and AD matrices in Matrix Market format for cellSNP-style analysis.
    Creates 'cellSNP' subdirectory in output_dir and saves:
        - cellSNP.tag.AD.mtx
        - cellSNP.tag.DP.mtx
        - cellSNP.tag.barcodes.txt
        - cellSNP.tag.vcf (or something similar)
    
    Parameters
    ----------
    cells : list of Cell objects
        Each cell has 'mutation_profile' dict with {mutation: {'DP': val, 'AD': val}}.
    mutations : list of str
        Sorted list of mutation identifiers.
    output_dir : str
        Main directory to save results.
    
    Returns
    -------
    dict
        Dictionary with file paths for the DP and AD matrices.
    """
    # Create subdirectory for cellSNP
    cellSNP_dir = os.path.join(output_dir, "cellSNP")
    os.makedirs(cellSNP_dir, exist_ok=True)

    # Sort cells by id to have consistent order
    sorted_cells = sorted(cells, key=lambda c: c.id)
    cell_ids = [c.id for c in sorted_cells]

    # Build DP and AD matrices (mutations x cells)
    DP_matrix = np.zeros((len(mutations), len(sorted_cells)), dtype=int)
    AD_matrix = np.zeros((len(mutations), len(sorted_cells)), dtype=int)

    # Fill matrices with mutation data
    for j, cell in enumerate(sorted_cells):
        for i, mut in enumerate(mutations):
            # Check if mutation exists in cell's profile
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                DP_matrix[i, j] = dp
                AD_matrix[i, j] = ad
            # If mutation doesn't exist in profile, leave as 0 (already initialized)

    # Convert to sparse
    DP_sparse = scipy.sparse.csr_matrix(DP_matrix)
    AD_sparse = scipy.sparse.csr_matrix(AD_matrix)

    # Save DP matrix
    dp_path = os.path.join(cellSNP_dir, "cellSNP.tag.DP.mtx")
    scipy.io.mmwrite(dp_path, DP_sparse)

    # Save AD matrix
    ad_path = os.path.join(cellSNP_dir, "cellSNP.tag.AD.mtx")
    scipy.io.mmwrite(ad_path, AD_sparse)

    # Save barcodes (cell IDs)
    barcodes_path = os.path.join(cellSNP_dir, "cellSNP.tag.barcodes.txt")
    with open(barcodes_path, 'w') as f:
        for cid in cell_ids:
            f.write(f"{cid}\n")

    # Save mutations list (now these are the row names)
    mutations_path = os.path.join(cellSNP_dir, "cellSNP.tag.mutations.txt")
    with open(mutations_path, 'w') as f:
        for mut in mutations:
            f.write(f"{mut}\n")

    # Save a minimal VCF
    vcf_path = os.path.join(cellSNP_dir, "cellSNP.tag.vcf")
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for mut in mutations:
            # For example, parse a mutation like "chrM:1234:A>G"
            f.write(f"chrM\t0\t{mut}\tN\tN\t.\t.\t.\n")

    logging.info(f"DP/AD matrices saved in '{cellSNP_dir}'")

    return {
        "DP": dp_path, 
        "AD": ad_path, 
        "barcodes": barcodes_path, 
        "mutations": mutations_path,
        "vcf": vcf_path
    }

def save_expression(expr_df, output_dir, filename="expression.csv"):
    """
    Save the expression DataFrame (cells x genes) in a dedicated subdirectory.
    
    Parameters
    ----------
    expr_df : pd.DataFrame
        Single-cell expression matrix or table.
    output_dir : str
        Main output directory.
    filename : str
        Name of the CSV file to save.
    """
    expr_dir = os.path.join(output_dir, "expression")
    os.makedirs(expr_dir, exist_ok=True)
    
    expr_path = os.path.join(expr_dir, filename)
    expr_df.to_csv(expr_path, index=False)
    logging.info(f"Expression data saved to '{expr_path}'")

def save_cell_metadata(cells, output_dir, filename="cell_metadata.csv"):
    """
    Save cell metadata including lineage information, generation, time, cell type,
    and mutation lists (baseline, de novo, and false mutations).
    
    Parameters
    ----------
    cells : list of Cell objects
        List of cells with attributes to save.
    output_dir : str
        Main output directory.
    filename : str
        Name of the CSV file to save.
    
    Returns
    -------
    str
        Path to the saved metadata file.
    """
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Extract metadata from each cell
    cell_metadata = []
    for cell in cells:
        # Separate mutations by type (baseline, de novo, false)
        baseline_mutations = []
        denovo_mutations = []
        false_mutations = []
        
        # Only count mutations with AD > 0
        for mut_id, mut_data in cell.mutation_profile.items():
            if mut_data['AD'] > 0:
                if mut_id.startswith('baseline_'):
                    baseline_mutations.append(mut_id)
                elif mut_id.startswith('false_m'):
                    false_mutations.append(mut_id)
                else:
                    denovo_mutations.append(mut_id)
        
        metadata = {
            'cell_id': cell.id,
            'parent_id': cell.parent_id,
            'generation': cell.generation,
            'time': cell.time,
            'cell_type': cell.cell_type,
            'children': ','.join(cell.children) if cell.children else '',
            'num_children': len(cell.children),
            'num_mutations': len(baseline_mutations) + len(denovo_mutations) + len(false_mutations),
            'num_baseline_mutations': len(baseline_mutations),
            'num_denovo_mutations': len(denovo_mutations),
            'num_false_mutations': len(false_mutations),
            'baseline_mutations': ','.join(baseline_mutations) if baseline_mutations else '',
            'denovo_mutations': ','.join(denovo_mutations) if denovo_mutations else '',
            'false_mutations': ','.join(false_mutations) if false_mutations else ''
        }
        cell_metadata.append(metadata)
    
    # print the number of cells in each cell type
    cell_type_counts = {}
    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in cell_type_counts:
            cell_type_counts[cell_type] = 0
        cell_type_counts[cell_type] += 1
    print(f"Number of cells in each cell type: {cell_type_counts}")
    
    # Create DataFrame and save
    metadata_df = pd.DataFrame(cell_metadata)
    metadata_path = os.path.join(metadata_dir, filename)
    metadata_df.to_csv(metadata_path, index=False)
    
    logging.info(f"Cell metadata saved to '{metadata_path}'")
    return metadata_path

def save_gene_metadata(gene_params, output_dir, filename="gene_metadata.csv"):
    """
    Save gene metadata including cell type specificity and expression parameters.
    
    Parameters
    ----------
    gene_params : dict
        Dictionary of gene parameters from generate_gene_params().
    output_dir : str
        Main output directory.
    filename : str
        Name of the CSV file to save.
    
    Returns
    -------
    str
        Path to the saved metadata file.
    """
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Extract metadata for each gene
    gene_metadata = []
    for gene_id, params in gene_params.items():
        metadata = {
            'gene_id': gene_id,
            'base_expression': params.get('base_expression', 0),
            'expression_rate': params.get('expression_rate', 0),
            'sigma': params.get('sigma', 0),
            'cell_type_specific': params.get('cell_type_specific', False),
            'specific_for': ','.join(params.get('specific_for', [])),
            'inherited_by': ';'.join([f"{src}->{dst}" for src, dst in params.get('inherited_by', [])])
        }
        gene_metadata.append(metadata)
    
    # Create DataFrame and save
    metadata_df = pd.DataFrame(gene_metadata)
    metadata_path = os.path.join(metadata_dir, filename)
    metadata_df.to_csv(metadata_path, index=False)
    
    logging.info(f"Gene metadata saved to '{metadata_path}'")
    return metadata_path

def visualize_mito_mutations(cells, mutations, output_file):
    """
    Visualize mitochondrial mutations as a heatmap.
    """
    sorted_cells = sorted(cells, key=lambda c: c.id)
    cell_ids = [c.id for c in sorted_cells]
    af_matrix = np.zeros((len(sorted_cells), len(mutations)), dtype=float)

    for i, cell in enumerate(sorted_cells):
        for j, mut in enumerate(mutations):
            # Check if mutation exists in cell's profile
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                af_matrix[i, j] = ad / dp if dp > 0 else 0.0
            else:
                # If mutation doesn't exist in profile, leave as 0
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

    logging.info(f"Mitochondrial mutation heatmap saved as '{output_file}'")

def visualize_gene_expression(cells, expr_df, output_prefix, color_by='cell_type'):
    """
    Visualize gene expression using scanpy, saving PCA & UMAP to output_prefix_pca.png, output_prefix_umap.png.
    (Only the saving mechanism is shown below.)
    """
    import scanpy as sc
    
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
    
    return None

def visualize_simulation_results(cells, mutations, expr_df, output_dir, color_by='cell_type'):
    """
    Visualize all simulation results, storing plots under 'figures/' inside output_dir.
    
    Parameters
    ----------
    cells : list of Cell objects
    mutations : list of str
    expr_df : pd.DataFrame
    output_dir : str
    color_by : str
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Mito heatmap
    mito_output_file = os.path.join(fig_dir, "simulation_heatmap.png")
    visualize_mito_mutations(cells, mutations, mito_output_file)

    # 2) Gene expression: PCA/UMAP
    gene_expr_prefix = os.path.join(fig_dir, "simulation")
    visualize_gene_expression(cells, expr_df, gene_expr_prefix, color_by=color_by)

def analyze_af_distribution(cells, mutations, output_dir):
    """
    Analyze the distribution of allele frequencies, saving plots & CSV under 'af_analysis/'.
    """
    af_dir = os.path.join(output_dir, "af_analysis")
    os.makedirs(af_dir, exist_ok=True)

    # Prepare data
    all_afs = []
    cell_type_afs = {}
    mutation_afs = {mut: [] for mut in mutations}

    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in cell_type_afs:
            cell_type_afs[cell_type] = []
        for mut in mutations:
            # Check if mutation exists in cell's profile
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                if dp > 0:
                    af = ad / dp
                    all_afs.append(af)
                    cell_type_afs[cell_type].append(af)
                    mutation_afs[mut].append(af)

    baseline_mutations = [mut for mut in mutations if mut.startswith('baseline_')]
    denovo_mutations = [mut for mut in mutations if not mut.startswith('baseline_') and not mut.startswith('false_m')]
    false_mutations = [mut for mut in mutations if mut.startswith('false_m')]

    baseline_afs = []
    denovo_afs = []
    false_afs = []
    baseline_cell_type_afs = {}
    denovo_cell_type_afs = {}
    false_cell_type_afs = {}

    for cell in cells:
        cell_type = cell.cell_type
        if cell_type not in baseline_cell_type_afs:
            baseline_cell_type_afs[cell_type] = []
            denovo_cell_type_afs[cell_type] = []
            false_cell_type_afs[cell_type] = []
            
        # Process baseline mutations
        for mut in baseline_mutations:
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                if dp > 0:
                    af = ad / dp
                    baseline_afs.append(af)
                    baseline_cell_type_afs[cell_type].append(af)
        
        # Process de novo mutations
        for mut in denovo_mutations:
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                if dp > 0:
                    af = ad / dp
                    denovo_afs.append(af)
                    denovo_cell_type_afs[cell_type].append(af)
                    
        # Process false mutations
        for mut in false_mutations:
            if mut in cell.mutation_profile:
                dp = cell.mutation_profile[mut]['DP']
                ad = cell.mutation_profile[mut]['AD']
                if dp > 0:
                    af = ad / dp
                    false_afs.append(af)
                    false_cell_type_afs[cell_type].append(af)

    # Create plots
    plt.figure(figsize=(12, 6))
    
    # Plot histograms for all three types of mutations
    if baseline_afs:
        plt.hist(baseline_afs, bins=30, alpha=0.5, label='Baseline Mutations', color='blue')
    if denovo_afs:
        plt.hist(denovo_afs, bins=30, alpha=0.5, label='De Novo Mutations', color='red')
    if false_afs:
        plt.hist(false_afs, bins=30, alpha=0.5, label='False Mutations', color='green')
    
    plt.title('Distribution of Allele Frequencies by Mutation Type')
    plt.xlabel('Allele Frequency')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)

    # Add statistics text
    stats_text = []
    if baseline_afs:
        stats_text.append(
            f"Baseline Mutations:\n"
            f"Count: {len(baseline_afs)}\n"
            f"Mean: {np.mean(baseline_afs):.4f}\n"
            f"Median: {np.median(baseline_afs):.4f}\n"
            f"Std Dev: {np.std(baseline_afs):.4f}\n"
        )
    if denovo_afs:
        stats_text.append(
            f"De Novo Mutations:\n"
            f"Count: {len(denovo_afs)}\n"
            f"Mean: {np.mean(denovo_afs):.4f}\n"
            f"Median: {np.median(denovo_afs):.4f}\n"
            f"Std Dev: {np.std(denovo_afs):.4f}\n"
        )
    if false_afs:
        stats_text.append(
            f"False Mutations:\n"
            f"Count: {len(false_afs)}\n"
            f"Mean: {np.mean(false_afs):.4f}\n"
            f"Median: {np.median(false_afs):.4f}\n"
            f"Std Dev: {np.std(false_afs):.4f}\n"
        )
    
    plt.text(0.95, 0.95, "\n".join(stats_text), 
             transform=plt.gca().transAxes,
             verticalalignment='top', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()

    hist_path = os.path.join(af_dir, "af_distribution.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # Create and save boxplots
    plt.figure(figsize=(15, 6))
    
    # Prepare data for boxplots
    boxplot_data = []
    for cell_type in baseline_cell_type_afs.keys():
        for af in baseline_cell_type_afs[cell_type]:
            boxplot_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'Baseline'})
        for af in denovo_cell_type_afs[cell_type]:
            boxplot_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'De Novo'})
        for af in false_cell_type_afs[cell_type]:
            boxplot_data.append({'Cell Type': cell_type, 'AF': af, 'Mutation Type': 'False'})

    combined_df = pd.DataFrame(boxplot_data)
    
    if not combined_df.empty:
        plt.subplot(1, 3, 1)
        sns.boxplot(data=combined_df[combined_df['Mutation Type'] == 'Baseline'],
                    x='Cell Type', y='AF')
        plt.title('Baseline Mutations by Cell Type')
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 2)
        sns.boxplot(data=combined_df[combined_df['Mutation Type'] == 'De Novo'],
                    x='Cell Type', y='AF')
        plt.title('De Novo Mutations by Cell Type')
        plt.xticks(rotation=45)

        plt.subplot(1, 3, 3)
        sns.boxplot(data=combined_df[combined_df['Mutation Type'] == 'False'],
                    x='Cell Type', y='AF')
        plt.title('False Mutations by Cell Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()

        celltype_path = os.path.join(af_dir, "af_celltype_comparison.png")
        plt.savefig(celltype_path, dpi=300)
        plt.close()

        # Save data
        csv_path = os.path.join(af_dir, "af_mutation_type_data.csv")
        combined_df.to_csv(csv_path, index=False)

    logging.info(f"AF analysis saved in '{af_dir}'")

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
        'false_summary': {
            'afs': false_afs,
            'mean': np.mean(false_afs) if false_afs else 0,
            'median': np.median(false_afs) if false_afs else 0,
            'std': np.std(false_afs) if false_afs else 0,
            'count': len(false_afs)
        }
    }

def save_simulation_data(cells, mutations, expr_df, gene_params, output_dir, prefix="simulation"):
    """
    Save all simulation data including expression, cell metadata, gene metadata, and mutation info.
    
    Parameters
    ----------
    cells : list of Cell objects
        List of cells from the simulation
    mutations : list of str
        List of mutation IDs
    expr_df : pd.DataFrame
        Expression matrix (cells x genes)
    gene_params : dict
        Dictionary of gene parameters from generate_gene_params()
    output_dir : str
        Directory to save results
    prefix : str
        Prefix for output files
    
    Returns
    -------
    dict
        Dictionary with paths to all saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save expression data
    expr_path = save_expression(expr_df, output_dir, f"{prefix}_expression.csv")
    
    # Save cell metadata
    cell_meta_path = save_cell_metadata(cells, output_dir, f"{prefix}_cell_metadata.csv")
    
    # Save gene metadata
    gene_meta_path = save_gene_metadata(gene_params, output_dir, f"{prefix}_gene_metadata.csv")
    
    # Save mutation data with explicit categorization
    mutation_info = pd.DataFrame({
        'mutation_id': mutations,
        'is_baseline': [mut.startswith('baseline_') for mut in mutations],
        'is_denovo': [not mut.startswith('baseline_') and not mut.startswith('false_m') for mut in mutations],
        'is_false': [mut.startswith('false_m') for mut in mutations],
        'mutation_type': [
            'Baseline' if mut.startswith('baseline_') else 
            'False' if mut.startswith('false_m') else 
            'De Novo' for mut in mutations
        ]
    })
    
    # Count the number of cells containing each mutation
    mutation_cell_counts = {'mutation_id': [], 'cells_with_mutation': []}
    
    for mut in mutations:
        count = sum(1 for cell in cells if mut in cell.mutation_profile and cell.mutation_profile[mut]['AD'] > 0)
        mutation_cell_counts['mutation_id'].append(mut)
        mutation_cell_counts['cells_with_mutation'].append(count)
    
    # Add cell counts to mutation info
    mutation_counts_df = pd.DataFrame(mutation_cell_counts)
    mutation_info = pd.merge(mutation_info, mutation_counts_df, on='mutation_id')
    
    metadata_dir = os.path.join(output_dir, "metadata")
    mutation_path = os.path.join(metadata_dir, f"{prefix}_mutation_info.csv")
    mutation_info.to_csv(mutation_path, index=False)
    
    # Save DP/AD matrices
    dp_ad_paths = export_mtx_for_dp_ad(cells, mutations, output_dir)
    
    # Create visualizations
    visualize_simulation_results(cells, mutations, expr_df, output_dir)
    
    # Analyze AF distributions
    af_analysis = analyze_af_distribution(cells, mutations, output_dir)
    
    logging.info(f"All simulation data saved to '{output_dir}'")
    
    return {
        'expression': expr_path,
        'cell_metadata': cell_meta_path,
        'gene_metadata': gene_meta_path,
        'mutation_info': mutation_path,
        'dp_ad': dp_ad_paths
    }
