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

    # Build DP and AD matrices
    DP_matrix = np.zeros((len(sorted_cells), len(mutations)), dtype=int)
    AD_matrix = np.zeros((len(sorted_cells), len(mutations)), dtype=int)

    for i, cell in enumerate(sorted_cells):
        for j, mut in enumerate(mutations):
            dp = cell.mutation_profile[mut]['DP']
            ad = cell.mutation_profile[mut]['AD']
            DP_matrix[i, j] = dp
            AD_matrix[i, j] = ad

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

    # (Optional) Save a minimal VCF or mutation list if needed
    # For demonstration, let's just store mutation IDs
    vcf_path = os.path.join(cellSNP_dir, "cellSNP.tag.vcf")
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for mut in mutations:
            # For example, parse a mutation like "chrM:1234:A>G"
            # Real logic depends on how your 'mutations' are stored
            f.write(f"chrM\t0\t{mut}\tN\tN\t.\t.\t.\n")

    logging.info(f"DP/AD matrices saved in '{cellSNP_dir}'")

    return {"DP": dp_path, "AD": ad_path, "barcodes": barcodes_path, "vcf": vcf_path}

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

def visualize_mito_mutations(cells, mutations, output_file):
    """
    Visualize mitochondrial mutations as a heatmap.
    (Unchanged except that you’ll pass an explicit output file path.)
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
    Visualize gene expression using scanpy, saving PCA & UMAP to output_prefix_pca.png, output_prefix_umap.png.
    (Only the saving mechanism is shown below.)
    """
    import scanpy as sc
    
    pca_output = f"{output_prefix}_pca.png"
    umap_output = f"{output_prefix}_umap.png"
    
    # (Same logic as before, omitted for brevity)
    # ...
    # Save final figures to pca_output and umap_output
    # ...
    
    # Return an AnnData object or None
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
    
    Parameters
    ----------
    cells : list of Cell objects
    mutations : list of str
    output_dir : str
    
    Returns
    -------
    dict
        A summary dictionary (same as your original function).
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

    # 1) Baseline vs De Novo histogram
    plt.figure(figsize=(12, 6))
    plt.hist(baseline_afs, bins=30, alpha=0.5, label='Baseline Mutations', color='blue')
    plt.hist(denovo_afs, bins=30, alpha=0.5, label='De Novo Mutations', color='red')
    plt.title('Distribution of Allele Frequencies: Baseline vs De Novo')
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

    hist_path = os.path.join(af_dir, "af_baseline_vs_denovo_af.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # 2) Boxplots by cell type
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

    celltype_path = os.path.join(af_dir, "af_celltype_comparison.png")
    plt.savefig(celltype_path, dpi=300)
    plt.close()

    # 3) Save combined data
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
        'cell_type_afs': {
            'baseline': baseline_cell_type_afs,
            'denovo': denovo_cell_type_afs
        }
    }
