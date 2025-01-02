import sys
import os
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import anndata
import cell2location
import torch
from matplotlib import rcParams

# Set matplotlib parameters
rcParams['pdf.fonttype'] = 42

# Define folder paths
folder = "/home/linxy29/data/CIVET"
sample_name = "visium_breastcancer_10x"
results_folder = os.path.join(folder, sample_name, "cSCC_cell2location")
ref_run_name = os.path.join(folder, "breastcancer_atlas", "reference_signatures")
run_name = os.path.join(results_folder, "cell2location_map")

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Load Visium data
matrix_h5_path = os.path.join(folder, sample_name, "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
spatial_dir = os.path.join(folder, sample_name)
adata_vis = sc.read_visium(path=spatial_dir, count_file=matrix_h5_path)
adata_vis.obs['sample'] = sample_name
adata_vis.var_names_make_unique()

# Find and remove mitochondria-encoded (MT) genes
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var_names]
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]

# Load reference data
adata_file = os.path.join(ref_run_name, "sc.h5ad")
adata_ref = sc.read_h5ad(adata_file)
mod = cell2location.models.RegressionModel.load(ref_run_name, adata_ref)

# Export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}' for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}' for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']

# Find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# Prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")

# Check for GPU availability
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Create and train the model
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver,
    N_cells_per_location=20,
    detection_alpha=200
)

mod.train(max_epochs=30000, batch_size=None, train_size=1)

# Plot ELBO loss history during training, removing first 100 epochs from the plot
mod.plot_history(1000)
plt.legend(labels=['full data training'])

# Save the plot
plot_path = os.path.join(results_folder, "elbo_loss_history.png")
plt.savefig(plot_path)

# Export the estimated cell abundance
adata_vis = mod.export_posterior(adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs})

# Save model and anndata object with results
mod.save(run_name, overwrite=True)
adata_file = os.path.join(run_name, "sp.h5ad")
adata_vis.write(adata_file)
