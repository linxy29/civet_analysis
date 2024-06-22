## This script runs SNP_VAE to get results while the input is the output of variants selected using existing labels.
## python ~/code/civet/selectedVariants_SNPVAE_CLI.py /home/linxy29/data/maester/oagct/gctb2/variant_selection/HEMO_pipeline_maester_cellSNP_celltype_gctb2TumorvsNorm_gctb2subset_analysis/
from utilities import load_mquad, load_cellsnp
import numpy as np
from SNP_VAE_modified import *
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Script to run SNP_VAE.')
parser.add_argument('input_path', help="The path of input folder.", type=str)
args = parser.parse_args()

base_path = args.input_path
#base_path = "/home/linxy29/data/maester/maester_paper/k562_bt142/variant_selection/mixed_effect/supervised_glm_Wald/"
#base_path = "/Users/linxy29/Documents/Data/maester/maester_paper/k562_bt142/maester/trimmed_starsolo_chrM_cellSNP0_WaldVariant_paperCell/

# Define the results directory path
SNP_VAE_res = base_path + "SNP_VAE_results/"
# Create the results directory if it doesn't exist
os.makedirs(SNP_VAE_res, exist_ok=True)


## read data
AD, DP, mtSNP_ids = load_mquad(base_path, "", "")
#AD = mmread(base_path + "cellSNP.tag.AD.mtx").tocsc()
#DP = mmread(base_path + "cellSNP.tag.DP.mtx").tocsc()
#mtSNP_ids = np.genfromtxt(base_path + "cellSNP.variants.tsv", dtype='str').tolist()

#exec(open("/home/linxy29/code/Ho_maester/SNP_VAE_modified.py").read())
#exec(open("/Users/linxy29/Documents/Code/Ho_maester/SNP_VAE_modified.py").read())
test1 = SNP_VAE(AD = AD.T, DP = DP.T, variant_name = mtSNP_ids, SNPread = "unnormalized") # output path of cellSNP-lite
test1.filtering(cell_SNPread_threshold = 0, SNP_DPmean_threshold = 0, SNP_logit_var_threshold = 0, figure_path = SNP_VAE_res)
test1.training()
test1.clustering(max_cluster = 10) # maximum number for k-means clustering

## get label and barcode
barcode_file = SNP_VAE_res + "passed_sample_names.txt"
### read barcode
barcode = np.genfromtxt(barcode_file, dtype=str)[test1.cell_filter]
print(len(barcode))
### convert cell label to vector
df = test1.labels
### convert to dataframe
df = pd.DataFrame(df).T
### add 'label' in front of column names
df.columns = ['label_' + str(x) for x in df.columns]
### make dataframe with barcode and cell label
df['barcode'] = barcode

## add embedding coordinates
### add PCA coordinates
df['PCA_1'] = test1.pc.T[0]
df['PCA_2'] = test1.pc.T[1]
### add UMAP coordinates
df['UMAP_1'] = test1.embedding_2d.T[0]
df['UMAP_2'] = test1.embedding_2d.T[1]

## save dataframe as csv
df.to_csv(SNP_VAE_res + 'SNP_VAE_results.csv', index=True)
## save test1 object
import pickle
with open(SNP_VAE_res + 'SNP_VAE_object.pkl', 'wb') as f:
    pickle.dump(test1, f)


