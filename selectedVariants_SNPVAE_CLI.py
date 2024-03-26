## This script runs SNP_VAE to get results while the input is the output of variants selected using existing labels.
## python selectedVariants_SNPVAE.py /home/linxy29/data/maester/maester_paper/k562_bt142/variant_selection/mixed_effect/supervised_glm_LRT/ /home/linxy29/data/maester/maester_paper/k562_bt142/cell_label.csv

from utilities import load_mquad
import numpy as np
from SNP_VAE_modified import *
import argparse

parser = argparse.ArgumentParser(description='Script to run SNP_VAE.')
parser.add_argument('input_path', help="The path of input folder.", type=str)
parser.add_argument('label_path', help="The path of the label file.", type=str)
args = parser.parse_args()

base_path = args.input_path
#base_path = "/home/linxy29/data/maester/maester_paper/k562_bt142/variant_selection/mixed_effect/supervised_glm_Wald/"
#base_path = "/Users/linxy29/Documents/Data/maester/maester_paper/k562_bt142/variant_selection/mixed_effect/supervised_glm_Wald"
SNP_VAE_res = base_path + "SNP_VAE_results/"

## detect SNP_VAE_res 
if not os.path.exists(SNP_VAE_res):
    os.makedirs(SNP_VAE_res)

## read data
AD, DP, mtSNP_ids = load_mquad(base_path, "", "")

#exec(open("/home/linxy29/code/Ho_maester/SNP_VAE_modified.py").read())
#exec(open("/Users/linxy29/Documents/Code/Ho_maester/SNP_VAE_modified.py").read())
test1 = SNP_VAE(AD = AD.T, DP = DP.T, variant_name = mtSNP_ids, SNPread = "unnormalized") # output path of cellSNP-lite
test1.filtering(cell_SNPread_threshold = 0, SNP_DPmean_threshold = 0, SNP_logit_var_threshold = 0, figure_path = SNP_VAE_res)
test1.training()
test1.clustering(max_cluster = 4) # maximum number for k-means clustering

## get label and barcode
barcode_file = base_path + "/passed_sample_names.txt"
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

## add true label
### get true label
true_label_path = args.label_path
true_label_df = pd.read_csv(true_label_path)
df = df.merge(true_label_df, left_on='barcode', right_on='barcode')

## save dataframe as csv
df.to_csv(SNP_VAE_res + '/SNP_VAE_results.csv', index=True)
## save test1 object
import pickle
with open(SNP_VAE_res + '/SNP_VAE_object.pkl', 'wb') as f:
    pickle.dump(test1, f)
## save the embedding plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Identify label columns
label_columns = [col for col in df.columns if 'label' in col]

### Iterate through label columns and create plots
for label_col in label_columns:
    plt.figure(figsize=(12, 6))

    # First subplot for PCA
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PCA_1', y='PCA_2', hue=label_col, data=df, palette='viridis')
    plt.title(f'PCA Plot for {label_col}')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # Second subplot for UMAP
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='UMAP_1', y='UMAP_2', hue=label_col, data=df, palette='viridis')
    plt.title(f'UMAP Plot for {label_col}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the plot
    plt.savefig(SNP_VAE_res + f'/{label_col}_embedding.png')

    # Show the plot
    plt.show()