## This script runs SNP_VAE to get results while the input is the output of variants selected using existing labels.
## python ~/code/civet/detect_germline.py /home/linxy29/data/maester/oagct/gct98/HEMO_pipeline/maester_cellSNP/

from utilities import load_cellsnp
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='Script to run SNP_VAE.')
parser.add_argument('input_path', help="The path of input folder.", type=str)
parser.add_argument('cutoff', help="The cutoff for germline mutation.", type=float)
args = parser.parse_args()
input_path = args.input_path
#input_path = "/home/linxy29/data/maester/oagct/gct98/HEMO_pipeline/maester_cellSNP/"

mdphd, barcode = load_cellsnp(input_path)

## calculate variant allele frequency
af = np.divide(mdphd.ad, mdphd.dp, out=np.zeros_like(mdphd.ad), where=mdphd.dp!=0)
af = af.astype(np.float32)

#### Germline mutation: if bulk af > 0.9 

## get sum of AD and DP
ad_sum = np.sum(mdphd.ad, axis=1)
dp_sum = np.sum(mdphd.dp, axis=1)
## calculate af
af_sum = np.divide(ad_sum, dp_sum, out=np.zeros_like(ad_sum), where=dp_sum!=0)
## make sure values in af_sum are floats
af_sum = af_sum.astype(float)
## get variant index where af_sum > 0.9
germline_idx1 = np.where(af_sum > 0.9)[0]

#### Germline mutation: if 90% cells have af > cutoff

# Boolean matrix where True indicates af > cutoff
af_gt_05 = af > args.cutoff
# Count the number of cells with af > cutoff for each variant
cells_af_gt_05_count = np.sum(af_gt_05, axis=1)
# Total number of cells
total_cells = af.shape[1]
# Variants where more than 90% of cells have af > cutoff
germline_idx2 = np.where(cells_af_gt_05_count / total_cells >= 0.9)[0]

germline_idx = np.concatenate((germline_idx1, germline_idx2)).astype(int)
#print(germline_idx[:5])
#print(mdphd.variants.head())
#germline_variants = mdphd.variants[germline_idx]
mdphd_variants_array = np.array(mdphd.variants)
germline_variants = mdphd_variants_array[germline_idx]

# Open the CSV file in write mode
with open(input_path + 'germline_variant' + str(parser.cutoff) + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the list as a row in the CSV file
    writer.writerow(germline_variants)
