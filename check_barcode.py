import pandas as pd

# Load the CSV file, assuming barcode is in the first column
cell_labels_df = pd.read_csv('/home/linxy29/data/maester/maester_paper/k562_bt142/cell_label.csv')

# Load the TSV file, assume it's in a directory named RNA/cellranger/outs/filtered_feature_bc_matrix/
# and remove the '-1' from the barcodes
barcodes_df = pd.read_csv('/home/linxy29/data/maester/maester_paper/k562_bt142/RNA/cellranger/outs/filtered_feature_bc_matrix/barcodes_copy.tsv', header=None, sep='\t')
barcodes_df[0] = barcodes_df[0].str.replace('-1$', '', regex=True)

# Extract the set of barcodes from both dataframes
cell_label_barcodes = set(cell_labels_df['barcode'])
barcodes = set(barcodes_df[0])

# Calculate the intersection of both sets
overlap_barcodes = cell_label_barcodes.intersection(barcodes)

# Output the result
print(f"Number of overlapping barcodes: {len(overlap_barcodes)}")

