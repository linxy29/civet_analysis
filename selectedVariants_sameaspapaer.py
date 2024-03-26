# generate paths_to_passed_variant_files.txt: find variant_selection/ -name passed_variant_names.txt > paths_to_passed_variant_files.txt

import os

# Set the working directory
os.chdir('/home/linxy29/data/maester/maester_paper/k562_bt142')

# Path to the file containing all the paths
paths_file = 'paths_to_passed_variant_files.txt'

# Path to the file containing the paper variants
paper_variant_file = 'variant_selection/paper_variant'

# Read the paper variants into a set for efficient lookups
with open(paper_variant_file, 'r') as file:
    paper_variants = set(file.read().splitlines())

# Function to check if a passed_variant file contains any paper variant
def check_variants_in_file(file_path, paper_variants):
    with open(file_path, 'r') as file:
        file_variants = set(file.read().splitlines())
    return not paper_variants.isdisjoint(file_variants)

# Read the paths to passed_variant_names.txt files
with open(paths_file, 'r') as file:
    paths = file.read().splitlines()

# Check each file and print the ones containing any paper variant
for path in paths:
    if check_variants_in_file(path, paper_variants):
        print(f"Variants from 'paper_variant' found in: {path}")
