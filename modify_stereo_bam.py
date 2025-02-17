import pysam
import pandas as pd
import time

# File paths
gem_file_path = '/home/linxy29/data/CIVET/stero_lung/B04373C2.adjusted.cellbin.MT.gem'
#gem_pickle_path = '/home/linxy29/data/CIVET/stero_lung/gem_lookup.pkl.gz'
bam_file_path = '/home/linxy29/data/CIVET/stero_lung/B04373C2.Aligned.sortedByCoord.out.merge.q10.dedup.target.sorted.bam'
#bam_file_path = '/home/linxy29/data/CIVET/stero_lung/example_sorted.bam'
output_bam_path = '/home/linxy29/data/CIVET/stero_lung/B04373C2.Aligned.sortedByCoord.out.merge.q10.dedup.target.addCB.bam'
cell_tag_file_path = '/home/linxy29/data/CIVET/stero_lung/cell_tags.txt'

# Load GEM file and create a lookup dictionary
gem_data = pd.read_csv(gem_file_path, sep='\t', comment='#')
gem_data['x'] = gem_data['x'].astype(int)
gem_data['y'] = gem_data['y'].astype(int)
#print(gem_data.head())
gem_lookup = {(row['x'], row['y']): f"cell{row['CellID']}" for _, row in gem_data.iterrows()}

print(f"Loaded GEM file and created lookup dictionary: {gem_file_path}")

# Open BAM file
input_bam = pysam.AlignmentFile(bam_file_path, 'rb')
output_bam = pysam.AlignmentFile(output_bam_path, 'wb', header=input_bam.header)

# Set to store unique cell tags
cell_tags_set = set()

start_time = time.time()
# Iterate through reads from chromosome MT
for read in input_bam.fetch('MT'):
    print("Processing read", read.query_name)
    
    # Check if XF tag exists and is either 0 or 3
    if read.has_tag('XF') and read.get_tag('XF') in [0, 3]:
        # Extract Cx:i and Cy:i tags
        if read.has_tag('Cx') and read.has_tag('Cy'):
            cx_tag = read.get_tag('Cx')
            cy_tag = read.get_tag('Cy')

            # Lookup cell_tag using the dictionary
            cell_tag = gem_lookup.get((cx_tag, cy_tag), None)
            if cell_tag:
                # Add CB and CR tags to the read
                read.set_tag('CB', cell_tag)
                read.set_tag('CR', cell_tag)

                # Add cell tag to the set
                cell_tags_set.add(cell_tag)
    
            # Write modified read to output BAM file
            output_bam.write(read)

print(f"Processed all reads in {time.time() - start_time:.2f} seconds")

# Write unique cell tags to the text file
with open(cell_tag_file_path, 'w') as cell_tag_file:
    cell_tag_file.write('\n'.join(sorted(cell_tags_set)))  # Write all at once

# Close BAM files
input_bam.close()
output_bam.close()

print(f"Modified BAM file written to: {output_bam_path}")
print(f"Cell tags written to: {cell_tag_file_path}")