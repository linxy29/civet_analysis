# utilities.py

## extract cloneID from the result of BinomMixtureVB
def get_cloneID_df(startDF, layer, AD, DP):
    cloneID_df = startDF.copy()
    for i in range(2, layer+1):
        clone_type = pd.unique(cloneID_df['cloneID'])
        for clone in clone_type:
            testDF = cloneID_df.loc[cloneID_df['cloneID'] == clone]
            #print(testDF[1:5])
            testAD = AD[:, testDF.index]
            testDP = DP[:, testDF.index]
            _model = BinomMixtureVB(n_var=testAD.shape[0], n_cell=testAD.shape[1], n_donor=2)
            _model.fit(testAD, testDP, min_iter=30, n_init=50)
            prop_max = _model.ID_prob.argmax(axis=1)
            cloneID_df.loc[cloneID_df['cloneID'] == clone, 'layer' + str(i)] = prop_max.astype(str)
        cloneID_df['cloneID'] = cloneID_df['cloneID'] + cloneID_df['layer' + str(i)]
    return cloneID_df

## read data from the output of mquad
def load_mquad(base_path, sub_dir, prefix):
    from scipy.io import mmread
    import numpy as np
    AD = mmread(base_path + sub_dir + "/passed_ad.mtx").tocsc()
    DP = mmread(base_path + sub_dir + "/passed_dp.mtx").tocsc()
    mtSNP_ids = np.array([prefix + id_ for id_ in np.genfromtxt(base_path + sub_dir + "/passed_variant_names.txt", dtype='str')])
    print("The number of variants: ", AD.shape[0])
    return AD, DP, mtSNP_ids

## read data from the output of cellsnp
def load_cellsnp(input_folder):
    from mquad.mquad import Mquad
    from vireoSNP.utils.vcf_utils import read_sparse_GeneINFO, load_VCF
    import numpy as np
    # Construct the VCF file path within the given input folder
    vcf_file = input_folder + "cellSNP.cells.vcf.gz"
    # Load the VCF file, specifying biallelic_only=True
    cell_vcf = load_VCF(vcf_file, biallelic_only=True)
    # Print the path of the loaded VCF file
    print("Loaded VCF file: %s" % vcf_file)
    # Read sparse gene information from the loaded VCF data
    cell_dat = read_sparse_GeneINFO(cell_vcf['GenoINFO'], keys=['AD', 'DP'])
    # Include variant names in the cell data
    cell_dat['variants'] = cell_vcf['variants']
    # Extract AD, DP, and variant names for the output
    AD = cell_dat['AD']
    DP = cell_dat['DP']
    variant_names = cell_dat['variants']
    ## read barcode
    barcode_file = input_folder + "cellSNP.samples.tsv"
    barcode = np.genfromtxt(barcode_file, dtype='str')
    mdphd = Mquad(AD = cell_dat['AD'], DP = cell_dat['DP'], variant_names = cell_dat['variants'])
    # Return the AD, DP, and variant_names
    return mdphd, barcode


## update anno_heat so that it can take two annotation columns
def anno_heat(X, row_anno=None, col_anno=None, col_anno2=None,
              row_order_ids=None, col_order_ids=None, col_order_ids2=None, 
              xticklabels=False, yticklabels=False,
              row_cluster=False, col_cluster=False,
              **kwargs):
    import seaborn as sns
    import numpy as np
    
    set1_colors = sns.color_palette("tab20", 20)
    set2_colors = sns.color_palette("gist_ncar", 6)
    
    idx_row = range(X.shape[0])
    
    if col_anno is not None:
        if col_order_ids is None:
            col_order_ids = list(np.unique(col_anno))
        else:
            col_order_ids = list(col_order_ids)
        col_num = np.array([col_order_ids.index(x) for x in col_anno])
        col_colors = np.array(set1_colors)[col_num]
    else:
        col_colors = None
        col_num = np.zeros(X.shape[1])
    
    if col_anno2 is not None:
        if col_order_ids2 is None:
            col_order_ids2 = list(np.unique(col_anno2))
        else:
            col_order_ids2 = list(col_order_ids2)
        col_num2 = np.array([col_order_ids2.index(x) for x in col_anno2])
        col_colors2 = np.array(set2_colors)[col_num2]
    else:
        col_colors2 = None
        col_num2 = np.zeros(X.shape[1])

    combined_col_num = col_num * len(col_order_ids2) + col_num2
    idx_col = np.argsort(combined_col_num)
    
    g = sns.clustermap(X[idx_row, :][:, idx_col], 
                       row_colors=None, col_colors=[col_colors[idx_col], col_colors2[idx_col]],
                       col_cluster=col_cluster, row_cluster=row_cluster,
                       xticklabels=xticklabels, yticklabels=yticklabels,
                       **kwargs)
    
    if col_anno is not None:
        for i, label in enumerate(col_order_ids):
            g.ax_col_dendrogram.bar(0, 0, color=set1_colors[i],
                                    label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=6)
    
    if col_anno2 is not None:
        for i, label in enumerate(col_order_ids2):
            g.ax_col_dendrogram.bar(0, 0, color=set2_colors[i],
                                    label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=6)
    
    g.cax.set_position([1.01, .2, .03, .45])
    
    return g

## subset mquad so that it only contains variants with given indices
def select_mquad(mquad, barcode, include_variant_names=None, exclude_variant_names=None, include_indices=None, exclude_indices=None, include_cell=None, exclude_cell=None):
    import numpy as np
    import copy
    from mquad.mquad import Mquad
    
    # Create a deep copy of the mquad object to ensure original is not modified
    new_mquad = copy.deepcopy(mquad)
    
    variants = np.array(new_mquad.variants)  # Convert to numpy array for advanced indexing

    if include_variant_names is not None:
        include_indices = [i for i, var in enumerate(variants) if var in include_variant_names]
    
    if exclude_variant_names is not None:
        exclude_indices = [i for i, var in enumerate(variants) if var in exclude_variant_names]

    if include_indices is not None:
        include_mask = np.zeros(variants.shape, dtype=bool)
        include_mask[include_indices] = True
        new_mquad.ad = new_mquad.ad[include_mask, :]
        new_mquad.dp = new_mquad.dp[include_mask, :]
        new_mquad.variants = variants[include_mask].tolist()  # Convert back to list
    
    if exclude_indices is not None:
        exclude_mask = np.ones(variants.shape, dtype=bool)
        exclude_mask[exclude_indices] = False
        new_mquad.ad = new_mquad.ad[exclude_mask, :]
        new_mquad.dp = new_mquad.dp[exclude_mask, :]
        new_mquad.variants = variants[exclude_mask].tolist()  # Convert back to list

    if include_cell is not None:
        new_mquad.ad = new_mquad.ad[:, include_cell]
        new_mquad.dp = new_mquad.dp[:, include_cell]
        new_barcode = barcode[include_cell]
    
    if exclude_cell is not None:
        new_mquad.ad = np.delete(new_mquad.ad, exclude_cell, axis=1)
        new_mquad.dp = np.delete(new_mquad.dp, exclude_cell, axis=1)
        new_barcode = np.delete(barcode, exclude_cell)
    
    return new_mquad, new_barcode


def write_mquad(mquad, barcode, out_dir):
    from scipy.io import mmwrite
    import os
    from scipy import sparse
    os.makedirs(out_dir, exist_ok=True)
    ## convert ad, dp to integer
    ad = mquad.ad.astype(int)
    dp = mquad.dp.astype(int)
    with open(out_dir + "/passed_variant_names.txt", "w") as f:
        ## reformat variant names, split by '_'
        if '>' in mquad.variants[0]:
            f.writelines(var + '\n' for var in mquad.variants)
        else:
            for var in mquad.variants:
                f.write(var.split('_')[1] + var.split('_')[2] + '>' + var.split('_')[3] + '\n')
    with open(out_dir + "/passed_sample_names.txt", "w") as f:
        f.writelines(cb + '\n' for cb in barcode)
    mmwrite(out_dir + "/passed_ad.mtx", sparse.csr_matrix(ad))
    mmwrite(out_dir + "/passed_dp.mtx", sparse.csr_matrix(dp))
    return
