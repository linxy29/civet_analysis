%load_ext autoreload
%autoreload 2

import xclone
import anndata as an
import pandas as pd
import numpy as np
import scipy
print("scipy", scipy.__version__)

import sys
print(sys.executable)

xclone.pp.efficiency_preview()

dataset_name = "gct86"
## output results dir
outdir = "/storage/xxxx/users/xxxx/xclone/tutorials/"

xconfig = xclone.XCloneConfig(dataset_name = dataset_name, module = "RDR")
xconfig.set_figure_params(xclone= True, fontsize = 18)
xconfig.outdir = outdir
xconfig.cell_anno_key = "cluster.pred"
xconfig.ref_celltype = "N"
xconfig.marker_group_anno_key = "cluster.pred"
xconfig.xclone_plot= True
xconfig.plot_cell_anno_key = "cluster"
xconfig.display()

RDR_Xdata = xclone.model.run_RDR(RDR_adata,
            config_file = xconfig)

xconfig = xclone.XCloneConfig(dataset_name = gct86, module = "Combine")
xconfig.set_figure_params(xclone= True, fontsize = 18)
xconfig.outdir = "/home/linxy29/data/maester/oagct/xclone/gct86_cnv"

xconfig.cell_anno_key = "cell_type"
xconfig.ref_celltype = "N"


xconfig.copygain_correct= False

xconfig.xclone_plot= True
xconfig.plot_cell_anno_key = "Clone_ID"
xconfig.merge_loss = False
xconfig.merge_loh = True

xconfig.BAF_denoise = True
xconfig.display()

combine_Xdata = xclone.model.run_combine(RDR_Xdata,
                BAF_merge_Xdata,
                verbose = True,
                run_verbose = True,
                config_file = xconfig)