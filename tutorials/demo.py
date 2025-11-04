# srun -p A800 -N 1 -n 8 --gres=gpu:1 -t 02:00:00 --pty bash
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import warnings
from matplotlib import pyplot as plt

from typing import Dict, Optional, Union
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
import scib

sys.path.insert(0, "../")
import scgpt_spatial
from scgpt_spatial.utils import eval_scib_metrics
warnings.filterwarnings("ignore", category=ResourceWarning)


# Load data
adata = sc.read_h5ad('../data/processed_fetal_lung_visium_xenium.h5ad')

# Run scGPT-spatial zero-shot inference 
model_dir = '../checkpoints/scGPT_spatial_v1'
gene_col = 'feature_name'
cell_type_col = 'celltype'
batch_id_col = 'batch_id'

ref_embed_adata = scgpt_spatial.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    obs_to_save=cell_type_col, 
    batch_size=64,
    return_new_adata=True,
)
ref_embed_adata.obsm['X'] = ref_embed_adata.X.copy()
ref_embed_adata.obs['batch_id'] = adata.obs['batch_id']

result_dict = eval_scib_metrics(ref_embed_adata, batch_key=batch_id_col, label_key=cell_type_col)


# UMAP projection
sc.pp.neighbors(ref_embed_adata, use_rep="X")
sc.tl.umap(ref_embed_adata)

# UMAP visualization with cell labels
custom_palette = ['#23b3b3', '#ff8a5b', '#6aa84f', '#8e44ad', '#f1c40f']
sc.pl.umap(ref_embed_adata, color=cell_type_col, palette=custom_palette, frameon=False, show=False, wspace=0.4)
plt.title('AVGBIO ({:.04f})'.format(result_dict["avg_bio"]))
plt.savefig('../figures/umap_cell_types.png', dpi=300, bbox_inches='tight')
plt.close()

# UMAP visualization with sequencing protocol labels
custom_palette = ['#4a6fa5', '#ea9999', '#9b6bd3','#52c3a3', '#d8c656',  '#6aa84f']
sc.pl.umap(ref_embed_adata, color='batch_id', palette=custom_palette, frameon=False, show=False, wspace=0.4)
plt.title('AVGBATCH ({:.04f})'.format(result_dict["avg_batch"]))
plt.savefig('../figures/umap_batch_protocol.png', dpi=300, bbox_inches='tight')
plt.close()
