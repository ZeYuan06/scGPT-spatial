import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Assumes the column representing "Visium" vs "Xenium" is named 'modality'
# *** Please replace 'modality' with the correct column name in your data ***
YOUR_MODALITY_COLUMN = 'batch' # e.g., 'batch', 'source', 'dataset_id', etc.

# Assumes the column representing cell type is named 'cell_type'
YOUR_CELL_TYPE_COLUMN = 'celltype'

# --- 1. Load and preprocess ---
adata = sc.read_h5ad('../data/processed_fetal_lung_visium_xenium.h5ad')

print(f"Original data shape: {adata.shape}") # Should show (n_cells, 333)
print(f"Data type: {type(adata.X)}")

sc.pp.filter_cells(adata, min_genes=50)
sc.pp.filter_genes(adata, min_cells=3)
print(f"After filtering: {adata.shape}")

# --- 2. Normalization ---
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# --- 3. Backup data ---
# We no longer select HVGs, so we backup before scaling
# This way .raw stores the log-normalized, unscaled full data
adata.raw = adata

# --- 4. Remove HVG selection ---
# sc.pp.highly_variable_genes(adata, ...)
# adata = adata[:, adata.var.highly_variable]
# print(f"Using all {adata.n_vars} genes for PCA.")

# --- 5. Scaling ---
# Now we scale all 333 genes
sc.pp.scale(adata, max_value=10)

# --- 6. Perform PCA ---
# n_comps=50 is reasonable since it's less than 333
sc.tl.pca(adata, n_comps=50) 

# --- 7. Visualize variance explained (elbow plot) ---
# Check this plot to decide 'n_pcs'
sc.pl.pca_variance_ratio(adata, log=False, save='.png')

# --- 9. Downstream analysis based on PCA ---
# (Recommended) Adjust 'n_pcs' based on the elbow plot from step 7
N_PCS_TO_USE = 10 # This is a placeholder, please modify based on the elbow plot
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=N_PCS_TO_USE)
sc.tl.umap(adata)

# --- 10. Calculate Silhouette Scores ---
# Get UMAP embeddings
umap_embeddings = adata.obsm['X_umap']

# Calculate ASW for cell type
if YOUR_CELL_TYPE_COLUMN in adata.obs.columns:
    labels_celltype = adata.obs[YOUR_CELL_TYPE_COLUMN].values
    # Filter out cells with missing labels
    valid_idx = pd.notna(labels_celltype)
    if valid_idx.sum() > 1 and len(np.unique(labels_celltype[valid_idx])) > 1:
        asw_celltype = silhouette_score(
            umap_embeddings[valid_idx], 
            labels_celltype[valid_idx]
        )
        print(f"ASW (Cell Type): {asw_celltype:.4f}")
    else:
        asw_celltype = None
        print("Cannot calculate ASW for cell type (insufficient data)")
else:
    asw_celltype = None
    print(f"Warning: Column '{YOUR_CELL_TYPE_COLUMN}' not found")

# Calculate ASW for modality
if YOUR_MODALITY_COLUMN in adata.obs.columns:
    labels_modality = adata.obs[YOUR_MODALITY_COLUMN].values
    valid_idx = pd.notna(labels_modality)
    if valid_idx.sum() > 1 and len(np.unique(labels_modality[valid_idx])) > 1:
        asw_modality = silhouette_score(
            umap_embeddings[valid_idx], 
            labels_modality[valid_idx]
        )
        print(f"ASW (Modality): {asw_modality:.4f}")
    else:
        asw_modality = None
        print("Cannot calculate ASW for modality (insufficient data)")
else:
    asw_modality = None
    print(f"Warning: Column '{YOUR_MODALITY_COLUMN}' not found")

# --- 11. UMAP visualization with ASW annotation ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: By cell type
if YOUR_CELL_TYPE_COLUMN in adata.obs.columns:
    sc.pl.umap(adata, color=YOUR_CELL_TYPE_COLUMN, ax=axes[0], show=False)
    if asw_celltype is not None:
        axes[0].text(0.02, 0.98, f'ASW = {asw_celltype:.4f}', 
                    transform=axes[0].transAxes, 
                    fontsize=12, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_title(f'UMAP colored by {YOUR_CELL_TYPE_COLUMN}')
else:
    axes[0].text(0.5, 0.5, 'Cell type column not found', 
                ha='center', va='center', transform=axes[0].transAxes)

# Plot 2: By modality
if YOUR_MODALITY_COLUMN in adata.obs.columns:
    sc.pl.umap(adata, color=YOUR_MODALITY_COLUMN, ax=axes[1], show=False)
    if asw_modality is not None:
        axes[1].text(0.02, 0.98, f'ASW = {asw_modality:.4f}', 
                    transform=axes[1].transAxes, 
                    fontsize=12, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].set_title(f'UMAP colored by {YOUR_MODALITY_COLUMN}')
else:
    axes[1].text(0.5, 0.5, 'Modality column not found', 
                ha='center', va='center', transform=axes[1].transAxes)
    
plt.tight_layout()
plt.savefig('./figures/umap_with_asw.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nUMAP plots with ASW saved to './figures/umap_with_asw.png'")
