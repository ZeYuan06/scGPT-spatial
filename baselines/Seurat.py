import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.io import mmwrite
import scib

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

pandas2ri.activate()
numpy2ri.activate()


class SeuratRPCAPipeline:
    def __init__(self, h5ad_path, output_dir="seurat_rpca_results"):
        self.h5ad_path = h5ad_path
        self.output_dir = output_dir
        self.mtx_dir = "../data/mtx_format"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.seurat = importr('Seurat')

    def convert_h5ad_to_mtx(self):
        """Convert h5ad to MTX format"""
        print("STEP 1: Converting h5ad to MTX format")
        
        if os.path.exists(f"{self.mtx_dir}/matrix.mtx") and os.path.exists(f"{self.mtx_dir}/metadata.csv"):
            print("MTX files already exist, skipping...\n")
            return
        
        print(f"Reading {self.h5ad_path}...")
        adata = sc.read_h5ad(self.h5ad_path)
        print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
        
        os.makedirs(self.mtx_dir, exist_ok=True)
        
        # Save matrix
        if sparse.issparse(adata.X):
            mtx_matrix = adata.X.T
        else:
            mtx_matrix = sparse.csr_matrix(adata.X.T)
        mmwrite(f"{self.mtx_dir}/matrix.mtx", mtx_matrix)
        
        # Save genes
        with open(f"{self.mtx_dir}/genes.tsv", 'w') as f:
            for gene in adata.var_names:
                f.write(f"{gene}\t{gene}\n")
        
        # Save barcodes
        with open(f"{self.mtx_dir}/barcodes.tsv", 'w') as f:
            for barcode in adata.obs_names:
                f.write(f"{barcode}\n")
        
        # Save metadata
        metadata = adata.obs.copy()
        for col in metadata.columns:
            if pd.api.types.is_categorical_dtype(metadata[col]):
                metadata[col] = metadata[col].astype(str)
        metadata.to_csv(f"{self.mtx_dir}/metadata.csv")
        
        print("Conversion completed!\n")
    
    def run_seurat_workflow(self):
        """Run Seurat RPCA integration via R"""
        print("STEP 2: Running Seurat RPCA workflow")
        
        # Load MTX data
        ro.r(f'''
            library(Seurat)
            counts <- ReadMtx(
                mtx = "{self.mtx_dir}/matrix.mtx",
                cells = "{self.mtx_dir}/barcodes.tsv",
                features = "{self.mtx_dir}/genes.tsv",
                feature.column = 1
            )
            metadata <- read.csv("{self.mtx_dir}/metadata.csv", row.names = 1)
            obj <- CreateSeuratObject(counts = counts, meta.data = metadata, project = "fetal_lung")
        ''')
        
        n_cells = int(ro.r('ncol(obj)')[0])
        n_genes = int(ro.r('nrow(obj)')[0])
        print(f"✓ Loaded {n_cells} cells and {n_genes} genes")
        
        # Check available columns
        has_celltype = ro.r('("celltype" %in% colnames(obj@meta.data))')[0]
        has_batch = ro.r('("batch" %in% colnames(obj@meta.data))')[0]
        
        if not has_celltype:
            raise ValueError("Celltype information ('celltype' column) is required in metadata.")
        if not has_batch:
            raise Warning("Batch information ('batch' column) not found in metadata. Proceeding without batch integration.")
        
        # QC filtering
        ro.r('''
            obj <- subset(obj, subset = nFeature_RNA > 50 & nFeature_RNA < 500)
        ''')
        n_cells_qc = int(ro.r('ncol(obj)')[0])
        print(f"After QC: {n_cells_qc} cells retained")
        
        # Preprocessing
        ro.r('''
            obj <- NormalizeData(obj)
            obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 50)
            obj <- ScaleData(obj)
            obj <- RunPCA(obj, npcs = 30, verbose = FALSE)
        ''')
        print(f"Preprocessing completed")
        
        # Check batch and integrate
        n_batches = int(ro.r('length(unique(obj@meta.data[["batch"]]))')[0])
        
        if n_batches <= 1:
            raise ValueError("Batch information ('batch' column) must contain at least 2 unique batches for integration.")

        print(f"Batch info found: {n_batches} batches")
        print("Running RPCA integration...")
        ro.r('''
            obj[["RNA"]] <- split(obj[["RNA"]], f = obj@meta.data[["batch"]])
            obj <- IntegrateLayers(
                object = obj,
                method = RPCAIntegration,
                orig.reduction = "pca",
                new.reduction = "integrated.rpca",
                verbose = FALSE
            )
            obj <- FindNeighbors(obj, reduction = "integrated.rpca", dims = 1:10)
            obj <- FindClusters(obj, resolution = 0.5)
            obj <- RunUMAP(obj, reduction = "integrated.rpca", dims = 1:10, reduction.name = "umap.rpca")
        ''')
        print("✓ RPCA integration completed")

        print("Seurat workflow completed!\n")
    
    def extract_results(self):
        """Extract high-dim and UMAP coordinates from R"""
        print("STEP 3: Extracting results from R")
        
        # Extract high-dimensional coordinates (for metrics calculation)
        reduction_coords = ro.r(f'''
            coords <- Embeddings(obj, reduction = "integrated.rpca")
            as.data.frame(coords)
        ''')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            reduction_df = ro.conversion.rpy2py(reduction_coords)
        
        # Extract UMAP coordinates (for visualization)
        umap_coords = ro.r(f'''
            coords <- Embeddings(obj, reduction = "umap.rpca")
            as.data.frame(coords)
        ''')
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            umap_df = ro.conversion.rpy2py(umap_coords)
        
        umap_df.columns = ['UMAP_1', 'UMAP_2']
        
        # Extract batch and celltype info
        batch_info = ro.r('obj@meta.data[["batch"]]')
        with localconverter(ro.default_converter + pandas2ri.converter):
            batch_data = ro.conversion.rpy2py(batch_info)
        reduction_df['batch'] = batch_data
        umap_df['batch'] = batch_data

        celltype_info = ro.r('obj@meta.data[["celltype"]]')
        with localconverter(ro.default_converter + pandas2ri.converter):
            celltype_data = ro.conversion.rpy2py(celltype_info)
        reduction_df['celltype'] = celltype_data
        umap_df['celltype'] = celltype_data
        
        # Get metadata
        n_cells = int(ro.r('ncol(obj)')[0])
        n_genes = int(ro.r('nrow(obj)')[0])
        
        # Get batch stats
        batch_table = ro.r('as.data.frame(table(obj@meta.data[["batch"]]))')
        with localconverter(ro.default_converter + pandas2ri.converter):
            batch_stats = ro.conversion.rpy2py(batch_table)
        batch_stats.columns = ['batch', 'count']
        
        # Get celltype stats
        celltype_table = ro.r('as.data.frame(table(obj@meta.data[["celltype"]]))')
        with localconverter(ro.default_converter + pandas2ri.converter):
            celltype_stats = ro.conversion.rpy2py(celltype_table)
        celltype_stats.columns = ['celltype', 'count']
        
        self.results = {
            'info': {
                'n_cells': n_cells,
                'n_genes': n_genes,
            },
            'reduction': reduction_df,  # High-dim for metrics
            'umap': umap_df,  # 2D for visualization
            'batch_stats': batch_stats,
            'celltype_stats': celltype_stats
        }
        
        print("Results extracted!\n")
        return self.results
    
    def compute_metrics_scib(self):
        """Compute batch mixing metrics using scib (standard method)"""
        print("STEP 4: Computing batch mixing metrics (scib)\n")
        
        reduction_df = self.results['reduction']
        umap_df = self.results['umap']
     
        n_batches = len(np.unique(reduction_df['batch']))
        
        # Create AnnData object for scib metrics
        coord_cols = [col for col in reduction_df.columns if col not in ['batch', 'celltype']]
        X_high_dim = reduction_df[coord_cols].values
        
        adata = sc.AnnData(X=X_high_dim)
        adata.obsm['X'] = X_high_dim
        adata.obs['batch'] = pd.Categorical(reduction_df['batch'].values)
        adata.obs['celltype'] = pd.Categorical(reduction_df['celltype'].values)
        adata.obsm['X_umap'] = umap_df[['UMAP_1', 'UMAP_2']].values
        
        print(f"Computing scib metrics for {n_batches} batches...")
        print(f"  Total cells: {X_high_dim.shape[0]}")
        print(f"  Cell types: {len(np.unique(adata.obs['celltype']))}\n")
        
        # Compute scib metrics
        results = scib.metrics.metrics(
            adata,
            adata_int=adata,
            batch_key='batch',
            label_key='celltype',
            embed='X',
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            graph_conn_=True,
            pcr_=False,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,
            ari_=True, 
            cell_cycle_=False,
            kBET_=False,
            ilisi_=True,
            clisi_=True,
        )
        
        result_dict = results[0].to_dict()
        
        # Extract all metrics
        asw_label_batch = result_dict.get('ASW_label/batch', np.nan)
        graph_conn = result_dict.get('graph_conn', np.nan)
        asw_label = result_dict.get('ASW_label', np.nan)
        nmi = result_dict.get('NMI_cluster/label', np.nan)
        ari = result_dict.get('ARI_cluster/label', np.nan)
        
        if not np.isnan(asw_label) and not np.isnan(nmi) and not np.isnan(ari):
            avg_bio = np.mean([asw_label, nmi, ari])
        else:
            avg_bio = np.nan
        # Compute AvgBAT: (ASW_label/batch + graph_conn) / 2
        if not np.isnan(asw_label_batch) and not np.isnan(graph_conn):
            avg_bat = np.mean([asw_label_batch, graph_conn])
        else:
            avg_bat = np.nan
        
        metrics = {
            'batch_asw': asw_label_batch,
            'graph_conn': graph_conn,
            'label_asw': asw_label,
            'nmi': nmi,
            'ari': ari,
            'avg_bat': avg_bat,
            'avg_bio': avg_bio,
            'n_batches': n_batches,
            'n_celltypes': len(np.unique(adata.obs['celltype'])),
            'n_cells': X_high_dim.shape[0],
            'scib_results': result_dict
        }
        
        # Print results
        print("="*70)
        print("BATCH MIXING & CELLTYPE PRESERVATION METRICS (scib)")
        print("="*70)
        print("\nBatch Metrics (mixing quality):")
        print(f"  ASW (label/batch):   {asw_label_batch:>8.4f}  (Range: [-1, 1], higher = better)")
        print(f"  Graph Connectivity:  {graph_conn:>8.4f}  (Range: [0, 1], higher = better)")
        
        print("\nCelltype Preservation:")
        print(f"  ASW                   {asw_label:>8.4f}")
        print(f"  NMI:                 {nmi:>8.4f}  (batch-celltype independence, lower = better)")
        print(f"  ARI:                 {ari:>8.4f}  (clustering agreement, higher = better)")
        
        print("\nIntegration Quality:")
        print(f"  AvgBAT:              {avg_bat:>8.4f}  (Range: [0, 1], higher = better) ⭐")
        print(f" AVGBIO                 {avg_bio:>8.4f}")
        print("="*70 + "\n")
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics):
        """Save metrics to JSON"""
        # Prepare data for JSON serialization
        save_dict = {
            'batch_metrics': {
                'asw_label_batch': float(metrics['batch_asw']) if not np.isnan(metrics['batch_asw']) else None,
                'graph_connectivity': float(metrics['graph_conn']) if not np.isnan(metrics['graph_conn']) else None,
            },
            'celltype_preservation': {
                'asw_label': float(metrics['label_asw']) if not np.isnan(metrics['label_asw']) else None,
                'nmi': float(metrics['nmi']) if not np.isnan(metrics['nmi']) else None,
                'ari': float(metrics['ari']) if not np.isnan(metrics['ari']) else None,
            },
            'integration_quality': {
                'avg_bat': float(metrics['avg_bat']) if not np.isnan(metrics['avg_bat']) else None,
                'avg_bio': float(metrics['avg_bio']) if not np.isnan(metrics['avg_bio']) else None,
            },
            'dataset_info': {
                'n_batches': int(metrics['n_batches']),
                'n_celltypes': int(metrics['n_celltypes']),
                'n_cells': int(metrics['n_cells']),
            },
        }
        
        with open(f"{self.output_dir}/batch_mixing_metrics.json", 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Metrics saved to {self.output_dir}/batch_mixing_metrics.json\n")
    
    def create_visualizations(self, metrics):
        """Create batch mixing and celltype preservation visualizations"""
        print("STEP 5: Creating visualizations")
        
        umap_df = self.results['umap']
        
        if 'batch' not in umap_df.columns or np.isnan(metrics.get('avg_bat', np.nan)):
            raise ValueError("The caculated AVGBATCH is NOT A NUMBER")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        saved_plots = []
        
        # Plot 1: UMAP colored by batch
        fig, ax = plt.subplots(figsize=(12, 9))
        
        unique_batches = sorted(umap_df['batch'].unique())
        n_batches = len(unique_batches)
        colors = plt.cm.Set2(np.linspace(0, 1, n_batches))
        
        for i, batch in enumerate(unique_batches):
            mask = umap_df['batch'] == batch
            ax.scatter(umap_df.loc[mask, 'UMAP_1'], umap_df.loc[mask, 'UMAP_2'],
                      c=[colors[i]], label=batch, s=15, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        
        title = f'Batch Mixing Quality\n'
        title += f'ASW: {metrics["batch_asw"]:.4f} | '
        title += f'GraphConn: {metrics["graph_conn"]:.4f} | '
        title += f'AvgBAT: {metrics["avg_bat"]:.4f}'
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend(title='Batch', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/01_batch_mixing_umap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot_path)
        print(f"✓ Saved: {plot_path}")
        plt.close()
        
        # Plot 2: UMAP colored by celltype
        fig, ax = plt.subplots(figsize=(12, 9))
        
        unique_celltypes = sorted(umap_df['celltype'].unique())
        n_celltypes = len(unique_celltypes)
        colors_ct = plt.cm.tab20(np.linspace(0, 1, n_celltypes))
        
        for i, ct in enumerate(unique_celltypes):
            mask = umap_df['celltype'] == ct
            ax.scatter(umap_df.loc[mask, 'UMAP_1'], umap_df.loc[mask, 'UMAP_2'],
                        c=[colors_ct[i]], label=ct, s=15, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        
        title = f'Celltype Preservation\n'
        title += f'ASW: {metrics["label_asw"]:.4f} | '
        title += f'NMI: {metrics["nmi"]:.4f} | '
        title += f'ARI: {metrics["ari"]:.4f} |'
        title += f'AVGBIO: {metrics["avg_bio"]:.4f}'
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend(title='Celltype', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/02_celltype_preservation_umap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot_path)
        print(f"✓ Saved: {plot_path}")
        plt.close()
        
        return saved_plots
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("SEURAT RPCA BATCH INTEGRATION & CELLTYPE EVALUATION")
        print("Pipeline: Seurat RPCA + scib Metrics")
        print("="*70 + "\n")
        
        self.convert_h5ad_to_mtx()
        self.run_seurat_workflow()
        self.extract_results()
        metrics = self.compute_metrics_scib()
        plots = self.create_visualizations(metrics)
        
        print("="*70)
        print("PIPELINE COMPLETED!")
        print("="*70 + "\n")
        
        return self.results, metrics, plots


if __name__ == "__main__":
    pipeline = SeuratRPCAPipeline("../data/processed_fetal_lung_visium_xenium.h5ad")
    results, metrics, plots = pipeline.run()
