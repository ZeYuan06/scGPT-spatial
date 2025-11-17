import os
import subprocess
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.io import mmwrite
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

R_SCRIPT = "helper.R"

class SeuratRPCAPipeline:
    def __init__(self, h5ad_path, output_dir="seurat_rpca_results"):
        self.h5ad_path = h5ad_path
        self.output_dir = output_dir
        self.mtx_dir = "../data/mtx_format"
        
    def convert_h5ad_to_mtx(self):
        """Step 1: Convert h5ad to MTX format"""
        print("STEP 1: Converting h5ad to MTX format")
        
        if os.path.exists(f"{self.mtx_dir}/matrix.mtx") and os.path.exists(f"{self.mtx_dir}/metadata.csv"):
            print("✓ MTX files already exist, skipping conversion...\n")
            return
        
        print(f"Reading {self.h5ad_path}...")
        adata = sc.read_h5ad(self.h5ad_path)
        print(f"✓ Loaded {adata.n_obs} cells and {adata.n_vars} genes")
        
        os.makedirs(self.mtx_dir, exist_ok=True)
        
        # Convert to sparse matrix
        if sparse.issparse(adata.X):
            mtx_matrix = adata.X.T
        else:
            mtx_matrix = sparse.csr_matrix(adata.X.T)
        
        # Save files
        print("Saving matrix...")
        mmwrite(f"{self.mtx_dir}/matrix.mtx", mtx_matrix)
        
        print("Saving genes...")
        with open(f"{self.mtx_dir}/genes.tsv", 'w') as f:
            for gene in adata.var_names:
                f.write(f"{gene}\t{gene}\n")
        
        print("Saving barcodes...")
        with open(f"{self.mtx_dir}/barcodes.tsv", 'w') as f:
            for barcode in adata.obs_names:
                f.write(f"{barcode}\n")
        
        print("Saving metadata...")
        metadata = adata.obs.copy()
        for col in metadata.columns:
            if pd.api.types.is_categorical_dtype(metadata[col]):
                print(f"  Converting categorical column '{col}' to string")
                metadata[col] = metadata[col].astype(str)
        
        metadata.to_csv(f"{self.mtx_dir}/metadata.csv")
        print("✓ Conversion completed!\n")
    
    def run_seurat_computation(self):
        """Step 2: Run R script for computation"""
        print("STEP 2: Running Seurat RPCA computation (R)")
        
        
        if not os.path.exists(R_SCRIPT):
            raise FileNotFoundError(f"R script '{R_SCRIPT}' not found!")
        print(f"Executing {R_SCRIPT}...\n")
        result = subprocess.run(
            ["Rscript", R_SCRIPT],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n--- R Script Errors ---")
            print(result.stderr)
            raise RuntimeError("R script execution failed!")
        
        print("✓ R computation completed!\n")
    
    def load_results(self):
        """Load computed results from R (simplified)"""
        print("STEP 3: Loading results from R")
        
        # Load analysis info
        with open(f"{self.output_dir}/analysis_info.json", 'r') as f:
            info = json.load(f)
        
        # Load UMAP coordinates
        umap_df = pd.read_csv(f"{self.output_dir}/umap_coordinates.csv")
        
        # Load batch stats if available
        batch_stats = None
        if os.path.exists(f"{self.output_dir}/batch_stats.csv"):
            batch_stats = pd.read_csv(f"{self.output_dir}/batch_stats.csv")
        
        print("✓ Results loaded!\n")
        
        return {
            'info': info,
            'umap': umap_df,
            'batch_stats': batch_stats
        }
    
    def compute_metrics(self, results):
        """Step 3: Compute batch mixing metrics"""
        print("STEP 3: Computing batch mixing metrics")
        
        umap_df = results['umap']
        
        # Extract UMAP coordinates
        X_umap = umap_df[['UMAP_1', 'UMAP_2']].values
        metrics = {}
        
        # Check if batch column exists
        if 'batch' not in umap_df.columns:
            print("No batch information found, skipping metrics computation.\n")
            return metrics, umap_df
        
        batch_labels = umap_df['batch'].values
        n_batches = len(np.unique(batch_labels))
        print(f"Number of batches: {n_batches}")
        
        if n_batches < 2:
            print("Only one batch found, batch mixing metrics not applicable.\n")
            return metrics, umap_df
        
        # Batch ASW (lower is better - indicates good mixing)
        print("\nComputing Batch ASW...")
        asw_batch = silhouette_score(X_umap, batch_labels)
        metrics['asw_batch'] = asw_batch
        
        print(f"✓ Batch ASW: {asw_batch:.4f}")
        print(f"  Interpretation: Lower values indicate better batch mixing")
        print(f"  Range: [-1, 1], closer to 0 or negative is better\n")
        
        # Save metrics
        metrics_path = f"{self.output_dir}/batch_mixing_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'batch_asw': asw_batch,
                'n_batches': int(n_batches),
                'interpretation': 'Lower is better (closer to 0 or negative)'
            }, f, indent=2)
        
        print(f"✓ Metrics saved to: {metrics_path}\n")
        
        return metrics, umap_df

    def create_visualizations(self, results, metrics, umap_df):
        """Step 4: Create batch mixing visualizations"""
        print("STEP 4: Creating visualizations")
        
        if 'batch' not in umap_df.columns or len(metrics) == 0:
            print("No batch information, skipping visualizations.\n")
            return []        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        
        saved_plots = []
        
        # =================================================================
        # Plot 1: UMAP colored by batch
        # =================================================================
        print("Creating UMAP batch plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_batches = sorted(umap_df['batch'].unique())
        n_batches = len(unique_batches)
        batch_colors = plt.cm.Set2(np.linspace(0, 1, n_batches))
        
        for i, batch in enumerate(unique_batches):
            mask = umap_df['batch'] == batch
            ax.scatter(
                umap_df.loc[mask, 'UMAP_1'],
                umap_df.loc[mask, 'UMAP_2'],
                c=[batch_colors[i]],
                label=batch,
                s=15,
                alpha=0.7
            )
        
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'RPCA Integration - Batch Mixing\n' +
                    f'Batch ASW = {metrics["asw_batch"]:.4f}',
                    fontsize=14, weight='bold')
        ax.legend(title='Batch', frameon=True, fontsize=10)
        
        plt.tight_layout()
        plot_path = f"{self.output_dir}/batch_mixing_umap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        saved_plots.append(plot_path)
        print(f"✓ Saved: {plot_path}")
        plt.close()

        return saved_plots

    def run(self):
        """Run complete pipeline"""
        print("SEURAT RPCA PIPELINE WITH ASW EVALUATION")
        
        # Step 1: Convert data
        self.convert_h5ad_to_mtx()
        
        # Step 2: Run R computation
        self.run_seurat_computation()
        
        # Step 3: Load results
        results = self.load_results()
        
        # Step 4: Compute metrics (ASW, etc.)
        metrics, umap_df = self.compute_metrics(results)
        
        # Step 5: Create visualizations
        plot_paths = self.create_visualizations(results, metrics, umap_df)

        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        
        return results, metrics, plot_paths

if __name__ == "__main__":
    # Configuration
    H5AD_PATH = "../data/processed_fetal_lung_visium_xenium.h5ad"
    
    # Run pipeline
    pipeline = SeuratRPCAPipeline(H5AD_PATH)
    results, metrics, plots = pipeline.run()
    
    print("✓ All done! Check the seurat_rpca_results/ directory.")
