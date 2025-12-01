import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scib


class PCAPipeline:
    def __init__(self, h5ad_path=None, mtx_path=None, output_dir="pca_results"):
        self.h5ad_path = h5ad_path
        self.mtx_dir = mtx_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def mtx_load_and_preprocess(self):
        """Load MTX data and perform basic preprocessing"""
        print("STEP 1: Loading data from MTX format")

        # 1. Define paths
        mtx_path = os.path.join(self.mtx_dir, "matrix.mtx")
        genes_path = os.path.join(self.mtx_dir, "genes.tsv")
        barcodes_path = os.path.join(self.mtx_dir, "barcodes.tsv")
        meta_path = os.path.join(self.mtx_dir, "combined_metadata.csv")

        if not os.path.exists(mtx_path):
            raise FileNotFoundError(f"MTX file not found: {mtx_path}")

        # 2. Load Matrix (Scanpy expects Cell x Gene, R saves Gene x Cell, so we transpose)
        print(f"Reading matrix from {mtx_path}...")
        adata = sc.read_mtx(mtx_path).T

        # 3. Load Genes and Barcodes
        genes = pd.read_csv(genes_path, header=None, sep="\t")
        barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")

        adata.var_names = genes[0].values
        adata.obs_names = barcodes[0].values

        # 4. Load Metadata
        print(f"Reading metadata from {meta_path}...")
        meta = pd.read_csv(meta_path, index_col=0)

        # Align metadata with adata
        # Ensure indices match (Scanpy is strict about this)
        adata.obs = meta.loc[adata.obs_names]

        print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

        # Normalization (Standard LogNormalize)
        print("Normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Scale data (Required for PCA)
        print("Scaling data...")
        sc.pp.scale(adata, max_value=10)

        self.adata = adata
        print("Preprocessing completed!\n")
        return adata

    def load_and_preprocess(self):
        """Load h5ad and perform basic preprocessing"""
        print("STEP 1: Loading and preprocessing data")

        print(f"Reading {self.h5ad_path}...")
        adata = sc.read_h5ad(self.h5ad_path)
        print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")

        # QC filtering (same as Seurat)
        sc.pp.filter_cells(adata, min_genes=50)
        sc.pp.filter_cells(adata, max_genes=500)
        print(f"After QC: {adata.n_obs} cells retained")

        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # # Feature selection
        # sc.pp.highly_variable_genes(adata, n_top_genes=50)
        # adata = adata[:, adata.var.highly_variable]
        # print(f"Selected {adata.n_vars} highly variable genes")

        # Scale data
        print("Scaling data...")
        sc.pp.scale(self.adata, max_value=10)

        self.adata = adata
        print("Preprocessing completed!\n")
        return adata

    def run_pca(self, n_pcs=50):
        """Run PCA dimensionality reduction"""
        print("STEP 2: Running PCA")

        # Run PCA
        sc.tl.pca(self.adata, n_comps=n_pcs)
        print(f"PCA completed with {n_pcs} components")

        # Compute neighbors and UMAP
        sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=10)
        sc.tl.umap(self.adata)
        print("UMAP completed")

        return self.adata

    def compute_metrics_scib(self):
        """Compute batch mixing and celltype preservation metrics using scib"""
        print("STEP 3: Computing metrics (scib)\n")

        adata = self.adata

        if "batch" not in adata.obs.columns:
            raise ValueError(
                "Batch information ('batch' column) is required in adata.obs for metrics computation."
            )

        if "celltype" not in adata.obs.columns:
            raise ValueError(
                "Celltype information ('celltype' column) is required in adata.obs for metrics computation."
            )

        n_batches = len(np.unique(adata.obs["batch"]))
        if n_batches < 2:
            raise ValueError(
                "At least two batches are required for batch mixing metrics computation."
            )

        # Ensure categorical types
        adata.obs["batch"] = pd.Categorical(adata.obs["batch"])
        adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"])

        # Store PCA in obsm['X'] for scib
        adata.obsm["X"] = adata.obsm["X_pca"].copy()

        print(f"Computing scib metrics for {n_batches} batches...")
        print(f"  Embedding space: PCA ({adata.obsm['X'].shape[1]} dims)")
        print(f"  Total cells: {adata.n_obs}")
        print(f"  Cell types: {len(adata.obs['celltype'].cat.categories)}\n")

        # Compute scib metrics (aligned with Seurat)
        results = scib.metrics.metrics(
            adata,
            adata_int=adata,
            batch_key="batch",
            label_key="celltype",
            embed="X",
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
            ilisi_=False,
            clisi_=False,
        )

        result_dict = results[0].to_dict()

        # Extract all metrics (same keys as Seurat)
        asw_label_batch = result_dict.get("ASW_label/batch", np.nan)
        graph_conn = result_dict.get("graph_conn", np.nan)
        asw_label = result_dict.get("ASW_label", np.nan)
        nmi = result_dict.get("NMI_cluster/label", np.nan)
        ari = result_dict.get("ARI_cluster/label", np.nan)

        # Compute AvgBIO: average of ASW_label, NMI, ARI
        if not np.isnan(asw_label) and not np.isnan(nmi) and not np.isnan(ari):
            avg_bio = np.mean([asw_label, nmi, ari])
        else:
            avg_bio = np.nan

        # Compute AvgBAT: average of ASW_label/batch and graph_conn
        if not np.isnan(asw_label_batch) and not np.isnan(graph_conn):
            avg_bat = np.mean([asw_label_batch, graph_conn])
        else:
            avg_bat = np.nan

        metrics = {
            "batch_asw": asw_label_batch,
            "graph_conn": graph_conn,
            "asw_label": asw_label,
            "nmi": nmi,
            "ari": ari,
            "avg_bio": avg_bio,
            "avg_bat": avg_bat,
            "n_batches": n_batches,
            "n_celltypes": len(adata.obs["celltype"].cat.categories),
            "n_cells": adata.n_obs,
            "scib_results": result_dict,
        }

        # Print results
        print("=" * 70)
        print("BATCH MIXING & CELLTYPE PRESERVATION METRICS (scib)")
        print("=" * 70)
        print("\nBatch Metrics (mixing quality):")
        print(f"  ASW (label/batch):   {asw_label_batch:>8.4f}")
        print(f"  Graph Connectivity:  {graph_conn:>8.4f}")
        print(f"  ➜ AvgBAT:            {avg_bat:>8.4f}")

        print("\nBio Conservation (celltype preservation):")
        print(f"  ASW (label):         {asw_label:>8.4f}")
        print(f"  NMI (cluster/label): {nmi:>8.4f}")
        print(f"  ARI (cluster/label): {ari:>8.4f}")
        print(f"  ➜ AvgBIO:            {avg_bio:>8.4f}")
        print("=" * 70 + "\n")

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _save_metrics(self, metrics):
        """Save metrics to JSON"""
        save_dict = {
            "batch_metrics": {
                "asw_label_batch": (
                    float(metrics["batch_asw"])
                    if not np.isnan(metrics["batch_asw"])
                    else None
                ),
                "graph_connectivity": (
                    float(metrics["graph_conn"])
                    if not np.isnan(metrics["graph_conn"])
                    else None
                ),
                "avg_bat": (
                    float(metrics["avg_bat"])
                    if not np.isnan(metrics["avg_bat"])
                    else None
                ),
            },
            "bio_conservation": {
                "asw_label": (
                    float(metrics["asw_label"])
                    if not np.isnan(metrics["asw_label"])
                    else None
                ),
                "nmi_cluster_label": (
                    float(metrics["nmi"]) if not np.isnan(metrics["nmi"]) else None
                ),
                "ari_cluster_label": (
                    float(metrics["ari"]) if not np.isnan(metrics["ari"]) else None
                ),
                "avg_bio": (
                    float(metrics["avg_bio"])
                    if not np.isnan(metrics["avg_bio"])
                    else None
                ),
            },
            "dataset_info": {
                "n_batches": int(metrics["n_batches"]),
                "n_celltypes": int(metrics["n_celltypes"]),
                "n_cells": int(metrics["n_cells"]),
                "method": "PCA",
            },
        }

        with open(f"{self.output_dir}/pca_metrics.json", "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"Metrics saved to {self.output_dir}/pca_metrics.json\n")

    def save_calculation_results(self, metrics):
        """Save UMAP coordinates and flat metrics for quick plotting later"""
        print(f"Saving intermediate results to {self.output_dir}...")

        adata = self.adata

        # Save UMAP DataFrame (contains UMAP_1, UMAP_2, batch, celltype)
        if "X_umap" in adata.obsm:
            umap_df = pd.DataFrame(
                adata.obsm["X_umap"],
                columns=["UMAP_1", "UMAP_2"],
                index=adata.obs_names,
            )
            umap_df["batch"] = adata.obs["batch"].values
            umap_df["celltype"] = adata.obs["celltype"].values
            umap_df.to_csv(f"{self.output_dir}/final_umap.csv", index=False)

        # Save Flat Metrics (for plot titles)
        # We convert numpy types to python types for JSON
        def convert(o):
            if isinstance(o, np.generic):
                return o.item()
            return o

        with open(f"{self.output_dir}/metrics_summary.json", "w") as f:
            json.dump(metrics, f, default=convert, indent=2)

        print("Results saved.\n")

    def load_calculation_results(self):
        """Load previously saved results to skip calculation"""
        print(f"Loading saved results from {self.output_dir}...")

        umap_path = f"{self.output_dir}/final_umap.csv"
        metrics_path = f"{self.output_dir}/metrics_summary.json"

        if not os.path.exists(umap_path) or not os.path.exists(metrics_path):
            raise FileNotFoundError(
                "Saved results not found. Please run without plot_only=True first."
            )

        # Load UMAP
        self.results = {}
        self.results["umap"] = pd.read_csv(umap_path)

        # Load Metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        print("✓ Loaded UMAP coordinates and metrics successfully.")
        return metrics

    def create_visualizations(self, metrics):
        """Create visualizations"""
        print("STEP 4: Creating visualizations")

        # Determine source of data: loaded DataFrame or current AnnData
        if hasattr(self, "results") and "umap" in self.results:
            umap_df = self.results["umap"]
        else:
            adata = self.adata
            # Prepare DataFrame for plotting
            umap_df = pd.DataFrame(
                adata.obsm["X_umap"],
                columns=["UMAP_1", "UMAP_2"],
                index=adata.obs_names,
            )
            umap_df["batch"] = adata.obs["batch"].values
            umap_df["celltype"] = adata.obs["celltype"].values

        # Shuffle for better visualization
        umap_df = umap_df.sample(frac=1, random_state=42).reset_index(drop=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        saved_plots = []

        # Plot 1: UMAP colored by batch
        scatter_kwargs = {"s": 1.5, "alpha": 0.8, "edgecolors": "none"}
        fig, ax = plt.subplots(figsize=(12, 9))

        unique_batches = sorted(umap_df["batch"].unique())
        n_batches = len(unique_batches)
        colors = (
            plt.cm.jet(np.linspace(0, 1, n_batches))
            if n_batches > 10
            else plt.cm.Set2(np.linspace(0, 1, n_batches))
        )
        colors = [tuple(c) for c in colors]
        batch_color_map = dict(zip(unique_batches, colors))

        c_array = umap_df["batch"].map(batch_color_map)
        ax.scatter(umap_df["UMAP_1"], umap_df["UMAP_2"], c=c_array, **scatter_kwargs)

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=b,
                markerfacecolor=batch_color_map[b],
                markersize=8,
            )
            for b in unique_batches
        ]

        title = f"PCA: Batch Mixing Quality\n"
        title += f'ASW: {metrics["batch_asw"]:.4f} | '
        title += f'GraphConn: {metrics["graph_conn"]:.4f} | '
        title += f'AvgBAT: {metrics["avg_bat"]:.4f}'

        ax.set_title(title, fontsize=14, weight="bold")
        if n_batches > 10:
            ax.legend(
                handles=legend_elements,
                title="Batch",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=9,
                ncol=1,
            )
        else:
            ax.legend(handles=legend_elements, title="Batch", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{self.output_dir}/01_batch_mixing_umap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        saved_plots.append(plot_path)
        print(f"Saved: {plot_path}")
        plt.close()

        # Plot 2: UMAP colored by celltype
        fig, ax = plt.subplots(figsize=(12, 9))

        unique_celltypes = sorted(umap_df["celltype"].unique())
        n_celltypes = len(unique_celltypes)
        colors_ct = plt.cm.tab20(np.linspace(0, 1, n_celltypes))
        if n_celltypes > 20:
            colors_ct = plt.cm.gist_ncar(np.linspace(0, 0.9, n_celltypes))

        ct_color_map = dict(zip(unique_celltypes, colors_ct))
        c_array_ct = umap_df["celltype"].map(ct_color_map)
        ax.scatter(umap_df["UMAP_1"], umap_df["UMAP_2"], c=c_array_ct, **scatter_kwargs)

        # Legend
        legend_elements_ct = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=ct,
                markerfacecolor=ct_color_map[ct],
                markersize=8,
            )
            for ct in unique_celltypes
        ]

        title = f"PCA: Celltype Preservation\n"
        title += f'ASW: {metrics["asw_label"]:.4f} | '
        title += f'NMI: {metrics["nmi"]:.4f} | '
        title += f'ARI: {metrics["ari"]:.4f} | '
        title += f'AvgBIO: {metrics["avg_bio"]:.4f}'

        ax.set_title(title, fontsize=14, weight="bold")
        ax.legend(
            handles=legend_elements_ct,
            title="Celltype",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
            ncol=1,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{self.output_dir}/02_celltype_preservation_umap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        saved_plots.append(plot_path)
        print(f"✓ Saved: {plot_path}")
        plt.close()

        return saved_plots

    def run(self, plot_only=False):
        """Run complete pipeline"""
        print("\n" + "=" * 70)
        print("PCA BATCH INTEGRATION & CELLTYPE EVALUATION")
        print("Pipeline: PCA + scib Metrics (aligned with Seurat)")
        print("=" * 70 + "\n")

        if plot_only:
            print("MODE: Plotting Only (Skipping Calculation)")
            metrics = self.load_calculation_results()
            plots = self.create_visualizations(metrics)
            print("=" * 70)
            print("PLOTTING COMPLETED!")
            print("=" * 70 + "\n")
            return None, metrics, plots

        if self.mtx_dir is not None:
            self.mtx_load_and_preprocess()
        else:
            self.load_and_preprocess()
        self.run_pca()
        metrics = self.compute_metrics_scib()
        self.save_calculation_results(metrics)
        plots = self.create_visualizations(metrics)

        print("=" * 70)
        print("✓ PIPELINE COMPLETED!")
        print("=" * 70 + "\n")

        if "avg_bat" in metrics and not np.isnan(metrics["avg_bat"]):
            print("📊 INTEGRATION QUALITY SUMMARY")
            print("-" * 70)
            print(
                f"  AvgBAT (Batch):     {metrics['avg_bat']:>8.4f}  (Range: [0, 1]) ⭐"
            )
            print(
                f"  AvgBIO (Celltype):  {metrics['avg_bio']:>8.4f}  (Range: [0, 1]) ⭐"
            )
            print("-" * 70 + "\n")

        return self.adata, metrics, plots


if __name__ == "__main__":
    pipeline = PCAPipeline(mtx_path="../data/mtx_format_from_rds")
    adata, metrics, plots = pipeline.run(plot_only=True)
