import os
import json
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    def __init__(self, h5ad_path=None, rds_path=None, output_dir="seurat_rpca_results"):
        self.h5ad_path = h5ad_path
        self.rds_path = rds_path
        self.output_dir = output_dir
        self.mtx_dir = "../data/mtx_format"
        os.makedirs(self.output_dir, exist_ok=True)

        self.seurat = importr("Seurat")

    def rds_to_mtx_format(self):
        """Convert RDS Seurat list to MTX format for faster loading"""
        print("STEP 1: Converting RDS to MTX format (one-time operation)")

        mtx_rds_dir = f"{self.mtx_dir}_from_rds"
        metadata_path = f"{mtx_rds_dir}/combined_metadata.csv"

        # Check if already converted
        if os.path.exists(f"{mtx_rds_dir}/matrix.mtx") and os.path.exists(
            metadata_path
        ):
            print("MTX files already exist, loading directly...\n")
            return mtx_rds_dir

        os.makedirs(mtx_rds_dir, exist_ok=True)

        print(f"Loading RDS: {self.rds_path}")
        ro.r(
            f"""
            library(Seurat)
            library(Matrix)
            
            seurat_list <- readRDS("{self.rds_path}")
            
            sample_names <- c("GW10", "GW23_L1", "GW23_L2", "GW23_L3", "GW16", "GW23_M2", "GW23_vM", "GW21", "GW23_PV", "GW21_2")
            names(seurat_list) <- sample_names

            # Filter for specific GW23 slides
            target_slides <- c("GW23_L1", "GW23_L2", "GW23_L3", "GW23_M2", "GW23_vM")
            seurat_list <- seurat_list[target_slides]
            
            cat("Filtered to", length(seurat_list), "target slides:", paste(names(seurat_list), collapse=", "), "\\n")
            
            # Extract count matrices and metadata
            counts_list <- list()
            metadata_list <- list()
            
            for(i in seq_along(seurat_list)) {{
                sample_name <- names(seurat_list)[i]
                cat("  Processing:", sample_name, "\\n")
                
                obj <- seurat_list[[i]]
                
                # 1. Get counts
                counts <- GetAssayData(obj, slot = "counts", assay = "Vizgen")
                
                # 2. Get metadata
                meta <- obj@meta.data
                meta$batch <- sample_name  # Batch is the sample identity
                meta$sample <- sample_name
                meta$cell_id <- paste0(sample_name, "_", rownames(meta))
                colnames(counts) <- meta$cell_id
                counts_list[[i]] <- counts
                
                # 3. Map Celltype
                meta$celltype <- as.character(meta$class)
                
                # 4. Map Slide Orientation (Optional: for specific analysis)
                if(sample_name == "GW23_L1") meta$orientation <- "Lateral 1"
                else if(sample_name == "GW23_L2") meta$orientation <- "Lateral 2"
                else if(sample_name == "GW23_L3") meta$orientation <- "Lateral 3"
                else if(sample_name == "GW23_M2") meta$orientation <- "Medial 2"
                else if(sample_name == "GW23_vM") meta$orientation <- "Medial 1"
                else meta$orientation <- "Other"
                
                metadata_list[[i]] <- meta
            }}
            
            # Combine matrices
            cat("\\nCombining matrices...\\n")
            
            # All samples have same 140 genes
            combined_counts <- do.call(cbind, counts_list)
            
            # Combine metadata
            # Find common columns first to avoid rbind error due to different resolution columns
            common_cols <- Reduce(intersect, lapply(metadata_list, colnames))
            cat("  Common metadata columns:", length(common_cols), "\\n")
            
            # Subset all metadata frames to only common columns
            metadata_list <- lapply(metadata_list, function(x) x[, common_cols])
            
            combined_meta <- do.call(rbind, metadata_list)
            rownames(combined_meta) <- combined_meta$cell_id
            
            cat("  Total cells:", ncol(combined_counts), "\\n")
            cat("  Total genes:", nrow(combined_counts), "\\n")
            
            # Print Celltype stats
            cat("\\nCelltype distribution (class):\\n")
            print(table(combined_meta$celltype))
        """
        )

        # Save as MTX
        print("\nSaving to MTX format...")
        ro.r(
            f"""
            # Save matrix
            writeMM(combined_counts, file = "{mtx_rds_dir}/matrix.mtx")
            
            # Save genes
            write.table(
                data.frame(gene_id = rownames(combined_counts), 
                          gene_name = rownames(combined_counts)),
                file = "{mtx_rds_dir}/genes.tsv",
                sep = "\\t", quote = FALSE, row.names = FALSE, col.names = FALSE
            )
            
            # Save barcodes
            write.table(
                data.frame(barcode = colnames(combined_counts)),
                file = "{mtx_rds_dir}/barcodes.tsv",
                sep = "\\t", quote = FALSE, row.names = FALSE, col.names = FALSE
            )
            
            # Save metadata
            write.csv(combined_meta, file = "{mtx_rds_dir}/combined_metadata.csv")
            
            cat("\\nConversion completed!\\n")
        """
        )

        print(f"MTX files saved to: {mtx_rds_dir}\n")
        return mtx_rds_dir

    def load_from_mtx_rds(self, mtx_rds_dir):
        """Load from pre-converted MTX format"""
        print("STEP 2: Loading from MTX format")

        ro.r(
            f"""
            library(Seurat)
            
            # Load MTX
            counts <- ReadMtx(
                mtx = "{mtx_rds_dir}/matrix.mtx",
                cells = "{mtx_rds_dir}/barcodes.tsv",
                features = "{mtx_rds_dir}/genes.tsv",
                feature.column = 1
            )
            
            # Load metadata
            metadata <- read.csv("{mtx_rds_dir}/combined_metadata.csv", row.names = 1)
            
            # Create Seurat object (Fix: Use Vizgen assay)
            obj <- CreateSeuratObject(
                counts = counts, 
                meta.data = metadata,
                project = "fetal_lung_merfish",
                assay = "Vizgen"
            )
        """
        )

        n_cells = int(ro.r("ncol(obj)")[0])
        n_genes = int(ro.r("nrow(obj)")[0])
        n_batches = int(ro.r("length(unique(obj$batch))")[0])

        print(f"Loaded successfully:")
        print(f"  Total cells: {n_cells}")
        print(f"  Total genes: {n_genes}")
        print(f"  Batches: {n_batches}\n")

        return True

    def run_seurat_workflow_from_rds(self):
        """Run Seurat RPCA integration on loaded RDS data"""
        print("STEP 2: Running Seurat RPCA workflow on RDS data")

        # QC filtering (adjust thresholds based on spatial data characteristics)
        ro.r(
            """
            # Check current feature range
            feature_range <- range(obj$nFeature_Vizgen)
            cat("Feature range before QC:", feature_range[1], "-", feature_range[2], "\n")
            
            # Adaptive QC thresholds for spatial data
            min_features <- max(50, quantile(obj$nFeature_Vizgen, 0.01))
            max_features <- quantile(obj$nFeature_Vizgen, 0.99)
            
            obj <- subset(obj, subset = nFeature_Vizgen > min_features & nFeature_Vizgen < max_features)
        """
        )

        n_cells_qc = int(ro.r("ncol(obj)")[0])
        print(f"✓ After QC: {n_cells_qc} cells retained")

        # Preprocessing
        ro.r(
            """
            obj <- NormalizeData(obj, normalization.method = "LogNormalize")
            obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000)
            obj <- ScaleData(obj)
            obj <- RunPCA(obj, npcs = 50, verbose = FALSE)
        """
        )
        print("✓ Preprocessing completed")

        # RPCA integration
        n_batches = int(ro.r("length(unique(obj$batch))")[0])
        print(f"✓ Running RPCA integration on {n_batches} batches...")

        ro.r(
            """
            # Split layers by batch
            obj[["Vizgen"]] <- split(obj[["Vizgen"]], f = obj$batch)
            
            # RPCA integration
            obj <- IntegrateLayers(
                object = obj,
                method = RPCAIntegration,
                orig.reduction = "pca",
                new.reduction = "integrated.rpca",
                dims = 1:30,
                verbose = FALSE
            )
            
            # Clustering and UMAP
            obj <- FindNeighbors(obj, reduction = "integrated.rpca", dims = 1:30)
            obj <- FindClusters(obj, resolution = 0.5)
            obj <- RunUMAP(obj, reduction = "integrated.rpca", dims = 1:30, 
                          reduction.name = "umap.rpca")
        """
        )

        print("✓ RPCA integration completed\n")

    def convert_h5ad_to_mtx(self):
        """Convert h5ad to MTX format"""
        print("STEP 1: Converting h5ad to MTX format")

        if os.path.exists(f"{self.mtx_dir}/matrix.mtx") and os.path.exists(
            f"{self.mtx_dir}/metadata.csv"
        ):
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
        with open(f"{self.mtx_dir}/genes.tsv", "w") as f:
            for gene in adata.var_names:
                f.write(f"{gene}\t{gene}\n")

        # Save barcodes
        with open(f"{self.mtx_dir}/barcodes.tsv", "w") as f:
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
        ro.r(
            f"""
            library(Seurat)
            counts <- ReadMtx(
                mtx = "{self.mtx_dir}/matrix.mtx",
                cells = "{self.mtx_dir}/barcodes.tsv",
                features = "{self.mtx_dir}/genes.tsv",
                feature.column = 1
            )
            metadata <- read.csv("{self.mtx_dir}/metadata.csv", row.names = 1)
            obj <- CreateSeuratObject(counts = counts, meta.data = metadata, project = "fetal_lung")
        """
        )

        n_cells = int(ro.r("ncol(obj)")[0])
        n_genes = int(ro.r("nrow(obj)")[0])
        print(f"✓ Loaded {n_cells} cells and {n_genes} genes")

        # Check available columns
        has_celltype = ro.r('("celltype" %in% colnames(obj@meta.data))')[0]
        has_batch = ro.r('("batch" %in% colnames(obj@meta.data))')[0]

        if not has_celltype:
            raise ValueError(
                "Celltype information ('celltype' column) is required in metadata."
            )
        if not has_batch:
            raise Warning(
                "Batch information ('batch' column) not found in metadata. Proceeding without batch integration."
            )

        # QC filtering
        ro.r(
            """
            obj <- subset(obj, subset = nFeature_Vizgen > 50 & nFeature_Vizgen < 500)
        """
        )
        n_cells_qc = int(ro.r("ncol(obj)")[0])
        print(f"After QC: {n_cells_qc} cells retained")

        # Preprocessing
        ro.r(
            """
            obj <- NormalizeData(obj)
            obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 50)
            obj <- ScaleData(obj)
            obj <- RunPCA(obj, npcs = 30, verbose = FALSE)
        """
        )
        print(f"Preprocessing completed")

        # Check batch and integrate
        n_batches = int(ro.r('length(unique(obj@meta.data[["batch"]]))')[0])

        if n_batches <= 1:
            raise ValueError(
                "Batch information ('batch' column) must contain at least 2 unique batches for integration."
            )

        print(f"Batch info found: {n_batches} batches")
        print("Running RPCA integration...")
        ro.r(
            """
            obj[["Vizgen"]] <- split(obj[["Vizgen"]], f = obj@meta.data[["batch"]])
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
        """
        )
        print("RPCA integration completed")

        print("Seurat workflow completed!\n")

    def extract_results(self):
        """Extract high-dim and UMAP coordinates from R"""
        print("STEP 3: Extracting results from R")

        # Extract high-dimensional coordinates (for metrics calculation)
        reduction_coords = ro.r(
            f"""
            coords <- Embeddings(obj, reduction = "integrated.rpca")
            as.data.frame(coords)
        """
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            reduction_df = ro.conversion.rpy2py(reduction_coords)

        # Extract UMAP coordinates (for visualization)
        umap_coords = ro.r(
            f"""
            coords <- Embeddings(obj, reduction = "umap.rpca")
            as.data.frame(coords)
        """
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            umap_df = ro.conversion.rpy2py(umap_coords)

        umap_df.columns = ["UMAP_1", "UMAP_2"]

        # Extract batch and celltype info
        batch_info = ro.r('obj@meta.data[["batch"]]')
        with localconverter(ro.default_converter + pandas2ri.converter):
            batch_data = ro.conversion.rpy2py(batch_info)
        reduction_df["batch"] = batch_data
        umap_df["batch"] = batch_data

        celltype_info = ro.r('obj@meta.data[["celltype"]]')
        with localconverter(ro.default_converter + pandas2ri.converter):
            celltype_data = ro.conversion.rpy2py(celltype_info)
        reduction_df["celltype"] = celltype_data
        umap_df["celltype"] = celltype_data

        # Get metadata
        n_cells = int(ro.r("ncol(obj)")[0])
        n_genes = int(ro.r("nrow(obj)")[0])

        # Get batch stats
        batch_table = ro.r('as.data.frame(table(obj@meta.data[["batch"]]))')
        with localconverter(ro.default_converter + pandas2ri.converter):
            batch_stats = ro.conversion.rpy2py(batch_table)
        batch_stats.columns = ["batch", "count"]

        # Get celltype stats
        celltype_table = ro.r('as.data.frame(table(obj@meta.data[["celltype"]]))')
        with localconverter(ro.default_converter + pandas2ri.converter):
            celltype_stats = ro.conversion.rpy2py(celltype_table)
        celltype_stats.columns = ["celltype", "count"]

        self.results = {
            "info": {
                "n_cells": n_cells,
                "n_genes": n_genes,
            },
            "reduction": reduction_df,  # High-dim for metrics
            "umap": umap_df,  # 2D for visualization
            "batch_stats": batch_stats,
            "celltype_stats": celltype_stats,
        }

        print("Results extracted!\n")
        return self.results

    def compute_metrics_scib(self):
        """Compute batch mixing metrics using scib (standard method)"""
        print("STEP 4: Computing batch mixing metrics (scib)\n")

        reduction_df = self.results["reduction"]
        umap_df = self.results["umap"]

        n_batches = len(np.unique(reduction_df["batch"]))

        # Create AnnData object for scib metrics
        coord_cols = [
            col for col in reduction_df.columns if col not in ["batch", "celltype"]
        ]
        X_high_dim = reduction_df[coord_cols].values

        adata = sc.AnnData(X=X_high_dim)
        adata.obsm["X"] = X_high_dim
        adata.obs["batch"] = pd.Categorical(reduction_df["batch"].values)
        adata.obs["celltype"] = pd.Categorical(reduction_df["celltype"].values)
        adata.obsm["X_umap"] = umap_df[["UMAP_1", "UMAP_2"]].values

        print(f"Computing scib metrics for {n_batches} batches...")
        print(f"  Total cells: {X_high_dim.shape[0]}")
        print(f"  Cell types: {len(np.unique(adata.obs['celltype']))}\n")

        # Compute scib metrics
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

        # Extract all metrics
        asw_label_batch = result_dict.get("ASW_label/batch", np.nan)
        graph_conn = result_dict.get("graph_conn", np.nan)
        asw_label = result_dict.get("ASW_label", np.nan)
        nmi = result_dict.get("NMI_cluster/label", np.nan)
        ari = result_dict.get("ARI_cluster/label", np.nan)

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
            "batch_asw": asw_label_batch,
            "graph_conn": graph_conn,
            "label_asw": asw_label,
            "nmi": nmi,
            "ari": ari,
            "avg_bat": avg_bat,
            "avg_bio": avg_bio,
            "n_batches": n_batches,
            "n_celltypes": len(np.unique(adata.obs["celltype"])),
            "n_cells": X_high_dim.shape[0],
            "scib_results": result_dict,
        }

        # Print results
        print("=" * 70)
        print("BATCH MIXING & CELLTYPE PRESERVATION METRICS (scib)")
        print("=" * 70)
        print("\nBatch Metrics (mixing quality):")
        print(
            f"  ASW (label/batch):   {asw_label_batch:>8.4f}  (Range: [-1, 1], higher = better)"
        )
        print(
            f"  Graph Connectivity:  {graph_conn:>8.4f}  (Range: [0, 1], higher = better)"
        )

        print("\nCelltype Preservation:")
        print(f"  ASW                   {asw_label:>8.4f}")
        print(
            f"  NMI:                 {nmi:>8.4f}  (batch-celltype independence, lower = better)"
        )
        print(
            f"  ARI:                 {ari:>8.4f}  (clustering agreement, higher = better)"
        )

        print("\nIntegration Quality:")
        print(
            f"  AvgBAT:              {avg_bat:>8.4f}  (Range: [0, 1], higher = better) ⭐"
        )
        print(f" AVGBIO                 {avg_bio:>8.4f}")
        print("=" * 70 + "\n")

        # Save metrics
        self._save_metrics(metrics)

        return metrics

    def _save_metrics(self, metrics):
        """Save metrics to JSON"""
        # Prepare data for JSON serialization
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
            },
            "celltype_preservation": {
                "asw_label": (
                    float(metrics["label_asw"])
                    if not np.isnan(metrics["label_asw"])
                    else None
                ),
                "nmi": float(metrics["nmi"]) if not np.isnan(metrics["nmi"]) else None,
                "ari": float(metrics["ari"]) if not np.isnan(metrics["ari"]) else None,
            },
            "integration_quality": {
                "avg_bat": (
                    float(metrics["avg_bat"])
                    if not np.isnan(metrics["avg_bat"])
                    else None
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
            },
        }

        with open(f"{self.output_dir}/batch_mixing_metrics.json", "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"Metrics saved to {self.output_dir}/batch_mixing_metrics.json\n")

    def save_calculation_results(self, metrics):
        """Save UMAP coordinates and flat metrics for quick plotting later"""
        print(f"Saving intermediate results to {self.output_dir}...")

        # 1. Save UMAP DataFrame (contains UMAP_1, UMAP_2, batch, celltype)
        if "umap" in self.results:
            self.results["umap"].to_csv(
                f"{self.output_dir}/final_umap.csv", index=False
            )

        # 2. Save Flat Metrics (for plot titles)
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

        # 1. Load UMAP
        self.results = {}
        self.results["umap"] = pd.read_csv(umap_path)

        # 2. Load Metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        print("Loaded UMAP coordinates and metrics successfully.")
        return metrics

    def create_visualizations(self, metrics):
        """Create batch mixing and celltype preservation visualizations"""
        print("STEP 5: Creating visualizations")

        umap_df = self.results["umap"]

        if "batch" not in umap_df.columns or np.isnan(metrics.get("avg_bat", np.nan)):
            raise ValueError("The caculated AVGBATCH is NOT A NUMBER")

        # Shuffle the dataframe to prevent plotting order bias (important for large datasets)
        umap_df = umap_df.sample(frac=1, random_state=42).reset_index(drop=True)

        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300

        saved_plots = []

        scatter_kwargs = {"s": 1.5, "alpha": 0.8, "edgecolors": "none"}

        # Plot 1: UMAP colored by batch
        fig, ax = plt.subplots(figsize=(12, 9))

        unique_batches = sorted(umap_df["batch"].unique())
        n_batches = len(unique_batches)
        colors = (
            plt.cm.jet(np.linspace(0, 1, n_batches))
            if n_batches > 10
            else plt.cm.Set2(np.linspace(0, 1, n_batches))
        )
        batch_color_map = dict(zip(unique_batches, colors))

        # Plot all points at once using list comprehension for colors (faster and respects shuffle)
        c_array = umap_df["batch"].map(batch_color_map)
        ax.scatter(umap_df["UMAP_1"], umap_df["UMAP_2"], c=c_array, **scatter_kwargs)

        # Create custom legend handles
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

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)

        title = f"Batch Mixing Quality\n"
        title += f'ASW: {metrics["batch_asw"]:.4f} | '
        title += f'GraphConn: {metrics["graph_conn"]:.4f} | '
        title += f'AvgBAT: {metrics["avg_bat"]:.4f}'

        ax.set_title(title, fontsize=14, weight="bold")
        # Move legend outside if too many batches
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
        print(f"✓ Saved: {plot_path}")
        plt.close()

        # Plot 2: UMAP colored by celltype
        fig, ax = plt.subplots(figsize=(12, 9))

        unique_celltypes = sorted(umap_df["celltype"].unique())
        n_celltypes = len(unique_celltypes)
        colors_ct = plt.cm.tab20(np.linspace(0, 1, n_celltypes))
        if n_celltypes > 20:
            colors_ct = plt.cm.gist_ncar(
                np.linspace(0, 0.9, n_celltypes)
            )  # More colors for many celltypes
        ct_color_map = dict(zip(unique_celltypes, colors_ct))

        c_array_ct = umap_df["celltype"].map(ct_color_map)
        ax.scatter(
            umap_df["UMAP_1"],
            umap_df["UMAP_2"],
            c=c_array_ct.tolist(),
            **scatter_kwargs,
        )

        # Legend handles
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

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)

        title = f"Celltype Preservation\n"
        title += f'ASW: {metrics["label_asw"]:.4f} | '
        title += f'NMI: {metrics["nmi"]:.4f} | '
        title += f'ARI: {metrics["ari"]:.4f} |'
        title += f'AVGBIO: {metrics["avg_bio"]:.4f}'

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

    def run(self, use_rds=False, plot_only=False):
        """Run complete pipeline"""
        print("\n" + "=" * 70)
        print("SEURAT RPCA BATCH INTEGRATION & CELLTYPE EVALUATION")
        print("Pipeline: Seurat RPCA + scib Metrics")
        print("=" * 70 + "\n")

        if plot_only:
            print("MODE: Plotting Only (Skipping Calculation)")
            metrics = self.load_calculation_results()
            plots = self.create_visualizations(metrics)
            print("=" * 70)
            print("PLOTTING COMPLETED!")
            print("=" * 70 + "\n")
            return self.results, metrics, plots

        if use_rds:
            # Convert RDS to MTX (one-time, cached)
            mtx_rds_dir = self.rds_to_mtx_format()
            # Load from MTX (fast)
            self.load_from_mtx_rds(mtx_rds_dir)
            # Run workflow
            self.run_seurat_workflow_from_rds()
        else:
            # Use h5ad workflow
            self.convert_h5ad_to_mtx()
            self.run_seurat_workflow()
        self.extract_results()
        metrics = self.compute_metrics_scib()
        self.save_calculation_results(metrics)
        plots = self.create_visualizations(metrics)

        print("=" * 70)
        print("PIPELINE COMPLETED!")
        print("=" * 70 + "\n")

        return self.results, metrics, plots


if __name__ == "__main__":
    pipeline = SeuratRPCAPipeline(rds_path="../data/merfish.rds")
    results, metrics, plots = pipeline.run(use_rds=True, plot_only=True)
