library(Seurat)
library(jsonlite)

options(future.globals.maxSize = 1e9)

# 1. Load my own dataset from MTX format
cat("Loading data from MTX format...\n")

counts <- ReadMtx(
    mtx = "../data/mtx_format/matrix.mtx",
    cells = "../data/mtx_format/barcodes.tsv",
    features = "../data/mtx_format/genes.tsv",
    feature.column = 1
)

metadata <- read.csv("../data/mtx_format/metadata.csv", row.names = 1)

obj <- CreateSeuratObject(
    counts = counts,
    meta.data = metadata,
    project = "fetal_lung"
)

cat(sprintf("✓ Loaded %d cells and %d genes\n\n", ncol(obj), nrow(obj)))

# 2. Quality Control (optional, adjust thresholds as needed)
cat("Performing QC filtering...\n")

if (!("nFeature_RNA" %in% colnames(obj@meta.data))) {
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
}

obj <- subset(obj, subset = nFeature_RNA > 50 & nFeature_RNA < 500)

cat(sprintf("✓ After QC: %d cells retained\n\n", ncol(obj)))
print(obj)

# 3. Split by batch
BATCH_VAR <- "batch"  # Change to your actual batch column name
if (BATCH_VAR %in% colnames(obj@meta.data)) {
    cat(sprintf("\nSplitting data by '%s'...\n", BATCH_VAR))
    
    # Check batch distribution
    cat("Batch distribution:\n")
    print(table(obj@meta.data[[BATCH_VAR]]))
    
    # Split layers by batch
    obj[["RNA"]] <- split(obj[["RNA"]], f = obj@meta.data[[BATCH_VAR]])
    
    cat("✓ Data split into layers\n")
    print(obj)
} else {
    cat(sprintf("\nWarning: Column '%s' not found in metadata.\n", BATCH_VAR))
    cat("\nSkipping batch splitting. Running without integration...\n")
    BATCH_VAR <- NULL
}

# 4. Standard preprocessing workflow
cat("\nRunning standard preprocessing...\n")

obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 50)
obj <- ScaleData(obj)
obj <- RunPCA(obj, npcs = 30, verbose = FALSE)

cat("✓ Preprocessing completed\n\n")

# 6. RPCA Integration (baseline method)
if (!is.null(BATCH_VAR)) {
    cat("Running RPCA integration...\n")
    
    obj <- IntegrateLayers(
        object = obj,
        method = RPCAIntegration,
        orig.reduction = "pca",
        new.reduction = "integrated.rpca",
        verbose = FALSE
    )
    
    obj <- FindNeighbors(obj, reduction = "integrated.rpca", dims = 1:10)
    obj <- FindClusters(obj, resolution = 0.5, cluster.name = "rpca_clusters")
    obj <- RunUMAP(
        obj,
        reduction = "integrated.rpca",
        dims = 1:10,
        reduction.name = "umap.rpca"
    )
    
    cat("✓ RPCA integration completed\n\n")
    
    REDUCTION <- "integrated.rpca"
    UMAP_NAME <- "umap.rpca"
    CLUSTER_NAME <- "rpca_clusters"
}

# 7. Export data for Python visualization
cat("Exporting data for Python...\n")

output_dir <- "seurat_rpca_results"
if (!dir.exists(output_dir)) {
    dir.create(output_dir)
}

# Export UMAP coordinates
umap_coords <- Embeddings(obj, reduction = UMAP_NAME)
umap_df <- data.frame(
    cell = rownames(umap_coords),
    UMAP_1 = umap_coords[, 1],
    UMAP_2 = umap_coords[, 2]
)

# Add batch information if available
if (!is.null(BATCH_VAR)) {
    umap_df$batch <- obj@meta.data[[BATCH_VAR]]
}

write.csv(umap_df, file.path(output_dir, "umap_coordinates.csv"), row.names = FALSE)
cat(sprintf("✓ Exported UMAP coordinates: %s\n", 
            file.path(output_dir, "umap_coordinates.csv")))

# Export batch statistics (only if batch info exists)
if (!is.null(BATCH_VAR)) {
    batch_stats <- as.data.frame(table(obj@meta.data[[BATCH_VAR]]))
    colnames(batch_stats) <- c("batch", "count")
    write.csv(batch_stats, file.path(output_dir, "batch_stats.csv"), row.names = FALSE)
    cat(sprintf("✓ Exported batch stats: %s\n", 
                file.path(output_dir, "batch_stats.csv")))
}

# Export analysis info
analysis_info <- list(
    n_cells = ncol(obj),
    n_genes = nrow(obj),
    batch_column = BATCH_VAR
)

write_json(analysis_info, 
           file.path(output_dir, "analysis_info.json"), 
           pretty = TRUE,
           auto_unbox = TRUE)
cat(sprintf("✓ Exported analysis info: %s\n", file.path(output_dir, "analysis_info.json")))

cat("✓ Ready for Python batch mixing evaluation!\n")
