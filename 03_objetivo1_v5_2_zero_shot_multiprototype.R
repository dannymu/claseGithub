# ============================================================
# 03_objetivo1_v5_2_zero_shot_multiprototype.R
# Objetivo 1 v5.2
# Zero-shot con taxonomía v4 multiprototipo usando embeddings limpios
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"

tables_dir <- file.path(base_dir, "tables")
script_dir <- file.path(base_dir, "script")
embeddings_dir <- file.path(base_dir, "embeddings")

python_bin <- path.expand("~/env_cluster/bin/python")

script_zero_v4 <- file.path(
  script_dir,
  "0030_zero_shot_v4_multiprototype.py"
)

doc_emb_dir <- file.path(
  embeddings_dir,
  "specter2_classification_v5_2_docs_interest_no_concepts"
)

doc_emb <- file.path(
  doc_emb_dir,
  "docs_embeddings.npy"
)

doc_meta_original <- file.path(
  doc_emb_dir,
  "docs_embeddings_meta.csv"
)

doc_meta_fixed <- file.path(
  doc_emb_dir,
  "docs_embeddings_meta_fixed.csv"
)

tax_emb <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings.npy"
)

tax_meta <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings_meta_fixed.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_zero_shot"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_sample <- file.path(
  out_dir,
  "zero_shot_v5_2_multiprototype_sample_10000.csv"
)

out_full <- file.path(
  out_dir,
  "zero_shot_v5_2_multiprototype_full.csv"
)

# Primero correr sample. Luego cambiar a "full".
# run_mode <- "sample"
run_mode <- "full"

# Parámetros
top_k <- 5
batch_size <- 5000
sample_n <- ifelse(run_mode == "sample", 10000, 0)
seed <- 123
min_score <- 0.35
min_margin <- 0.03

# ============================================================
# 2) Funciones
# ============================================================

stop_if_missing <- function(path, label) {
  if (!file.exists(path)) {
    stop(paste0("No existe ", label, ": ", path), call. = FALSE)
  }
}

run_command <- function(command, args, step_name) {
  
  cat("\n============================================================\n")
  cat("Ejecutando:", step_name, "\n")
  cat("Comando:", command, "\n")
  cat("Args:", paste(args, collapse = " "), "\n")
  cat("============================================================\n")
  
  res <- system2(
    command = command,
    args = args,
    stdout = TRUE,
    stderr = TRUE
  )
  
  status <- attr(res, "status")
  if (is.null(status)) status <- 0
  
  cat(paste(res, collapse = "\n"), "\n")
  
  if (status != 0) {
    stop(paste0("Falló el paso: ", step_name), call. = FALSE)
  }
  
  invisible(res)
}

inspect_npy_shape <- function(npy_file) {
  
  py_code <- paste0(
    "import numpy as np\n",
    "x=np.load(r'", npy_file, "')\n",
    "print(x.shape)\n"
  )
  
  res <- system2(
    python_bin,
    args = c("-c", shQuote(py_code)),
    stdout = TRUE,
    stderr = TRUE
  )
  
  cat("\nDimensión NPY:\n")
  cat(paste(res, collapse = "\n"), "\n")
  
  invisible(res)
}

# ============================================================
# 3) Validaciones
# ============================================================

stop_if_missing(python_bin, "Python")
stop_if_missing(script_zero_v4, "script zero-shot v4 multiprototype")
stop_if_missing(doc_emb, "embeddings documentos v5.2")
stop_if_missing(doc_meta_original, "metadata documentos v5.2")
stop_if_missing(tax_emb, "embeddings taxonomía v4")
stop_if_missing(tax_meta, "metadata taxonomía v4 fixed")

# ============================================================
# 4) Crear metadata fixed con doc_id
# ============================================================

meta <- fread(doc_meta_original)
setDT(meta)

if (!"doc_id" %in% names(meta)) {
  
  if ("doc_id_v5" %in% names(meta)) {
    meta[, doc_id := as.integer(doc_id_v5)]
  } else {
    stop("La metadata no tiene doc_id ni doc_id_v5.")
  }
}

# Mantener doc_id como primera columna
setcolorder(
  meta,
  c("doc_id", setdiff(names(meta), "doc_id"))
)

fwrite(meta, doc_meta_fixed)

cat("\nMetadata fixed creada:\n")
cat(doc_meta_fixed, "\n")
cat("Filas metadata:", nrow(meta), "\n")

inspect_npy_shape(doc_emb)

# ============================================================
# 5) Ejecutar zero-shot
# ============================================================

if (run_mode == "sample") {
  output_file <- out_sample
  step_label <- "Zero-shot v5.2 multiprototipo - muestra 10000"
} else if (run_mode == "full") {
  output_file <- out_full
  step_label <- "Zero-shot v5.2 multiprototipo - corpus completo"
} else {
  stop("run_mode debe ser 'sample' o 'full'.")
}

args_zero <- c(
  script_zero_v4,
  
  "--doc_emb", doc_emb,
  "--doc_meta", doc_meta_fixed,
  
  "--tax_emb", tax_emb,
  "--tax_meta", tax_meta,
  
  "--output", output_file,
  
  "--top_k", as.character(top_k),
  "--batch_size", as.character(batch_size),
  "--sample_n", as.character(sample_n),
  "--seed", as.character(seed),
  "--min_score", as.character(min_score),
  "--min_margin", as.character(min_margin)
)

run_command(
  command = python_bin,
  args = args_zero,
  step_name = step_label
)

# ============================================================
# 6) Verificar salida
# ============================================================

stop_if_missing(output_file, "salida zero-shot v5.2")

pred <- fread(output_file)
setDT(pred)

cat("\n================ ZERO-SHOT v5.2 COMPLETADO ================\n")
cat("Modo:", run_mode, "\n")
cat("Archivo:", output_file, "\n")
cat("Filas:", nrow(pred), "\n")

cat("\nColumnas:\n")
print(names(pred))

cat("\nPrimeras filas:\n")
print(head(pred))

if ("top1_area_name" %in% names(pred)) {
  cat("\nDistribución top áreas:\n")
  print(
    pred[
      ,
      .N,
      by = top1_area_name
    ][order(-N)][1:30]
  )
}

if ("top1_prototype_language" %in% names(pred)) {
  cat("\nUso de prototipos por idioma:\n")
  print(
    pred[
      ,
      .N,
      by = top1_prototype_language
    ][order(-N)]
  )
}