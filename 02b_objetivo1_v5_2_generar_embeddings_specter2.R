# ============================================================
# 02b_objetivo1_v5_2_generar_embeddings_specter2.R
# Objetivo 1 v5.2
# Generar embeddings SPECTER2 usando texto limpio sin Concepts
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Configuración
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"

tables_dir <- file.path(base_dir, "tables")
script_dir <- file.path(base_dir, "script")
embeddings_dir <- file.path(base_dir, "embeddings")

v5_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

input_full <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_2_for_embeddings_no_concepts.csv"
)

input_sample <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_2_for_embeddings_no_concepts_sample_1000.csv"
)

python_bin <- path.expand("~/env_cluster/bin/python")

script_embeddings <- file.path(
  script_dir,
  "0002-generar_embeddings_specter2.py"
)

# Primero correr sample. Luego cambiar a "full".
# run_mode <- "sample"
run_mode <- "full"

batch_size <- 16
max_length <- 512

out_dir_sample <- file.path(
  embeddings_dir,
  "specter2_classification_v5_2_docs_interest_no_concepts_sample_1000"
)

out_dir_full <- file.path(
  embeddings_dir,
  "specter2_classification_v5_2_docs_interest_no_concepts"
)

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
  
  if (!file.exists(npy_file)) {
    cat("No existe archivo NPY:", npy_file, "\n")
    return(invisible(NULL))
  }
  
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
  
  cat("\nDimensión del archivo NPY:\n")
  cat(paste(res, collapse = "\n"), "\n")
  
  invisible(res)
}

# ============================================================
# 3) Validaciones
# ============================================================

stop_if_missing(input_full, "dataset v5.2 para embeddings")
stop_if_missing(python_bin, "Python")
stop_if_missing(script_embeddings, "script de embeddings SPECTER2")

dt_header <- fread(input_full, nrows = 0)

required_cols <- c(
  "doc_id_v5",
  "text_for_embedding"
)

missing_cols <- setdiff(required_cols, names(dt_header))

if (length(missing_cols) > 0) {
  stop(
    paste0(
      "Faltan columnas en dataset v5.2: ",
      paste(missing_cols, collapse = ", ")
    ),
    call. = FALSE
  )
}

# ============================================================
# 4) Crear muestra de prueba
# ============================================================

if (!file.exists(input_sample)) {
  
  cat("\nCreando muestra de 1,000 documentos v5.2...\n")
  
  dt <- fread(input_full)
  setDT(dt)
  
  set.seed(123)
  sample_n <- min(1000, nrow(dt))
  
  dt_sample <- dt[
    sample(.N, sample_n)
  ]
  
  fwrite(dt_sample, input_sample)
  
  cat("Muestra creada:\n")
  cat(input_sample, "\n")
}

# ============================================================
# 5) Seleccionar modo
# ============================================================

if (run_mode == "sample") {
  
  input_run <- input_sample
  output_dir <- out_dir_sample
  step_label <- "Generar embeddings SPECTER2 v5.2 - muestra 1000"
  
} else if (run_mode == "full") {
  
  input_run <- input_full
  output_dir <- out_dir_full
  step_label <- "Generar embeddings SPECTER2 v5.2 - corpus completo"
  
} else {
  stop("run_mode debe ser 'sample' o 'full'.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("\nModo de ejecución:", run_mode, "\n")
cat("Input:", input_run, "\n")
cat("Output:", output_dir, "\n")

# ============================================================
# 6) Ejecutar SPECTER2
# ============================================================

args_emb <- c(
  script_embeddings,
  
  "--input", input_run,
  "--output_dir", output_dir,
  
  "--id_col", "doc_id_v5",
  "--text_col", "text_for_embedding",
  
  "--model_name", "allenai/specter2_base",
  "--adapter_name", "allenai/specter2_classification",
  "--adapter_load_as", "specter2_classification",
  
  "--batch_size", as.character(batch_size),
  "--max_length", as.character(max_length),
  
  "--normalize",
  "--overwrite"
)

run_command(
  command = python_bin,
  args = args_emb,
  step_name = step_label
)

# ============================================================
# 7) Verificar salidas
# ============================================================

emb_file <- file.path(output_dir, "docs_embeddings.npy")
meta_file <- file.path(output_dir, "docs_embeddings_meta.csv")

stop_if_missing(emb_file, "docs_embeddings.npy")
stop_if_missing(meta_file, "docs_embeddings_meta.csv")

meta_out <- fread(meta_file)

cat("\n================ VERIFICACIÓN EMBEDDINGS v5.2 ================\n")
cat("Modo:", run_mode, "\n")
cat("Archivo embeddings:", emb_file, "\n")
cat("Archivo metadata:", meta_file, "\n")
cat("Filas metadata:", nrow(meta_out), "\n")

inspect_npy_shape(emb_file)

cat("\nPrimeras filas metadata:\n")
print(head(meta_out))

cat("\nProceso completado correctamente.\n")