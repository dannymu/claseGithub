# ============================================================
# 30_objetivo3_perfil_cli.R
# Evaluación automática de perfil de investigador
# Objetivo 3
# ============================================================
# Uso:
# Rscript 30_objetivo3_perfil_cli.R <profile_file> <researcher_id> <base_dir>
#
# Ejemplo:
# Rscript script/30_objetivo3_perfil_cli.R \
#   tables/perfiles_uploads/gs-martin.csv \
#   "Alberto Martin-Martin" \
#   /mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN
#
# Este script:
# 1. Lee un CSV de publicaciones.
# 2. Normaliza columnas.
# 3. Crea input para embeddings.
# 4. Genera embeddings SPECTER2.
# 5. Aplica modelo supervisado jerárquico v3.
# 6. Aplica zero-shot v2.
# 7. Integra resultados.
# 8. Genera resúmenes.
# 9. Genera mapas semánticos y señales de revisión.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

# ============================================================
# 0) Funciones auxiliares
# ============================================================

msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "|", ..., "\n")
}

stop_if_missing <- function(path, label = "archivo") {
  if (!file.exists(path)) {
    stop(paste0("No existe ", label, ": ", path), call. = FALSE)
  }
}

stop_if_missing_dir <- function(path, label = "directorio") {
  if (!dir.exists(path)) {
    stop(paste0("No existe ", label, ": ", path), call. = FALSE)
  }
}

safe_id <- function(x) {
  x <- as.character(x)
  x <- enc2utf8(x)
  x <- stringi::stri_trans_general(x, "Latin-ASCII")
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  if (x == "") x <- paste0("perfil_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  x
}

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

normalize_title <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- stringi::stri_trans_general(x, "Latin-ASCII")
  x <- gsub("[[:punct:]]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

find_col <- function(dt, candidates) {
  nms <- names(dt)
  nms_lower <- tolower(nms)
  candidates_lower <- tolower(candidates)
  
  # Coincidencia exacta ignorando mayúsculas/minúsculas
  for (cand in candidates_lower) {
    idx <- which(nms_lower == cand)
    if (length(idx) > 0) return(nms[idx[1]])
  }
  
  # Coincidencia parcial
  for (cand in candidates_lower) {
    idx <- grep(cand, nms_lower, fixed = TRUE)
    if (length(idx) > 0) return(nms[idx[1]])
  }
  
  NA_character_
}

read_csv_safe <- function(path) {
  msg("Leyendo CSV:", path)
  
  dt <- tryCatch(
    fread(path, encoding = "UTF-8"),
    error = function(e) {
      msg("No se pudo leer como UTF-8. Intentando Latin-1...")
      fread(path, encoding = "Latin-1")
    }
  )
  
  setDT(dt)
  dt
}

run_command <- function(command, args, log_file = NULL, step_name = "comando") {
  msg("Ejecutando:", step_name)
  msg("Comando:", command)
  msg("Args:", paste(args, collapse = " "))
  
  res <- system2(
    command = command,
    args = args,
    stdout = TRUE,
    stderr = TRUE
  )
  
  status <- attr(res, "status")
  if (is.null(status)) status <- 0
  
  if (!is.null(log_file)) {
    cat(
      "\n\n============================================================\n",
      "PASO: ", step_name, "\n",
      "FECHA: ", as.character(Sys.time()), "\n",
      "COMANDO: ", command, " ", paste(args, collapse = " "), "\n",
      "STATUS: ", status, "\n",
      "SALIDA:\n",
      paste(res, collapse = "\n"),
      "\n",
      file = log_file,
      append = TRUE,
      sep = ""
    )
  }
  
  if (status != 0) {
    cat(paste(res, collapse = "\n"), "\n")
    stop(paste0("Falló el paso: ", step_name, ". Código de salida: ", status), call. = FALSE)
  }
  
  msg("Paso completado:", step_name)
  invisible(res)
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

safe_integer <- function(x) {
  suppressWarnings(as.integer(x))
}

# ============================================================
# 1) Argumentos CLI
# ============================================================

args_cli <- commandArgs(trailingOnly = TRUE)

if (length(args_cli) < 3) {
  stop(
    paste0(
      "Uso incorrecto.\n",
      "Debe ejecutar:\n",
      "Rscript 30_objetivo3_perfil_cli.R <profile_file> <researcher_id> <base_dir>\n"
    ),
    call. = FALSE
  )
}

profile_file <- args_cli[1]
researcher_id <- args_cli[2]
base_dir <- args_cli[3]

researcher_safe_id <- safe_id(researcher_id)

msg("============================================================")
msg("INICIO DEL PIPELINE OBJETIVO 3")
msg("Investigador:", researcher_id)
msg("ID seguro:", researcher_safe_id)
msg("Archivo CSV:", profile_file)
msg("Base dir:", base_dir)
msg("============================================================")

# ============================================================
# 2) Rutas del proyecto
# ============================================================

tables_dir <- file.path(base_dir, "tables")
script_dir <- file.path(base_dir, "script")
embeddings_dir <- file.path(base_dir, "embeddings")
models_dir <- file.path(base_dir, "models")

python_bin <- path.expand("~/env_cluster/bin/python")

script_embeddings <- file.path(
  script_dir,
  "0002-generar_embeddings_specter2.py"
)

script_supervised <- file.path(
  script_dir,
  "0015_predict_hierarchical_area_classifier_v3.py"
)

script_zero_shot <- file.path(
  script_dir,
  "0003-zero_shot_classification.py"
)

script_maps <- file.path(
  script_dir,
  "0020_profile_semantic_maps_anomalies.py"
)

model_dir <- file.path(
  models_dir,
  "hierarchical_area_classifier_v3_supported"
)

coverage_file <- file.path(
  tables_dir,
  "area_coverage_plan_v3.csv"
)

tax_emb <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v2",
  "docs_embeddings.npy"
)

tax_meta <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v2",
  "docs_embeddings_meta_fixed.csv"
)

profile_out_dir <- file.path(
  tables_dir,
  "perfiles_resultados",
  researcher_safe_id
)

profile_emb_dir <- file.path(
  embeddings_dir,
  paste0("profile_specter2_", researcher_safe_id)
)

profile_maps_dir <- file.path(
  profile_out_dir,
  "semantic_maps_anomalies"
)

dir.create(profile_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(profile_emb_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(profile_maps_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(profile_out_dir, paste0("pipeline_log_", researcher_safe_id, ".txt"))

cat(
  "LOG DEL PIPELINE OBJETIVO 3\n",
  "Investigador: ", researcher_id, "\n",
  "Fecha inicio: ", as.character(Sys.time()), "\n",
  "CSV: ", profile_file, "\n",
  "Base dir: ", base_dir, "\n",
  file = log_file,
  sep = ""
)

# ============================================================
# 3) Validaciones iniciales
# ============================================================

stop_if_missing(profile_file, "CSV del perfil")
stop_if_missing(python_bin, "Python")
stop_if_missing(script_embeddings, "script de embeddings")
stop_if_missing(script_supervised, "script supervised v3")
stop_if_missing(script_zero_shot, "script zero-shot")
stop_if_missing(script_maps, "script de mapas")
stop_if_missing_dir(model_dir, "modelo jerárquico v3")
stop_if_missing(coverage_file, "area_coverage_plan_v3.csv")
stop_if_missing(tax_emb, "embeddings de taxonomía")
stop_if_missing(tax_meta, "metadata de taxonomía")

# ============================================================
# 4) Leer perfil
# ============================================================

dt_raw <- read_csv_safe(profile_file)

msg("Filas originales:", nrow(dt_raw))
msg("Columnas detectadas:")
print(names(dt_raw))

if (nrow(dt_raw) == 0) {
  stop("El archivo CSV no contiene filas.", call. = FALSE)
}

# ============================================================
# 5) Detectar columnas
# ============================================================

title_col <- find_col(
  dt_raw,
  c(
    "title", "titulo", "título",
    "publication_title", "document_title",
    "article title", "paper title"
  )
)

abstract_col <- find_col(
  dt_raw,
  c(
    "abstract", "resumen", "description",
    "descripcion", "descripción"
  )
)

keyword_col <- find_col(
  dt_raw,
  c(
    "keyword", "keywords",
    "palabras_clave", "palabras clave",
    "author_keywords", "index_keywords"
  )
)

year_col <- find_col(
  dt_raw,
  c(
    "year", "publication_year",
    "anio", "año", "date", "publication date"
  )
)

cites_col <- find_col(
  dt_raw,
  c(
    "cites", "citations", "cited_by_count",
    "cited by", "citas", "times cited"
  )
)

journal_col <- find_col(
  dt_raw,
  c(
    "journal", "source", "revista",
    "venue", "publication", "container"
  )
)

authors_col <- find_col(
  dt_raw,
  c(
    "authors", "author",
    "autores", "autor"
  )
)

doi_col <- find_col(
  dt_raw,
  c(
    "doi", "DOI"
  )
)

url_col <- find_col(
  dt_raw,
  c(
    "url", "link", "source_url", "publication_url"
  )
)

if (is.na(title_col)) {
  stop(
    paste0(
      "No se detectó una columna de título.\n",
      "Columnas disponibles: ", paste(names(dt_raw), collapse = ", ")
    ),
    call. = FALSE
  )
}

msg("Columna título:", title_col)
msg("Columna abstract:", ifelse(is.na(abstract_col), "NO DETECTADA", abstract_col))
msg("Columna keywords:", ifelse(is.na(keyword_col), "NO DETECTADA", keyword_col))
msg("Columna año:", ifelse(is.na(year_col), "NO DETECTADA", year_col))
msg("Columna citas:", ifelse(is.na(cites_col), "NO DETECTADA", cites_col))
msg("Columna revista:", ifelse(is.na(journal_col), "NO DETECTADA", journal_col))
msg("Columna autores:", ifelse(is.na(authors_col), "NO DETECTADA", authors_col))
msg("Columna DOI:", ifelse(is.na(doi_col), "NO DETECTADA", doi_col))
msg("Columna URL:", ifelse(is.na(url_col), "NO DETECTADA", url_col))

# ============================================================
# 6) Construir tabla normalizada
# ============================================================

dt_profile <- data.table(
  doc_id = seq_len(nrow(dt_raw)),
  profile_doc_id = paste0(researcher_safe_id, "_", seq_len(nrow(dt_raw))),
  researcher_id = researcher_id,
  title = clean_text(dt_raw[[title_col]])
)

if (!is.na(abstract_col)) {
  dt_profile[, abstract := clean_text(dt_raw[[abstract_col]])]
} else {
  dt_profile[, abstract := ""]
}

if (!is.na(keyword_col)) {
  dt_profile[, keyword := clean_text(dt_raw[[keyword_col]])]
} else {
  dt_profile[, keyword := ""]
}

if (!is.na(year_col)) {
  dt_profile[, year := safe_integer(dt_raw[[year_col]])]
} else {
  dt_profile[, year := NA_integer_]
}

if (!is.na(cites_col)) {
  dt_profile[, citas := safe_numeric(dt_raw[[cites_col]])]
} else {
  dt_profile[, citas := NA_real_]
}

if (!is.na(journal_col)) {
  dt_profile[, revista := clean_text(dt_raw[[journal_col]])]
} else {
  dt_profile[, revista := ""]
}

if (!is.na(authors_col)) {
  dt_profile[, authors := clean_text(dt_raw[[authors_col]])]
} else {
  dt_profile[, authors := ""]
}

if (!is.na(doi_col)) {
  dt_profile[, doi := clean_text(dt_raw[[doi_col]])]
} else {
  dt_profile[, doi := ""]
}

if (!is.na(url_col)) {
  dt_profile[, url := clean_text(dt_raw[[url_col]])]
} else {
  dt_profile[, url := ""]
}

# Quitar registros sin título
dt_profile <- dt_profile[title != ""]

if (nrow(dt_profile) == 0) {
  stop("Después de quitar registros sin título no quedan publicaciones.", call. = FALSE)
}

dt_profile[, title_norm := normalize_title(title)]

# Quitar duplicados por título normalizado
before_dedup <- nrow(dt_profile)
dt_profile <- unique(dt_profile, by = "title_norm")
after_dedup <- nrow(dt_profile)

msg("Publicaciones antes de deduplicar:", before_dedup)
msg("Publicaciones después de deduplicar por título:", after_dedup)

# Conteos de palabras
dt_profile[, n_words_title := stringi::stri_count_words(title)]
dt_profile[, n_words_abstract := stringi::stri_count_words(abstract)]
dt_profile[, n_words_keyword := stringi::stri_count_words(keyword)]

# Calidad textual
dt_profile[, quality_flag := fcase(
  title != "" & abstract != "" & keyword != "",
  "complete_text",
  
  title != "" & abstract != "" & keyword == "",
  "title_abstract",
  
  title != "" & abstract == "" & keyword != "",
  "title_keywords",
  
  title != "" & abstract == "" & keyword == "",
  "title_only",
  
  default = "limited_text"
)]

# Campo principal para embeddings
dt_profile[, text_for_embedding := paste(
  paste0("[TITLE] ", title),
  paste0("[ABSTRACT] ", abstract),
  paste0("[KEYWORDS] ", keyword),
  sep = " [SEP] "
)]

dt_profile[, n_words_text := stringi::stri_count_words(text_for_embedding)]

# Detección de idioma opcional
if (requireNamespace("cld3", quietly = TRUE)) {
  dt_profile[, language_detected := cld3::detect_language(title)]
  dt_profile[, language_group := fifelse(
    language_detected %in% c("en", "es", "pt", "fr", "de"),
    language_detected,
    fifelse(is.na(language_detected), "unknown", "other")
  )]
} else {
  dt_profile[, language_detected := NA_character_]
  dt_profile[, language_group := "unknown"]
}

msg("Resumen de calidad textual:")
print(dt_profile[, .N, by = quality_flag][order(-N)])

msg("Resumen de idioma:")
print(dt_profile[, .N, by = language_group][order(-N)])

# ============================================================
# 7) Exportar input para embeddings
# ============================================================

profile_input_file <- file.path(
  profile_out_dir,
  paste0("profile_input_for_embeddings_", researcher_safe_id, ".csv")
)

fwrite(dt_profile, profile_input_file)

msg("Input para embeddings creado:", profile_input_file)

# ============================================================
# 8) Generar embeddings SPECTER2
# ============================================================

args_emb <- c(
  script_embeddings,
  
  "--input", profile_input_file,
  "--output_dir", profile_emb_dir,
  
  "--id_col", "doc_id",
  "--text_col", "text_for_embedding",
  
  "--model_name", "allenai/specter2_base",
  "--adapter_name", "allenai/specter2_classification",
  "--adapter_load_as", "specter2_classification",
  
  "--batch_size", "16",
  "--max_length", "512",
  
  "--normalize",
  "--overwrite"
)

run_command(
  command = python_bin,
  args = args_emb,
  log_file = log_file,
  step_name = "Generar embeddings SPECTER2 del perfil"
)

profile_emb <- file.path(profile_emb_dir, "docs_embeddings.npy")
profile_meta <- file.path(profile_emb_dir, "docs_embeddings_meta.csv")

stop_if_missing(profile_emb, "embeddings del perfil")
stop_if_missing(profile_meta, "metadata de embeddings del perfil")

# ============================================================
# 9) Aplicar modelo supervisado jerárquico v3
# ============================================================

profile_supervised_file <- file.path(
  profile_out_dir,
  paste0("profile_supervised_v3_", researcher_safe_id, ".csv")
)

args_sup <- c(
  script_supervised,
  "--emb", profile_emb,
  "--meta", profile_meta,
  "--model_dir", model_dir,
  "--output", profile_supervised_file,
  "--batch_size", "2000"
)

run_command(
  command = python_bin,
  args = args_sup,
  log_file = log_file,
  step_name = "Aplicar modelo supervisado jerárquico v3"
)

stop_if_missing(profile_supervised_file, "predicciones supervisadas del perfil")

# ============================================================
# 10) Aplicar zero-shot v2
# ============================================================

profile_zero_file <- file.path(
  profile_out_dir,
  paste0("profile_zero_shot_v2_", researcher_safe_id, ".csv")
)

args_zero <- c(
  script_zero_shot,
  
  "--doc_emb", profile_emb,
  "--doc_meta", profile_meta,
  "--tax_emb", tax_emb,
  "--tax_meta", tax_meta,
  "--output", profile_zero_file,
  
  "--top_k", "5",
  "--batch_size", "1000",
  "--sample_n", "0",
  "--seed", "123",
  
  "--min_score", "0.35",
  "--min_margin", "0.03"
)

run_command(
  command = python_bin,
  args = args_zero,
  log_file = log_file,
  step_name = "Aplicar zero-shot v2 al perfil"
)

stop_if_missing(profile_zero_file, "predicciones zero-shot del perfil")

# ============================================================
# 11) Integrar supervised v3 + zero-shot v2
# ============================================================

dt_sup <- fread(profile_supervised_file)
dt_zero <- fread(profile_zero_file)
coverage <- fread(coverage_file)

setDT(dt_sup)
setDT(dt_zero)
setDT(coverage)

dt_sup[, doc_id := as.integer(doc_id)]
dt_zero[, doc_id := as.integer(doc_id)]

if (!"supported_now" %in% names(coverage)) {
  stop("area_coverage_plan_v3.csv no contiene la columna supported_now.", call. = FALSE)
}

if (!"area_label" %in% names(coverage)) {
  stop("area_coverage_plan_v3.csv no contiene la columna area_label.", call. = FALSE)
}

supported_areas <- coverage[supported_now == TRUE, area_label]

# Crear confidence_level zero-shot si no existe
if (!"confidence_level" %in% names(dt_zero)) {
  dt_zero[, confidence_level := fcase(
    top1_score < 0.35,
    "baja_similitud",
    
    top1_top2_margin < 0.03,
    "ambiguo",
    
    top1_score >= 0.70 & top1_top2_margin >= 0.07,
    "alta_confianza",
    
    top1_score >= 0.55 & top1_top2_margin >= 0.03,
    "confianza_media",
    
    default = "revisar"
  )]
}

cols_zero <- c(
  "doc_id",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level"
)

cols_zero <- cols_zero[cols_zero %in% names(dt_zero)]
dt_zero_small <- dt_zero[, ..cols_zero]

setnames(
  dt_zero_small,
  old = setdiff(names(dt_zero_small), "doc_id"),
  new = paste0("zero_", setdiff(names(dt_zero_small), "doc_id"))
)

dt_final <- merge(
  dt_sup,
  dt_zero_small,
  by = "doc_id",
  all.x = TRUE
)

# Verificar columnas del modelo supervisado
needed_sup_cols <- c("pred_area_hierarchical", "pred_area_hierarchical_score")
missing_sup <- setdiff(needed_sup_cols, names(dt_final))
if (length(missing_sup) > 0) {
  stop(
    paste0(
      "Faltan columnas del modelo supervisado: ",
      paste(missing_sup, collapse = ", ")
    ),
    call. = FALSE
  )
}

dt_final[, supervised_area_supported := pred_area_hierarchical %in% supported_areas]
dt_final[, zero_area_supported := zero_top1_area_name %in% supported_areas]

# Regla final del sistema integrado
dt_final[, final_decision := fcase(
  
  supervised_area_supported == TRUE &
    pred_area_hierarchical_score >= 0.90,
  "aceptar_supervisado_v3",
  
  supervised_area_supported == TRUE &
    pred_area_hierarchical_score >= 0.75 &
    pred_area_hierarchical_score < 0.90,
  "sugerencia_supervisada_revisar",
  
  supervised_area_supported == FALSE &
    !is.na(zero_top1_area_name),
  "fuera_cobertura_supervisada_usar_zero_shot_sugerencia",
  
  supervised_area_supported == TRUE &
    pred_area_hierarchical_score < 0.75 &
    !is.na(zero_top1_area_name),
  "baja_confianza_supervisada_usar_zero_shot_sugerencia",
  
  default = "revision_manual"
)]

dt_final[, final_area_label := fcase(
  
  final_decision == "aceptar_supervisado_v3",
  pred_area_hierarchical,
  
  final_decision == "sugerencia_supervisada_revisar",
  pred_area_hierarchical,
  
  final_decision %in% c(
    "fuera_cobertura_supervisada_usar_zero_shot_sugerencia",
    "baja_confianza_supervisada_usar_zero_shot_sugerencia"
  ),
  zero_top1_area_name,
  
  default = NA_character_
)]

dt_final[, final_subarea_label := fcase(
  
  final_decision %in% c(
    "fuera_cobertura_supervisada_usar_zero_shot_sugerencia",
    "baja_confianza_supervisada_usar_zero_shot_sugerencia"
  ),
  zero_top1_subarea_name,
  
  default = NA_character_
)]

dt_final[, final_label_source := fcase(
  
  final_decision == "aceptar_supervisado_v3",
  "supervised_v3",
  
  final_decision == "sugerencia_supervisada_revisar",
  "supervised_v3_suggestion",
  
  final_decision %in% c(
    "fuera_cobertura_supervisada_usar_zero_shot_sugerencia",
    "baja_confianza_supervisada_usar_zero_shot_sugerencia"
  ),
  "zero_shot_v2_suggestion",
  
  default = "manual_review"
)]

dt_final[, final_confidence_score := fcase(
  
  final_label_source %in% c("supervised_v3", "supervised_v3_suggestion"),
  pred_area_hierarchical_score,
  
  final_label_source == "zero_shot_v2_suggestion",
  zero_top1_score,
  
  default = NA_real_
)]

# Agregar columnas del perfil original
cols_profile_keep <- c(
  "doc_id",
  "profile_doc_id",
  "researcher_id",
  "title",
  "title_norm",
  "abstract",
  "keyword",
  "year",
  "citas",
  "revista",
  "authors",
  "doi",
  "url",
  "quality_flag",
  "language_group",
  "language_detected",
  "n_words_title",
  "n_words_abstract",
  "n_words_keyword",
  "n_words_text"
)

cols_profile_keep <- cols_profile_keep[cols_profile_keep %in% names(dt_profile)]

# Eliminar columnas duplicadas para conservar las del perfil limpio
overlap_cols <- intersect(
  names(dt_final),
  setdiff(cols_profile_keep, "doc_id")
)

if (length(overlap_cols) > 0) {
  dt_final[, (overlap_cols) := NULL]
}

dt_final <- merge(
  dt_final,
  dt_profile[, ..cols_profile_keep],
  by = "doc_id",
  all.x = TRUE
)

# Ordenar columnas importantes al inicio
important_cols <- c(
  "doc_id",
  "profile_doc_id",
  "researcher_id",
  "title",
  "year",
  "citas",
  "revista",
  "doi",
  "url",
  "quality_flag",
  "language_group",
  "final_area_label",
  "final_subarea_label",
  "final_label_source",
  "final_confidence_score",
  "final_decision",
  "pred_macroarea_label",
  "pred_macroarea_score",
  "pred_area_hierarchical",
  "pred_area_hierarchical_score",
  "pred_area_hierarchical_top2",
  "pred_area_hierarchical_top2_score",
  "zero_top1_area_name",
  "zero_top1_subarea_name",
  "zero_top1_score",
  "zero_top2_area_name",
  "zero_top2_subarea_name",
  "zero_top2_score",
  "zero_top1_top2_margin"
)

important_cols <- important_cols[important_cols %in% names(dt_final)]
other_cols <- setdiff(names(dt_final), important_cols)

setcolorder(dt_final, c(important_cols, other_cols))

profile_final_file <- file.path(
  profile_out_dir,
  paste0("profile_final_predictions_integrated_v3_", researcher_safe_id, ".csv")
)

fwrite(dt_final, profile_final_file)

msg("Archivo final integrado creado:", profile_final_file)

# ============================================================
# 12) Resúmenes del perfil
# ============================================================

summary_decision <- dt_final[
  ,
  .N,
  by = .(final_label_source, final_decision)
][order(-N)]

summary_decision[, pct := round(N / sum(N) * 100, 2)]

summary_area <- dt_final[
  !is.na(final_area_label),
  .(
    publications = .N,
    mean_confidence = round(mean(final_confidence_score, na.rm = TRUE), 3),
    mean_year = round(mean(year, na.rm = TRUE), 1),
    total_citations = sum(citas, na.rm = TRUE)
  ),
  by = final_area_label
][order(-publications)]

if (nrow(summary_area) > 0) {
  summary_area[, pct := round(publications / sum(publications) * 100, 2)]
}

summary_year_area <- dt_final[
  !is.na(year) & !is.na(final_area_label),
  .N,
  by = .(year, final_area_label)
][order(year, -N)]

summary_source_area <- dt_final[
  !is.na(final_area_label),
  .N,
  by = .(final_label_source, final_area_label)
][order(final_label_source, -N)]

summary_quality <- dt_final[
  ,
  .N,
  by = quality_flag
][order(-N)]

summary_language <- dt_final[
  ,
  .N,
  by = language_group
][order(-N)]

fwrite(
  summary_decision,
  file.path(profile_out_dir, paste0("profile_summary_decision_", researcher_safe_id, ".csv"))
)

fwrite(
  summary_area,
  file.path(profile_out_dir, paste0("profile_summary_area_", researcher_safe_id, ".csv"))
)

fwrite(
  summary_year_area,
  file.path(profile_out_dir, paste0("profile_summary_year_area_", researcher_safe_id, ".csv"))
)

fwrite(
  summary_source_area,
  file.path(profile_out_dir, paste0("profile_summary_source_area_", researcher_safe_id, ".csv"))
)

fwrite(
  summary_quality,
  file.path(profile_out_dir, paste0("profile_summary_quality_", researcher_safe_id, ".csv"))
)

fwrite(
  summary_language,
  file.path(profile_out_dir, paste0("profile_summary_language_", researcher_safe_id, ".csv"))
)

msg("================ RESUMEN DECISIÓN ================")
print(summary_decision)

msg("================ PERFIL POR ÁREA ================")
print(summary_area)

# ============================================================
# 13) Ejecutar mapas semánticos y señales de revisión
# ============================================================

args_maps <- c(
  script_maps,
  "--emb", profile_emb,
  "--meta", profile_meta,
  "--pred", profile_final_file,
  "--output_dir", profile_maps_dir,
  "--id_col", "doc_id",
  "--title_col", "title",
  "--year_col", "year",
  "--area_col", "final_area_label",
  "--confidence_col", "final_confidence_score",
  "--source_col", "final_label_source",
  "--k", "0",
  "--max_k", "8",
  "--seed", "123"
)

run_command(
  command = python_bin,
  args = args_maps,
  log_file = log_file,
  step_name = "Generar mapas semánticos y señales de revisión"
)

# ============================================================
# 14) Cargar señales de revisión si existen
# ============================================================

points_file <- file.path(profile_maps_dir, "profile_semantic_points.csv")
cluster_file <- file.path(profile_maps_dir, "profile_kmeans_cluster_summary.csv")
cluster_area_file <- file.path(profile_maps_dir, "profile_kmeans_area_matrix.csv")
review_signals_file <- file.path(profile_maps_dir, "profile_possible_anomalies.csv")

if (file.exists(review_signals_file)) {
  dt_review <- fread(review_signals_file)
  setDT(dt_review)
  
  # Copia con nombre menos alarmista
  review_signals_clean_file <- file.path(
    profile_maps_dir,
    "profile_semantic_review_signals.csv"
  )
  
  fwrite(dt_review, review_signals_clean_file)
  
  msg("Señales de revisión semántica:", nrow(dt_review))
  msg("Archivo:", review_signals_clean_file)
}

# ============================================================
# 15) Crear manifiesto del resultado
# ============================================================

manifest <- data.table(
  item = c(
    "researcher_id",
    "researcher_safe_id",
    "input_csv",
    "output_dir",
    "embeddings_dir",
    "maps_dir",
    "final_predictions",
    "summary_decision",
    "summary_area",
    "summary_year_area",
    "summary_source_area",
    "semantic_points",
    "cluster_summary",
    "cluster_area_matrix",
    "semantic_review_signals",
    "log_file"
  ),
  file_path = c(
    researcher_id,
    researcher_safe_id,
    profile_file,
    profile_out_dir,
    profile_emb_dir,
    profile_maps_dir,
    profile_final_file,
    file.path(profile_out_dir, paste0("profile_summary_decision_", researcher_safe_id, ".csv")),
    file.path(profile_out_dir, paste0("profile_summary_area_", researcher_safe_id, ".csv")),
    file.path(profile_out_dir, paste0("profile_summary_year_area_", researcher_safe_id, ".csv")),
    file.path(profile_out_dir, paste0("profile_summary_source_area_", researcher_safe_id, ".csv")),
    points_file,
    cluster_file,
    cluster_area_file,
    file.path(profile_maps_dir, "profile_semantic_review_signals.csv"),
    log_file
  )
)

manifest_file <- file.path(
  profile_out_dir,
  paste0("profile_result_manifest_", researcher_safe_id, ".csv")
)

fwrite(manifest, manifest_file)

# ============================================================
# 16) Resumen final en consola
# ============================================================

msg("============================================================")
msg("PIPELINE OBJETIVO 3 FINALIZADO CORRECTAMENTE")
msg("Investigador:", researcher_id)
msg("Publicaciones procesadas:", nrow(dt_final))
msg("Directorio de salida:", profile_out_dir)
msg("Manifiesto:", manifest_file)
msg("============================================================")

cat("\nARCHIVOS PRINCIPALES CREADOS:\n")
cat("1. Predicciones finales:\n   ", profile_final_file, "\n", sep = "")
cat("2. Resumen por decisión:\n   ", file.path(profile_out_dir, paste0("profile_summary_decision_", researcher_safe_id, ".csv")), "\n", sep = "")
cat("3. Resumen por área:\n   ", file.path(profile_out_dir, paste0("profile_summary_area_", researcher_safe_id, ".csv")), "\n", sep = "")
cat("4. Resumen año-área:\n   ", file.path(profile_out_dir, paste0("profile_summary_year_area_", researcher_safe_id, ".csv")), "\n", sep = "")
cat("5. Mapas y señales:\n   ", profile_maps_dir, "\n", sep = "")
cat("6. Señales de revisión semántica:\n   ", file.path(profile_maps_dir, "profile_semantic_review_signals.csv"), "\n", sep = "")
cat("7. Manifiesto:\n   ", manifest_file, "\n", sep = "")
cat("8. Log:\n   ", log_file, "\n", sep = "")

cat("\nRESUMEN DE DECISIONES:\n")
print(summary_decision)

cat("\nRESUMEN DE ÁREAS:\n")
print(summary_area)