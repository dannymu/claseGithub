# ============================================================
# 22b_objetivo1_v5_2_clasificar_no_encontrados_perfil_pipeline_oficial.R
#
# Clasificar publicaciones NO emparejadas de un perfil usando
# el pipeline oficial:
#   0002-generar_embeddings_specter2.py
#   0030_zero_shot_v4_multiprototype.py
#
# Regla:
# - Son sugerencias title-only.
# - Nunca se aceptan automáticamente.
# - Todas requieren revisión.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

# ============================================================
# 1) Configuración
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")
script_dir <- file.path(base_dir, "script")

if (!exists("researcher_safe_id") || is.null(researcher_safe_id)) {
  researcher_safe_id <- "martin"
}

profile_eval_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_evaluation_fast",
  researcher_safe_id
)

unmatched_csv <- file.path(
  profile_eval_dir,
  "profile_publications_unmatched.csv"
)

matched_csv <- file.path(
  profile_eval_dir,
  "profile_publications_matched_final_qa.csv"
)

python_bin <- path.expand("~/env_cluster/bin/python")

script_embeddings <- file.path(
  script_dir,
  "0002-generar_embeddings_specter2.py"
)

script_zero <- file.path(
  script_dir,
  "0030_zero_shot_v4_multiprototype.py"
)

tax_emb_file <- file.path(
  base_dir,
  "embeddings",
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings.npy"
)

tax_meta_file <- file.path(
  base_dir,
  "embeddings",
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings_meta_fixed.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_unmatched_classified_pipeline",
  researcher_safe_id
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

embedding_input_file <- file.path(out_dir, "unmatched_title_only_embedding_input.csv")

doc_emb_dir <- file.path(out_dir, "embeddings_title_only")
dir.create(doc_emb_dir, recursive = TRUE, showWarnings = FALSE)

doc_emb_file <- file.path(doc_emb_dir, "docs_embeddings.npy")
doc_meta_file <- file.path(doc_emb_dir, "docs_embeddings_meta.csv")
doc_meta_fixed_file <- file.path(doc_emb_dir, "docs_embeddings_meta_fixed.csv")

zero_output_file <- file.path(out_dir, "unmatched_zero_shot_title_only_suggestions.csv")

top_k <- 5
min_score <- 0.30
min_margin <- 0.02

# ============================================================
# 2) Funciones
# ============================================================

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

run_command <- function(command, args, step_name) {
  
  cat("\n============================================================\n")
  cat("Ejecutando:", step_name, "\n")
  cat("Comando:", command, "\n")
  cat("Args:", paste(args, collapse = " "), "\n")
  cat("============================================================\n")
  
  t0 <- proc.time()
  
  res <- system2(
    command = command,
    args = args,
    stdout = TRUE,
    stderr = TRUE
  )
  
  elapsed <- proc.time() - t0
  
  cat(paste(res, collapse = "\n"), "\n")
  cat(sprintf("\nTiempo %s: %.1f segundos\n", step_name, elapsed["elapsed"]))
  
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop("Falló el paso: ", step_name)
  }
  
  invisible(res)
}

contains_biblio_topk <- function(dt) {
  out <- rep(FALSE, nrow(dt))
  
  for (k in 1:5) {
    area_col <- paste0("top", k, "_area_name")
    sub_col <- paste0("top", k, "_subarea_name")
    
    if (area_col %in% names(dt) && sub_col %in% names(dt)) {
      out <- out |
        clean_text(dt[[area_col]]) == "Social Sciences" &
        clean_text(dt[[sub_col]]) == "Bibliometrics and Scientometrics"
    }
  }
  
  out
}

is_biblio_top1 <- function(dt) {
  if (!all(c("top1_area_name", "top1_subarea_name") %in% names(dt))) {
    return(rep(FALSE, nrow(dt)))
  }
  
  clean_text(dt$top1_area_name) == "Social Sciences" &
    clean_text(dt$top1_subarea_name) == "Bibliometrics and Scientometrics"
}

has_metascience_terms <- function(x) {
  pattern <- paste(
    c(
      "bibliometric",
      "bibliometr",
      "scientometric",
      "cienciometr",
      "informetric",
      "altmetric",
      "altm[eé]tric",
      "citation analysis",
      "citation impact",
      "citation index",
      "citation count",
      "co-citation",
      "cocitation",
      "co-word",
      "h-index",
      "\\bindice h\\b",
      "google scholar",
      "google scholar metrics",
      "web of science",
      "scopus",
      "journal citation reports",
      "impact factor",
      "factor de impacto",
      "journal ranking",
      "ranking de revistas",
      "research evaluation",
      "research assessment",
      "evaluaci[oó]n cient[ií]fica",
      "producci[oó]n cient[ií]fica",
      "scientific production",
      "scientific output",
      "scholarly communication",
      "comunicaci[oó]n cient[ií]fica",
      "science mapping",
      "mapas? de ciencia",
      "data citation",
      "data sharing",
      "data reuse",
      "research data",
      "open science",
      "ciencia abierta",
      "open access",
      "acceso abierto",
      "repositories",
      "repositorios",
      "academic profile",
      "academic search engine",
      "research visibility",
      "visibilidad cient[ií]fica",
      "orcid",
      "doi",
      "crossref",
      "openalex",
      "semantic scholar",
      "mendeley",
      "plumx"
    ),
    collapse = "|"
  )
  
  grepl(pattern, clean_text(x), ignore.case = TRUE, perl = TRUE)
}

# ============================================================
# 3) Validaciones
# ============================================================

cat("\n================ TITLE-ONLY CON PIPELINE OFICIAL ================\n")
cat("Perfil:", researcher_safe_id, "\n")
cat("Directorio perfil:", profile_eval_dir, "\n")

required_files <- c(
  unmatched_csv,
  tax_emb_file,
  tax_meta_file,
  python_bin,
  script_embeddings,
  script_zero
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    "Faltan archivos requeridos:\n",
    paste(missing_files, collapse = "\n")
  )
}

unmatched <- fread(unmatched_csv)
setDT(unmatched)

cat("Publicaciones no emparejadas:", nrow(unmatched), "\n")

if (nrow(unmatched) == 0) {
  stop("No hay publicaciones no emparejadas.")
}

if (!"profile_title" %in% names(unmatched)) {
  stop("El archivo unmatched no contiene profile_title.")
}

unmatched[, profile_title := clean_text(profile_title)]
unmatched <- unmatched[profile_title != ""]

# ============================================================
# 4) Crear input para embeddings title-only
# ============================================================

embedding_input <- copy(unmatched)

embedding_input[, doc_id := as.integer(profile_row_id)]

# Formato SPECTER2 esperado:
# title [SEP] abstract vacío
embedding_input[, text_for_embedding := paste0(profile_title, " [SEP] ")]

embedding_input[, text_input_type := "title_sep_empty_abstract"]
embedding_input[, classification_origin := "profile_unmatched_title_only_suggestion"]

fwrite(embedding_input, embedding_input_file)

cat("\nInput embeddings creado:\n")
cat(embedding_input_file, "\n")
cat("Filas:", nrow(embedding_input), "\n")

# ============================================================
# 5) Generar embeddings con script oficial 0002
# ============================================================

args_embeddings <- c(
  script_embeddings,
  "--input", embedding_input_file,
  "--output_dir", doc_emb_dir,
  "--id_col", "doc_id",
  "--text_col", "text_for_embedding",
  "--model_name", "allenai/specter2_base",
  "--adapter_name", "allenai/specter2_classification",
  "--adapter_load_as", "specter2_classification",
  "--batch_size", "32",
  "--max_length", "512",
  "--normalize",
  "--overwrite"
)

run_command(
  python_bin,
  args_embeddings,
  "Generar embeddings title-only con 0002"
)

if (!file.exists(doc_emb_file)) {
  stop("No se generó: ", doc_emb_file)
}

if (!file.exists(doc_meta_file)) {
  stop("No se generó: ", doc_meta_file)
}

# ============================================================
# 6) Corregir metadata embeddings local
# ============================================================

doc_meta <- fread(doc_meta_file)
setDT(doc_meta)

if (!"doc_id" %in% names(doc_meta)) {
  stop("La metadata generada no tiene doc_id.")
}

doc_meta[, doc_id := as.integer(doc_id)]

# Asegurar emb_row local
doc_meta[, emb_row := seq_len(.N) - 1L]

fwrite(doc_meta, doc_meta_fixed_file)

cat("\nMetadata local fixed:\n")
cat(doc_meta_fixed_file, "\n")
cat("Filas:", nrow(doc_meta), "\n")
cat("Rango emb_row:", min(doc_meta$emb_row), "-", max(doc_meta$emb_row), "\n")

# ============================================================
# 7) Zero-shot con script oficial 0030
# ============================================================

args_zero <- c(
  script_zero,
  "--doc_emb", doc_emb_file,
  "--doc_meta", doc_meta_fixed_file,
  "--tax_emb", tax_emb_file,
  "--tax_meta", tax_meta_file,
  "--output", zero_output_file,
  "--top_k", as.character(top_k),
  "--batch_size", "500",
  "--sample_n", "0",
  "--seed", "123",
  "--min_score", as.character(min_score),
  "--min_margin", as.character(min_margin)
)

run_command(
  python_bin,
  args_zero,
  "Zero-shot title-only con 0030"
)

if (!file.exists(zero_output_file)) {
  stop("No se generó: ", zero_output_file)
}

# ============================================================
# 8) Leer predicciones y unir metadata del perfil
# ============================================================

pred <- fread(zero_output_file)
setDT(pred)

if (!"doc_id" %in% names(pred)) {
  stop("La salida zero-shot no contiene doc_id.")
}

pred[, doc_id := as.integer(doc_id)]

profile_cols <- c(
  "profile_row_id",
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_doi",
  "profile_source"
)

for (col in profile_cols) {
  if (!col %in% names(unmatched)) unmatched[, (col) := NA]
}

profile_meta <- unmatched[
  ,
  .(
    doc_id = as.integer(profile_row_id),
    profile_row_id,
    researcher_name,
    researcher_safe_id,
    profile_file,
    profile_title,
    profile_year,
    profile_cites,
    profile_doi,
    profile_source
  )
]

pred <- merge(
  pred,
  profile_meta,
  by = "doc_id",
  all.x = TRUE
)

# ============================================================
# 9) Enriquecer salida title-only con reglas metodológicas
# ============================================================

pred[, profile_title := clean_text(profile_title)]

pred[, title_only_has_metascience_terms := has_metascience_terms(profile_title)]
pred[, title_only_top1_is_bibliometrics := is_biblio_top1(.SD)]
pred[, title_only_top5_contains_bibliometrics := contains_biblio_topk(.SD)]

pred[
  ,
  title_only_possible_metascience_conflict :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE
]

pred[
  ,
  title_only_possible_metascience_top5_failure :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE &
    title_only_top5_contains_bibliometrics == FALSE
]

pred[
  ,
  title_only_possible_metascience_rerank_case :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE &
    title_only_top5_contains_bibliometrics == TRUE
]

# Nunca aceptar automáticamente title-only
pred[, qa_auto_accept := FALSE]
pred[, qa_requires_review := TRUE]

pred[, qa_final_decision := fifelse(
  title_only_possible_metascience_conflict == TRUE,
  "requires_review_title_only_metascience_conflict",
  "requires_review_title_only_zero_shot"
)]

pred[, qa_final_confidence_tier := fifelse(
  title_only_possible_metascience_conflict == TRUE,
  "exploratory_title_only_metascience_review",
  "exploratory_title_only"
)]

pred[, suggested_area_title_only := clean_text(top1_area_name)]
pred[, suggested_subarea_title_only := clean_text(top1_subarea_name)]
pred[, suggested_label_title_only := paste(
  suggested_area_title_only,
  suggested_subarea_title_only,
  sep = " / "
)]

pred[, title_only_evidence_level := fcase(
  
  zero_shot_status == "low_similarity_review",
  "titulo_solo_baja_similitud_revision",
  
  zero_shot_status == "ambiguous_review",
  "titulo_solo_ambiguo_revision",
  
  title_only_possible_metascience_conflict == TRUE,
  "titulo_solo_metaciencia_revision_prioritaria",
  
  confidence_level %in% c("alta_confianza", "high_confidence", "alta_confianza_title_only"),
  "titulo_solo_sugerencia_alta_senal",
  
  confidence_level %in% c("confianza_media", "medium_confidence", "confianza_media_title_only"),
  "titulo_solo_sugerencia_media_senal",
  
  default = "titulo_solo_sugerencia_revision"
)]

pred[, review_reason := fcase(
  
  title_only_possible_metascience_top5_failure == TRUE,
  "title_only_no_corpus_match; possible_metascience_bibliometrics_top5_failure",
  
  title_only_possible_metascience_rerank_case == TRUE,
  "title_only_no_corpus_match; possible_metascience_bibliometrics_rerank_case",
  
  title_only_possible_metascience_conflict == TRUE,
  "title_only_no_corpus_match; possible_metascience_bibliometrics_conflict",
  
  default = "title_only_no_corpus_match; exploratory_zero_shot_suggestion"
)]

pred[, profile_classification_origin := "profile_unmatched_title_only_suggestion"]
pred[, profile_output_type := "review_required_title_only_suggestion"]
pred[, profile_accepted_area_label := NA_character_]
pred[, profile_accepted_subarea_label := NA_character_]
pred[, profile_suggested_area_label := suggested_area_title_only]
pred[, profile_suggested_subarea_label := suggested_subarea_title_only]
pred[, profile_requires_review := TRUE]

# ============================================================
# 10) Combinar con emparejadas del perfil
# ============================================================

matched_dt <- if (file.exists(matched_csv)) {
  fread(matched_csv)
} else {
  data.table()
}

if (nrow(matched_dt) > 0) {
  setDT(matched_dt)
  
  matched_dt[, profile_classification_origin := "matched_in_universe_v5_2_qa"]
  matched_dt[, profile_output_type := fifelse(
    qa_auto_accept == TRUE,
    "accepted_from_corpus_qa",
    "review_required_from_corpus_qa"
  )]
  
  matched_dt[, profile_accepted_area_label := fifelse(
    qa_auto_accept == TRUE,
    qa_final_area_label,
    NA_character_
  )]
  
  matched_dt[, profile_accepted_subarea_label := fifelse(
    qa_auto_accept == TRUE,
    qa_final_subarea_label,
    NA_character_
  )]
  
  matched_dt[, profile_suggested_area_label := top1_area_name]
  matched_dt[, profile_suggested_subarea_label := top1_subarea_name]
  matched_dt[, profile_requires_review := qa_requires_review]
}

common_profile_cols <- c(
  "profile_row_id",
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_doi",
  "profile_source",
  "profile_classification_origin",
  "profile_output_type",
  "profile_accepted_area_label",
  "profile_accepted_subarea_label",
  "profile_suggested_area_label",
  "profile_suggested_subarea_label",
  "profile_requires_review",
  "qa_final_decision",
  "qa_final_confidence_tier",
  "qa_auto_accept",
  "qa_requires_review",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top1_top2_margin",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "zero_shot_status",
  "confidence_level",
  "review_reason",
  "title_only_evidence_level",
  "title_only_possible_metascience_conflict"
)

for (col in common_profile_cols) {
  if (nrow(matched_dt) > 0 && !col %in% names(matched_dt)) matched_dt[, (col) := NA]
  if (!col %in% names(pred)) pred[, (col) := NA]
}

profile_combined <- rbindlist(
  list(
    if (nrow(matched_dt) > 0) matched_dt[, ..common_profile_cols] else NULL,
    pred[, ..common_profile_cols]
  ),
  fill = TRUE
)

# ============================================================
# 11) Resúmenes
# ============================================================

summary_area_unmatched <- pred[
  ,
  .(
    publications = .N,
    citations = sum(profile_cites, na.rm = TRUE),
    mean_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4),
    review_required = sum(qa_requires_review == TRUE, na.rm = TRUE),
    metascience_conflict = sum(title_only_possible_metascience_conflict == TRUE, na.rm = TRUE)
  ),
  by = .(
    suggested_area_title_only,
    suggested_subarea_title_only
  )
][order(-publications)]

summary_area_unmatched[, pct := round(publications / sum(publications) * 100, 2)]

summary_status_unmatched <- pred[
  ,
  .N,
  by = .(
    zero_shot_status,
    confidence_level,
    title_only_evidence_level
  )
][order(-N)]

summary_metascience_unmatched <- data.table(
  metric = c(
    "title_only_total",
    "title_only_has_metascience_terms",
    "title_only_top1_is_bibliometrics",
    "title_only_top5_contains_bibliometrics",
    "title_only_possible_metascience_conflict",
    "title_only_possible_metascience_top5_failure",
    "title_only_possible_metascience_rerank_case"
  ),
  value = c(
    nrow(pred),
    pred[title_only_has_metascience_terms == TRUE, .N],
    pred[title_only_top1_is_bibliometrics == TRUE, .N],
    pred[title_only_top5_contains_bibliometrics == TRUE, .N],
    pred[title_only_possible_metascience_conflict == TRUE, .N],
    pred[title_only_possible_metascience_top5_failure == TRUE, .N],
    pred[title_only_possible_metascience_rerank_case == TRUE, .N]
  )
)

summary_metascience_unmatched[, pct := round(value / nrow(pred) * 100, 2)]

summary_profile_coverage <- data.table(
  metric = c(
    "matched_in_corpus",
    "unmatched_title_only_suggestions",
    "profile_total_covered",
    "pct_matched_in_corpus",
    "pct_title_only_suggestions"
  ),
  value = c(
    nrow(matched_dt),
    nrow(pred),
    nrow(profile_combined),
    round(nrow(matched_dt) / nrow(profile_combined) * 100, 2),
    round(nrow(pred) / nrow(profile_combined) * 100, 2)
  )
)

summary_profile_output <- profile_combined[
  ,
  .N,
  by = .(
    profile_classification_origin,
    profile_output_type,
    profile_requires_review
  )
][order(-N)]

summary_profile_suggested_area <- profile_combined[
  ,
  .(
    publications = .N,
    citations = sum(profile_cites, na.rm = TRUE),
    accepted = sum(!is.na(profile_accepted_area_label) & profile_accepted_area_label != "", na.rm = TRUE),
    review_required = sum(profile_requires_review == TRUE, na.rm = TRUE)
  ),
  by = .(
    profile_suggested_area_label,
    profile_suggested_subarea_label
  )
][order(-publications)]

summary_profile_accepted_area <- profile_combined[
  !is.na(profile_accepted_area_label) &
    profile_accepted_area_label != "",
  .(
    publications = .N,
    citations = sum(profile_cites, na.rm = TRUE)
  ),
  by = .(
    profile_accepted_area_label,
    profile_accepted_subarea_label
  )
][order(-publications)]

# ============================================================
# 12) Revisión manual
# ============================================================

review_dt <- copy(profile_combined)

review_dt[
  ,
  candidate_top5_text := paste(
    paste0("1) ", clean_text(candidate_top1)),
    paste0("2) ", clean_text(candidate_top2)),
    paste0("3) ", clean_text(candidate_top3)),
    paste0("4) ", clean_text(candidate_top4)),
    paste0("5) ", clean_text(candidate_top5)),
    sep = "\n"
  )
]

review_dt[
  ,
  text_for_review_short := paste(
    paste0("Título: ", profile_title),
    paste0("Año: ", profile_year),
    paste0("Fuente: ", profile_source),
    paste0("Origen: ", profile_classification_origin),
    paste0("Tipo salida: ", profile_output_type),
    paste0("Área aceptada: ", profile_accepted_area_label),
    paste0("Subárea aceptada: ", profile_accepted_subarea_label),
    paste0("Sugerencia top1: ", profile_suggested_area_label, " / ", profile_suggested_subarea_label),
    paste0("Top-k:\n", candidate_top5_text),
    sep = "\n\n"
  )
]

review_dt[, manual_area_correct := ""]
review_dt[, manual_subarea_correct := ""]
review_dt[, manual_area_label := ""]
review_dt[, manual_subarea_label := ""]
review_dt[, manual_decision := ""]
review_dt[, manual_comment := ""]

review_cols <- c(
  "profile_row_id",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_source",
  "profile_classification_origin",
  "profile_output_type",
  "profile_accepted_area_label",
  "profile_accepted_subarea_label",
  "profile_suggested_area_label",
  "profile_suggested_subarea_label",
  "profile_requires_review",
  "qa_final_decision",
  "qa_final_confidence_tier",
  "top1_score",
  "top1_top2_margin",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "zero_shot_status",
  "confidence_level",
  "review_reason",
  "title_only_evidence_level",
  "title_only_possible_metascience_conflict",
  "text_for_review_short",
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

for (col in review_cols) {
  if (!col %in% names(review_dt)) review_dt[, (col) := NA]
}

review_dt <- review_dt[, ..review_cols]

# ============================================================
# 13) Exportar
# ============================================================

fwrite(pred, file.path(out_dir, "unmatched_title_only_suggestions_enriched.csv"))
fwrite(profile_combined, file.path(out_dir, "profile_combined_matched_plus_title_only_suggestions.csv"))
fwrite(review_dt, file.path(out_dir, "profile_review_all_matched_and_title_only.csv"))

fwrite(summary_area_unmatched, file.path(out_dir, "summary_area_unmatched_title_only.csv"))
fwrite(summary_status_unmatched, file.path(out_dir, "summary_status_unmatched_title_only.csv"))
fwrite(summary_metascience_unmatched, file.path(out_dir, "summary_metascience_unmatched_title_only.csv"))
fwrite(summary_profile_coverage, file.path(out_dir, "summary_profile_coverage.csv"))
fwrite(summary_profile_output, file.path(out_dir, "summary_profile_output.csv"))
fwrite(summary_profile_suggested_area, file.path(out_dir, "summary_profile_suggested_area.csv"))
fwrite(summary_profile_accepted_area, file.path(out_dir, "summary_profile_accepted_area.csv"))

if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  xlsx_file <- file.path(out_dir, "profile_review_all_matched_and_title_only.xlsx")
  
  wb <- openxlsx::createWorkbook()
  
  openxlsx::addWorksheet(wb, "revision")
  openxlsx::addWorksheet(wb, "cobertura")
  openxlsx::addWorksheet(wb, "sugeridas_title_only")
  openxlsx::addWorksheet(wb, "aceptadas_corpus")
  openxlsx::addWorksheet(wb, "metascience_title_only")
  openxlsx::addWorksheet(wb, "instrucciones")
  
  openxlsx::writeData(wb, "revision", review_dt)
  openxlsx::writeData(wb, "cobertura", summary_profile_coverage)
  openxlsx::writeData(wb, "sugeridas_title_only", summary_area_unmatched)
  openxlsx::writeData(wb, "aceptadas_corpus", summary_profile_accepted_area)
  openxlsx::writeData(wb, "metascience_title_only", summary_metascience_unmatched)
  
  instrucciones <- data.table(
    campo = c(
      "manual_area_correct",
      "manual_subarea_correct",
      "manual_area_label",
      "manual_subarea_label",
      "manual_decision",
      "manual_comment"
    ),
    instruccion = c(
      "TRUE si el área aceptada o sugerida es correcta; FALSE si no.",
      "TRUE si la subárea aceptada o sugerida es correcta; FALSE si no.",
      "Si es incorrecta, escribir el área correcta.",
      "Si es incorrecta, escribir la subárea correcta.",
      "Usar: correct, area_correct_subarea_wrong, area_wrong, insufficient_information, ambiguous_or_both_valid, top5_contains_correct_label, top5_does_not_contain_correct_label.",
      "Comentario breve. En title-only recordar que es sugerencia exploratoria."
    )
  )
  
  openxlsx::writeData(wb, "instrucciones", instrucciones)
  openxlsx::freezePane(wb, "revision", firstActiveRow = 2, firstActiveCol = 4)
  openxlsx::setColWidths(wb, "revision", cols = 1:ncol(review_dt), widths = "auto")
  
  text_col <- which(names(review_dt) == "text_for_review_short")
  if (length(text_col) > 0) {
    openxlsx::setColWidths(wb, "revision", cols = text_col, widths = 90)
  }
  
  openxlsx::saveWorkbook(wb, xlsx_file, overwrite = TRUE)
}

# ============================================================
# 14) Imprimir
# ============================================================

cat("\n================ RESULTADOS TITLE-ONLY PIPELINE OFICIAL ================\n")

cat("\nCobertura perfil:\n")
print(summary_profile_coverage)

cat("\nSalida por tipo:\n")
print(summary_profile_output)

cat("\nÁreas sugeridas para no emparejados title-only:\n")
print(summary_area_unmatched)

cat("\nEstado/confianza title-only:\n")
print(summary_status_unmatched)

cat("\nMetaciencia/bibliometría title-only:\n")
print(summary_metascience_unmatched)

cat("\nÁreas aceptadas desde corpus v5.2 + QA:\n")
print(summary_profile_accepted_area)

cat("\nÁreas sugeridas en perfil completo:\n")
print(summary_profile_suggested_area)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")