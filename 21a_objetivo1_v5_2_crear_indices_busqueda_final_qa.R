# ============================================================
# 21a_objetivo1_v5_2_crear_indices_busqueda_final_qa.R
# Crear índices rápidos para evaluar perfiles individuales
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

final_qa_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe_qa",
  "classification_final_universe_v5_2_corrected_984318_with_metascience_qa_preserve_original.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_indexes"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_index_rds <- file.path(out_dir, "final_qa_search_indexes_v5_2.rds")
out_summary <- file.path(out_dir, "summary_final_qa_search_indexes_v5_2.csv")

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

normalize_doi <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- gsub("^https?://(dx\\.)?doi\\.org/", "", x)
  x <- gsub("^doi:", "", x)
  trimws(x)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))
safe_integer <- function(x) suppressWarnings(as.integer(x))

if (!file.exists(final_qa_file)) {
  stop("No existe archivo final QA: ", final_qa_file)
}

needed_cols <- c(
  "doc_id_v5", "doc_id_interest", "work_id", "doi", "title", "year",
  "citations", "source", "source_clean", "type",
  "original_final_decision", "original_final_confidence_tier",
  "original_auto_accept", "original_requires_review",
  "original_critical_conflict",
  "original_final_area_label", "original_final_subarea_label",
  "qa_final_decision", "qa_final_confidence_tier",
  "qa_auto_accept", "qa_requires_review",
  "qa_final_area_label", "qa_final_subarea_label",
  "qa_reportable_label",
  "qa_metascience_flag", "qa_metascience_priority_review",
  "qa_metascience_review_type", "qa_changed_auto_accept",
  "qa_changed_decision",
  "top1_area_name", "top1_subarea_name", "top1_score",
  "top2_area_name", "top2_subarea_name", "top2_score",
  "top3_area_name", "top3_subarea_name", "top3_score",
  "top4_area_name", "top4_subarea_name", "top4_score",
  "top5_area_name", "top5_subarea_name", "top5_score",
  "candidate_top1", "candidate_top2", "candidate_top3",
  "candidate_top4", "candidate_top5",
  "top1_top2_margin", "zero_shot_status", "confidence_level",
  "review_type", "review_reason",
  "has_metascience_bibliometrics_terms",
  "top1_is_bibliometrics",
  "top5_contains_bibliometrics",
  "possible_bibliometrics_metascience_conflict",
  "possible_bibliometrics_metascience_top5_failure",
  "possible_bibliometrics_metascience_rerank_case"
)

available_cols <- names(fread(final_qa_file, nrows = 0))
read_cols <- intersect(needed_cols, available_cols)

cat("Leyendo archivo final QA solo con columnas necesarias...\n")

dt <- fread(final_qa_file, select = read_cols)
setDT(dt)

for (col in needed_cols) {
  if (!col %in% names(dt)) dt[, (col) := NA]
}

cat("Filas leídas:", nrow(dt), "\n")

final_idx <- dt[
  ,
  .(
    final_doc_id_v5 = as.integer(doc_id_v5),
    doc_id_interest,
    final_work_id = clean_text(work_id),
    final_doi = clean_text(doi),
    final_title = clean_text(title),
    final_year = safe_integer(year),
    final_citations = safe_numeric(citations),
    final_source = clean_text(source),
    final_source_clean = clean_text(source_clean),
    final_type = clean_text(type),
    
    original_final_decision = clean_text(original_final_decision),
    original_final_confidence_tier = clean_text(original_final_confidence_tier),
    original_auto_accept = as.logical(original_auto_accept),
    original_requires_review = as.logical(original_requires_review),
    original_critical_conflict = as.logical(original_critical_conflict),
    original_final_area_label = clean_text(original_final_area_label),
    original_final_subarea_label = clean_text(original_final_subarea_label),
    
    qa_final_decision = clean_text(qa_final_decision),
    qa_final_confidence_tier = clean_text(qa_final_confidence_tier),
    qa_auto_accept = as.logical(qa_auto_accept),
    qa_requires_review = as.logical(qa_requires_review),
    qa_final_area_label = clean_text(qa_final_area_label),
    qa_final_subarea_label = clean_text(qa_final_subarea_label),
    qa_reportable_label = clean_text(qa_reportable_label),
    
    qa_metascience_flag = as.logical(qa_metascience_flag),
    qa_metascience_priority_review = as.logical(qa_metascience_priority_review),
    qa_metascience_review_type = clean_text(qa_metascience_review_type),
    qa_changed_auto_accept = as.logical(qa_changed_auto_accept),
    qa_changed_decision = as.logical(qa_changed_decision),
    
    top1_area_name = clean_text(top1_area_name),
    top1_subarea_name = clean_text(top1_subarea_name),
    top1_score = safe_numeric(top1_score),
    top2_area_name = clean_text(top2_area_name),
    top2_subarea_name = clean_text(top2_subarea_name),
    top2_score = safe_numeric(top2_score),
    top3_area_name = clean_text(top3_area_name),
    top3_subarea_name = clean_text(top3_subarea_name),
    top3_score = safe_numeric(top3_score),
    top4_area_name = clean_text(top4_area_name),
    top4_subarea_name = clean_text(top4_subarea_name),
    top4_score = safe_numeric(top4_score),
    top5_area_name = clean_text(top5_area_name),
    top5_subarea_name = clean_text(top5_subarea_name),
    top5_score = safe_numeric(top5_score),
    candidate_top1 = clean_text(candidate_top1),
    candidate_top2 = clean_text(candidate_top2),
    candidate_top3 = clean_text(candidate_top3),
    candidate_top4 = clean_text(candidate_top4),
    candidate_top5 = clean_text(candidate_top5),
    top1_top2_margin = safe_numeric(top1_top2_margin),
    zero_shot_status = clean_text(zero_shot_status),
    confidence_level = clean_text(confidence_level),
    review_type = clean_text(review_type),
    review_reason = clean_text(review_reason),
    
    has_metascience_bibliometrics_terms = as.logical(has_metascience_bibliometrics_terms),
    top1_is_bibliometrics = as.logical(top1_is_bibliometrics),
    top5_contains_bibliometrics = as.logical(top5_contains_bibliometrics),
    possible_bibliometrics_metascience_conflict = as.logical(possible_bibliometrics_metascience_conflict),
    possible_bibliometrics_metascience_top5_failure = as.logical(possible_bibliometrics_metascience_top5_failure),
    possible_bibliometrics_metascience_rerank_case = as.logical(possible_bibliometrics_metascience_rerank_case)
  )
]

final_idx[, final_title_norm := normalize_title(final_title)]
final_idx[, final_doi_norm := normalize_doi(final_doi)]

final_idx[, final_rank := fcase(
  qa_auto_accept == TRUE, 1L,
  qa_requires_review == TRUE, 2L,
  default = 3L
)]

# Muy importante:
# NO usar .SD por 984k grupos.
# Ordenar y luego unique() es mucho más rápido.

cat("Creando índice DOI...\n")
final_by_doi <- final_idx[final_doi_norm != ""]
setorder(final_by_doi, final_doi_norm, final_rank, -top1_score)
final_by_doi <- unique(final_by_doi, by = "final_doi_norm")

cat("Creando índice título + año...\n")
final_by_title_year <- final_idx[final_title_norm != "" & !is.na(final_year)]
setorder(final_by_title_year, final_title_norm, final_year, final_rank, -top1_score)
final_by_title_year <- unique(final_by_title_year, by = c("final_title_norm", "final_year"))

cat("Creando índice título...\n")
final_by_title <- final_idx[final_title_norm != ""]
setorder(final_by_title, final_title_norm, final_rank, -top1_score)
final_by_title <- unique(final_by_title, by = "final_title_norm")

setkey(final_by_doi, final_doi_norm)
setkey(final_by_title_year, final_title_norm, final_year)
setkey(final_by_title, final_title_norm)

indexes <- list(
  final_by_doi = final_by_doi,
  final_by_title_year = final_by_title_year,
  final_by_title = final_by_title,
  created_at = Sys.time()
)

saveRDS(indexes, out_index_rds, compress = FALSE)

summary_index <- data.table(
  index = c("final_by_doi", "final_by_title_year", "final_by_title"),
  rows = c(nrow(final_by_doi), nrow(final_by_title_year), nrow(final_by_title))
)

fwrite(summary_index, out_summary)

cat("\n================ ÍNDICES CREADOS ================\n")
print(summary_index)
cat("\nArchivo RDS:\n")
cat(out_index_rds, "\n")