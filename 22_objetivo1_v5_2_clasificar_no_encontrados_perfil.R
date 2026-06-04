# ============================================================
# 22_objetivo1_v5_2_clasificar_no_encontrados_perfil_v2.R
#
# Clasificar publicaciones NO emparejadas de un perfil
# usando título-solo + SPECTER2 classification + zero-shot.
#
# Regla metodológica:
# - No produce clasificación final.
# - Produce sugerencias title-only.
# - Todo resultado requiere revisión.
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

# Cambia este valor para otro perfil.
# Para Alberto Martín-Martín:
if (!exists("researcher_safe_id") || is.null(researcher_safe_id)) {
  researcher_safe_id <- "alberto_martin_martin"
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

# Rutas reales usadas en el pipeline v4/v5.2
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

python_bin <- path.expand("~/env_cluster/bin/python")

python_script <- file.path(
  script_dir,
  "0060_classify_profile_unmatched_titles_v2.py"
)

unmatched_out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_unmatched_classified_v2",
  researcher_safe_id
)

dir.create(unmatched_out_dir, recursive = TRUE, showWarnings = FALSE)

# Parámetros title-only
top_k <- 5
min_score <- 0.30
min_margin <- 0.02
batch_size <- 32
overwrite <- TRUE

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

make_top5_text <- function(dt) {
  paste(
    paste0("1) ", clean_text(dt$candidate_top1)),
    paste0("2) ", clean_text(dt$candidate_top2)),
    paste0("3) ", clean_text(dt$candidate_top3)),
    paste0("4) ", clean_text(dt$candidate_top4)),
    paste0("5) ", clean_text(dt$candidate_top5)),
    sep = "\n"
  )
}

# ============================================================
# 3) Validaciones
# ============================================================

cat("\n================ CLASIFICACIÓN TITLE-ONLY NO EMPAREJADOS ================\n")
cat("Perfil:", researcher_safe_id, "\n")
cat("Directorio perfil:", profile_eval_dir, "\n")

required_files <- c(
  unmatched_csv,
  tax_emb_file,
  tax_meta_file,
  python_bin,
  python_script
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    "Faltan archivos requeridos:\n",
    paste(missing_files, collapse = "\n")
  )
}

unmatched_check <- fread(unmatched_csv)
setDT(unmatched_check)

cat("Publicaciones no emparejadas:", nrow(unmatched_check), "\n")

if (nrow(unmatched_check) == 0) {
  cat("No hay publicaciones no emparejadas. Nada que clasificar.\n")
  quit(save = "no")
}

if (!"profile_title" %in% names(unmatched_check)) {
  stop("El archivo unmatched no contiene profile_title.")
}

empty_title_pct <- round(
  mean(clean_text(unmatched_check$profile_title) == "") * 100,
  2
)

cat("Porcentaje de títulos vacíos:", empty_title_pct, "%\n")

# ============================================================
# 4) Ejecutar Python
# ============================================================

py_args <- c(
  python_script,
  "--unmatched_csv", unmatched_csv,
  "--tax_emb", tax_emb_file,
  "--tax_meta", tax_meta_file,
  "--output_dir", unmatched_out_dir,
  "--top_k", as.character(top_k),
  "--min_score", as.character(min_score),
  "--min_margin", as.character(min_margin),
  "--batch_size", as.character(batch_size)
)

if (overwrite) {
  py_args <- c(py_args, "--overwrite")
}

cat("\nEjecutando Python title-only...\n")
cat("Salida:", unmatched_out_dir, "\n")

t0 <- proc.time()

ret <- system2(
  command = python_bin,
  args = py_args,
  stdout = TRUE,
  stderr = TRUE
)

elapsed <- proc.time() - t0

cat(paste(ret, collapse = "\n"), "\n")
cat(sprintf("\nTiempo transcurrido: %.1f segundos\n", elapsed["elapsed"]))

status <- attr(ret, "status")
if (!is.null(status) && status != 0) {
  stop("Falló la clasificación Python title-only.")
}

pred_file <- file.path(
  unmatched_out_dir,
  "unmatched_zero_shot_title_only_suggestions.csv"
)

if (!file.exists(pred_file)) {
  stop("No se generó el archivo esperado: ", pred_file)
}

unmatched_classified <- fread(pred_file)
setDT(unmatched_classified)

# ============================================================
# 5) Enriquecer salida title-only con QA y metaciencia
# ============================================================

unmatched_classified[, profile_title := clean_text(profile_title)]

unmatched_classified[
  ,
  title_only_has_metascience_terms := has_metascience_terms(profile_title)
]

unmatched_classified[
  ,
  title_only_top1_is_bibliometrics := is_biblio_top1(.SD)
]

unmatched_classified[
  ,
  title_only_top5_contains_bibliometrics := contains_biblio_topk(.SD)
]

unmatched_classified[
  ,
  title_only_possible_metascience_conflict :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE
]

unmatched_classified[
  ,
  title_only_possible_metascience_top5_failure :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE &
    title_only_top5_contains_bibliometrics == FALSE
]

unmatched_classified[
  ,
  title_only_possible_metascience_rerank_case :=
    title_only_has_metascience_terms == TRUE &
    title_only_top1_is_bibliometrics == FALSE &
    title_only_top5_contains_bibliometrics == TRUE
]

# Reglas de salida:
# Nunca aceptar automáticamente title-only.
unmatched_classified[, qa_auto_accept := FALSE]
unmatched_classified[, qa_requires_review := TRUE]
unmatched_classified[, qa_final_decision := "requires_review_title_only_zero_shot"]
unmatched_classified[, qa_final_confidence_tier := "exploratory_title_only"]
unmatched_classified[, needs_profile_review := TRUE]

unmatched_classified[
  title_only_possible_metascience_conflict == TRUE,
  qa_final_decision := "requires_review_title_only_metascience_conflict"
]

unmatched_classified[
  title_only_possible_metascience_conflict == TRUE,
  qa_final_confidence_tier := "exploratory_title_only_metascience_review"
]

unmatched_classified[
  ,
  suggested_area_title_only := clean_text(top1_area_name)
]

unmatched_classified[
  ,
  suggested_subarea_title_only := clean_text(top1_subarea_name)
]

unmatched_classified[
  ,
  suggested_label_title_only := paste(
    suggested_area_title_only,
    suggested_subarea_title_only,
    sep = " / "
  )
]

unmatched_classified[
  ,
  title_only_evidence_level := fcase(
    
    zero_shot_status == "low_similarity_review",
    "titulo_solo_baja_similitud_revision",
    
    zero_shot_status == "ambiguous_review",
    "titulo_solo_ambiguo_revision",
    
    title_only_possible_metascience_conflict == TRUE,
    "titulo_solo_metaciencia_revision_prioritaria",
    
    confidence_level == "alta_confianza_title_only",
    "titulo_solo_sugerencia_alta_senal",
    
    confidence_level == "confianza_media_title_only",
    "titulo_solo_sugerencia_media_senal",
    
    default = "titulo_solo_sugerencia_revision"
  )
]

unmatched_classified[
  ,
  review_reason := fcase(
    
    title_only_possible_metascience_top5_failure == TRUE,
    "title_only_no_corpus_match; possible_metascience_bibliometrics_top5_failure",
    
    title_only_possible_metascience_rerank_case == TRUE,
    "title_only_no_corpus_match; possible_metascience_bibliometrics_rerank_case",
    
    title_only_possible_metascience_conflict == TRUE,
    "title_only_no_corpus_match; possible_metascience_bibliometrics_conflict",
    
    default = "title_only_no_corpus_match; exploratory_zero_shot_suggestion"
  )
]

# No crear final_area_label ni final_subarea_label.
# Si vinieran del Python por alguna razón, se eliminan o vacían.
if ("final_area_label" %in% names(unmatched_classified)) {
  unmatched_classified[, final_area_label := NA_character_]
}

if ("final_subarea_label" %in% names(unmatched_classified)) {
  unmatched_classified[, final_subarea_label := NA_character_]
}

# ============================================================
# 6) Resúmenes title-only
# ============================================================

summary_area_unmatched <- unmatched_classified[
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

summary_area_unmatched[
  ,
  pct := round(publications / sum(publications) * 100, 2)
]

summary_status_unmatched <- unmatched_classified[
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
    nrow(unmatched_classified),
    unmatched_classified[title_only_has_metascience_terms == TRUE, .N],
    unmatched_classified[title_only_top1_is_bibliometrics == TRUE, .N],
    unmatched_classified[title_only_top5_contains_bibliometrics == TRUE, .N],
    unmatched_classified[title_only_possible_metascience_conflict == TRUE, .N],
    unmatched_classified[title_only_possible_metascience_top5_failure == TRUE, .N],
    unmatched_classified[title_only_possible_metascience_rerank_case == TRUE, .N]
  )
)

summary_metascience_unmatched[
  ,
  pct := round(value / nrow(unmatched_classified) * 100, 2)
]

# ============================================================
# 7) Combinar con emparejadas del perfil
# ============================================================

matched_dt <- if (file.exists(matched_csv)) {
  fread(matched_csv)
} else {
  data.table()
}

if (nrow(matched_dt) > 0) {
  setDT(matched_dt)
  
  matched_dt[
    ,
    profile_classification_origin := "matched_in_universe_v5_2_qa"
  ]
  
  matched_dt[
    ,
    profile_output_type := fifelse(
      qa_auto_accept == TRUE,
      "accepted_from_corpus_qa",
      "review_required_from_corpus_qa"
    )
  ]
  
  matched_dt[
    ,
    profile_accepted_area_label := fifelse(
      qa_auto_accept == TRUE,
      qa_final_area_label,
      NA_character_
    )
  ]
  
  matched_dt[
    ,
    profile_accepted_subarea_label := fifelse(
      qa_auto_accept == TRUE,
      qa_final_subarea_label,
      NA_character_
    )
  ]
  
  matched_dt[
    ,
    profile_suggested_area_label := top1_area_name
  ]
  
  matched_dt[
    ,
    profile_suggested_subarea_label := top1_subarea_name
  ]
  
  matched_dt[
    ,
    profile_requires_review := qa_requires_review
  ]
}

if (nrow(unmatched_classified) > 0) {
  
  unmatched_profile <- copy(unmatched_classified)
  
  unmatched_profile[
    ,
    profile_classification_origin := "profile_unmatched_title_only_suggestion"
  ]
  
  unmatched_profile[
    ,
    profile_output_type := "review_required_title_only_suggestion"
  ]
  
  unmatched_profile[
    ,
    profile_accepted_area_label := NA_character_
  ]
  
  unmatched_profile[
    ,
    profile_accepted_subarea_label := NA_character_
  ]
  
  unmatched_profile[
    ,
    profile_suggested_area_label := suggested_area_title_only
  ]
  
  unmatched_profile[
    ,
    profile_suggested_subarea_label := suggested_subarea_title_only
  ]
  
  unmatched_profile[
    ,
    profile_requires_review := TRUE
  ]
  
} else {
  unmatched_profile <- data.table()
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
  if (nrow(unmatched_profile) > 0 && !col %in% names(unmatched_profile)) unmatched_profile[, (col) := NA]
}

profile_combined <- rbindlist(
  list(
    if (nrow(matched_dt) > 0) matched_dt[, ..common_profile_cols] else NULL,
    if (nrow(unmatched_profile) > 0) unmatched_profile[, ..common_profile_cols] else NULL
  ),
  fill = TRUE
)

# ============================================================
# 8) Resúmenes combinados
# ============================================================

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
    nrow(unmatched_profile),
    nrow(profile_combined),
    round(nrow(matched_dt) / nrow(profile_combined) * 100, 2),
    round(nrow(unmatched_profile) / nrow(profile_combined) * 100, 2)
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
# 9) Muestra de revisión manual
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
# 10) Exportar
# ============================================================

fwrite(
  unmatched_classified,
  file.path(unmatched_out_dir, "unmatched_title_only_suggestions_enriched.csv")
)

fwrite(
  profile_combined,
  file.path(unmatched_out_dir, "profile_combined_matched_plus_title_only_suggestions.csv")
)

fwrite(
  review_dt,
  file.path(unmatched_out_dir, "profile_review_all_matched_and_title_only.csv")
)

fwrite(
  summary_area_unmatched,
  file.path(unmatched_out_dir, "summary_area_unmatched_title_only.csv")
)

fwrite(
  summary_status_unmatched,
  file.path(unmatched_out_dir, "summary_status_unmatched_title_only.csv")
)

fwrite(
  summary_metascience_unmatched,
  file.path(unmatched_out_dir, "summary_metascience_unmatched_title_only.csv")
)

fwrite(
  summary_profile_coverage,
  file.path(unmatched_out_dir, "summary_profile_coverage_v2.csv")
)

fwrite(
  summary_profile_output,
  file.path(unmatched_out_dir, "summary_profile_output_v2.csv")
)

fwrite(
  summary_profile_suggested_area,
  file.path(unmatched_out_dir, "summary_profile_suggested_area_v2.csv")
)

fwrite(
  summary_profile_accepted_area,
  file.path(unmatched_out_dir, "summary_profile_accepted_area_v2.csv")
)

# XLSX de revisión
if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  xlsx_file <- file.path(
    unmatched_out_dir,
    "profile_review_all_matched_and_title_only.xlsx"
  )
  
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
# 11) Imprimir
# ============================================================

cat("\n================ RESULTADOS TITLE-ONLY v2 ================\n")

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
cat(unmatched_out_dir, "\n")