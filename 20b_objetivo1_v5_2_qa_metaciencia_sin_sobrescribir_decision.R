# ============================================================
# 20b_objetivo1_v5_2_qa_metaciencia_sin_sobrescribir_decision.R
# Objetivo 1 v5.2
# Agregar capa QA de metaciencia/bibliometría SIN sobrescribir
# la decisión final original.
#
# Entrada:
# classification_final_universe_v5_2_corrected_984318.csv
#
# Salida:
# classification_final_universe_v5_2_corrected_984318_with_metascience_qa_preserve_original.csv
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

final_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe"
)

input_final <- file.path(
  final_dir,
  "classification_final_universe_v5_2_corrected_984318.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe_qa"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_final_qa <- file.path(
  out_dir,
  "classification_final_universe_v5_2_corrected_984318_with_metascience_qa_preserve_original.csv"
)

out_priority_review <- file.path(
  out_dir,
  "metascience_bibliometrics_priority_review_preserve_original_v5_2.csv"
)

out_metascience_cases <- file.path(
  out_dir,
  "metascience_bibliometrics_cases_preserve_original_v5_2.csv"
)

out_summary_qa <- file.path(
  out_dir,
  "summary_metascience_bibliometrics_qa_preserve_original_v5_2.csv"
)

out_summary_original_decision <- file.path(
  out_dir,
  "summary_original_final_decision_v5_2.csv"
)

out_summary_qa_decision <- file.path(
  out_dir,
  "summary_qa_final_decision_v5_2.csv"
)

out_summary_cross <- file.path(
  out_dir,
  "summary_original_vs_qa_decision_v5_2.csv"
)

out_summary_area_qa <- file.path(
  out_dir,
  "summary_metascience_flag_by_predicted_area_preserve_original_v5_2.csv"
)

out_summary_topk_qa <- file.path(
  out_dir,
  "summary_metascience_topk_presence_preserve_original_v5_2.csv"
)

if (!file.exists(input_final)) {
  stop("No existe archivo final v5.2 corregido: ", input_final)
}

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

has_pattern <- function(x, pattern) {
  grepl(pattern, x, ignore.case = TRUE, perl = TRUE)
}

append_reason <- function(current, condition, reason) {
  out <- current
  
  idx_empty <- condition & (is.na(out) | out == "")
  out[idx_empty] <- reason
  
  idx_existing <- condition & !is.na(out) & out != "" & !grepl(reason, out, fixed = TRUE)
  out[idx_existing] <- paste(out[idx_existing], reason, sep = "; ")
  
  out
}

contains_bibliometrics_topk <- function(dt) {
  
  out <- rep(FALSE, nrow(dt))
  
  candidate_cols <- c(
    "candidate_top1",
    "candidate_top2",
    "candidate_top3",
    "candidate_top4",
    "candidate_top5"
  )
  
  for (col in candidate_cols) {
    if (col %in% names(dt)) {
      out <- out | grepl(
        "Social Sciences\\s*/\\s*Bibliometrics and Scientometrics|Bibliometrics and Scientometrics",
        clean_text(dt[[col]]),
        ignore.case = TRUE
      )
    }
  }
  
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

is_bibliometrics_top1 <- function(dt) {
  
  out <- rep(FALSE, nrow(dt))
  
  if (all(c("top1_area_name", "top1_subarea_name") %in% names(dt))) {
    out <- out |
      clean_text(dt$top1_area_name) == "Social Sciences" &
      clean_text(dt$top1_subarea_name) == "Bibliometrics and Scientometrics"
  }
  
  if (all(c("final_area_label", "final_subarea_label") %in% names(dt))) {
    out <- out |
      clean_text(dt$final_area_label) == "Social Sciences" &
      clean_text(dt$final_subarea_label) == "Bibliometrics and Scientometrics"
  }
  
  out
}

# ============================================================
# 3) Leer archivo final v5.2 original
# ============================================================

dt <- fread(input_final)
setDT(dt)

cat("Universo final leído:", nrow(dt), "\n")

needed_cols <- c(
  "doc_id_v5",
  "work_id",
  "doi",
  "title",
  "year",
  "source",
  "source_clean",
  "type",
  "final_area_label",
  "final_subarea_label",
  "reportable_label",
  "final_decision",
  "final_confidence_tier",
  "auto_accept",
  "requires_review",
  "critical_conflict",
  "classified_or_candidate",
  "review_type",
  "review_reason",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top3_area_name",
  "top3_subarea_name",
  "top3_score",
  "top4_area_name",
  "top4_subarea_name",
  "top4_score",
  "top5_area_name",
  "top5_subarea_name",
  "top5_score",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "text_quality_v5_2",
  "embedding_eligible_v5_2",
  "keyword_clean",
  "topics_clean",
  "conceptos_clean",
  "text_for_embedding"
)

for (col in needed_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := NA]
  }
}

text_cols <- c(
  "work_id",
  "doi",
  "title",
  "source",
  "source_clean",
  "type",
  "final_area_label",
  "final_subarea_label",
  "reportable_label",
  "final_decision",
  "final_confidence_tier",
  "review_type",
  "review_reason",
  "top1_area_name",
  "top1_subarea_name",
  "top2_area_name",
  "top2_subarea_name",
  "top3_area_name",
  "top3_subarea_name",
  "top4_area_name",
  "top4_subarea_name",
  "top5_area_name",
  "top5_subarea_name",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "zero_shot_status",
  "confidence_level",
  "text_quality_v5_2",
  "keyword_clean",
  "topics_clean",
  "conceptos_clean",
  "text_for_embedding"
)

for (col in text_cols) {
  dt[, (col) := clean_text(get(col))]
}

dt[, auto_accept := as.logical(auto_accept)]
dt[, requires_review := as.logical(requires_review)]
dt[, critical_conflict := as.logical(critical_conflict)]
dt[, classified_or_candidate := as.logical(classified_or_candidate)]

score_cols <- c(
  "top1_score",
  "top2_score",
  "top3_score",
  "top4_score",
  "top5_score",
  "top1_top2_margin"
)

for (col in score_cols) {
  dt[, (col) := safe_numeric(get(col))]
}

# ============================================================
# 4) Preservar decisión original
# ============================================================

dt[, original_final_decision := final_decision]
dt[, original_final_confidence_tier := final_confidence_tier]
dt[, original_auto_accept := auto_accept]
dt[, original_requires_review := requires_review]
dt[, original_critical_conflict := critical_conflict]
dt[, original_review_type := review_type]
dt[, original_review_reason := review_reason]
dt[, original_final_area_label := final_area_label]
dt[, original_final_subarea_label := final_subarea_label]
dt[, original_reportable_label := reportable_label]

# ============================================================
# 5) Patrones de metaciencia/bibliometría
# ============================================================

metascience_pattern <- paste(
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
    "citedness",
    "co-citation",
    "cocitation",
    "co-word",
    "coword",
    "bibliographic coupling",
    "h-index",
    "\\bindice h\\b",
    "google scholar",
    "google scholar metrics",
    "web of science",
    "scopus",
    "journal citation reports",
    "impact factor",
    "factor de impacto",
    "journal impact",
    "journal ranking",
    "ranking de revistas",
    "research evaluation",
    "research assessment",
    "evaluaci[oó]n cient[ií]fica",
    "evaluaci[oó]n de la investigaci[oó]n",
    "producci[oó]n cient[ií]fica",
    "scientific production",
    "scientific output",
    "scholarly communication",
    "comunicaci[oó]n cient[ií]fica",
    "communication of science",
    "science mapping",
    "mapas? de ciencia",
    "research front",
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

dt[, metascience_text := paste(
  title,
  source,
  source_clean,
  keyword_clean,
  topics_clean,
  conceptos_clean,
  text_for_embedding,
  sep = " "
)]

dt[, has_metascience_bibliometrics_terms := has_pattern(
  metascience_text,
  metascience_pattern
)]

dt[, top1_is_bibliometrics := is_bibliometrics_top1(.SD)]
dt[, top5_contains_bibliometrics := contains_bibliometrics_topk(.SD)]

dt[, possible_bibliometrics_metascience_conflict := 
     has_metascience_bibliometrics_terms == TRUE &
     top1_is_bibliometrics == FALSE]

dt[, possible_bibliometrics_metascience_top5_failure := 
     has_metascience_bibliometrics_terms == TRUE &
     top1_is_bibliometrics == FALSE &
     top5_contains_bibliometrics == FALSE]

dt[, possible_bibliometrics_metascience_rerank_case := 
     has_metascience_bibliometrics_terms == TRUE &
     top1_is_bibliometrics == FALSE &
     top5_contains_bibliometrics == TRUE]

# ============================================================
# 6) Capa QA sin sobrescribir decisión original
# ============================================================

dt[, qa_metascience_flag := possible_bibliometrics_metascience_conflict]

dt[, qa_metascience_priority_review := 
     possible_bibliometrics_metascience_top5_failure == TRUE |
     (
       possible_bibliometrics_metascience_conflict == TRUE &
         original_auto_accept == TRUE
     )]

dt[, qa_metascience_review_type := fcase(
  
  qa_metascience_priority_review == TRUE &
    possible_bibliometrics_metascience_top5_failure == TRUE,
  "metascience_bibliometrics_top5_failure",
  
  qa_metascience_priority_review == TRUE &
    possible_bibliometrics_metascience_rerank_case == TRUE,
  "metascience_bibliometrics_rerank_case",
  
  qa_metascience_priority_review == TRUE,
  "metascience_bibliometrics_priority_review",
  
  possible_bibliometrics_metascience_rerank_case == TRUE,
  "metascience_bibliometrics_rerank_candidate",
  
  possible_bibliometrics_metascience_conflict == TRUE,
  "metascience_bibliometrics_conflict",
  
  default = "no_metascience_qa_flag"
)]

# Iniciar campos QA copiando la decisión original
dt[, qa_auto_accept := original_auto_accept]
dt[, qa_requires_review := original_requires_review]
dt[, qa_final_decision := original_final_decision]
dt[, qa_final_confidence_tier := original_final_confidence_tier]
dt[, qa_reportable_label := original_reportable_label]
dt[, qa_final_area_label := original_final_area_label]
dt[, qa_final_subarea_label := original_final_subarea_label]
dt[, qa_review_type := original_review_type]
dt[, qa_review_reason := original_review_reason]

# Si QA prioritaria afecta aceptados automáticos, retirar aceptación automática
dt[
  qa_metascience_priority_review == TRUE &
    original_auto_accept == TRUE,
  qa_auto_accept := FALSE
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_requires_review := TRUE
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_final_decision := "requires_assisted_review_metascience_bibliometrics_conflict"
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_final_confidence_tier := "not_accepted_metascience_review"
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_reportable_label := "No aceptado automáticamente; posible conflicto bibliometría/metaciencia"
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_final_area_label := NA_character_
]

dt[
  qa_metascience_priority_review == TRUE,
  qa_final_subarea_label := NA_character_
]

dt[, qa_review_type := append_reason(
  qa_review_type,
  qa_metascience_priority_review == TRUE,
  "metascience_bibliometrics_priority_review"
)]

dt[, qa_review_reason := append_reason(
  qa_review_reason,
  qa_metascience_priority_review == TRUE,
  "possible_bibliometrics_metascience_conflict_detected_by_qa"
)]

dt[, qa_changed_auto_accept := original_auto_accept == TRUE & qa_auto_accept == FALSE]
dt[, qa_changed_decision := original_final_decision != qa_final_decision]

# ============================================================
# 7) Subconjuntos
# ============================================================

metascience_cases <- dt[
  has_metascience_bibliometrics_terms == TRUE |
    top1_is_bibliometrics == TRUE |
    top5_contains_bibliometrics == TRUE |
    possible_bibliometrics_metascience_conflict == TRUE
]

priority_review <- dt[
  qa_metascience_priority_review == TRUE
]

# ============================================================
# 8) Resúmenes
# ============================================================

summary_qa <- data.table(
  metric = c(
    "universe_documents",
    "documents_with_metascience_bibliometrics_terms",
    "top1_is_bibliometrics",
    "top5_contains_bibliometrics",
    "possible_bibliometrics_metascience_conflict",
    "possible_bibliometrics_metascience_top5_failure",
    "possible_bibliometrics_metascience_rerank_case",
    "qa_metascience_priority_review",
    "original_auto_accept",
    "qa_auto_accept",
    "auto_accept_removed_by_metascience_qa",
    "original_review_required",
    "qa_review_required",
    "qa_changed_decision"
  ),
  value = c(
    nrow(dt),
    dt[has_metascience_bibliometrics_terms == TRUE, .N],
    dt[top1_is_bibliometrics == TRUE, .N],
    dt[top5_contains_bibliometrics == TRUE, .N],
    dt[possible_bibliometrics_metascience_conflict == TRUE, .N],
    dt[possible_bibliometrics_metascience_top5_failure == TRUE, .N],
    dt[possible_bibliometrics_metascience_rerank_case == TRUE, .N],
    dt[qa_metascience_priority_review == TRUE, .N],
    dt[original_auto_accept == TRUE, .N],
    dt[qa_auto_accept == TRUE, .N],
    dt[qa_changed_auto_accept == TRUE, .N],
    dt[original_requires_review == TRUE, .N],
    dt[qa_requires_review == TRUE, .N],
    dt[qa_changed_decision == TRUE, .N]
  )
)

summary_qa[, pct_universe := round(value / nrow(dt) * 100, 4)]

summary_original_decision <- dt[
  ,
  .N,
  by = .(
    original_final_decision,
    original_final_confidence_tier,
    original_auto_accept,
    original_requires_review
  )
][order(-N)]

summary_original_decision[, pct := round(N / sum(N) * 100, 2)]

summary_qa_decision <- dt[
  ,
  .N,
  by = .(
    qa_final_decision,
    qa_final_confidence_tier,
    qa_auto_accept,
    qa_requires_review
  )
][order(-N)]

summary_qa_decision[, pct := round(N / sum(N) * 100, 2)]

summary_cross <- dt[
  ,
  .N,
  by = .(
    original_final_decision,
    qa_final_decision,
    qa_changed_decision
  )
][order(-N)]

summary_cross[, pct := round(N / sum(N) * 100, 2)]

summary_area_qa <- dt[
  has_metascience_bibliometrics_terms == TRUE &
    top1_is_bibliometrics == FALSE,
  .N,
  by = .(
    top1_area_name,
    top1_subarea_name,
    top5_contains_bibliometrics,
    qa_metascience_priority_review,
    original_auto_accept
  )
][order(-N)]

summary_topk_qa <- dt[
  has_metascience_bibliometrics_terms == TRUE,
  .N,
  by = .(
    top1_is_bibliometrics,
    top5_contains_bibliometrics,
    possible_bibliometrics_metascience_top5_failure,
    possible_bibliometrics_metascience_rerank_case
  )
][order(-N)]

# ============================================================
# 9) Exportar
# ============================================================

fwrite(dt, out_final_qa)
fwrite(priority_review, out_priority_review)
fwrite(metascience_cases, out_metascience_cases)
fwrite(summary_qa, out_summary_qa)
fwrite(summary_original_decision, out_summary_original_decision)
fwrite(summary_qa_decision, out_summary_qa_decision)
fwrite(summary_cross, out_summary_cross)
fwrite(summary_area_qa, out_summary_area_qa)
fwrite(summary_topk_qa, out_summary_topk_qa)

# ============================================================
# 10) Imprimir
# ============================================================

cat("\n================ QA METACIENCIA / BIBLIOMETRÍA v5.2 SIN SOBRESCRIBIR ================\n")

cat("\nResumen QA:\n")
print(summary_qa)

cat("\nResumen decisión ORIGINAL:\n")
print(summary_original_decision)

cat("\nResumen decisión QA:\n")
print(summary_qa_decision)

cat("\nCruce original vs QA:\n")
print(summary_cross)

cat("\nÁreas top1 con términos de metaciencia pero no top1 bibliometría:\n")
print(head(summary_area_qa, 40))

cat("\nPresencia de bibliometría en top-k:\n")
print(summary_topk_qa)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")