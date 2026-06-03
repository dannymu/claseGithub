# ============================================================
# 20_objetivo1_v5_2_agregar_flag_metaciencia_bibliometria.R
# Objetivo 1 v5.2
# Agregar bandera de control de calidad para casos de
# bibliometría/metaciencia no capturados en top-k.
#
# No modifica la clasificación automáticamente.
# Solo marca revisión prioritaria.
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
  "classification_final_universe_v5_2_corrected_984318_with_metascience_qa.csv"
)

out_metascience_cases <- file.path(
  out_dir,
  "metascience_bibliometrics_flagged_cases_v5_2.csv"
)

out_priority_review <- file.path(
  out_dir,
  "metascience_bibliometrics_priority_review_v5_2.csv"
)

out_summary_qa <- file.path(
  out_dir,
  "summary_metascience_bibliometrics_qa_v5_2.csv"
)

out_summary_decision_qa <- file.path(
  out_dir,
  "summary_final_decision_with_metascience_qa_v5_2.csv"
)

out_summary_area_qa <- file.path(
  out_dir,
  "summary_metascience_flag_by_predicted_area_v5_2.csv"
)

out_summary_topk_qa <- file.path(
  out_dir,
  "summary_metascience_topk_presence_v5_2.csv"
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
  
  # También revisar columnas topk separadas
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

append_reason <- function(current, condition, reason) {
  out <- current
  
  idx_empty <- condition & (is.na(out) | out == "")
  out[idx_empty] <- reason
  
  idx_existing <- condition & !is.na(out) & out != "" & !grepl(reason, out, fixed = TRUE)
  out[idx_existing] <- paste(out[idx_existing], reason, sep = "; ")
  
  out
}

# ============================================================
# 3) Leer archivo final
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
# 4) Patrones de metaciencia/bibliometría
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

# ============================================================
# 5) Bandera QA
# ============================================================

dt[, possible_bibliometrics_metascience_conflict := FALSE]

dt[
  has_metascience_bibliometrics_terms == TRUE &
    top1_is_bibliometrics == FALSE,
  possible_bibliometrics_metascience_conflict := TRUE
]

dt[, possible_bibliometrics_metascience_top5_failure := FALSE]

dt[
  has_metascience_bibliometrics_terms == TRUE &
    top1_is_bibliometrics == FALSE &
    top5_contains_bibliometrics == FALSE,
  possible_bibliometrics_metascience_top5_failure := TRUE
]

dt[, possible_bibliometrics_metascience_rerank_case := FALSE]

dt[
  has_metascience_bibliometrics_terms == TRUE &
    top1_is_bibliometrics == FALSE &
    top5_contains_bibliometrics == TRUE,
  possible_bibliometrics_metascience_rerank_case := TRUE
]

# Revisión prioritaria:
# 1) términos de metaciencia y bibliometría no están en top5
# 2) o estaban aceptados automáticamente fuera de bibliometría
dt[, metascience_qa_review_priority := FALSE]

dt[
  possible_bibliometrics_metascience_top5_failure == TRUE |
    (
      possible_bibliometrics_metascience_conflict == TRUE &
        auto_accept == TRUE
    ),
  metascience_qa_review_priority := TRUE
]

# ============================================================
# 6) Actualizar campos de revisión sin cambiar etiqueta final
# ============================================================

dt[, final_decision_original := final_decision]
dt[, final_confidence_tier_original := final_confidence_tier]
dt[, auto_accept_original := auto_accept]
dt[, requires_review_original := requires_review]
dt[, review_type_original := review_type]
dt[, review_reason_original := review_reason]

# No reclasificar automáticamente.
# Solo bajar a revisión prioritaria los auto-aceptados con conflicto metaciencia.
dt[
  metascience_qa_review_priority == TRUE & auto_accept == TRUE,
  auto_accept := FALSE
]

dt[
  metascience_qa_review_priority == TRUE,
  requires_review := TRUE
]

dt[
  metascience_qa_review_priority == TRUE,
  final_decision := "requires_assisted_review_metascience_bibliometrics_conflict"
]

dt[
  metascience_qa_review_priority == TRUE,
  final_confidence_tier := "not_accepted_metascience_review"
]

dt[
  metascience_qa_review_priority == TRUE,
  reportable_label := "No aceptado automáticamente; posible conflicto bibliometría/metaciencia"
]

dt[
  metascience_qa_review_priority == TRUE,
  final_area_label := NA_character_
]

dt[
  metascience_qa_review_priority == TRUE,
  final_subarea_label := NA_character_
]

dt[, review_type := append_reason(
  review_type,
  metascience_qa_review_priority == TRUE,
  "metascience_bibliometrics_priority_review"
)]

dt[, review_reason := append_reason(
  review_reason,
  metascience_qa_review_priority == TRUE,
  "possible_bibliometrics_metascience_conflict_detected_by_qa"
)]

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
  metascience_qa_review_priority == TRUE
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
    "metascience_qa_review_priority",
    "auto_accept_original",
    "auto_accept_after_qa",
    "auto_accept_removed_by_metascience_qa",
    "review_required_after_qa"
  ),
  value = c(
    nrow(dt),
    dt[has_metascience_bibliometrics_terms == TRUE, .N],
    dt[top1_is_bibliometrics == TRUE, .N],
    dt[top5_contains_bibliometrics == TRUE, .N],
    dt[possible_bibliometrics_metascience_conflict == TRUE, .N],
    dt[possible_bibliometrics_metascience_top5_failure == TRUE, .N],
    dt[possible_bibliometrics_metascience_rerank_case == TRUE, .N],
    dt[metascience_qa_review_priority == TRUE, .N],
    dt[auto_accept_original == TRUE, .N],
    dt[auto_accept == TRUE, .N],
    dt[auto_accept_original == TRUE & auto_accept == FALSE, .N],
    dt[requires_review == TRUE, .N]
  )
)

summary_qa[, pct_universe := round(value / nrow(dt) * 100, 4)]

summary_decision_qa <- dt[
  ,
  .N,
  by = .(
    final_decision,
    final_confidence_tier,
    auto_accept,
    requires_review
  )
][order(-N)]

summary_decision_qa[, pct := round(N / sum(N) * 100, 2)]

summary_area_qa <- dt[
  has_metascience_bibliometrics_terms == TRUE &
    top1_is_bibliometrics == FALSE,
  .N,
  by = .(
    top1_area_name,
    top1_subarea_name,
    top5_contains_bibliometrics,
    metascience_qa_review_priority
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
fwrite(metascience_cases, out_metascience_cases)
fwrite(priority_review, out_priority_review)
fwrite(summary_qa, out_summary_qa)
fwrite(summary_decision_qa, out_summary_decision_qa)
fwrite(summary_area_qa, out_summary_area_qa)
fwrite(summary_topk_qa, out_summary_topk_qa)

# ============================================================
# 10) Imprimir
# ============================================================

cat("\n================ QA METACIENCIA / BIBLIOMETRÍA v5.2 ================\n")

cat("\nResumen QA:\n")
print(summary_qa)

cat("\nResumen decisión después QA:\n")
print(summary_decision_qa)

cat("\nÁreas top1 con términos de metaciencia pero no top1 bibliometría:\n")
print(head(summary_area_qa, 40))

cat("\nPresencia de bibliometría en top-k:\n")
print(summary_topk_qa)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")