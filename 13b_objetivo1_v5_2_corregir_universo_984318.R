# ============================================================
# 13b_objetivo1_v5_2_corregir_universo_984318.R
# Objetivo 1 v5.2
# Corregir cierre final para cubrir exactamente el universo
# oficial docs_interest = 984,318 documentos.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

v5_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

final_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe"
)

# Universo completo generado antes de filtrar embeddings
full_universe_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_docs_interest_enriched_all_clean_text.csv"
)

# Cierre actual
current_final_file <- file.path(
  final_dir,
  "classification_final_universe_v5_2.csv"
)

out_final_corrected <- file.path(
  final_dir,
  "classification_final_universe_v5_2_corrected_984318.csv"
)

out_missing_docs <- file.path(
  final_dir,
  "classification_final_universe_missing_docs_added_v5_2.csv"
)

out_summary_corrected <- file.path(
  final_dir,
  "summary_final_universe_v5_2_corrected_984318.csv"
)

out_summary_decision_corrected <- file.path(
  final_dir,
  "summary_final_universe_decision_v5_2_corrected_984318.csv"
)

if (!file.exists(full_universe_file)) {
  stop("No existe: ", full_universe_file)
}

if (!file.exists(current_final_file)) {
  stop("No existe: ", current_final_file)
}

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

# ============================================================
# 1) Leer datos
# ============================================================

full_universe <- fread(full_universe_file)
final_current <- fread(current_final_file)

setDT(full_universe)
setDT(final_current)

full_universe[, doc_id_v5 := as.integer(doc_id_v5)]
final_current[, doc_id_v5 := as.integer(doc_id_v5)]

cat("Universo completo:", nrow(full_universe), "\n")
cat("Cierre actual:", nrow(final_current), "\n")

# ============================================================
# 2) Detectar documentos faltantes
# ============================================================

missing_docs <- full_universe[
  !doc_id_v5 %in% final_current$doc_id_v5
]

cat("Documentos faltantes:", nrow(missing_docs), "\n")

# ============================================================
# 3) Preparar registros faltantes con la misma estructura final
# ============================================================

if (nrow(missing_docs) > 0) {
  
  # Crear una tabla vacía con la MISMA estructura y tipos de final_current
  # y luego generar tantas filas NA como documentos faltantes.
  missing_docs_final <- final_current[rep(NA_integer_, nrow(missing_docs))]
  
  # Copiar columnas comunes desde missing_docs hacia la estructura final
  common_cols <- intersect(names(missing_docs_final), names(missing_docs))
  
  for (col in common_cols) {
    missing_docs_final[, (col) := missing_docs[[col]]]
  }
  
  # Asegurar doc_id_v5
  missing_docs_final[, doc_id_v5 := as.integer(missing_docs$doc_id_v5)]
  
  # Campos básicos, si existen en missing_docs
  if ("doc_id_interest" %in% names(missing_docs) &&
      "doc_id_interest" %in% names(missing_docs_final)) {
    missing_docs_final[, doc_id_interest := clean_text(missing_docs$doc_id_interest)]
  }
  
  if ("work_id" %in% names(missing_docs) &&
      "work_id" %in% names(missing_docs_final)) {
    missing_docs_final[, work_id := clean_text(missing_docs$work_id)]
  }
  
  if ("title" %in% names(missing_docs) &&
      "title" %in% names(missing_docs_final)) {
    missing_docs_final[, title := clean_text(missing_docs$title)]
  }
  
  # Decisión final para documentos faltantes
  missing_docs_final[, final_area_label := NA_character_]
  missing_docs_final[, final_subarea_label := NA_character_]
  
  missing_docs_final[
    ,
    reportable_label := "No clasificado: metadatos insuficientes o no elegible para embeddings"
  ]
  
  missing_docs_final[, final_decision := "not_classified_insufficient_metadata"]
  missing_docs_final[, final_confidence_tier := "not_classified"]
  missing_docs_final[, auto_accept := FALSE]
  missing_docs_final[, requires_review := TRUE]
  missing_docs_final[, critical_conflict := FALSE]
  missing_docs_final[, classified_or_candidate := FALSE]
  missing_docs_final[, review_type := "insufficient_metadata_or_not_embedding_eligible"]
  missing_docs_final[, review_reason := "not_embedding_eligible_v5_2"]
  
  # Calidad textual disponible
  if ("text_quality_v5_2" %in% names(missing_docs)) {
    missing_docs_final[, text_quality_v5_2 := clean_text(missing_docs$text_quality_v5_2)]
  } else if ("text_quality_v5_clean" %in% names(missing_docs)) {
    missing_docs_final[, text_quality_v5_2 := clean_text(missing_docs$text_quality_v5_clean)]
  } else if ("text_quality_v5" %in% names(missing_docs)) {
    missing_docs_final[, text_quality_v5_2 := clean_text(missing_docs$text_quality_v5)]
  } else {
    missing_docs_final[, text_quality_v5_2 := "not_available"]
  }
  
  missing_docs_final[, embedding_eligible_v5_2 := FALSE]
  
  if ("n_words_embedding_v5_2" %in% names(missing_docs_final)) {
    missing_docs_final[, n_words_embedding_v5_2 := NA_real_]
  }
  
  # Mantener exactamente las mismas columnas que final_current
  missing_docs_final <- missing_docs_final[, names(final_current), with = FALSE]
  
} else {
  
  missing_docs_final <- final_current[0]
}
# ============================================================
# 4) Unir cierre actual + faltantes
# ============================================================

final_corrected <- rbindlist(
  list(
    final_current,
    missing_docs_final
  ),
  fill = TRUE
)

# Eliminar duplicados defensivamente
setorder(final_corrected, doc_id_v5)
final_corrected <- unique(final_corrected, by = "doc_id_v5")

# ============================================================
# 5) Recalcular subconjuntos y resúmenes
# ============================================================

auto_corrected <- final_corrected[
  auto_accept == TRUE
]

review_corrected <- final_corrected[
  requires_review == TRUE
]

not_classified_corrected <- final_corrected[
  final_decision == "not_classified_insufficient_metadata"
]

summary_corrected <- data.table(
  metric = c(
    "official_universe_documents",
    "classified_or_candidate_documents",
    "auto_accepted_documents",
    "review_required_documents",
    "critical_conflict_documents",
    "not_classified_documents",
    "auto_accepted_pct_of_universe",
    "review_required_pct_of_universe",
    "not_classified_pct_of_universe"
  ),
  value = c(
    nrow(final_corrected),
    final_corrected[classified_or_candidate == TRUE, .N],
    nrow(auto_corrected),
    nrow(review_corrected),
    final_corrected[critical_conflict == TRUE, .N],
    nrow(not_classified_corrected),
    round(nrow(auto_corrected) / nrow(final_corrected) * 100, 2),
    round(nrow(review_corrected) / nrow(final_corrected) * 100, 2),
    round(nrow(not_classified_corrected) / nrow(final_corrected) * 100, 4)
  )
)

summary_decision_corrected <- final_corrected[
  ,
  .N,
  by = .(
    final_decision,
    final_confidence_tier,
    auto_accept,
    requires_review
  )
][order(-N)]

summary_decision_corrected[
  ,
  pct := round(N / sum(N) * 100, 2)
]

# ============================================================
# 6) Exportar
# ============================================================

fwrite(final_corrected, out_final_corrected)
fwrite(missing_docs_final, out_missing_docs)
fwrite(summary_corrected, out_summary_corrected)
fwrite(summary_decision_corrected, out_summary_decision_corrected)

# ============================================================
# 7) Imprimir
# ============================================================

cat("\n================ UNIVERSO FINAL CORREGIDO v5.2 ================\n")

cat("\nResumen corregido:\n")
print(summary_corrected)

cat("\nResumen decisión corregido:\n")
print(summary_decision_corrected)

cat("\nArchivos creados:\n")
cat(out_final_corrected, "\n")
cat(out_missing_docs, "\n")
cat(out_summary_corrected, "\n")