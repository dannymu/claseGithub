# ============================================================
# 13_objetivo1_v5_2_cerrar_universo_final.R
# Objetivo 1 v5.2
# Cerrar universo oficial completo:
# - documentos elegibles clasificados / con candidatos top-k
# - documentos no elegibles por insuficiencia de metadatos
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

v5_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

integrated_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_integrated"
)

universe_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_2_all_no_concepts.csv"
)

classified_file <- file.path(
  integrated_dir,
  "classification_final_integrated_v5_2.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_final_universe <- file.path(
  out_dir,
  "classification_final_universe_v5_2.csv"
)

out_auto_universe <- file.path(
  out_dir,
  "classification_final_universe_auto_accepted_v5_2.csv"
)

out_review_universe <- file.path(
  out_dir,
  "classification_final_universe_review_required_v5_2.csv"
)

out_not_classified <- file.path(
  out_dir,
  "classification_final_universe_not_classified_v5_2.csv"
)

out_summary_universe <- file.path(
  out_dir,
  "summary_final_universe_v5_2.csv"
)

out_summary_decision <- file.path(
  out_dir,
  "summary_final_universe_decision_v5_2.csv"
)

out_summary_coverage <- file.path(
  out_dir,
  "summary_final_universe_coverage_v5_2.csv"
)

out_summary_area_auto <- file.path(
  out_dir,
  "summary_final_universe_auto_area_v5_2.csv"
)

if (!file.exists(universe_file)) {
  stop("No existe dataset_objetivo1_v5_2_all_no_concepts.csv")
}

if (!file.exists(classified_file)) {
  stop("No existe classification_final_integrated_v5_2.csv")
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

# ============================================================
# 3) Leer datos
# ============================================================

universe <- fread(universe_file)
classified <- fread(classified_file)

setDT(universe)
setDT(classified)

cat("Universo oficial:", nrow(universe), "\n")
cat("Clasificados/integrados:", nrow(classified), "\n")

# ============================================================
# 4) Preparar universo
# ============================================================

needed_universe_cols <- c(
  "doc_id_v5",
  "doc_id_interest",
  "work_id",
  "doi",
  "title",
  "abstract",
  "keyword",
  "keyword_clean",
  "conceptos",
  "conceptos_clean",
  "topics",
  "topics_clean",
  "source",
  "source_clean",
  "type",
  "authors",
  "year",
  "citations",
  "language",
  "text_quality_v5",
  "text_quality_v5_clean",
  "text_quality_v5_2",
  "embedding_eligible_v5",
  "embedding_eligible_v5_clean",
  "embedding_eligible_v5_2",
  "contextual_metadata_count",
  "contextual_metadata_count_clean",
  "contextual_metadata_count_v5_2",
  "text_for_embedding",
  "n_words_embedding_v5_2",
  "metadata_score"
)

for (col in needed_universe_cols) {
  if (!col %in% names(universe)) {
    universe[, (col) := NA]
  }
}

universe[, doc_id_v5 := as.integer(doc_id_v5)]
universe[, title := clean_text(title)]
universe[, work_id := clean_text(work_id)]
universe[, doi := clean_text(doi)]
universe[, source := clean_text(source)]
universe[, source_clean := clean_text(source_clean)]
universe[, type := clean_text(type)]
universe[, language := clean_text(language)]
universe[, text_quality_v5_2 := clean_text(text_quality_v5_2)]
universe[, embedding_eligible_v5_2 := as.logical(embedding_eligible_v5_2)]
universe[, n_words_embedding_v5_2 := safe_numeric(n_words_embedding_v5_2)]

# ============================================================
# 5) Preparar clasificados
# ============================================================

needed_classified_cols <- c(
  "doc_id",
  "doc_id_v5",
  "work_id",
  "title",
  "final_area_label",
  "final_subarea_label",
  "reportable_label",
  "final_decision",
  "final_confidence_tier",
  "auto_accept",
  "requires_review",
  "critical_conflict",
  "review_type",
  "review_reason",
  "decision_layer",
  "decision_strength",
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
  "top1_prototype_language",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "low_margin_cross_area",
  "same_area_low_margin",
  "has_virus_context",
  "has_animal_context",
  "has_metascience_method",
  "text_for_embedding"
)

for (col in needed_classified_cols) {
  if (!col %in% names(classified)) {
    classified[, (col) := NA]
  }
}

classified[, doc_id_v5 := as.integer(doc_id_v5)]

# Remover text_for_embedding duplicado desde classified para evitar conflicto
classified_small <- classified[
  ,
  .(
    doc_id,
    doc_id_v5,
    final_area_label,
    final_subarea_label,
    reportable_label,
    final_decision,
    final_confidence_tier,
    auto_accept,
    requires_review,
    critical_conflict,
    review_type,
    review_reason,
    decision_layer,
    decision_strength,
    top1_area_name,
    top1_subarea_name,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top3_area_name,
    top3_subarea_name,
    top3_score,
    top4_area_name,
    top4_subarea_name,
    top4_score,
    top5_area_name,
    top5_subarea_name,
    top5_score,
    candidate_top1,
    candidate_top2,
    candidate_top3,
    candidate_top4,
    candidate_top5,
    top1_top2_margin,
    zero_shot_status,
    confidence_level,
    top1_prototype_language,
    possible_biomed_veterinary_conflict,
    possible_method_domain_conflict,
    low_margin_cross_area,
    same_area_low_margin,
    has_virus_context,
    has_animal_context,
    has_metascience_method
  )
]

# ============================================================
# 6) Unir universo con salida clasificada
# ============================================================

final_universe <- merge(
  universe,
  classified_small,
  by = "doc_id_v5",
  all.x = TRUE
)

# ============================================================
# 7) Asignar decisión a documentos no clasificados
# ============================================================

final_universe[, classified_or_candidate := !is.na(final_decision) & final_decision != ""]

final_universe[
  classified_or_candidate == FALSE,
  final_decision := "not_classified_insufficient_metadata"
]

final_universe[
  classified_or_candidate == FALSE,
  final_confidence_tier := "not_classified"
]

final_universe[
  classified_or_candidate == FALSE,
  auto_accept := FALSE
]

final_universe[
  classified_or_candidate == FALSE,
  requires_review := TRUE
]

final_universe[
  classified_or_candidate == FALSE,
  critical_conflict := FALSE
]

final_universe[
  classified_or_candidate == FALSE,
  review_type := "insufficient_metadata_or_not_embedding_eligible"
]

final_universe[
  classified_or_candidate == FALSE,
  review_reason := "not_embedding_eligible_v5_2"
]

final_universe[
  classified_or_candidate == FALSE,
  reportable_label := "No clasificado: metadatos insuficientes o no elegible para embeddings"
]

# Asegurar etiquetas vacías en no clasificados
final_universe[
  classified_or_candidate == FALSE,
  final_area_label := NA_character_
]

final_universe[
  classified_or_candidate == FALSE,
  final_subarea_label := NA_character_
]

# ============================================================
# 8) Ordenar columnas finales
# ============================================================

final_cols <- c(
  "doc_id_v5",
  "doc_id_interest",
  "doc_id",
  "work_id",
  "doi",
  "title",
  "year",
  "citations",
  "source",
  "source_clean",
  "type",
  "language",
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
  "decision_layer",
  "decision_strength",
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
  "top1_prototype_language",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "low_margin_cross_area",
  "same_area_low_margin",
  "has_virus_context",
  "has_animal_context",
  "has_metascience_method",
  "text_quality_v5_2",
  "embedding_eligible_v5_2",
  "contextual_metadata_count_v5_2",
  "n_words_embedding_v5_2",
  "keyword_clean",
  "topics_clean",
  "conceptos_clean",
  "text_for_embedding"
)

for (col in final_cols) {
  if (!col %in% names(final_universe)) {
    final_universe[, (col) := NA]
  }
}

final_universe <- final_universe[, ..final_cols]

# ============================================================
# 9) Subconjuntos
# ============================================================

auto_universe <- final_universe[
  auto_accept == TRUE
]

review_universe <- final_universe[
  requires_review == TRUE
]

not_classified <- final_universe[
  final_decision == "not_classified_insufficient_metadata"
]

# ============================================================
# 10) Resúmenes
# ============================================================

summary_universe <- data.table(
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
    nrow(final_universe),
    final_universe[classified_or_candidate == TRUE, .N],
    nrow(auto_universe),
    nrow(review_universe),
    final_universe[critical_conflict == TRUE, .N],
    nrow(not_classified),
    round(nrow(auto_universe) / nrow(final_universe) * 100, 2),
    round(nrow(review_universe) / nrow(final_universe) * 100, 2),
    round(nrow(not_classified) / nrow(final_universe) * 100, 4)
  )
)

summary_decision <- final_universe[
  ,
  .N,
  by = .(
    final_decision,
    final_confidence_tier,
    auto_accept,
    requires_review
  )
][order(-N)]

summary_decision[, pct := round(N / sum(N) * 100, 2)]

summary_coverage <- final_universe[
  ,
  .N,
  by = .(
    text_quality_v5_2,
    embedding_eligible_v5_2,
    final_decision
  )
][order(text_quality_v5_2, -N)]

summary_coverage[
  ,
  pct_within_quality := round(N / sum(N) * 100, 2),
  by = text_quality_v5_2
]

summary_area_auto <- auto_universe[
  ,
  .(
    publications = .N,
    mean_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4)
  ),
  by = .(
    final_area_label,
    final_subarea_label
  )
][order(-publications)]

summary_area_auto[
  ,
  pct := round(publications / sum(publications) * 100, 2)
]

# ============================================================
# 11) Exportar
# ============================================================

fwrite(final_universe, out_final_universe)
fwrite(auto_universe, out_auto_universe)
fwrite(review_universe, out_review_universe)
fwrite(not_classified, out_not_classified)

fwrite(summary_universe, out_summary_universe)
fwrite(summary_decision, out_summary_decision)
fwrite(summary_coverage, out_summary_coverage)
fwrite(summary_area_auto, out_summary_area_auto)

# ============================================================
# 12) Imprimir
# ============================================================

cat("\n================ CIERRE UNIVERSO FINAL v5.2 ================\n")

cat("\nResumen universo:\n")
print(summary_universe)

cat("\nResumen decisión:\n")
print(summary_decision)

cat("\nResumen cobertura:\n")
print(summary_coverage)

cat("\nÁreas/subáreas aceptadas automáticamente:\n")
print(head(summary_area_auto, 40))

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")