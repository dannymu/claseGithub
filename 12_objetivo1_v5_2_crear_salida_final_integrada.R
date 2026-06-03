# ============================================================
# 12_objetivo1_v5_2_crear_salida_final_integrada.R
# Objetivo 1 v5.2
# Crear salida final integrada:
# - aceptados automáticamente
# - candidatos top-k para revisión asistida
# - conflictos críticos
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

zero_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_zero_shot"
)

decision_dir <- file.path(
  zero_dir,
  "decision_layer_v5_2"
)

pred_file <- file.path(
  zero_dir,
  "zero_shot_v5_2_multiprototype_full.csv"
)

decision_file <- file.path(
  decision_dir,
  "zero_shot_v5_2_decision_layer.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_integrated"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_final <- file.path(out_dir, "classification_final_integrated_v5_2.csv")
out_auto <- file.path(out_dir, "classification_auto_accepted_v5_2.csv")
out_review <- file.path(out_dir, "classification_review_required_v5_2.csv")
out_conflicts <- file.path(out_dir, "classification_critical_conflicts_v5_2.csv")

out_summary_final <- file.path(out_dir, "summary_final_decision_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_final_area_v5_2.csv")
out_summary_review <- file.path(out_dir, "summary_review_type_v5_2.csv")
out_summary_conflicts <- file.path(out_dir, "summary_critical_conflicts_v5_2.csv")

if (!file.exists(pred_file)) {
  stop("No existe zero_shot_v5_2_multiprototype_full.csv")
}

if (!file.exists(decision_file)) {
  stop("No existe zero_shot_v5_2_decision_layer.csv")
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

make_candidate <- function(area, subarea, score) {
  area <- clean_text(area)
  subarea <- clean_text(subarea)
  score <- safe_numeric(score)
  
  out <- ifelse(
    area == "" | is.na(area),
    "",
    paste0(area, " / ", subarea, " (", round(score, 4), ")")
  )
  
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
# 3) Leer datos
# ============================================================

pred <- fread(pred_file)
decision <- fread(decision_file)

setDT(pred)
setDT(decision)

cat("Predicciones:", nrow(pred), "\n")
cat("Decision layer:", nrow(decision), "\n")

# ============================================================
# 4) Preparar columnas
# ============================================================

needed_pred <- c(
  "doc_id",
  "doc_id_v5",
  "work_id",
  "title",
  "text_for_embedding",
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
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "top1_prototype_language",
  "top1_prototype_id",
  "top1_prototype_type"
)

needed_decision <- c(
  "doc_id",
  "doc_id_v5",
  "decision_layer",
  "decision_strength",
  "review_reason",
  "accepted_for_training_strict",
  "accepted_for_training_expanded",
  "needs_manual_validation",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "low_margin_cross_area",
  "same_area_low_margin",
  "has_virus_context",
  "has_animal_context",
  "has_metascience_method"
)

for (col in needed_pred) {
  if (!col %in% names(pred)) pred[, (col) := NA]
}

for (col in needed_decision) {
  if (!col %in% names(decision)) decision[, (col) := NA]
}

# Evitar duplicados de columnas al unir
decision_small <- decision[, ..needed_decision]

dt <- merge(
  pred,
  decision_small,
  by = c("doc_id", "doc_id_v5"),
  all.x = TRUE
)

# Normalización
text_cols <- c(
  "work_id", "title", "text_for_embedding",
  "top1_area_name", "top1_subarea_name",
  "top2_area_name", "top2_subarea_name",
  "top3_area_name", "top3_subarea_name",
  "top4_area_name", "top4_subarea_name",
  "top5_area_name", "top5_subarea_name",
  "zero_shot_status", "confidence_level",
  "top1_prototype_language", "top1_prototype_id", "top1_prototype_type",
  "decision_layer", "decision_strength", "review_reason"
)

for (col in text_cols) {
  if (col %in% names(dt)) {
    dt[, (col) := clean_text(get(col))]
  }
}

score_cols <- c(
  "top1_score", "top2_score", "top3_score",
  "top4_score", "top5_score", "top1_top2_margin"
)

for (col in score_cols) {
  dt[, (col) := safe_numeric(get(col))]
}

# ============================================================
# 5) Crear candidatos top-k
# ============================================================

dt[, candidate_top1 := make_candidate(top1_area_name, top1_subarea_name, top1_score)]
dt[, candidate_top2 := make_candidate(top2_area_name, top2_subarea_name, top2_score)]
dt[, candidate_top3 := make_candidate(top3_area_name, top3_subarea_name, top3_score)]
dt[, candidate_top4 := make_candidate(top4_area_name, top4_subarea_name, top4_score)]
dt[, candidate_top5 := make_candidate(top5_area_name, top5_subarea_name, top5_score)]

# ============================================================
# 6) Reglas finales de decisión
# ============================================================

dt[, auto_accept := decision_layer %in% c(
  "seed_strict_high_precision",
  "seed_expanded_with_caution"
)]

# Nunca aceptar automáticamente conflictos críticos
dt[
  possible_biomed_veterinary_conflict == TRUE |
    possible_method_domain_conflict == TRUE |
    zero_shot_status == "low_similarity_review",
  auto_accept := FALSE
]

dt[, requires_review := auto_accept == FALSE]

dt[, critical_conflict := FALSE]

dt[
  possible_biomed_veterinary_conflict == TRUE |
    possible_method_domain_conflict == TRUE |
    zero_shot_status == "low_similarity_review",
  critical_conflict := TRUE
]

dt[, review_type := ""]

dt[, review_type := append_reason(
  review_type,
  possible_biomed_veterinary_conflict == TRUE,
  "biomed_veterinary_conflict"
)]

dt[, review_type := append_reason(
  review_type,
  possible_method_domain_conflict == TRUE,
  "method_domain_conflict"
)]

dt[, review_type := append_reason(
  review_type,
  low_margin_cross_area == TRUE,
  "low_margin_cross_area"
)]

dt[, review_type := append_reason(
  review_type,
  same_area_low_margin == TRUE,
  "same_area_low_margin_subarea"
)]

dt[, review_type := append_reason(
  review_type,
  zero_shot_status == "ambiguous_review",
  "ambiguous_top1_top2"
)]

dt[, review_type := append_reason(
  review_type,
  zero_shot_status == "low_similarity_review",
  "low_similarity"
)]

dt[, review_type := append_reason(
  review_type,
  decision_layer == "candidate_review_light",
  "candidate_review_light"
)]

dt[, review_type := append_reason(
  review_type,
  decision_layer == "manual_or_assisted_review",
  "manual_or_assisted_review"
)]

dt[review_type == "" | is.na(review_type), review_type := "no_review_required"]

# ============================================================
# 7) Etiqueta final reportable
# ============================================================

dt[, final_area_label := fifelse(
  auto_accept == TRUE,
  top1_area_name,
  NA_character_
)]

dt[, final_subarea_label := fifelse(
  auto_accept == TRUE,
  top1_subarea_name,
  NA_character_
)]

dt[, final_decision := fcase(
  
  auto_accept == TRUE &
    decision_layer == "seed_strict_high_precision",
  "accepted_automatic_high_precision",
  
  auto_accept == TRUE &
    decision_layer == "seed_expanded_with_caution",
  "accepted_automatic_with_caution",
  
  critical_conflict == TRUE,
  "requires_assisted_review_critical_conflict",
  
  requires_review == TRUE,
  "requires_assisted_review_top5_candidates",
  
  default = "requires_assisted_review_top5_candidates"
)]

dt[, final_confidence_tier := fcase(
  
  final_decision == "accepted_automatic_high_precision",
  "high_precision",
  
  final_decision == "accepted_automatic_with_caution",
  "medium_high_precision",
  
  final_decision == "requires_assisted_review_critical_conflict",
  "not_accepted_critical_review",
  
  final_decision == "requires_assisted_review_top5_candidates",
  "not_accepted_review_top5",
  
  default = "not_accepted_review_top5"
)]

dt[, reportable_label := fifelse(
  auto_accept == TRUE,
  paste0(final_area_label, " / ", final_subarea_label),
  "No aceptado automáticamente; revisar candidatos top-k"
)]

# ============================================================
# 8) Tabla final
# ============================================================

final_cols <- c(
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

for (col in final_cols) {
  if (!col %in% names(dt)) dt[, (col) := NA]
}

final_dt <- dt[, ..final_cols]

auto_dt <- final_dt[auto_accept == TRUE]
review_dt <- final_dt[requires_review == TRUE]
conflict_dt <- final_dt[critical_conflict == TRUE]

# ============================================================
# 9) Resúmenes
# ============================================================

summary_final <- final_dt[
  ,
  .N,
  by = .(
    final_decision,
    final_confidence_tier,
    auto_accept,
    requires_review
  )
][order(-N)]

summary_final[, pct := round(N / sum(N) * 100, 2)]

summary_area <- final_dt[
  auto_accept == TRUE,
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

summary_area[
  ,
  pct := round(publications / sum(publications) * 100, 2)
]

summary_review <- final_dt[
  requires_review == TRUE,
  .N,
  by = review_type
][order(-N)]

summary_review[, pct := round(N / sum(N) * 100, 2)]

summary_conflicts <- final_dt[
  critical_conflict == TRUE,
  .N,
  by = .(
    possible_biomed_veterinary_conflict,
    possible_method_domain_conflict,
    zero_shot_status
  )
][order(-N)]

summary_conflicts[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 10) Exportar
# ============================================================

fwrite(final_dt, out_final)
fwrite(auto_dt, out_auto)
fwrite(review_dt, out_review)
fwrite(conflict_dt, out_conflicts)

fwrite(summary_final, out_summary_final)
fwrite(summary_area, out_summary_area)
fwrite(summary_review, out_summary_review)
fwrite(summary_conflicts, out_summary_conflicts)

# ============================================================
# 11) Imprimir
# ============================================================

cat("\n================ SALIDA FINAL INTEGRADA v5.2 ================\n")
cat("Total documentos:", nrow(final_dt), "\n")
cat("Aceptados automáticamente:", nrow(auto_dt), "\n")
cat("Requieren revisión:", nrow(review_dt), "\n")
cat("Conflictos críticos:", nrow(conflict_dt), "\n")

cat("\nResumen final:\n")
print(summary_final)

cat("\nResumen áreas aceptadas automáticamente:\n")
print(head(summary_area, 40))

cat("\nResumen tipos de revisión:\n")
print(head(summary_review, 30))

cat("\nResumen conflictos críticos:\n")
print(summary_conflicts)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")