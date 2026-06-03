# ============================================================
# 11_objetivo1_v5_2_comparar_zero_shot_supervisado.R
# Objetivo 1 v5.2
# Comparar zero-shot top1 vs modelos supervisados sobre test manual
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

decision_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_zero_shot",
  "decision_layer_v5_2"
)

validation_dir <- file.path(
  decision_dir,
  "validation_results_v5_2"
)

model_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_supervised_models"
)

manual_file <- file.path(
  validation_dir,
  "manual_validation_clean_v5_2.csv"
)

pred_supervised_file <- file.path(
  model_dir,
  "manual_test_predictions_supervised_v5_2.csv"
)

out_dir <- file.path(
  model_dir,
  "comparison_zero_shot_supervised_v5_2"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_comparison <- file.path(out_dir, "manual_test_comparison_zero_shot_supervised_v5_2.csv")
out_summary_overall <- file.path(out_dir, "summary_comparison_overall_v5_2.csv")
out_summary_group <- file.path(out_dir, "summary_comparison_by_validation_group_v5_2.csv")
out_summary_decision <- file.path(out_dir, "summary_comparison_by_decision_layer_v5_2.csv")
out_summary_conflict <- file.path(out_dir, "summary_comparison_by_conflict_flags_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_comparison_by_manual_area_v5_2.csv")
out_improvement_area <- file.path(out_dir, "summary_area_improvement_patterns_v5_2.csv")
out_error_cases <- file.path(out_dir, "error_cases_supervised_v5_2.csv")
out_zero_correct_supervised_wrong <- file.path(out_dir, "zero_correct_supervised_wrong_v5_2.csv")
out_zero_wrong_supervised_correct <- file.path(out_dir, "zero_wrong_supervised_correct_v5_2.csv")
out_biomed_errors <- file.path(out_dir, "biomed_veterinary_errors_supervised_v5_2.csv")

if (!file.exists(manual_file)) {
  stop("No existe manual_validation_clean_v5_2.csv. Ejecuta script 08.")
}

if (!file.exists(pred_supervised_file)) {
  stop("No existe manual_test_predictions_supervised_v5_2.csv. Ejecuta script 10.")
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

mean_bool <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  round(mean(x, na.rm = TRUE), 4)
}

parse_suggestion_area <- function(x) {
  x <- clean_text(x)
  out <- sub(" / .*", "", x)
  out[out == x & !grepl(" / ", x, fixed = TRUE)] <- ""
  out
}

parse_suggestion_subarea <- function(x) {
  x <- clean_text(x)
  out <- sub(".* / ", "", x)
  out[out == x & !grepl(" / ", x, fixed = TRUE)] <- ""
  out
}

# ============================================================
# 3) Leer datos
# ============================================================

manual <- fread(manual_file)
sup <- fread(pred_supervised_file)

setDT(manual)
setDT(sup)

cat("Manual:", nrow(manual), "\n")
cat("Supervisado:", nrow(sup), "\n")

# ============================================================
# 4) Preparar columnas
# ============================================================

needed_manual <- c(
  "doc_id",
  "doc_id_v5",
  "validation_group",
  "title",
  "top1_area_name",
  "top1_subarea_name",
  "top2_area_name",
  "top2_subarea_name",
  "top1_score",
  "top2_score",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "decision_layer",
  "review_reason",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "has_virus_context",
  "has_animal_context",
  "has_metascience_method",
  "corrected_area_label",
  "corrected_subarea_label",
  "manual_decision_clean",
  "manual_comment"
)

for (col in needed_manual) {
  if (!col %in% names(manual)) {
    manual[, (col) := ""]
  }
}

needed_sup <- c(
  "doc_id",
  "doc_id_v5",
  "label_area",
  "label_subarea",
  "pred_area_model",
  "pred_pair_area",
  "pred_pair_subarea",
  "pred_pair_model",
  "area_seen_in_train",
  "pair_seen_in_train"
)

for (col in needed_sup) {
  if (!col %in% names(sup)) {
    sup[, (col) := ""]
  }
}

# Normalizar
for (col in names(manual)) {
  if (is.character(manual[[col]])) manual[, (col) := clean_text(get(col))]
}

for (col in names(sup)) {
  if (is.character(sup[[col]])) sup[, (col) := clean_text(get(col))]
}

manual[, top1_score := safe_numeric(top1_score)]
manual[, top2_score := safe_numeric(top2_score)]
manual[, top1_top2_margin := safe_numeric(top1_top2_margin)]

# ============================================================
# 5) Unir
# ============================================================

sup_small <- sup[
  ,
  .(
    doc_id,
    doc_id_v5,
    label_area,
    label_subarea,
    pred_area_model,
    pred_pair_area,
    pred_pair_subarea,
    pred_pair_model,
    area_seen_in_train,
    pair_seen_in_train
  )
]

dt <- merge(
  manual,
  sup_small,
  by = c("doc_id", "doc_id_v5"),
  all.x = TRUE
)

# Etiquetas manuales de referencia
dt[, manual_area := fifelse(
  corrected_area_label != "",
  corrected_area_label,
  label_area
)]

dt[, manual_subarea := fifelse(
  corrected_subarea_label != "",
  corrected_subarea_label,
  label_subarea
)]

dt <- dt[
  manual_decision_clean != "insufficient_information" &
    manual_area != "" &
    manual_subarea != ""
]

# ============================================================
# 6) Top-k zero-shot si existen suggested_top3-top5
# ============================================================

suggested_cols <- c("suggested_top1", "suggested_top2", "suggested_top3", "suggested_top4", "suggested_top5")

for (col in suggested_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := ""]
  }
}

dt[, top1_area_from_suggestion := parse_suggestion_area(suggested_top1)]
dt[, top2_area_from_suggestion := parse_suggestion_area(suggested_top2)]
dt[, top3_area_from_suggestion := parse_suggestion_area(suggested_top3)]
dt[, top4_area_from_suggestion := parse_suggestion_area(suggested_top4)]
dt[, top5_area_from_suggestion := parse_suggestion_area(suggested_top5)]

dt[, top1_subarea_from_suggestion := parse_suggestion_subarea(suggested_top1)]
dt[, top2_subarea_from_suggestion := parse_suggestion_subarea(suggested_top2)]
dt[, top3_subarea_from_suggestion := parse_suggestion_subarea(suggested_top3)]
dt[, top4_subarea_from_suggestion := parse_suggestion_subarea(suggested_top4)]
dt[, top5_subarea_from_suggestion := parse_suggestion_subarea(suggested_top5)]

# Si suggested_top1 está vacío, usar top1_area_name/top1_subarea_name
dt[top1_area_from_suggestion == "", top1_area_from_suggestion := top1_area_name]
dt[top1_subarea_from_suggestion == "", top1_subarea_from_suggestion := top1_subarea_name]
dt[top2_area_from_suggestion == "", top2_area_from_suggestion := top2_area_name]
dt[top2_subarea_from_suggestion == "", top2_subarea_from_suggestion := top2_subarea_name]

dt[, manual_area_in_top2 := manual_area %in% c(
  top1_area_from_suggestion,
  top2_area_from_suggestion
), by = .I]

dt[, manual_area_in_top5 := manual_area %in% c(
  top1_area_from_suggestion,
  top2_area_from_suggestion,
  top3_area_from_suggestion,
  top4_area_from_suggestion,
  top5_area_from_suggestion
), by = .I]

dt[, manual_pair_in_top5 := paste(manual_area, manual_subarea, sep = " / ") %in% c(
  suggested_top1,
  suggested_top2,
  suggested_top3,
  suggested_top4,
  suggested_top5
), by = .I]

# ============================================================
# 7) Métricas de corrección
# ============================================================

dt[, zero_area_correct := top1_area_name == manual_area]
dt[, zero_subarea_correct := top1_subarea_name == manual_subarea]
dt[, zero_pair_correct := zero_area_correct & zero_subarea_correct]

dt[, supervised_area_correct := pred_area_model == manual_area]

dt[, pair_model_area_correct := pred_pair_area == manual_area]
dt[, pair_model_subarea_correct := pred_pair_subarea == manual_subarea]
dt[, pair_model_exact_correct := pair_model_area_correct & pair_model_subarea_correct]

dt[, supervised_vs_zero_area := fcase(
  zero_area_correct == TRUE & supervised_area_correct == TRUE,
  "both_correct_area",
  
  zero_area_correct == TRUE & supervised_area_correct == FALSE,
  "supervised_worsens_area",
  
  zero_area_correct == FALSE & supervised_area_correct == TRUE,
  "supervised_improves_area",
  
  zero_area_correct == FALSE & supervised_area_correct == FALSE,
  "both_wrong_area",
  
  default = "review"
)]

dt[, pair_vs_zero_area := fcase(
  zero_area_correct == TRUE & pair_model_area_correct == TRUE,
  "both_correct_area",
  
  zero_area_correct == TRUE & pair_model_area_correct == FALSE,
  "pair_model_worsens_area",
  
  zero_area_correct == FALSE & pair_model_area_correct == TRUE,
  "pair_model_improves_area",
  
  zero_area_correct == FALSE & pair_model_area_correct == FALSE,
  "both_wrong_area",
  
  default = "review"
)]

# ============================================================
# 8) Resúmenes
# ============================================================

summary_overall <- data.table(
  metric = c(
    "test_rows",
    "zero_shot_area_accuracy",
    "zero_shot_subarea_accuracy",
    "zero_shot_pair_accuracy",
    "supervised_area_accuracy",
    "pair_model_area_accuracy",
    "pair_model_subarea_accuracy",
    "pair_model_exact_accuracy",
    "manual_area_in_top2_rate",
    "manual_area_in_top5_rate",
    "manual_pair_in_top5_rate",
    "supervised_improves_area_cases",
    "supervised_worsens_area_cases",
    "pair_model_improves_area_cases",
    "pair_model_worsens_area_cases"
  ),
  value = c(
    nrow(dt),
    mean_bool(dt$zero_area_correct),
    mean_bool(dt$zero_subarea_correct),
    mean_bool(dt$zero_pair_correct),
    mean_bool(dt$supervised_area_correct),
    mean_bool(dt$pair_model_area_correct),
    mean_bool(dt$pair_model_subarea_correct),
    mean_bool(dt$pair_model_exact_correct),
    mean_bool(dt$manual_area_in_top2),
    mean_bool(dt$manual_area_in_top5),
    mean_bool(dt$manual_pair_in_top5),
    dt[supervised_vs_zero_area == "supervised_improves_area", .N],
    dt[supervised_vs_zero_area == "supervised_worsens_area", .N],
    dt[pair_vs_zero_area == "pair_model_improves_area", .N],
    dt[pair_vs_zero_area == "pair_model_worsens_area", .N]
  )
)

summary_group <- dt[
  ,
  .(
    N = .N,
    zero_area_accuracy = mean_bool(zero_area_correct),
    supervised_area_accuracy = mean_bool(supervised_area_correct),
    pair_model_area_accuracy = mean_bool(pair_model_area_correct),
    zero_subarea_accuracy = mean_bool(zero_subarea_correct),
    pair_model_subarea_accuracy = mean_bool(pair_model_subarea_correct),
    pair_model_exact_accuracy = mean_bool(pair_model_exact_correct),
    manual_area_in_top2_rate = mean_bool(manual_area_in_top2),
    manual_area_in_top5_rate = mean_bool(manual_area_in_top5),
    supervised_improves = sum(supervised_vs_zero_area == "supervised_improves_area"),
    supervised_worsens = sum(supervised_vs_zero_area == "supervised_worsens_area")
  ),
  by = validation_group
][order(validation_group)]

summary_decision <- dt[
  ,
  .(
    N = .N,
    zero_area_accuracy = mean_bool(zero_area_correct),
    supervised_area_accuracy = mean_bool(supervised_area_correct),
    pair_model_area_accuracy = mean_bool(pair_model_area_correct),
    zero_subarea_accuracy = mean_bool(zero_subarea_correct),
    pair_model_exact_accuracy = mean_bool(pair_model_exact_correct),
    manual_area_in_top5_rate = mean_bool(manual_area_in_top5),
    supervised_improves = sum(supervised_vs_zero_area == "supervised_improves_area"),
    supervised_worsens = sum(supervised_vs_zero_area == "supervised_worsens_area")
  ),
  by = decision_layer
][order(-N)]

summary_conflict <- dt[
  ,
  .(
    N = .N,
    zero_area_accuracy = mean_bool(zero_area_correct),
    supervised_area_accuracy = mean_bool(supervised_area_correct),
    pair_model_area_accuracy = mean_bool(pair_model_area_correct),
    zero_subarea_accuracy = mean_bool(zero_subarea_correct),
    pair_model_exact_accuracy = mean_bool(pair_model_exact_correct),
    manual_area_in_top5_rate = mean_bool(manual_area_in_top5),
    supervised_improves = sum(supervised_vs_zero_area == "supervised_improves_area"),
    supervised_worsens = sum(supervised_vs_zero_area == "supervised_worsens_area")
  ),
  by = .(
    possible_biomed_veterinary_conflict,
    possible_method_domain_conflict
  )
][order(-N)]

summary_area <- dt[
  ,
  .(
    N = .N,
    zero_area_accuracy = mean_bool(zero_area_correct),
    supervised_area_accuracy = mean_bool(supervised_area_correct),
    pair_model_area_accuracy = mean_bool(pair_model_area_correct),
    zero_subarea_accuracy = mean_bool(zero_subarea_correct),
    pair_model_exact_accuracy = mean_bool(pair_model_exact_correct),
    supervised_improves = sum(supervised_vs_zero_area == "supervised_improves_area"),
    supervised_worsens = sum(supervised_vs_zero_area == "supervised_worsens_area")
  ),
  by = manual_area
][order(-N)]

summary_improvement_area <- dt[
  ,
  .N,
  by = supervised_vs_zero_area
][order(-N)]

summary_improvement_area[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 9) Casos de error
# ============================================================

error_cases <- dt[
  supervised_area_correct == FALSE |
    pair_model_exact_correct == FALSE,
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    manual_area,
    manual_subarea,
    zero_area = top1_area_name,
    zero_subarea = top1_subarea_name,
    pred_area_model,
    pred_pair_area,
    pred_pair_subarea,
    zero_area_correct,
    supervised_area_correct,
    pair_model_exact_correct,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    confidence_level,
    zero_shot_status,
    decision_layer,
    review_reason,
    manual_comment
  )
][order(validation_group, manual_area)]

zero_correct_supervised_wrong <- dt[
  zero_area_correct == TRUE &
    supervised_area_correct == FALSE,
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    manual_area,
    manual_subarea,
    zero_area = top1_area_name,
    zero_subarea = top1_subarea_name,
    pred_area_model,
    pred_pair_area,
    pred_pair_subarea,
    top1_score,
    top1_top2_margin,
    decision_layer,
    review_reason,
    manual_comment
  )
][order(validation_group, manual_area)]

zero_wrong_supervised_correct <- dt[
  zero_area_correct == FALSE &
    supervised_area_correct == TRUE,
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    manual_area,
    manual_subarea,
    zero_area = top1_area_name,
    zero_subarea = top1_subarea_name,
    pred_area_model,
    pred_pair_area,
    pred_pair_subarea,
    top1_score,
    top1_top2_margin,
    decision_layer,
    review_reason,
    manual_comment
  )
][order(validation_group, manual_area)]

biomed_errors <- dt[
  possible_biomed_veterinary_conflict == TRUE |
    manual_area %in% c("Medicine", "Immunology and Microbiology", "Biochemistry, Genetics and Molecular Biology") |
    top1_area_name == "Veterinary" |
    pred_area_model == "Veterinary",
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    manual_area,
    manual_subarea,
    zero_area = top1_area_name,
    zero_subarea = top1_subarea_name,
    pred_area_model,
    pred_pair_area,
    pred_pair_subarea,
    zero_area_correct,
    supervised_area_correct,
    pair_model_exact_correct,
    has_virus_context,
    has_animal_context,
    possible_biomed_veterinary_conflict,
    top1_score,
    top1_top2_margin,
    decision_layer,
    review_reason,
    manual_comment
  )
][order(manual_area, zero_area_correct, supervised_area_correct)]

# ============================================================
# 10) Exportar
# ============================================================

fwrite(dt, out_comparison)
fwrite(summary_overall, out_summary_overall)
fwrite(summary_group, out_summary_group)
fwrite(summary_decision, out_summary_decision)
fwrite(summary_conflict, out_summary_conflict)
fwrite(summary_area, out_summary_area)
fwrite(summary_improvement_area, out_improvement_area)
fwrite(error_cases, out_error_cases)
fwrite(zero_correct_supervised_wrong, out_zero_correct_supervised_wrong)
fwrite(zero_wrong_supervised_correct, out_zero_wrong_supervised_correct)
fwrite(biomed_errors, out_biomed_errors)

# ============================================================
# 11) Imprimir
# ============================================================

cat("\n================ COMPARACIÓN ZERO-SHOT vs SUPERVISADO v5.2 ================\n")

cat("\nResumen general:\n")
print(summary_overall)

cat("\nPor grupo de validación:\n")
print(summary_group)

cat("\nPor decision_layer:\n")
print(summary_decision)

cat("\nPor conflictos:\n")
print(summary_conflict)

cat("\nPor área manual:\n")
print(summary_area)

cat("\nPatrones de mejora/empeora supervisado vs zero-shot:\n")
print(summary_improvement_area)

cat("\nCasos zero correcto y supervisado incorrecto:", nrow(zero_correct_supervised_wrong), "\n")
cat("Casos zero incorrecto y supervisado correcto:", nrow(zero_wrong_supervised_correct), "\n")
cat("Errores biomédicos/veterinaria revisados:", nrow(biomed_errors), "\n")

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")