# ============================================================
# 08_objetivo1_v5_2_evaluar_validacion_manual.R
# Objetivo 1 v5.2
# Evaluar resultados de validación manual
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

xlsx_file <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2_for_review.xlsx"
)

csv_file <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2_for_review.csv"
)

out_dir <- file.path(
  decision_dir,
  "validation_results_v5_2"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_validation_clean <- file.path(out_dir, "manual_validation_clean_v5_2.csv")
out_summary_completeness <- file.path(out_dir, "summary_validation_completeness_v5_2.csv")
out_summary_overall <- file.path(out_dir, "summary_validation_overall_v5_2.csv")
out_summary_group <- file.path(out_dir, "summary_validation_by_group_v5_2.csv")
out_summary_decision_layer <- file.path(out_dir, "summary_validation_by_decision_layer_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_validation_by_area_v5_2.csv")
out_summary_manual_decision <- file.path(out_dir, "summary_manual_decision_v5_2.csv")
out_area_confusion <- file.path(out_dir, "confusion_area_top1_vs_manual_v5_2.csv")
out_subarea_errors <- file.path(out_dir, "summary_subarea_errors_v5_2.csv")
out_area_errors <- file.path(out_dir, "summary_area_errors_v5_2.csv")
out_corrections <- file.path(out_dir, "manual_corrections_v5_2.csv")
out_recommendation <- file.path(out_dir, "summary_training_recommendation_v5_2.csv")

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

to_bool_manual <- function(x) {
  x <- clean_text(x)
  x_low <- tolower(x)
  
  out <- rep(NA, length(x_low))
  
  out[x_low %in% c("true", "t", "1", "si", "sí", "yes", "y", "correcto", "correcta")] <- TRUE
  out[x_low %in% c("false", "f", "0", "no", "n", "incorrecto", "incorrecta")] <- FALSE
  
  out
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

pct <- function(x, total) {
  round(x / total * 100, 2)
}

mean_bool <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  round(mean(x, na.rm = TRUE), 4)
}

# ============================================================
# 3) Leer archivo validado
# ============================================================

if (file.exists(xlsx_file) && requireNamespace("openxlsx", quietly = TRUE)) {
  
  val <- as.data.table(openxlsx::read.xlsx(xlsx_file, sheet = "validacion"))
  input_used <- xlsx_file
  
} else if (file.exists(csv_file)) {
  
  val <- fread(csv_file)
  input_used <- csv_file
  
} else {
  
  stop("No se encontró archivo XLSX o CSV de validación.")
}

setDT(val)

cat("Archivo usado:\n")
cat(input_used, "\n")
cat("Filas leídas:", nrow(val), "\n")

# ============================================================
# 4) Normalizar columnas manuales
# ============================================================

needed_cols <- c(
  "validation_group",
  "doc_id",
  "doc_id_v5",
  "work_id",
  "title",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "decision_layer",
  "decision_strength",
  "review_reason",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "text_for_embedding_short",
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

for (col in needed_cols) {
  if (!col %in% names(val)) {
    val[, (col) := ""]
  }
}

text_cols <- c(
  "validation_group",
  "work_id",
  "title",
  "top1_area_name",
  "top1_subarea_name",
  "top2_area_name",
  "top2_subarea_name",
  "zero_shot_status",
  "confidence_level",
  "decision_layer",
  "decision_strength",
  "review_reason",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

for (col in text_cols) {
  val[, (col) := clean_text(get(col))]
}

val[, top1_score := safe_numeric(top1_score)]
val[, top2_score := safe_numeric(top2_score)]
val[, top1_top2_margin := safe_numeric(top1_top2_margin)]

val[, manual_area_correct_bool := to_bool_manual(manual_area_correct)]
val[, manual_subarea_correct_bool := to_bool_manual(manual_subarea_correct)]

val[, manual_decision_clean := tolower(clean_text(manual_decision))]

# Normalizar variantes posibles
val[manual_decision_clean %in% c("correcto", "correcta", "ok"), manual_decision_clean := "correct"]
val[manual_decision_clean %in% c("area wrong", "área incorrecta", "area_incorrecta"), manual_decision_clean := "area_wrong"]
val[manual_decision_clean %in% c("insufficient", "sin informacion", "sin información"), manual_decision_clean := "insufficient_information"]
val[manual_decision_clean %in% c("ambiguous", "ambigua", "ambiguo"), manual_decision_clean := "ambiguous_or_both_valid"]

# ============================================================
# 5) Validar completitud
# ============================================================

val[, is_validated := manual_decision_clean != "" |
      !is.na(manual_area_correct_bool) |
      !is.na(manual_subarea_correct_bool)]

val[, is_excluded_from_accuracy := manual_decision_clean %in% c(
  "insufficient_information",
  "ambiguous_or_both_valid"
)]

val[, is_ambiguous_manual := manual_decision_clean == "ambiguous_or_both_valid"]
val[, is_insufficient_manual := manual_decision_clean == "insufficient_information"]

# Inferir correctitud si faltó TRUE/FALSE pero sí llenaron manual_decision
val[
  is.na(manual_area_correct_bool) &
    manual_decision_clean %in% c("correct", "area_correct_subarea_wrong"),
  manual_area_correct_bool := TRUE
]

val[
  is.na(manual_subarea_correct_bool) &
    manual_decision_clean == "correct",
  manual_subarea_correct_bool := TRUE
]

val[
  is.na(manual_area_correct_bool) &
    manual_decision_clean == "area_wrong",
  manual_area_correct_bool := FALSE
]

val[
  is.na(manual_subarea_correct_bool) &
    manual_decision_clean %in% c("area_correct_subarea_wrong", "area_wrong"),
  manual_subarea_correct_bool := FALSE
]

# Para ambiguous_or_both_valid, no lo metemos al accuracy estricto.
# Pero si manual_area_correct_bool = TRUE, se puede usar en accuracy ampliado.

# ============================================================
# 6) Métricas de accuracy
# ============================================================

val[, area_correct_strict := fifelse(
  is_validated == TRUE &
    is_excluded_from_accuracy == FALSE,
  manual_area_correct_bool,
  NA
)]

val[, subarea_correct_strict := fifelse(
  is_validated == TRUE &
    is_excluded_from_accuracy == FALSE,
  manual_subarea_correct_bool,
  NA
)]

val[, area_correct_including_ambiguous := fifelse(
  is_validated == TRUE &
    manual_decision_clean != "insufficient_information",
  manual_area_correct_bool,
  NA
)]

val[, subarea_correct_including_ambiguous := fifelse(
  is_validated == TRUE &
    manual_decision_clean != "insufficient_information",
  manual_subarea_correct_bool,
  NA
)]

# Etiquetas corregidas
val[, corrected_area_label := fifelse(
  manual_area_correct_bool == FALSE & manual_area_label != "",
  manual_area_label,
  top1_area_name
)]

val[, corrected_subarea_label := fifelse(
  manual_subarea_correct_bool == FALSE & manual_subarea_label != "",
  manual_subarea_label,
  top1_subarea_name
)]

val[
  manual_decision_clean == "insufficient_information",
  corrected_area_label := ""
]

val[
  manual_decision_clean == "insufficient_information",
  corrected_subarea_label := ""
]

# ============================================================
# 7) Resúmenes
# ============================================================

summary_completeness <- data.table(
  metric = c(
    "rows_total",
    "rows_validated",
    "rows_not_validated",
    "pct_validated",
    "insufficient_information",
    "ambiguous_or_both_valid"
  ),
  value = c(
    nrow(val),
    val[is_validated == TRUE, .N],
    val[is_validated == FALSE, .N],
    round(val[is_validated == TRUE, .N] / nrow(val) * 100, 2),
    val[manual_decision_clean == "insufficient_information", .N],
    val[manual_decision_clean == "ambiguous_or_both_valid", .N]
  )
)

summary_overall <- data.table(
  metric = c(
    "validated_for_strict_accuracy",
    "area_accuracy_strict",
    "subarea_accuracy_strict",
    "validated_including_ambiguous",
    "area_accuracy_including_ambiguous",
    "subarea_accuracy_including_ambiguous",
    "area_wrong_cases",
    "subarea_wrong_cases"
  ),
  value = c(
    val[!is.na(area_correct_strict), .N],
    mean_bool(val$area_correct_strict),
    mean_bool(val$subarea_correct_strict),
    val[!is.na(area_correct_including_ambiguous), .N],
    mean_bool(val$area_correct_including_ambiguous),
    mean_bool(val$subarea_correct_including_ambiguous),
    val[manual_area_correct_bool == FALSE, .N],
    val[manual_subarea_correct_bool == FALSE, .N]
  )
)

summary_group <- val[
  is_validated == TRUE,
  .(
    N = .N,
    strict_N = sum(!is.na(area_correct_strict)),
    area_accuracy_strict = mean_bool(area_correct_strict),
    subarea_accuracy_strict = mean_bool(subarea_correct_strict),
    area_accuracy_including_ambiguous = mean_bool(area_correct_including_ambiguous),
    subarea_accuracy_including_ambiguous = mean_bool(subarea_correct_including_ambiguous),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE),
    insufficient = sum(manual_decision_clean == "insufficient_information", na.rm = TRUE),
    ambiguous = sum(manual_decision_clean == "ambiguous_or_both_valid", na.rm = TRUE)
  ),
  by = validation_group
][order(validation_group)]

summary_decision_layer <- val[
  is_validated == TRUE,
  .(
    N = .N,
    strict_N = sum(!is.na(area_correct_strict)),
    area_accuracy_strict = mean_bool(area_correct_strict),
    subarea_accuracy_strict = mean_bool(subarea_correct_strict),
    area_accuracy_including_ambiguous = mean_bool(area_correct_including_ambiguous),
    subarea_accuracy_including_ambiguous = mean_bool(subarea_correct_including_ambiguous),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE)
  ),
  by = decision_layer
][order(-N)]

summary_area <- val[
  is_validated == TRUE,
  .(
    N = .N,
    strict_N = sum(!is.na(area_correct_strict)),
    area_accuracy_strict = mean_bool(area_correct_strict),
    subarea_accuracy_strict = mean_bool(subarea_correct_strict),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE)
  ),
  by = top1_area_name
][order(-N)]

summary_manual_decision <- val[
  ,
  .N,
  by = manual_decision_clean
][order(-N)]

summary_manual_decision[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 8) Matrices y errores
# ============================================================

area_confusion <- val[
  is_validated == TRUE &
    manual_area_correct_bool == FALSE &
    manual_area_label != "",
  .N,
  by = .(
    predicted_area = top1_area_name,
    manual_area = manual_area_label
  )
][order(-N)]

area_errors <- val[
  manual_area_correct_bool == FALSE,
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    predicted_area = top1_area_name,
    predicted_subarea = top1_subarea_name,
    corrected_area = manual_area_label,
    corrected_subarea = manual_subarea_label,
    manual_decision = manual_decision_clean,
    manual_comment,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    review_reason
  )
][order(validation_group, predicted_area)]

subarea_errors <- val[
  manual_subarea_correct_bool == FALSE,
  .(
    doc_id,
    doc_id_v5,
    validation_group,
    title,
    predicted_area = top1_area_name,
    predicted_subarea = top1_subarea_name,
    corrected_area = corrected_area_label,
    corrected_subarea = manual_subarea_label,
    manual_decision = manual_decision_clean,
    manual_comment,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    review_reason
  )
][order(validation_group, predicted_area)]

manual_corrections <- val[
  is_validated == TRUE,
  .(
    doc_id,
    doc_id_v5,
    work_id,
    title,
    validation_group,
    decision_layer,
    predicted_area = top1_area_name,
    predicted_subarea = top1_subarea_name,
    corrected_area = corrected_area_label,
    corrected_subarea = corrected_subarea_label,
    manual_area_correct = manual_area_correct_bool,
    manual_subarea_correct = manual_subarea_correct_bool,
    manual_decision = manual_decision_clean,
    manual_comment,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    confidence_level,
    zero_shot_status,
    review_reason
  )
]

# ============================================================
# 9) Recomendación de uso para entrenamiento
# ============================================================

get_group_acc <- function(group_name, metric_col = "area_accuracy_strict") {
  x <- summary_group[validation_group == group_name, get(metric_col)]
  if (length(x) == 0) return(NA_real_)
  x
}

strict_acc <- get_group_acc("01_seed_strict_by_area", "area_accuracy_strict")
expanded_acc <- get_group_acc("02_seed_expanded_by_area", "area_accuracy_strict")
candidate_acc <- get_group_acc("03_candidate_review_light_by_area", "area_accuracy_strict")
biomed_acc <- get_group_acc("05_possible_biomed_veterinary_conflict", "area_accuracy_strict")
method_acc <- get_group_acc("06_possible_method_domain_conflict", "area_accuracy_strict")

training_recommendation <- data.table(
  component = c(
    "seed_strict",
    "seed_expanded",
    "candidate_review_light",
    "biomed_veterinary_conflict",
    "method_domain_conflict"
  ),
  area_accuracy_strict = c(
    strict_acc,
    expanded_acc,
    candidate_acc,
    biomed_acc,
    method_acc
  )
)

training_recommendation[, recommendation := fcase(
  component == "seed_strict" & area_accuracy_strict >= 0.90,
  "usar_para_entrenamiento_principal",
  
  component == "seed_strict" & area_accuracy_strict < 0.90,
  "no_usar_directamente_ajustar_umbral_o_revisar",
  
  component == "seed_expanded" & area_accuracy_strict >= 0.85,
  "usar_como_entrenamiento_secundario_con_cautela",
  
  component == "seed_expanded" & area_accuracy_strict < 0.85,
  "no_usar_para_entrenamiento_sin_revision",
  
  component %in% c("candidate_review_light", "biomed_veterinary_conflict", "method_domain_conflict"),
  "usar_para_diagnostico_reranker_o_reglas_no_como_semilla",
  
  default = "revisar"
)]

# ============================================================
# 10) Exportar
# ============================================================

fwrite(val, out_validation_clean)
fwrite(summary_completeness, out_summary_completeness)
fwrite(summary_overall, out_summary_overall)
fwrite(summary_group, out_summary_group)
fwrite(summary_decision_layer, out_summary_decision_layer)
fwrite(summary_area, out_summary_area)
fwrite(summary_manual_decision, out_summary_manual_decision)
fwrite(area_confusion, out_area_confusion)
fwrite(subarea_errors, out_subarea_errors)
fwrite(area_errors, out_area_errors)
fwrite(manual_corrections, out_corrections)
fwrite(training_recommendation, out_recommendation)

# ============================================================
# 11) Imprimir resultados
# ============================================================

cat("\n================ EVALUACIÓN VALIDACIÓN MANUAL v5.2 ================\n")

cat("\nCompletitud:\n")
print(summary_completeness)

cat("\nResumen general:\n")
print(summary_overall)

cat("\nResumen por grupo:\n")
print(summary_group)

cat("\nResumen por decision_layer:\n")
print(summary_decision_layer)

cat("\nResumen por área predicha:\n")
print(summary_area)

cat("\nDecisiones manuales:\n")
print(summary_manual_decision)

cat("\nConfusión de áreas incorrectas:\n")
print(head(area_confusion, 30))

cat("\nRecomendación de entrenamiento:\n")
print(training_recommendation)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")