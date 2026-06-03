# ============================================================
# 16_objetivo1_v5_2_evaluar_validacion_perfiles.R
# Objetivo 1 v5.2
# Evaluar validación externa por perfiles seleccionados
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

profile_val_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_validation",
  "selected_profiles_external_validation"
)

xlsx_file <- file.path(
  profile_val_dir,
  "manual_validation_selected_profiles_v5_2.xlsx"
)

csv_file <- file.path(
  profile_val_dir,
  "manual_validation_selected_profiles_v5_2.csv"
)

out_dir <- file.path(
  profile_val_dir,
  "validation_results_selected_profiles_v5_2"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_clean <- file.path(out_dir, "manual_validation_selected_profiles_clean_v5_2.csv")
out_summary_overall <- file.path(out_dir, "summary_profiles_validation_overall_v5_2.csv")
out_summary_group <- file.path(out_dir, "summary_profiles_validation_by_group_v5_2.csv")
out_summary_profile <- file.path(out_dir, "summary_profiles_validation_by_profile_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_profiles_validation_by_area_v5_2.csv")
out_summary_decision <- file.path(out_dir, "summary_profiles_manual_decision_v5_2.csv")
out_errors <- file.path(out_dir, "profile_validation_errors_v5_2.csv")
out_top5_failures <- file.path(out_dir, "profile_validation_top5_failures_v5_2.csv")

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

mean_bool <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  round(mean(x, na.rm = TRUE), 4)
}

# ============================================================
# 3) Leer archivo validado
# ============================================================

if (file.exists(xlsx_file) && requireNamespace("openxlsx", quietly = TRUE)) {
  val <- as.data.table(openxlsx::read.xlsx(xlsx_file, sheet = "validacion_perfiles"))
  input_used <- xlsx_file
} else if (file.exists(csv_file)) {
  val <- fread(csv_file)
  input_used <- csv_file
} else {
  stop("No se encontró archivo XLSX o CSV de validación por perfiles.")
}

setDT(val)

cat("Archivo usado:\n")
cat(input_used, "\n")
cat("Filas leídas:", nrow(val), "\n")

# ============================================================
# 4) Asegurar columnas
# ============================================================

needed_cols <- c(
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_source",
  "final_decision",
  "final_confidence_tier",
  "auto_accept",
  "requires_review",
  "critical_conflict",
  "review_type",
  "reportable_label",
  "final_area_label",
  "final_subarea_label",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top1_top2_margin",
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment",
  "manual_profile_observation"
)

for (col in needed_cols) {
  if (!col %in% names(val)) {
    val[, (col) := ""]
  }
}

text_cols <- c(
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_title",
  "profile_source",
  "final_decision",
  "final_confidence_tier",
  "review_type",
  "reportable_label",
  "final_area_label",
  "final_subarea_label",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "top1_area_name",
  "top1_subarea_name",
  "top2_area_name",
  "top2_subarea_name",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment",
  "manual_profile_observation"
)

for (col in text_cols) {
  val[, (col) := clean_text(get(col))]
}

val[, top1_score := safe_numeric(top1_score)]
val[, top2_score := safe_numeric(top2_score)]
val[, top1_top2_margin := safe_numeric(top1_top2_margin)]

val[, auto_accept := as.logical(auto_accept)]
val[, requires_review := as.logical(requires_review)]
val[, critical_conflict := as.logical(critical_conflict)]

val[, manual_area_correct_bool := to_bool_manual(manual_area_correct)]
val[, manual_subarea_correct_bool := to_bool_manual(manual_subarea_correct)]

val[, manual_decision_clean := tolower(clean_text(manual_decision))]

# Normalizar variantes frecuentes
val[manual_decision_clean %in% c("correcto", "correcta", "ok"), manual_decision_clean := "correct"]
val[manual_decision_clean %in% c("area wrong", "área incorrecta", "area_incorrecta"), manual_decision_clean := "area_wrong"]
val[manual_decision_clean %in% c("subarea wrong", "subárea incorrecta"), manual_decision_clean := "area_correct_subarea_wrong"]
val[manual_decision_clean %in% c("insufficient", "sin informacion", "sin información"), manual_decision_clean := "insufficient_information"]
val[manual_decision_clean %in% c("ambiguous", "ambigua", "ambiguo"), manual_decision_clean := "ambiguous_or_both_valid"]

# ============================================================
# 5) Inferencias para casos no llenados completamente
# ============================================================

val[, is_validated := manual_decision_clean != "" |
      !is.na(manual_area_correct_bool) |
      !is.na(manual_subarea_correct_bool)]

# Si manual_decision permite inferir TRUE/FALSE
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

# Para top5:
# top5_contains_correct_label equivale a que el sistema fue útil en revisión top-k.
val[, top5_area_contains_correct := NA]
val[, top5_subarea_contains_correct := NA]

val[
  manual_decision_clean %in% c("top5_contains_correct_label", "correct", "ambiguous_or_both_valid"),
  top5_area_contains_correct := TRUE
]

val[
  manual_decision_clean %in% c("top5_contains_correct_label", "correct", "ambiguous_or_both_valid"),
  top5_subarea_contains_correct := TRUE
]

val[
  manual_decision_clean == "top5_does_not_contain_correct_label",
  top5_area_contains_correct := FALSE
]

val[
  manual_decision_clean == "top5_does_not_contain_correct_label",
  top5_subarea_contains_correct := FALSE
]

# Si el usuario llenó TRUE/FALSE en casos review, usarlo como señal top5
val[
  is.na(top5_area_contains_correct) &
    validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict"),
  top5_area_contains_correct := manual_area_correct_bool
]

val[
  is.na(top5_subarea_contains_correct) &
    validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict"),
  top5_subarea_contains_correct := manual_subarea_correct_bool
]

# Para auto_accept, accuracy directo
val[, auto_area_correct := fifelse(
  validation_group == "01_profile_auto_accepted" & is_validated == TRUE,
  manual_area_correct_bool,
  NA
)]

val[, auto_subarea_correct := fifelse(
  validation_group == "01_profile_auto_accepted" & is_validated == TRUE,
  manual_subarea_correct_bool,
  NA
)]

# Para review, utilidad top5
val[, review_top5_area_useful := fifelse(
  validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict") &
    is_validated == TRUE,
  top5_area_contains_correct,
  NA
)]

val[, review_top5_subarea_useful := fifelse(
  validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict") &
    is_validated == TRUE,
  top5_subarea_contains_correct,
  NA
)]

# Correcciones manuales
val[, corrected_area_label := fifelse(
  manual_area_correct_bool == FALSE & manual_area_label != "",
  manual_area_label,
  fifelse(final_area_label != "", final_area_label, top1_area_name)
)]

val[, corrected_subarea_label := fifelse(
  manual_subarea_correct_bool == FALSE & manual_subarea_label != "",
  manual_subarea_label,
  fifelse(final_subarea_label != "", final_subarea_label, top1_subarea_name)
)]

# ============================================================
# 6) Resúmenes
# ============================================================

summary_overall <- data.table(
  metric = c(
    "rows_total",
    "rows_validated",
    "rows_not_validated",
    "pct_validated",
    "auto_accepted_rows",
    "review_top5_rows",
    "critical_conflict_rows",
    "auto_area_accuracy",
    "auto_subarea_accuracy",
    "review_top5_area_usefulness",
    "review_top5_subarea_usefulness",
    "area_wrong_cases",
    "subarea_wrong_cases",
    "top5_not_contains_correct_cases"
  ),
  value = c(
    nrow(val),
    val[is_validated == TRUE, .N],
    val[is_validated == FALSE, .N],
    round(val[is_validated == TRUE, .N] / nrow(val) * 100, 2),
    val[validation_group == "01_profile_auto_accepted", .N],
    val[validation_group == "02_profile_review_top5", .N],
    val[validation_group == "03_profile_critical_conflict", .N],
    mean_bool(val$auto_area_correct),
    mean_bool(val$auto_subarea_correct),
    mean_bool(val$review_top5_area_useful),
    mean_bool(val$review_top5_subarea_useful),
    val[manual_area_correct_bool == FALSE, .N],
    val[manual_subarea_correct_bool == FALSE, .N],
    val[manual_decision_clean == "top5_does_not_contain_correct_label", .N]
  )
)

summary_group <- val[
  is_validated == TRUE,
  .(
    N = .N,
    auto_area_accuracy = mean_bool(auto_area_correct),
    auto_subarea_accuracy = mean_bool(auto_subarea_correct),
    top5_area_usefulness = mean_bool(review_top5_area_useful),
    top5_subarea_usefulness = mean_bool(review_top5_subarea_useful),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE),
    top5_failures = sum(manual_decision_clean == "top5_does_not_contain_correct_label", na.rm = TRUE)
  ),
  by = validation_group
][order(validation_group)]

summary_profile <- val[
  is_validated == TRUE,
  .(
    N = .N,
    auto_n = sum(validation_group == "01_profile_auto_accepted"),
    review_n = sum(validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict")),
    auto_area_accuracy = mean_bool(auto_area_correct),
    auto_subarea_accuracy = mean_bool(auto_subarea_correct),
    top5_area_usefulness = mean_bool(review_top5_area_useful),
    top5_subarea_usefulness = mean_bool(review_top5_subarea_useful),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE),
    top5_failures = sum(manual_decision_clean == "top5_does_not_contain_correct_label", na.rm = TRUE)
  ),
  by = .(
    researcher_name,
    researcher_safe_id,
    profile_file
  )
][order(researcher_name)]

summary_area <- val[
  is_validated == TRUE,
  .(
    N = .N,
    auto_area_accuracy = mean_bool(auto_area_correct),
    auto_subarea_accuracy = mean_bool(auto_subarea_correct),
    top5_area_usefulness = mean_bool(review_top5_area_useful),
    top5_subarea_usefulness = mean_bool(review_top5_subarea_useful),
    area_wrong = sum(manual_area_correct_bool == FALSE, na.rm = TRUE),
    subarea_wrong = sum(manual_subarea_correct_bool == FALSE, na.rm = TRUE)
  ),
  by = corrected_area_label
][order(-N)]

summary_decision <- val[
  ,
  .N,
  by = manual_decision_clean
][order(-N)]

summary_decision[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 7) Errores y fallos top5
# ============================================================

errors <- val[
  manual_area_correct_bool == FALSE |
    manual_subarea_correct_bool == FALSE,
  .(
    validation_group,
    researcher_name,
    researcher_safe_id,
    profile_title,
    final_decision,
    final_area_label,
    final_subarea_label,
    top1_area_name,
    top1_subarea_name,
    candidate_top1,
    candidate_top2,
    candidate_top3,
    candidate_top4,
    candidate_top5,
    corrected_area_label,
    corrected_subarea_label,
    manual_decision_clean,
    manual_comment,
    manual_profile_observation
  )
][order(researcher_name, validation_group)]

top5_failures <- val[
  manual_decision_clean == "top5_does_not_contain_correct_label" |
    review_top5_area_useful == FALSE |
    review_top5_subarea_useful == FALSE,
  .(
    validation_group,
    researcher_name,
    researcher_safe_id,
    profile_title,
    candidate_top1,
    candidate_top2,
    candidate_top3,
    candidate_top4,
    candidate_top5,
    corrected_area_label,
    corrected_subarea_label,
    manual_decision_clean,
    manual_comment,
    manual_profile_observation
  )
][order(researcher_name, validation_group)]

# ============================================================
# 8) Exportar
# ============================================================

fwrite(val, out_clean)
fwrite(summary_overall, out_summary_overall)
fwrite(summary_group, out_summary_group)
fwrite(summary_profile, out_summary_profile)
fwrite(summary_area, out_summary_area)
fwrite(summary_decision, out_summary_decision)
fwrite(errors, out_errors)
fwrite(top5_failures, out_top5_failures)

# ============================================================
# 9) Imprimir
# ============================================================

cat("\n================ EVALUACIÓN VALIDACIÓN EXTERNA POR PERFILES v5.2 ================\n")

cat("\nResumen general:\n")
print(summary_overall)

cat("\nResumen por grupo:\n")
print(summary_group)

cat("\nResumen por perfil:\n")
print(summary_profile)

cat("\nResumen por área corregida:\n")
print(summary_area)

cat("\nDecisiones manuales:\n")
print(summary_decision)

cat("\nErrores:", nrow(errors), "\n")
cat("Fallos top5:", nrow(top5_failures), "\n")

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")