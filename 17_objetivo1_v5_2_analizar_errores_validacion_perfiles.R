# ============================================================
# 17_objetivo1_v5_2_analizar_errores_validacion_perfiles.R
# Objetivo 1 v5.2
# Analizar errores y fallos top5 en validación externa por perfiles
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

profile_val_results_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_validation",
  "selected_profiles_external_validation",
  "validation_results_selected_profiles_v5_2"
)

clean_file <- file.path(
  profile_val_results_dir,
  "manual_validation_selected_profiles_clean_v5_2.csv"
)

errors_file <- file.path(
  profile_val_results_dir,
  "profile_validation_errors_v5_2.csv"
)

top5_failures_file <- file.path(
  profile_val_results_dir,
  "profile_validation_top5_failures_v5_2.csv"
)

out_dir <- file.path(
  profile_val_results_dir,
  "error_analysis"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_summary_error_profile <- file.path(out_dir, "summary_error_by_profile_v5_2.csv")
out_summary_error_group <- file.path(out_dir, "summary_error_by_group_v5_2.csv")
out_summary_error_area <- file.path(out_dir, "summary_error_by_area_v5_2.csv")
out_summary_top5_failure_profile <- file.path(out_dir, "summary_top5_failure_by_profile_v5_2.csv")
out_summary_top5_failure_area <- file.path(out_dir, "summary_top5_failure_by_area_v5_2.csv")
out_cases_to_review <- file.path(out_dir, "cases_to_review_before_report_v5_2.csv")

if (!file.exists(clean_file)) {
  stop("No existe: ", clean_file)
}

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

val <- fread(clean_file)
setDT(val)

# Asegurar columnas
needed_cols <- c(
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_title",
  "final_decision",
  "final_area_label",
  "final_subarea_label",
  "top1_area_name",
  "top1_subarea_name",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "manual_area_correct_bool",
  "manual_subarea_correct_bool",
  "corrected_area_label",
  "corrected_subarea_label",
  "manual_decision_clean",
  "manual_comment",
  "manual_profile_observation"
)

for (col in needed_cols) {
  if (!col %in% names(val)) {
    val[, (col) := NA]
  }
}

for (col in names(val)) {
  if (is.character(val[[col]])) {
    val[, (col) := clean_text(get(col))]
  }
}

# Marcas de error
val[, is_area_error := manual_area_correct_bool == FALSE]
val[, is_subarea_error := manual_subarea_correct_bool == FALSE]

val[, is_top5_failure := manual_decision_clean == "top5_does_not_contain_correct_label"]

val[
  validation_group %in% c("02_profile_review_top5", "03_profile_critical_conflict") &
    (manual_area_correct_bool == FALSE | manual_subarea_correct_bool == FALSE),
  is_review_failure := TRUE
]

val[is.na(is_review_failure), is_review_failure := FALSE]

# ============================================================
# Resúmenes
# ============================================================

summary_error_profile <- val[
  ,
  .(
    N = .N,
    area_errors = sum(is_area_error, na.rm = TRUE),
    subarea_errors = sum(is_subarea_error, na.rm = TRUE),
    top5_failures = sum(is_top5_failure, na.rm = TRUE),
    review_failures = sum(is_review_failure, na.rm = TRUE)
  ),
  by = .(
    researcher_name,
    researcher_safe_id
  )
][order(-area_errors, -top5_failures)]

summary_error_profile[
  ,
  area_error_pct := round(area_errors / N * 100, 2)
]

summary_error_profile[
  ,
  top5_failure_pct := round(top5_failures / N * 100, 2)
]

summary_error_group <- val[
  ,
  .(
    N = .N,
    area_errors = sum(is_area_error, na.rm = TRUE),
    subarea_errors = sum(is_subarea_error, na.rm = TRUE),
    top5_failures = sum(is_top5_failure, na.rm = TRUE),
    review_failures = sum(is_review_failure, na.rm = TRUE)
  ),
  by = validation_group
][order(validation_group)]

summary_error_group[
  ,
  area_error_pct := round(area_errors / N * 100, 2)
]

summary_error_group[
  ,
  top5_failure_pct := round(top5_failures / N * 100, 2)
]

summary_error_area <- val[
  is_area_error == TRUE | is_subarea_error == TRUE,
  .N,
  by = .(
    predicted_area = fifelse(final_area_label != "", final_area_label, top1_area_name),
    predicted_subarea = fifelse(final_subarea_label != "", final_subarea_label, top1_subarea_name),
    corrected_area_label,
    corrected_subarea_label
  )
][order(-N)]

summary_top5_failure_profile <- val[
  is_top5_failure == TRUE,
  .N,
  by = .(
    researcher_name,
    researcher_safe_id
  )
][order(-N)]

summary_top5_failure_area <- val[
  is_top5_failure == TRUE,
  .N,
  by = .(
    corrected_area_label,
    corrected_subarea_label
  )
][order(-N)]

# Casos que conviene revisar antes de redactar
cases_to_review <- val[
  is_area_error == TRUE |
    is_subarea_error == TRUE |
    is_top5_failure == TRUE |
    validation_group == "03_profile_critical_conflict",
  .(
    validation_group,
    researcher_name,
    researcher_safe_id,
    profile_title,
    final_decision,
    predicted_area = fifelse(final_area_label != "", final_area_label, top1_area_name),
    predicted_subarea = fifelse(final_subarea_label != "", final_subarea_label, top1_subarea_name),
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
# Exportar
# ============================================================

fwrite(summary_error_profile, out_summary_error_profile)
fwrite(summary_error_group, out_summary_error_group)
fwrite(summary_error_area, out_summary_error_area)
fwrite(summary_top5_failure_profile, out_summary_top5_failure_profile)
fwrite(summary_top5_failure_area, out_summary_top5_failure_area)
fwrite(cases_to_review, out_cases_to_review)

# ============================================================
# Imprimir
# ============================================================

cat("\n================ ANÁLISIS DE ERRORES VALIDACIÓN PERFILES v5.2 ================\n")

cat("\nErrores por perfil:\n")
print(summary_error_profile)

cat("\nErrores por grupo:\n")
print(summary_error_group)

cat("\nConfusión/corrección de áreas y subáreas:\n")
print(summary_error_area)

cat("\nFallos top5 por perfil:\n")
print(summary_top5_failure_profile)

cat("\nFallos top5 por área corregida:\n")
print(summary_top5_failure_area)

cat("\nCasos a revisar antes del informe:", nrow(cases_to_review), "\n")

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")