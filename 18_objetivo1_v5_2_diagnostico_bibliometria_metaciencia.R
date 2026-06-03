# ============================================================
# 18_objetivo1_v5_2_diagnostico_bibliometria_metaciencia.R
# Objetivo 1 v5.2
# Diagnóstico dirigido de errores en Bibliometrics and Scientometrics
# a partir de la validación externa por perfiles.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

profile_val_results_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_validation",
  "selected_profiles_external_validation",
  "validation_results_selected_profiles_v5_2"
)

error_analysis_dir <- file.path(
  profile_val_results_dir,
  "error_analysis"
)

clean_file <- file.path(
  profile_val_results_dir,
  "manual_validation_selected_profiles_clean_v5_2.csv"
)

cases_review_file <- file.path(
  error_analysis_dir,
  "cases_to_review_before_report_v5_2.csv"
)

out_dir <- file.path(
  profile_val_results_dir,
  "bibliometrics_metascience_diagnosis"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_biblio_cases <- file.path(out_dir, "biblio_metascience_error_cases_v5_2.csv")
out_biblio_top5_failures <- file.path(out_dir, "biblio_metascience_top5_failures_v5_2.csv")
out_summary_biblio <- file.path(out_dir, "summary_biblio_metascience_diagnosis_v5_2.csv")
out_summary_by_profile <- file.path(out_dir, "summary_biblio_errors_by_profile_v5_2.csv")
out_summary_predicted_area <- file.path(out_dir, "summary_biblio_errors_by_predicted_area_v5_2.csv")
out_terms_context <- file.path(out_dir, "biblio_metascience_terms_context_v5_2.csv")
out_recommendations <- file.path(out_dir, "biblio_metascience_recommendations_v5_2.csv")

if (!file.exists(clean_file)) {
  stop("No existe: ", clean_file)
}

if (!file.exists(cases_review_file)) {
  stop("No existe: ", cases_review_file)
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

to_bool_manual <- function(x) {
  x <- clean_text(x)
  x_low <- tolower(x)
  
  out <- rep(NA, length(x_low))
  out[x_low %in% c("true", "t", "1", "si", "sí", "yes", "y")] <- TRUE
  out[x_low %in% c("false", "f", "0", "no", "n")] <- FALSE
  
  out
}

has_pattern <- function(x, pattern) {
  grepl(pattern, x, ignore.case = TRUE, perl = TRUE)
}

contains_biblio_candidate <- function(...) {
  txt <- paste(clean_text(c(...)), collapse = " ")
  grepl("Bibliometrics and Scientometrics|Bibliometrics|Scientometrics", txt, ignore.case = TRUE)
}

# ============================================================
# 3) Leer datos
# ============================================================

val <- fread(clean_file)
cases_review <- fread(cases_review_file)

setDT(val)
setDT(cases_review)

cat("Validación perfiles:", nrow(val), "\n")
cat("Casos a revisar:", nrow(cases_review), "\n")

# ============================================================
# 4) Asegurar columnas
# ============================================================

needed_cols <- c(
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_title",
  "profile_source",
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
  "manual_profile_observation",
  "text_for_review_short"
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

# Por si TRUE/FALSE vino como texto
if (!is.logical(val$manual_area_correct_bool)) {
  val[, manual_area_correct_bool := to_bool_manual(manual_area_correct_bool)]
}

if (!is.logical(val$manual_subarea_correct_bool)) {
  val[, manual_subarea_correct_bool := to_bool_manual(manual_subarea_correct_bool)]
}

# ============================================================
# 5) Patrones de metaciencia/bibliometría
# ============================================================

biblio_pattern <- paste(
  c(
    "bibliometric",
    "bibliometr",
    "scientometric",
    "cienciometr",
    "informetric",
    "altmetric",
    "citation analysis",
    "citation impact",
    "citation index",
    "citation count",
    "citedness",
    "co-citation",
    "cocitation",
    "co-word",
    "coword",
    "h-index",
    "google scholar",
    "web of science",
    "scopus",
    "journal impact",
    "impact factor",
    "research evaluation",
    "evaluaci[oó]n cient[ií]fica",
    "producci[oó]n cient[ií]fica",
    "comunicaci[oó]n cient[ií]fica",
    "scholarly communication",
    "science mapping",
    "mapas? de ciencia",
    "research data",
    "data citation",
    "data sharing",
    "open science",
    "ciencia abierta",
    "academic search engine",
    "scientific output",
    "scientific production",
    "research performance",
    "ranking"
  ),
  collapse = "|"
)

val[, text_diagnosis := paste(
  profile_title,
  profile_source,
  candidate_top1,
  candidate_top2,
  candidate_top3,
  candidate_top4,
  candidate_top5,
  manual_comment,
  manual_profile_observation,
  text_for_review_short,
  sep = " "
)]

val[, has_biblio_terms := has_pattern(text_diagnosis, biblio_pattern)]

val[, predicted_area_for_diagnosis := fifelse(
  final_area_label != "",
  final_area_label,
  top1_area_name
)]

val[, predicted_subarea_for_diagnosis := fifelse(
  final_subarea_label != "",
  final_subarea_label,
  top1_subarea_name
)]

val[, corrected_is_biblio := corrected_area_label == "Social Sciences" &
      corrected_subarea_label == "Bibliometrics and Scientometrics"]

val[, predicted_is_biblio := predicted_area_for_diagnosis == "Social Sciences" &
      predicted_subarea_for_diagnosis == "Bibliometrics and Scientometrics"]

val[, top5_contains_biblio := mapply(
  contains_biblio_candidate,
  candidate_top1,
  candidate_top2,
  candidate_top3,
  candidate_top4,
  candidate_top5
)]

# ============================================================
# 6) Casos de interés
# ============================================================

biblio_cases <- val[
  corrected_is_biblio == TRUE |
    has_biblio_terms == TRUE |
    predicted_is_biblio == TRUE
]

biblio_error_cases <- biblio_cases[
  manual_area_correct_bool == FALSE |
    manual_subarea_correct_bool == FALSE |
    manual_decision_clean == "top5_does_not_contain_correct_label"
]

biblio_top5_failures <- biblio_error_cases[
  corrected_is_biblio == TRUE &
    top5_contains_biblio == FALSE
]

biblio_top5_contains_but_not_top1 <- biblio_error_cases[
  corrected_is_biblio == TRUE &
    top5_contains_biblio == TRUE &
    predicted_is_biblio == FALSE
]

# ============================================================
# 7) Resúmenes
# ============================================================

summary_biblio <- data.table(
  metric = c(
    "profile_validation_rows",
    "biblio_related_cases",
    "biblio_error_cases",
    "corrected_biblio_cases",
    "corrected_biblio_errors",
    "corrected_biblio_top5_failures",
    "corrected_biblio_top5_contains_but_not_top1",
    "biblio_cases_with_biblio_terms",
    "biblio_error_cases_with_biblio_terms"
  ),
  value = c(
    nrow(val),
    nrow(biblio_cases),
    nrow(biblio_error_cases),
    val[corrected_is_biblio == TRUE, .N],
    biblio_error_cases[corrected_is_biblio == TRUE, .N],
    nrow(biblio_top5_failures),
    nrow(biblio_top5_contains_but_not_top1),
    biblio_cases[has_biblio_terms == TRUE, .N],
    biblio_error_cases[has_biblio_terms == TRUE, .N]
  )
)

summary_by_profile <- biblio_error_cases[
  ,
  .(
    biblio_error_cases = .N,
    corrected_biblio_errors = sum(corrected_is_biblio == TRUE, na.rm = TRUE),
    top5_failures = sum(top5_contains_biblio == FALSE & corrected_is_biblio == TRUE, na.rm = TRUE),
    top5_contains_but_not_top1 = sum(top5_contains_biblio == TRUE & predicted_is_biblio == FALSE & corrected_is_biblio == TRUE, na.rm = TRUE),
    with_biblio_terms = sum(has_biblio_terms == TRUE, na.rm = TRUE)
  ),
  by = .(
    researcher_name,
    researcher_safe_id
  )
][order(-biblio_error_cases)]

summary_predicted_area <- biblio_error_cases[
  corrected_is_biblio == TRUE,
  .N,
  by = .(
    predicted_area_for_diagnosis,
    predicted_subarea_for_diagnosis
  )
][order(-N)]

# ============================================================
# 8) Contexto de términos
# ============================================================

terms_context <- biblio_error_cases[
  ,
  .(
    validation_group,
    researcher_name,
    researcher_safe_id,
    profile_title,
    profile_source,
    predicted_area = predicted_area_for_diagnosis,
    predicted_subarea = predicted_subarea_for_diagnosis,
    corrected_area_label,
    corrected_subarea_label,
    top5_contains_biblio,
    has_biblio_terms,
    candidate_top1,
    candidate_top2,
    candidate_top3,
    candidate_top4,
    candidate_top5,
    manual_decision_clean,
    manual_comment,
    manual_profile_observation
  )
][order(researcher_name, validation_group)]

# ============================================================
# 9) Recomendaciones automáticas
# ============================================================

recommendations <- data.table(
  finding = c(
    "biblio_errors_concentrated_in_profiles",
    "biblio_top5_failures",
    "biblio_signal_detection",
    "critical_conflicts_policy",
    "prototype_improvement"
  ),
  recommendation = c(
    "Reportar que los errores de bibliometría/metaciencia se concentran en perfiles especializados de bibliometría y evaluación científica.",
    "Si corrected_biblio_top5_failures es alto, reforzar prototipos de Social Sciences / Bibliometrics and Scientometrics antes de reentrenar o rerankear.",
    "Usar patrones textuales de bibliometría/metaciencia como señal auxiliar para marcar revisión prioritaria cuando el top1 no sea Bibliometrics and Scientometrics.",
    "Mantener los conflictos críticos como revisión obligatoria; no convertirlos en clasificación automática.",
    "Crear prototipos adicionales para bibliometría: evaluación científica, Google Scholar, Scopus, Web of Science, indicadores de citación, comunicación científica, ciencia abierta, data citation y scholarly communication."
  )
)

# Añadir una decisión preliminar según resultados
biblio_top5_failure_n <- nrow(biblio_top5_failures)

if (biblio_top5_failure_n >= 5) {
  recommendations <- rbind(
    recommendations,
    data.table(
      finding = "decision_preliminary",
      recommendation = "Conviene reforzar prototipos de Bibliometrics and Scientometrics y repetir una prueba localizada antes de redactar el informe final."
    )
  )
} else {
  recommendations <- rbind(
    recommendations,
    data.table(
      finding = "decision_preliminary",
      recommendation = "Los errores de bibliometría pueden reportarse como limitación controlada, sin repetir el pipeline completo."
    )
  )
}

# ============================================================
# 10) Exportar
# ============================================================

fwrite(biblio_error_cases, out_biblio_cases)
fwrite(biblio_top5_failures, out_biblio_top5_failures)
fwrite(summary_biblio, out_summary_biblio)
fwrite(summary_by_profile, out_summary_by_profile)
fwrite(summary_predicted_area, out_summary_predicted_area)
fwrite(terms_context, out_terms_context)
fwrite(recommendations, out_recommendations)

# ============================================================
# 11) Imprimir
# ============================================================

cat("\n================ DIAGNÓSTICO BIBLIOMETRÍA / METACIENCIA v5.2 ================\n")

cat("\nResumen bibliometría/metaciencia:\n")
print(summary_biblio)

cat("\nErrores por perfil:\n")
print(summary_by_profile)

cat("\nErrores de bibliometría por área/subárea predicha:\n")
print(summary_predicted_area)

cat("\nFallos top5 bibliometría:", nrow(biblio_top5_failures), "\n")
cat("Top5 contiene bibliometría pero no top1:", nrow(biblio_top5_contains_but_not_top1), "\n")

cat("\nRecomendaciones:\n")
print(recommendations)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")