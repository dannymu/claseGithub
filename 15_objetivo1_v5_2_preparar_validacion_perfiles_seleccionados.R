# ============================================================
# 15_objetivo1_v5_2_preparar_validacion_perfiles_seleccionados.R
# Objetivo 1 v5.2
# Preparar muestra de validación externa por perfiles seleccionados
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
  "objetivo1_v5_2_profile_validation"
)

matches_file <- file.path(
  profile_val_dir,
  "profile_publications_matched_final_v5_2.csv"
)

summary_profiles_file <- file.path(
  profile_val_dir,
  "summary_profiles_matching_v5_2.csv"
)

out_dir <- file.path(
  profile_val_dir,
  "selected_profiles_external_validation"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_sample <- file.path(
  out_dir,
  "manual_validation_selected_profiles_v5_2.csv"
)

out_sample_xlsx <- file.path(
  out_dir,
  "manual_validation_selected_profiles_v5_2.xlsx"
)

out_summary <- file.path(
  out_dir,
  "summary_selected_profiles_validation_sample_v5_2.csv"
)

out_summary_profile <- file.path(
  out_dir,
  "summary_selected_profiles_by_profile_v5_2.csv"
)

if (!file.exists(matches_file)) {
  stop("No existe profile_publications_matched_final_v5_2.csv. Ejecuta primero el script 14.")
}

if (!file.exists(summary_profiles_file)) {
  stop("No existe summary_profiles_matching_v5_2.csv. Ejecuta primero el script 14.")
}

# ============================================================
# 2) Perfiles seleccionados
# ============================================================

selected_profiles <- c(
  "lopez_cozar",
  "torres_salina",
  "corchado",
  "antonio_gabriel",
  "carmen_batanero",
  "enrique_viedna",
  "luque",
  "jacqueline_schmidt",
  "evaristo_contreras"
)

# Muestra por perfil
n_auto_per_profile <- 10
n_review_per_profile <- 15
n_critical_per_profile <- 5

# ============================================================
# 3) Funciones
# ============================================================

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

short_text <- function(x, n = 1500) {
  x <- clean_text(x)
  ifelse(nchar(x) > n, paste0(substr(x, 1, n), " ..."), x)
}

sample_n_safe <- function(dt, n) {
  if (nrow(dt) == 0) return(dt)
  dt[sample(.N, min(.N, n))]
}

make_candidate_text <- function(c1, c2, c3, c4, c5) {
  paste(
    paste0("1) ", clean_text(c1)),
    paste0("2) ", clean_text(c2)),
    paste0("3) ", clean_text(c3)),
    paste0("4) ", clean_text(c4)),
    paste0("5) ", clean_text(c5)),
    sep = "\n"
  )
}

# ============================================================
# 4) Leer datos
# ============================================================

matches <- fread(matches_file)
summary_profiles <- fread(summary_profiles_file)

setDT(matches)
setDT(summary_profiles)

cat("Publicaciones emparejadas:", nrow(matches), "\n")

# ============================================================
# 5) Validar columnas necesarias
# ============================================================

needed_cols <- c(
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_row_id",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_source",
  "final_doc_id_v5",
  "final_work_id",
  "final_title",
  "final_year",
  "final_area_label",
  "final_subarea_label",
  "reportable_label",
  "final_decision",
  "final_confidence_tier",
  "auto_accept",
  "requires_review",
  "critical_conflict",
  "review_type",
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
  "profile_match_method"
)

for (col in needed_cols) {
  if (!col %in% names(matches)) {
    matches[, (col) := NA]
  }
}

matches[, researcher_safe_id := clean_text(researcher_safe_id)]
matches[, researcher_name := clean_text(researcher_name)]
matches[, profile_title := clean_text(profile_title)]
matches[, final_title := clean_text(final_title)]
matches[, auto_accept := as.logical(auto_accept)]
matches[, requires_review := as.logical(requires_review)]
matches[, critical_conflict := as.logical(critical_conflict)]

# ============================================================
# 6) Filtrar perfiles seleccionados
# ============================================================

dt <- matches[
  researcher_safe_id %in% selected_profiles
]

if (nrow(dt) == 0) {
  stop("No se encontraron perfiles seleccionados en los matches. Revisa researcher_safe_id.")
}

cat("Publicaciones de perfiles seleccionados:", nrow(dt), "\n")

# ============================================================
# 7) Crear muestras por perfil
# ============================================================

set.seed(123)

sample_list <- list()

for (pid in selected_profiles) {
  
  prof_dt <- dt[researcher_safe_id == pid]
  
  if (nrow(prof_dt) == 0) next
  
  # 7.1 Aceptados automáticamente
  s_auto <- sample_n_safe(
    prof_dt[auto_accept == TRUE],
    n_auto_per_profile
  )
  s_auto[, validation_group := "01_profile_auto_accepted"]
  
  # 7.2 Revisión top-k no crítica
  s_review <- sample_n_safe(
    prof_dt[
      requires_review == TRUE &
        (critical_conflict == FALSE | is.na(critical_conflict))
    ],
    n_review_per_profile
  )
  s_review[, validation_group := "02_profile_review_top5"]
  
  # 7.3 Conflictos críticos
  s_critical <- sample_n_safe(
    prof_dt[critical_conflict == TRUE],
    n_critical_per_profile
  )
  s_critical[, validation_group := "03_profile_critical_conflict"]
  
  sample_list[[pid]] <- rbindlist(
    list(s_auto, s_review, s_critical),
    fill = TRUE
  )
}

validation_sample <- rbindlist(sample_list, fill = TRUE)

validation_sample <- unique(
  validation_sample,
  by = c("researcher_safe_id", "profile_row_id", "final_doc_id_v5")
)

# ============================================================
# 8) Preparar columnas de revisión manual
# ============================================================

validation_sample[, candidate_top5_text := make_candidate_text(
  candidate_top1,
  candidate_top2,
  candidate_top3,
  candidate_top4,
  candidate_top5
)]

validation_sample[, text_for_review := paste(
  paste0("Título perfil: ", profile_title),
  paste0("Título universo: ", final_title),
  paste0("Fuente perfil: ", clean_text(profile_source)),
  paste0("Año perfil: ", profile_year),
  paste0("Candidatos top-k:\n", candidate_top5_text),
  sep = "\n\n"
)]

validation_sample[, text_for_review_short := short_text(text_for_review, 1800)]

validation_sample[, manual_area_correct := ""]
validation_sample[, manual_subarea_correct := ""]
validation_sample[, manual_area_label := ""]
validation_sample[, manual_subarea_label := ""]
validation_sample[, manual_decision := ""]
validation_sample[, manual_comment := ""]
validation_sample[, manual_profile_observation := ""]

# Valores recomendados:
# correct
# area_correct_subarea_wrong
# area_wrong
# insufficient_information
# ambiguous_or_both_valid
# top5_contains_correct_label
# top5_does_not_contain_correct_label

review_cols <- c(
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_file",
  "profile_row_id",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_source",
  "profile_match_method",
  "final_doc_id_v5",
  "final_work_id",
  "final_title",
  "final_year",
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
  "text_for_review_short",
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment",
  "manual_profile_observation"
)

for (col in review_cols) {
  if (!col %in% names(validation_sample)) {
    validation_sample[, (col) := ""]
  }
}

validation_out <- validation_sample[, ..review_cols]

setorder(
  validation_out,
  researcher_name,
  validation_group,
  profile_year,
  profile_title
)

# ============================================================
# 9) Resúmenes
# ============================================================

summary_sample <- validation_out[
  ,
  .N,
  by = validation_group
][order(validation_group)]

summary_sample[, pct := round(N / sum(N) * 100, 2)]

summary_profile <- validation_out[
  ,
  .(
    sample_n = .N,
    auto_accepted = sum(auto_accept == TRUE, na.rm = TRUE),
    review_required = sum(requires_review == TRUE, na.rm = TRUE),
    critical_conflict = sum(critical_conflict == TRUE, na.rm = TRUE)
  ),
  by = .(
    researcher_name,
    researcher_safe_id,
    profile_file
  )
][order(researcher_name)]

# Añadir datos generales del perfil
summary_profile <- merge(
  summary_profile,
  summary_profiles[
    ,
    .(
      researcher_safe_id,
      profile_publications,
      matched_publications,
      match_rate_pct,
      matched_auto_accept,
      auto_accept_pct_of_matched,
      matched_review_required,
      dominant_area,
      dominant_subarea,
      dominant_area_pct_auto
    )
  ],
  by = "researcher_safe_id",
  all.x = TRUE
)

# ============================================================
# 10) Exportar
# ============================================================

fwrite(validation_out, out_sample)
fwrite(summary_sample, out_summary)
fwrite(summary_profile, out_summary_profile)

# Crear XLSX si openxlsx existe
if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  wb <- openxlsx::createWorkbook()
  
  openxlsx::addWorksheet(wb, "validacion_perfiles")
  openxlsx::addWorksheet(wb, "resumen_perfiles")
  openxlsx::addWorksheet(wb, "instrucciones")
  
  openxlsx::writeData(wb, "validacion_perfiles", validation_out)
  openxlsx::writeData(wb, "resumen_perfiles", summary_profile)
  
  instrucciones <- data.table(
    campo = c(
      "manual_area_correct",
      "manual_subarea_correct",
      "manual_area_label",
      "manual_subarea_label",
      "manual_decision",
      "manual_comment",
      "manual_profile_observation"
    ),
    instruccion = c(
      "TRUE si el área sugerida o aceptada es correcta; FALSE si es incorrecta.",
      "TRUE si la subárea sugerida o aceptada es correcta; FALSE si es incorrecta.",
      "Si el área es incorrecta, escribir el área correcta.",
      "Si la subárea es incorrecta, escribir la subárea correcta.",
      "Usar: correct, area_correct_subarea_wrong, area_wrong, insufficient_information, ambiguous_or_both_valid, top5_contains_correct_label, top5_does_not_contain_correct_label.",
      "Comentario breve sobre la corrección temática.",
      "Observación general sobre el perfil o patrón detectado."
    )
  )
  
  openxlsx::writeData(wb, "instrucciones", instrucciones)
  
  openxlsx::freezePane(wb, "validacion_perfiles", firstActiveRow = 2, firstActiveCol = 6)
  openxlsx::setColWidths(wb, "validacion_perfiles", cols = 1:ncol(validation_out), widths = "auto")
  
  text_col <- which(names(validation_out) == "text_for_review_short")
  if (length(text_col) > 0) {
    openxlsx::setColWidths(wb, "validacion_perfiles", cols = text_col, widths = 90)
  }
  
  openxlsx::saveWorkbook(wb, out_sample_xlsx, overwrite = TRUE)
}

# ============================================================
# 11) Imprimir
# ============================================================

cat("\n================ VALIDACIÓN EXTERNA POR PERFILES v5.2 ================\n")
cat("Perfiles seleccionados:", length(unique(validation_out$researcher_safe_id)), "\n")
cat("Registros en muestra:", nrow(validation_out), "\n")

cat("\nResumen muestra:\n")
print(summary_sample)

cat("\nResumen por perfil:\n")
print(summary_profile)

cat("\nArchivos creados:\n")
cat(out_sample, "\n")
cat(out_sample_xlsx, "\n")
cat(out_summary, "\n")
cat(out_summary_profile, "\n")