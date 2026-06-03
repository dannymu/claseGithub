# ============================================================
# 04_objetivo1_v5_2_diagnostico_zero_shot_full.R
# Objetivo 1 v5.2
# Diagnóstico del zero-shot full antes de integración o entrenamiento
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
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

pred_file <- file.path(
  zero_dir,
  "zero_shot_v5_2_multiprototype_full.csv"
)

out_dir <- file.path(
  zero_dir,
  "diagnostico_zero_shot_v5_2"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(pred_file)) {
  stop("No existe zero_shot_v5_2_multiprototype_full.csv")
}

# Salidas
out_summary_global <- file.path(out_dir, "summary_global_zero_shot_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_area_zero_shot_v5_2.csv")
out_summary_confidence <- file.path(out_dir, "summary_confidence_zero_shot_v5_2.csv")
out_summary_status <- file.path(out_dir, "summary_status_zero_shot_v5_2.csv")
out_top1_top2_pairs <- file.path(out_dir, "summary_top1_top2_area_pairs.csv")
out_boundary_conflicts <- file.path(out_dir, "boundary_conflicts_low_margin.csv")
out_biomed_veterinary <- file.path(out_dir, "possible_biomed_veterinary_conflicts.csv")
out_method_domain <- file.path(out_dir, "possible_method_domain_conflicts.csv")
out_critical_cases <- file.path(out_dir, "critical_pattern_cases.csv")
out_validation_sample <- file.path(out_dir, "validation_sample_stratified_v5_2.csv")

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

has_pattern <- function(text, pattern) {
  grepl(pattern, text, ignore.case = TRUE, perl = TRUE)
}

# ============================================================
# 3) Leer predicciones
# ============================================================

pred <- fread(pred_file)
setDT(pred)

cat("Documentos leídos:", nrow(pred), "\n")

needed_cols <- c(
  "doc_id",
  "doc_id_v5",
  "title",
  "text_for_embedding",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "top1_prototype_language"
)

for (col in needed_cols) {
  if (!col %in% names(pred)) {
    pred[, (col) := NA]
  }
}

pred[, title := clean_text(title)]
pred[, text_for_embedding := clean_text(text_for_embedding)]
pred[, top1_score := safe_numeric(top1_score)]
pred[, top2_score := safe_numeric(top2_score)]
pred[, top1_top2_margin := safe_numeric(top1_top2_margin)]

pred[, text_lower := tolower(paste(title, text_for_embedding))]

# ============================================================
# 4) Resumen global
# ============================================================

summary_global <- data.table(
  metric = c(
    "documents_total",
    "preliminary_label",
    "ambiguous_review",
    "low_similarity_review",
    "alta_confianza",
    "confianza_media",
    "ambiguo",
    "revisar",
    "baja_similitud",
    "mean_top1_score",
    "median_top1_score",
    "mean_margin",
    "median_margin"
  ),
  value = c(
    nrow(pred),
    pred[zero_shot_status == "preliminary_label", .N],
    pred[zero_shot_status == "ambiguous_review", .N],
    pred[zero_shot_status == "low_similarity_review", .N],
    pred[confidence_level == "alta_confianza", .N],
    pred[confidence_level == "confianza_media", .N],
    pred[confidence_level == "ambiguo", .N],
    pred[confidence_level == "revisar", .N],
    pred[confidence_level == "baja_similitud", .N],
    round(mean(pred$top1_score, na.rm = TRUE), 4),
    round(median(pred$top1_score, na.rm = TRUE), 4),
    round(mean(pred$top1_top2_margin, na.rm = TRUE), 4),
    round(median(pred$top1_top2_margin, na.rm = TRUE), 4)
  )
)

# ============================================================
# 5) Resúmenes por área, confianza y estado
# ============================================================

summary_area <- pred[
  ,
  .(
    N = .N,
    mean_top1_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4),
    ambiguous = sum(zero_shot_status == "ambiguous_review", na.rm = TRUE),
    low_similarity = sum(zero_shot_status == "low_similarity_review", na.rm = TRUE),
    high_confidence = sum(confidence_level == "alta_confianza", na.rm = TRUE),
    spanish_prototype = sum(top1_prototype_language == "es", na.rm = TRUE),
    english_prototype = sum(top1_prototype_language == "en", na.rm = TRUE)
  ),
  by = top1_area_name
][order(-N)]

summary_area[, pct := round(N / sum(N) * 100, 2)]
summary_area[, ambiguous_pct := round(ambiguous / N * 100, 2)]
summary_area[, high_confidence_pct := round(high_confidence / N * 100, 2)]
summary_area[, spanish_prototype_pct := round(spanish_prototype / N * 100, 2)]

summary_confidence <- pred[
  ,
  .N,
  by = confidence_level
][order(-N)]

summary_confidence[, pct := round(N / sum(N) * 100, 2)]

summary_status <- pred[
  ,
  .N,
  by = zero_shot_status
][order(-N)]

summary_status[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 6) Pares top1-top2
# ============================================================

top1_top2_pairs <- pred[
  ,
  .(
    N = .N,
    mean_top1_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_top2_score = round(mean(top2_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4),
    ambiguous = sum(zero_shot_status == "ambiguous_review", na.rm = TRUE)
  ),
  by = .(
    top1_area_name,
    top2_area_name
  )
][order(-N)]

top1_top2_pairs[, ambiguous_pct := round(ambiguous / N * 100, 2)]

boundary_conflicts <- pred[
  top1_area_name != top2_area_name &
    !is.na(top1_top2_margin) &
    top1_top2_margin < 0.05,
  .(
    N = .N,
    mean_top1_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_top2_score = round(mean(top2_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4)
  ),
  by = .(
    top1_area_name,
    top2_area_name
  )
][order(-N)]

# ============================================================
# 7) Diagnóstico general de conflictos semánticos
# ============================================================

# No corrige todavía. Solo identifica casos sospechosos para evaluar.

virus_pattern <- paste(
  c(
    "covid",
    "coronavirus",
    "sarscov",
    "sars-cov",
    "h1n1",
    "influenza",
    "ebola",
    "pandemic",
    "pandemia",
    "virus",
    "viral"
  ),
  collapse = "|"
)

animal_context_pattern <- paste(
  c(
    "veterinary",
    "veterinaria",
    "animal",
    "animals",
    "livestock",
    "bovine",
    "porcine",
    "swine",
    "cattle",
    "goat",
    "goats",
    "sheep",
    "poultry",
    "canine",
    "feline",
    "equine",
    "zoo",
    "wildlife"
  ),
  collapse = "|"
)

pred[, has_virus_context := has_pattern(text_lower, virus_pattern)]
pred[, has_animal_context := has_pattern(text_lower, animal_context_pattern)]

possible_biomed_veterinary <- pred[
  top1_area_name == "Veterinary" &
    has_virus_context == TRUE &
    has_animal_context == FALSE,
  .(
    doc_id,
    doc_id_v5,
    title,
    top1_area_name,
    top1_subarea_name,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    zero_shot_status,
    confidence_level,
    top1_prototype_language,
    text_for_embedding
  )
][order(top1_top2_margin)]

# Método/enfoque vs dominio empírico.
# Sirve para detectar cuando una palabra de dominio arrastra la clasificación.
method_science_pattern <- paste(
  c(
    "bibliometric",
    "bibliometr",
    "scientometric",
    "cienciometr",
    "informetric",
    "altmetric",
    "citation analysis",
    "citation index",
    "data citation",
    "data reuse",
    "research data",
    "data sharing",
    "open science",
    "scholarly communication",
    "science communication",
    "research evaluation",
    "journal impact",
    "h-index",
    "co-citation",
    "coword",
    "co-word",
    "science mapping",
    "research data management"
  ),
  collapse = "|"
)

pred[, has_metascience_method := has_pattern(text_lower, method_science_pattern)]

possible_method_domain <- pred[
  has_metascience_method == TRUE &
    !top1_area_name %in% c(
      "Social Sciences",
      "Computer Science",
      "Decision Sciences",
      "Business, Management and Accounting"
    ),
  .(
    doc_id,
    doc_id_v5,
    title,
    top1_area_name,
    top1_subarea_name,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    zero_shot_status,
    confidence_level,
    top1_prototype_language,
    text_for_embedding
  )
][order(top1_area_name, top1_top2_margin)]

# ============================================================
# 8) Casos críticos por patrones
# ============================================================

critical_pattern <- paste(
  c(
    "do journal data sharing mandates work",
    "data citation and reuse practice",
    "does data sharing influence data reuse",
    "coronavirus research before 2020",
    "data sharing",
    "data citation",
    "data reuse",
    "dryad",
    "covid",
    "coronavirus"
  ),
  collapse = "|"
)

critical_cases <- pred[
  has_pattern(text_lower, critical_pattern),
  .(
    doc_id,
    doc_id_v5,
    title,
    top1_area_name,
    top1_subarea_name,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    zero_shot_status,
    confidence_level,
    top1_prototype_language,
    has_metascience_method,
    has_virus_context,
    has_animal_context,
    text_for_embedding
  )
][order(title)]

# ============================================================
# 9) Muestra estratificada para validación manual
# ============================================================

set.seed(123)

validation_sample_main <- pred[
  ,
  .SD[sample(.N, min(.N, 10))],
  by = .(
    top1_area_name,
    confidence_level
  )
]

validation_sample_critical <- rbindlist(
  list(
    possible_biomed_veterinary[1:min(.N, 100)],
    possible_method_domain[1:min(.N, 100)],
    critical_cases[1:min(.N, 100)]
  ),
  fill = TRUE
)

validation_sample <- rbindlist(
  list(
    validation_sample_main,
    validation_sample_critical
  ),
  fill = TRUE
)

validation_sample <- unique(
  validation_sample,
  by = c("doc_id", "doc_id_v5")
)

validation_sample <- validation_sample[
  ,
  .(
    doc_id,
    doc_id_v5,
    title,
    top1_area_name,
    top1_subarea_name,
    top1_score,
    top2_area_name,
    top2_subarea_name,
    top2_score,
    top1_top2_margin,
    zero_shot_status,
    confidence_level,
    top1_prototype_language,
    has_metascience_method,
    has_virus_context,
    has_animal_context,
    text_for_embedding
  )
]

# ============================================================
# 10) Resúmenes específicos
# ============================================================

summary_biomed_veterinary <- data.table(
  metric = c(
    "possible_biomed_veterinary_conflicts",
    "veterinary_total",
    "veterinary_with_virus_context",
    "veterinary_with_virus_without_animal_context"
  ),
  value = c(
    nrow(possible_biomed_veterinary),
    pred[top1_area_name == "Veterinary", .N],
    pred[top1_area_name == "Veterinary" & has_virus_context == TRUE, .N],
    pred[top1_area_name == "Veterinary" & has_virus_context == TRUE & has_animal_context == FALSE, .N]
  )
)

summary_method_domain <- data.table(
  metric = c(
    "possible_method_domain_conflicts",
    "documents_with_metascience_method",
    "metascience_method_classified_social_sciences",
    "metascience_method_classified_computer_science",
    "metascience_method_classified_other"
  ),
  value = c(
    nrow(possible_method_domain),
    pred[has_metascience_method == TRUE, .N],
    pred[has_metascience_method == TRUE & top1_area_name == "Social Sciences", .N],
    pred[has_metascience_method == TRUE & top1_area_name == "Computer Science", .N],
    pred[has_metascience_method == TRUE &
           !top1_area_name %in% c("Social Sciences", "Computer Science"), .N]
  )
)

# ============================================================
# 11) Exportar
# ============================================================

fwrite(summary_global, out_summary_global)
fwrite(summary_area, out_summary_area)
fwrite(summary_confidence, out_summary_confidence)
fwrite(summary_status, out_summary_status)
fwrite(top1_top2_pairs, out_top1_top2_pairs)
fwrite(boundary_conflicts, out_boundary_conflicts)
fwrite(possible_biomed_veterinary, out_biomed_veterinary)
fwrite(possible_method_domain, out_method_domain)
fwrite(critical_cases, out_critical_cases)
fwrite(validation_sample, out_validation_sample)

fwrite(summary_biomed_veterinary, file.path(out_dir, "summary_biomed_veterinary_conflicts.csv"))
fwrite(summary_method_domain, file.path(out_dir, "summary_method_domain_conflicts.csv"))

# ============================================================
# 12) Imprimir
# ============================================================

cat("\n================ DIAGNÓSTICO ZERO-SHOT v5.2 ================\n")

cat("\nResumen global:\n")
print(summary_global)

cat("\nÁreas top 30:\n")
print(head(summary_area, 30))

cat("\nConfianza:\n")
print(summary_confidence)

cat("\nEstado zero-shot:\n")
print(summary_status)

cat("\nConflictos top1-top2 de bajo margen top 30:\n")
print(head(boundary_conflicts, 30))

cat("\nResumen posible conflicto biomédico/veterinaria:\n")
print(summary_biomed_veterinary)

cat("\nResumen posible conflicto método/dominio:\n")
print(summary_method_domain)

cat("\nCasos críticos:", nrow(critical_cases), "\n")
cat("Muestra de validación:", nrow(validation_sample), "\n")

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")