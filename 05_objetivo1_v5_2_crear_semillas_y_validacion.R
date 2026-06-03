# ============================================================
# 05_objetivo1_v5_2_crear_semillas_y_validacion.R
# Objetivo 1 v5.2
# Crear capa de decisión, semillas de entrenamiento y muestra
# de validación manual a partir del zero-shot multiprototipo.
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
  "decision_layer_v5_2"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(pred_file)) {
  stop("No existe zero_shot_v5_2_multiprototype_full.csv")
}

out_decision <- file.path(out_dir, "zero_shot_v5_2_decision_layer.csv")
out_seed_strict <- file.path(out_dir, "dataset_seed_strict_v5_2.csv")
out_seed_expanded <- file.path(out_dir, "dataset_seed_expanded_v5_2.csv")
out_review <- file.path(out_dir, "dataset_review_v5_2.csv")
out_validation <- file.path(out_dir, "manual_validation_sample_v5_2.csv")

out_summary_decision <- file.path(out_dir, "summary_decision_v5_2.csv")
out_summary_area_decision <- file.path(out_dir, "summary_area_decision_v5_2.csv")
out_summary_review_reason <- file.path(out_dir, "summary_review_reason_v5_2.csv")
out_summary_seed_area <- file.path(out_dir, "summary_seed_area_v5_2.csv")
out_summary_validation <- file.path(out_dir, "summary_validation_sample_v5_2.csv")

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

append_reason <- function(current, condition, reason) {
  out <- current
  
  idx_empty <- condition & (is.na(out) | out == "")
  out[idx_empty] <- reason
  
  idx_existing <- condition & !is.na(out) & out != "" & !grepl(reason, out, fixed = TRUE)
  out[idx_existing] <- paste(out[idx_existing], reason, sep = "; ")
  
  out
}

sample_n_safe <- function(dt, n) {
  if (nrow(dt) == 0) return(dt)
  dt[sample(.N, min(.N, n))]
}

# ============================================================
# 3) Leer predicciones
# ============================================================

dt <- fread(pred_file)
setDT(dt)

cat("Documentos leídos:", nrow(dt), "\n")

needed_cols <- c(
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
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "top1_prototype_language",
  "top1_prototype_id",
  "top1_prototype_type"
)

for (col in needed_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := NA]
  }
}

dt[, title := clean_text(title)]
dt[, text_for_embedding := clean_text(text_for_embedding)]
dt[, top1_score := safe_numeric(top1_score)]
dt[, top2_score := safe_numeric(top2_score)]
dt[, top1_top2_margin := safe_numeric(top1_top2_margin)]

dt[, text_lower := tolower(paste(title, text_for_embedding))]

# ============================================================
# 4) Patrones de conflicto
# ============================================================

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

metascience_pattern <- paste(
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

dt[, has_virus_context := has_pattern(text_lower, virus_pattern)]
dt[, has_animal_context := has_pattern(text_lower, animal_context_pattern)]
dt[, has_metascience_method := has_pattern(text_lower, metascience_pattern)]

dt[, possible_biomed_veterinary_conflict :=
     top1_area_name == "Veterinary" &
     has_virus_context == TRUE &
     has_animal_context == FALSE]

dt[, possible_method_domain_conflict :=
     has_metascience_method == TRUE &
     !top1_area_name %in% c(
       "Social Sciences",
       "Computer Science",
       "Decision Sciences",
       "Business, Management and Accounting"
     )]

dt[, low_margin_cross_area :=
     top1_area_name != top2_area_name &
     !is.na(top1_top2_margin) &
     top1_top2_margin < 0.05]

dt[, same_area_low_margin :=
     top1_area_name == top2_area_name &
     !is.na(top1_top2_margin) &
     top1_top2_margin < 0.03]

# ============================================================
# 5) Capa de decisión
# ============================================================

dt[, review_reason := ""]

dt[, review_reason := append_reason(
  review_reason,
  zero_shot_status == "ambiguous_review",
  "zero_shot_ambiguous"
)]

dt[, review_reason := append_reason(
  review_reason,
  zero_shot_status == "low_similarity_review",
  "zero_shot_low_similarity"
)]

dt[, review_reason := append_reason(
  review_reason,
  confidence_level %in% c("revisar", "baja_similitud"),
  "low_confidence_level"
)]

dt[, review_reason := append_reason(
  review_reason,
  possible_biomed_veterinary_conflict == TRUE,
  "possible_biomed_veterinary_conflict"
)]

dt[, review_reason := append_reason(
  review_reason,
  possible_method_domain_conflict == TRUE,
  "possible_method_domain_conflict"
)]

dt[, review_reason := append_reason(
  review_reason,
  low_margin_cross_area == TRUE,
  "low_margin_cross_area"
)]

dt[, review_reason := append_reason(
  review_reason,
  same_area_low_margin == TRUE,
  "same_area_low_margin_subarea"
)]

dt[review_reason == "" | is.na(review_reason), review_reason := "sin_revision_obligatoria"]

# ------------------------------------------------------------
# Decisiones de uso
# ------------------------------------------------------------

dt[, decision_layer := fcase(
  
  zero_shot_status == "preliminary_label" &
    confidence_level == "alta_confianza" &
    !is.na(top1_score) &
    !is.na(top1_top2_margin) &
    top1_score >= 0.70 &
    top1_top2_margin >= 0.07 &
    possible_biomed_veterinary_conflict == FALSE &
    possible_method_domain_conflict == FALSE,
  "seed_strict_high_precision",
  
  zero_shot_status == "preliminary_label" &
    confidence_level %in% c("alta_confianza", "confianza_media") &
    !is.na(top1_score) &
    !is.na(top1_top2_margin) &
    top1_score >= 0.65 &
    top1_top2_margin >= 0.05 &
    possible_biomed_veterinary_conflict == FALSE &
    possible_method_domain_conflict == FALSE,
  "seed_expanded_with_caution",
  
  zero_shot_status == "preliminary_label" &
    confidence_level == "confianza_media" &
    !is.na(top1_score) &
    top1_score >= 0.60 &
    possible_biomed_veterinary_conflict == FALSE &
    possible_method_domain_conflict == FALSE,
  "candidate_review_light",
  
  default = "manual_or_assisted_review"
)]

dt[, decision_strength := fcase(
  
  decision_layer == "seed_strict_high_precision",
  "alta_precision",
  
  decision_layer == "seed_expanded_with_caution",
  "precision_media_con_cautela",
  
  decision_layer == "candidate_review_light",
  "candidato_revisable",
  
  decision_layer == "manual_or_assisted_review",
  "revision_asistida",
  
  default = "revision_asistida"
)]

dt[, accepted_for_training_strict := decision_layer == "seed_strict_high_precision"]

dt[, accepted_for_training_expanded := decision_layer %in% c(
  "seed_strict_high_precision",
  "seed_expanded_with_caution"
)]

dt[, needs_manual_validation := decision_layer %in% c(
  "candidate_review_light",
  "manual_or_assisted_review"
)]

# ============================================================
# 6) Crear datasets
# ============================================================

common_cols <- c(
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
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "top1_prototype_language",
  "top1_prototype_id",
  "top1_prototype_type",
  "has_virus_context",
  "has_animal_context",
  "has_metascience_method",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "low_margin_cross_area",
  "same_area_low_margin",
  "decision_layer",
  "decision_strength",
  "review_reason",
  "accepted_for_training_strict",
  "accepted_for_training_expanded",
  "needs_manual_validation"
)

for (col in common_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := NA]
  }
}

decision_out <- dt[, ..common_cols]

seed_strict <- decision_out[
  accepted_for_training_strict == TRUE
]

seed_expanded <- decision_out[
  accepted_for_training_expanded == TRUE
]

review_set <- decision_out[
  needs_manual_validation == TRUE |
    review_reason != "sin_revision_obligatoria"
]

# ============================================================
# 7) Muestra de validación manual
# ============================================================

set.seed(123)

# 7.1 Muestra de semillas estrictas por área para estimar precisión
sample_strict <- seed_strict[
  ,
  .SD[sample(.N, min(.N, 25))],
  by = top1_area_name
]

sample_strict[, validation_group := "01_seed_strict_by_area"]

# 7.2 Muestra de semilla expandida por área
sample_expanded <- seed_expanded[
  decision_layer == "seed_expanded_with_caution",
  .SD[sample(.N, min(.N, 25))],
  by = top1_area_name
]

sample_expanded[, validation_group := "02_seed_expanded_by_area"]

# 7.3 Ambiguos frontera top1-top2
sample_boundary <- decision_out[
  low_margin_cross_area == TRUE,
  .SD[sample(.N, min(.N, 10))],
  by = .(top1_area_name, top2_area_name)
]

sample_boundary[, validation_group := "03_boundary_top1_top2"]

# 7.4 Conflicto biomédico/veterinaria
sample_biomed_vet <- sample_n_safe(
  decision_out[possible_biomed_veterinary_conflict == TRUE],
  300
)

sample_biomed_vet[, validation_group := "04_possible_biomed_veterinary_conflict"]

# 7.5 Conflicto método/dominio
sample_method_domain <- sample_n_safe(
  decision_out[possible_method_domain_conflict == TRUE],
  400
)

sample_method_domain[, validation_group := "05_possible_method_domain_conflict"]

# 7.6 Baja similitud
sample_low_similarity <- sample_n_safe(
  decision_out[zero_shot_status == "low_similarity_review"],
  200
)

sample_low_similarity[, validation_group := "06_low_similarity"]

validation_sample <- rbindlist(
  list(
    sample_strict,
    sample_expanded,
    sample_boundary,
    sample_biomed_vet,
    sample_method_domain,
    sample_low_similarity
  ),
  fill = TRUE
)

validation_sample <- unique(
  validation_sample,
  by = c("doc_id", "doc_id_v5")
)

# Columnas para revisión manual
validation_sample[, manual_area_label := ""]
validation_sample[, manual_subarea_label := ""]
validation_sample[, manual_decision := ""]
validation_sample[, manual_comment := ""]

validation_cols <- c(
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
  "has_metascience_method",
  "has_virus_context",
  "has_animal_context",
  "possible_biomed_veterinary_conflict",
  "possible_method_domain_conflict",
  "text_for_embedding",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

validation_sample <- validation_sample[, ..validation_cols]

# ============================================================
# 8) Resúmenes
# ============================================================

summary_decision <- decision_out[
  ,
  .N,
  by = .(
    decision_layer,
    decision_strength
  )
][order(-N)]

summary_decision[, pct := round(N / sum(N) * 100, 2)]

summary_area_decision <- decision_out[
  ,
  .N,
  by = .(
    top1_area_name,
    decision_layer
  )
][order(top1_area_name, -N)]

summary_area_decision[
  ,
  pct_within_area := round(N / sum(N) * 100, 2),
  by = top1_area_name
]

summary_review_reason <- decision_out[
  ,
  .N,
  by = review_reason
][order(-N)]

summary_review_reason[, pct := round(N / sum(N) * 100, 2)]

summary_seed_area <- seed_strict[
  ,
  .N,
  by = top1_area_name
][order(-N)]

summary_seed_area[, pct := round(N / sum(N) * 100, 2)]

summary_validation <- validation_sample[
  ,
  .N,
  by = validation_group
][order(validation_group)]

summary_validation[, pct := round(N / sum(N) * 100, 2)]

# ============================================================
# 9) Exportar
# ============================================================

fwrite(decision_out, out_decision)
fwrite(seed_strict, out_seed_strict)
fwrite(seed_expanded, out_seed_expanded)
fwrite(review_set, out_review)
fwrite(validation_sample, out_validation)

fwrite(summary_decision, out_summary_decision)
fwrite(summary_area_decision, out_summary_area_decision)
fwrite(summary_review_reason, out_summary_review_reason)
fwrite(summary_seed_area, out_summary_seed_area)
fwrite(summary_validation, out_summary_validation)

# ============================================================
# 10) Imprimir
# ============================================================

cat("\n================ CAPA DE DECISIÓN v5.2 ================\n")

cat("\nResumen de decisión:\n")
print(summary_decision)

cat("\nSemilla estricta por área:\n")
print(head(summary_seed_area, 30))

cat("\nRazones de revisión:\n")
print(head(summary_review_reason, 30))

cat("\nMuestra de validación:\n")
print(summary_validation)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")

cat("\nTamaños:\n")
cat("decision_out:", nrow(decision_out), "\n")
cat("seed_strict:", nrow(seed_strict), "\n")
cat("seed_expanded:", nrow(seed_expanded), "\n")
cat("review_set:", nrow(review_set), "\n")
cat("validation_sample:", nrow(validation_sample), "\n")