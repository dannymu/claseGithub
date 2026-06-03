# ============================================================
# 06_objetivo1_v5_2_muestra_validacion_balanceada.R
# Objetivo 1 v5.2
# Crear muestra equilibrada y manejable para validación manual
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

decision_file <- file.path(
  decision_dir,
  "zero_shot_v5_2_decision_layer.csv"
)

if (!file.exists(decision_file)) {
  stop("No existe zero_shot_v5_2_decision_layer.csv. Ejecuta primero el script 05.")
}

out_validation_balanced <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2.csv"
)

out_summary_balanced <- file.path(
  decision_dir,
  "summary_manual_validation_sample_balanced_v5_2.csv"
)

out_summary_area_balanced <- file.path(
  decision_dir,
  "summary_manual_validation_area_balanced_v5_2.csv"
)

# ============================================================
# 2) Funciones
# ============================================================

sample_n_safe <- function(dt, n) {
  if (nrow(dt) == 0) return(dt)
  dt[sample(.N, min(.N, n))]
}

# ============================================================
# 3) Leer capa de decisión
# ============================================================

dt <- fread(decision_file)
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

for (col in needed_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := NA]
  }
}

# ============================================================
# 4) Crear submuestras
# ============================================================

set.seed(123)

# 4.1 Semilla estricta: 20 por área
sample_seed_strict <- dt[
  decision_layer == "seed_strict_high_precision",
  .SD[sample(.N, min(.N, 20))],
  by = top1_area_name
]

sample_seed_strict[, validation_group := "01_seed_strict_by_area"]

# 4.2 Semilla expandida: 20 por área
sample_seed_expanded <- dt[
  decision_layer == "seed_expanded_with_caution",
  .SD[sample(.N, min(.N, 20))],
  by = top1_area_name
]

sample_seed_expanded[, validation_group := "02_seed_expanded_by_area"]

# 4.3 Candidate review light: 10 por área
sample_candidate_light <- dt[
  decision_layer == "candidate_review_light",
  .SD[sample(.N, min(.N, 10))],
  by = top1_area_name
]

sample_candidate_light[, validation_group := "03_candidate_review_light_by_area"]

# 4.4 Pares frontera top1-top2 más frecuentes
boundary_pairs <- dt[
  low_margin_cross_area == TRUE,
  .N,
  by = .(
    top1_area_name,
    top2_area_name
  )
][order(-N)]

top_boundary_pairs <- boundary_pairs[1:min(.N, 50)]

sample_boundary <- merge(
  dt[low_margin_cross_area == TRUE],
  top_boundary_pairs[
    ,
    .(
      top1_area_name,
      top2_area_name
    )
  ],
  by = c("top1_area_name", "top2_area_name"),
  all = FALSE
)

sample_boundary <- sample_boundary[
  ,
  .SD[sample(.N, min(.N, 8))],
  by = .(
    top1_area_name,
    top2_area_name
  )
]

sample_boundary[, validation_group := "04_boundary_top1_top2"]

# 4.5 Conflicto biomédico/veterinaria
sample_biomed_vet <- sample_n_safe(
  dt[possible_biomed_veterinary_conflict == TRUE],
  200
)

sample_biomed_vet[, validation_group := "05_possible_biomed_veterinary_conflict"]

# 4.6 Conflicto método/dominio
sample_method_domain <- sample_n_safe(
  dt[possible_method_domain_conflict == TRUE],
  250
)

sample_method_domain[, validation_group := "06_possible_method_domain_conflict"]

# 4.7 Baja similitud
sample_low_similarity <- sample_n_safe(
  dt[zero_shot_status == "low_similarity_review"],
  150
)

sample_low_similarity[, validation_group := "07_low_similarity"]

# ============================================================
# 5) Consolidar muestra
# ============================================================

validation_sample <- rbindlist(
  list(
    sample_seed_strict,
    sample_seed_expanded,
    sample_candidate_light,
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

# ============================================================
# 6) Añadir columnas de validación manual
# ============================================================

validation_sample[, manual_area_label := ""]
validation_sample[, manual_subarea_label := ""]
validation_sample[, manual_area_correct := ""]
validation_sample[, manual_subarea_correct := ""]
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
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

validation_sample <- validation_sample[, ..validation_cols]

# ============================================================
# 7) Resúmenes
# ============================================================

summary_balanced <- validation_sample[
  ,
  .N,
  by = validation_group
][order(validation_group)]

summary_balanced[, pct := round(N / sum(N) * 100, 2)]

summary_area_balanced <- validation_sample[
  ,
  .N,
  by = .(
    validation_group,
    top1_area_name
  )
][order(validation_group, top1_area_name)]

# ============================================================
# 8) Exportar
# ============================================================

fwrite(validation_sample, out_validation_balanced)
fwrite(summary_balanced, out_summary_balanced)
fwrite(summary_area_balanced, out_summary_area_balanced)

cat("\n================ MUESTRA BALANCEADA DE VALIDACIÓN v5.2 ================\n")
cat("Total muestra:", nrow(validation_sample), "\n")

cat("\nResumen por grupo:\n")
print(summary_balanced)

cat("\nArchivo creado:\n")
cat(out_validation_balanced, "\n")