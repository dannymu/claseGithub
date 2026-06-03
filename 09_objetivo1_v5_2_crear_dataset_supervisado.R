# ============================================================
# 09_objetivo1_v5_2_crear_dataset_supervisado.R
# Objetivo 1 v5.2
# Crear dataset supervisado a partir de seed_strict + seed_expanded
# validado manualmente
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

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

decision_file <- file.path(
  decision_dir,
  "zero_shot_v5_2_decision_layer.csv"
)

manual_validation_file <- file.path(
  validation_dir,
  "manual_validation_clean_v5_2.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_supervised_dataset"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_training <- file.path(out_dir, "dataset_supervised_train_v5_2.csv")
out_manual_test <- file.path(out_dir, "dataset_manual_test_v5_2.csv")
out_all_labeled <- file.path(out_dir, "dataset_supervised_all_labeled_v5_2.csv")

out_summary_training <- file.path(out_dir, "summary_training_dataset_v5_2.csv")
out_summary_area <- file.path(out_dir, "summary_training_area_v5_2.csv")
out_summary_subarea <- file.path(out_dir, "summary_training_subarea_v5_2.csv")
out_summary_manual <- file.path(out_dir, "summary_manual_test_v5_2.csv")

if (!file.exists(decision_file)) {
  stop("No existe zero_shot_v5_2_decision_layer.csv")
}

if (!file.exists(manual_validation_file)) {
  stop("No existe manual_validation_clean_v5_2.csv. Ejecuta primero el script 08.")
}

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

# ============================================================
# 1) Leer datos
# ============================================================

decision <- fread(decision_file)
manual <- fread(manual_validation_file)

setDT(decision)
setDT(manual)

cat("Decision layer:", nrow(decision), "\n")
cat("Manual validation:", nrow(manual), "\n")

# ============================================================
# 2) Preparar semilla supervisada
# ============================================================

decision[, top1_area_name := clean_text(top1_area_name)]
decision[, top1_subarea_name := clean_text(top1_subarea_name)]
decision[, decision_layer := clean_text(decision_layer)]
decision[, top1_score := safe_numeric(top1_score)]
decision[, top1_top2_margin := safe_numeric(top1_top2_margin)]

seed <- decision[
  decision_layer %in% c(
    "seed_strict_high_precision",
    "seed_expanded_with_caution"
  )
]

seed[, supervised_label_source := fifelse(
  decision_layer == "seed_strict_high_precision",
  "zero_shot_v5_2_seed_strict_validated",
  "zero_shot_v5_2_seed_expanded_validated"
)]

seed[, sample_weight := fifelse(
  decision_layer == "seed_strict_high_precision",
  1.00,
  0.85
)]

seed[, label_area := top1_area_name]
seed[, label_subarea := top1_subarea_name]

seed[, label_confidence_group := fifelse(
  decision_layer == "seed_strict_high_precision",
  "high_precision_seed",
  "expanded_seed"
)]

# ============================================================
# 3) Excluir muestra manual del entrenamiento
# ============================================================

manual_ids <- unique(manual[, .(doc_id, doc_id_v5)])

seed[, in_manual_validation := paste(doc_id, doc_id_v5) %in%
       paste(manual_ids$doc_id, manual_ids$doc_id_v5)]

train <- seed[in_manual_validation == FALSE]

heldout_from_seed <- seed[in_manual_validation == TRUE]

# ============================================================
# 4) Crear test manual con etiquetas corregidas
# ============================================================

manual[, manual_decision_clean := clean_text(manual_decision_clean)]
manual[, top1_area_name := clean_text(top1_area_name)]
manual[, top1_subarea_name := clean_text(top1_subarea_name)]
manual[, corrected_area_label := clean_text(corrected_area_label)]
manual[, corrected_subarea_label := clean_text(corrected_subarea_label)]

manual_test <- manual[
  manual_decision_clean != "insufficient_information" &
    corrected_area_label != "" &
    corrected_subarea_label != ""
]

manual_test[, label_area := corrected_area_label]
manual_test[, label_subarea := corrected_subarea_label]
manual_test[, supervised_label_source := "manual_validation_v5_2"]
manual_test[, sample_weight := 1.00]
manual_test[, label_confidence_group := "manual_gold"]

# ============================================================
# 5) Columnas finales
# ============================================================

final_cols <- c(
  "doc_id",
  "doc_id_v5",
  "work_id",
  "title",
  "text_for_embedding",
  "label_area",
  "label_subarea",
  "supervised_label_source",
  "sample_weight",
  "label_confidence_group",
  "decision_layer",
  "decision_strength",
  "top1_score",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "top1_prototype_language",
  "review_reason"
)

for (col in final_cols) {
  if (!col %in% names(train)) train[, (col) := NA]
  if (!col %in% names(manual_test)) manual_test[, (col) := NA]
}

train_out <- train[, ..final_cols]
manual_test_out <- manual_test[, ..final_cols]

all_labeled <- rbindlist(
  list(
    train_out,
    manual_test_out
  ),
  fill = TRUE
)

# ============================================================
# 6) Resúmenes
# ============================================================

summary_training <- data.table(
  metric = c(
    "seed_total_strict_plus_expanded",
    "excluded_because_manual_validation",
    "training_rows",
    "manual_test_rows",
    "all_labeled_rows",
    "strict_training_rows",
    "expanded_training_rows"
  ),
  value = c(
    nrow(seed),
    nrow(heldout_from_seed),
    nrow(train_out),
    nrow(manual_test_out),
    nrow(all_labeled),
    train_out[supervised_label_source == "zero_shot_v5_2_seed_strict_validated", .N],
    train_out[supervised_label_source == "zero_shot_v5_2_seed_expanded_validated", .N]
  )
)

summary_area <- train_out[
  ,
  .N,
  by = .(
    label_area,
    supervised_label_source
  )
][order(label_area, -N)]

summary_area[
  ,
  pct_within_source := round(N / sum(N) * 100, 2),
  by = supervised_label_source
]

summary_subarea <- train_out[
  ,
  .N,
  by = .(
    label_area,
    label_subarea,
    supervised_label_source
  )
][order(label_area, label_subarea, -N)]

summary_manual <- manual_test_out[
  ,
  .N,
  by = .(
    label_area,
    label_subarea
  )
][order(label_area, label_subarea)]

# ============================================================
# 7) Exportar
# ============================================================

fwrite(train_out, out_training)
fwrite(manual_test_out, out_manual_test)
fwrite(all_labeled, out_all_labeled)

fwrite(summary_training, out_summary_training)
fwrite(summary_area, out_summary_area)
fwrite(summary_subarea, out_summary_subarea)
fwrite(summary_manual, out_summary_manual)

cat("\n================ DATASET SUPERVISADO v5.2 ================\n")

cat("\nResumen entrenamiento:\n")
print(summary_training)

cat("\nÁreas entrenamiento:\n")
print(head(summary_area, 40))

cat("\nManual test:\n")
print(head(summary_manual, 40))

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")