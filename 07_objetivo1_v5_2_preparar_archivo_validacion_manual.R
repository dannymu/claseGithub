# ============================================================
# 07_objetivo1_v5_2_preparar_archivo_validacion_manual.R
# Objetivo 1 v5.2
# Preparar archivo de validación manual con campos claros
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")
embeddings_dir <- file.path(base_dir, "embeddings")

decision_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_zero_shot",
  "decision_layer_v5_2"
)

validation_file <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2.csv"
)

pred_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_zero_shot",
  "zero_shot_v5_2_multiprototype_full.csv"
)

taxonomy_file <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings_meta_fixed.csv"
)

out_csv <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2_for_review.csv"
)

out_taxonomy <- file.path(
  decision_dir,
  "taxonomy_area_subarea_for_manual_review.csv"
)

out_xlsx <- file.path(
  decision_dir,
  "manual_validation_sample_balanced_v5_2_for_review.xlsx"
)

if (!file.exists(validation_file)) {
  stop("No existe manual_validation_sample_balanced_v5_2.csv")
}

if (!file.exists(pred_file)) {
  stop("No existe zero_shot_v5_2_multiprototype_full.csv")
}

if (!file.exists(taxonomy_file)) {
  stop("No existe taxonomía v4")
}

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

short_text <- function(x, n = 1200) {
  x <- clean_text(x)
  ifelse(nchar(x) > n, paste0(substr(x, 1, n), " ..."), x)
}

# ============================================================
# 1) Leer muestra, predicciones completas y taxonomía
# ============================================================

val <- fread(validation_file)
pred <- fread(pred_file)
tax <- fread(taxonomy_file)

setDT(val)
setDT(pred)
setDT(tax)

# ============================================================
# 2) Añadir top3-top5 desde pred_full
# ============================================================

extra_cols <- c(
  "doc_id",
  "doc_id_v5",
  "top3_area_name",
  "top3_subarea_name",
  "top3_score",
  "top4_area_name",
  "top4_subarea_name",
  "top4_score",
  "top5_area_name",
  "top5_subarea_name",
  "top5_score"
)

extra_cols <- extra_cols[extra_cols %in% names(pred)]

pred_extra <- pred[, ..extra_cols]

val2 <- merge(
  val,
  pred_extra,
  by = c("doc_id", "doc_id_v5"),
  all.x = TRUE
)

# ============================================================
# 3) Preparar taxonomía única área/subárea
# ============================================================

area_col <- if ("area_name" %in% names(tax)) "area_name" else "area_label"
subarea_col <- if ("subarea_name" %in% names(tax)) "subarea_name" else "subarea_label"

taxonomy_review <- unique(
  tax[
    ,
    .(
      area_label = clean_text(get(area_col)),
      subarea_label = clean_text(get(subarea_col))
    )
  ]
)

taxonomy_review <- taxonomy_review[
  area_label != "" &
    subarea_label != ""
][order(area_label, subarea_label)]

fwrite(taxonomy_review, out_taxonomy)

# ============================================================
# 4) Crear archivo de validación manual
# ============================================================

val2[, title := clean_text(title)]
val2[, text_for_embedding_short := short_text(text_for_embedding, 1400)]

val2[, suggested_top1 := paste0(top1_area_name, " / ", top1_subarea_name)]
val2[, suggested_top2 := paste0(top2_area_name, " / ", top2_subarea_name)]

if ("top3_area_name" %in% names(val2)) {
  val2[, suggested_top3 := paste0(top3_area_name, " / ", top3_subarea_name)]
} else {
  val2[, suggested_top3 := ""]
}

if ("top4_area_name" %in% names(val2)) {
  val2[, suggested_top4 := paste0(top4_area_name, " / ", top4_subarea_name)]
} else {
  val2[, suggested_top4 := ""]
}

if ("top5_area_name" %in% names(val2)) {
  val2[, suggested_top5 := paste0(top5_area_name, " / ", top5_subarea_name)]
} else {
  val2[, suggested_top5 := ""]
}

val2[, manual_area_correct := ""]
val2[, manual_subarea_correct := ""]
val2[, manual_area_label := ""]
val2[, manual_subarea_label := ""]
val2[, manual_decision := ""]
val2[, manual_comment := ""]

# Decisiones sugeridas para llenar manual_decision:
# correct
# area_correct_subarea_wrong
# area_wrong
# insufficient_information
# ambiguous_or_both_valid

review_cols <- c(
  "validation_group",
  "doc_id",
  "doc_id_v5",
  "work_id",
  "title",
  "suggested_top1",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "suggested_top2",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "suggested_top3",
  "top3_score",
  "suggested_top4",
  "top4_score",
  "suggested_top5",
  "top5_score",
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
  "text_for_embedding_short",
  "manual_area_correct",
  "manual_subarea_correct",
  "manual_area_label",
  "manual_subarea_label",
  "manual_decision",
  "manual_comment"
)

for (col in review_cols) {
  if (!col %in% names(val2)) {
    val2[, (col) := ""]
  }
}

review_dt <- val2[, ..review_cols]

setorder(review_dt, validation_group, top1_area_name, title)

fwrite(review_dt, out_csv)

# ============================================================
# 5) Crear XLSX si openxlsx está disponible
# ============================================================

if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  wb <- openxlsx::createWorkbook()
  
  openxlsx::addWorksheet(wb, "validacion")
  openxlsx::addWorksheet(wb, "taxonomia")
  openxlsx::addWorksheet(wb, "instrucciones")
  
  openxlsx::writeData(wb, "validacion", review_dt)
  openxlsx::writeData(wb, "taxonomia", taxonomy_review)
  
  instrucciones <- data.table(
    campo = c(
      "manual_area_correct",
      "manual_subarea_correct",
      "manual_area_label",
      "manual_subarea_label",
      "manual_decision",
      "manual_comment"
    ),
    instruccion = c(
      "Escribir TRUE si el área top1 es correcta; FALSE si es incorrecta.",
      "Escribir TRUE si la subárea top1 es correcta; FALSE si es incorrecta.",
      "Si el área top1 es incorrecta, escribir el área correcta según taxonomía.",
      "Si la subárea top1 es incorrecta, escribir la subárea correcta según taxonomía.",
      "Usar: correct, area_correct_subarea_wrong, area_wrong, insufficient_information, ambiguous_or_both_valid.",
      "Agregar observación breve cuando sea necesario."
    )
  )
  
  openxlsx::writeData(wb, "instrucciones", instrucciones)
  
  openxlsx::freezePane(wb, "validacion", firstActiveRow = 2, firstActiveCol = 6)
  openxlsx::setColWidths(wb, "validacion", cols = 1:ncol(review_dt), widths = "auto")
  openxlsx::setColWidths(wb, "validacion", cols = which(names(review_dt) == "text_for_embedding_short"), widths = 80)
  
  openxlsx::saveWorkbook(wb, out_xlsx, overwrite = TRUE)
  
  cat("\nXLSX creado:\n")
  cat(out_xlsx, "\n")
  
} else {
  cat("\nPaquete openxlsx no instalado. Se creó solo CSV.\n")
}

cat("\n================ ARCHIVO DE VALIDACIÓN MANUAL ================\n")
cat("Filas:", nrow(review_dt), "\n")
cat("CSV:", out_csv, "\n")
cat("Taxonomía:", out_taxonomy, "\n")

cat("\nDistribución por grupo:\n")
print(review_dt[, .N, by = validation_group][order(validation_group)])