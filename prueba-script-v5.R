saveRDS(
  dt_documents_All_GS_OpenA,
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/dt_documents_All_GS_OpenA.rds"
)


source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/01_objetivo1_v5_preparar_dataset_enriquecido_GS_OpenAlex.R")
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/01b_objetivo1_v5_universo_docs_interest_enriquecido.R")
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/02_objetivo1_v5_generar_embeddings_specter2.R")


summary_enrichment
summary_quality
summary_match_method
cat("Registros problemáticos:", nrow(problematic_records), "\n")


meta_out <- fread("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/embeddings/specter2_classification_v5_docs_interest_enriched_sample_1000/docs_embeddings_meta.csv")
nrow(meta_out)
head(meta_out)


# ===================================
library(data.table)

v5_file <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/objetivo1_v5_docs_interest_enriched/dataset_objetivo1_v5_docs_interest_for_embeddings.csv"

dt_v5 <- fread(v5_file)

cases_check <- dt_v5[
  grepl(
    "data sharing|data citation|data reuse|dryad|biodiversity|coronavirus|covid",
    text_for_embedding,
    ignore.case = TRUE
  ),
  .(
    doc_id_v5,
    work_id,
    title,
    year,
    source,
    text_quality_v5,
    abstract = substr(abstract, 1, 400),
    keyword = substr(keyword, 1, 300),
    conceptos = substr(conceptos, 1, 300),
    topics = substr(topics, 1, 300),
    text_for_embedding = substr(text_for_embedding, 1, 800)
  )
]

print(cases_check[1:min(.N, 30)])
cat("Casos encontrados:", nrow(cases_check), "\n")

# 3========================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/01c_objetivo1_v5_limpiar_text_for_embedding.R")

summary_compare
summary_quality_clean
cat("Casos de revisión temática encontrados:", nrow(check_cases), "\n")


check_cases[
  grepl(
    "data sharing|data citation|data reuse|dryad|coronavirus research before 2020",
    title,
    ignore.case = TRUE
  ),
  .(
    title,
    source,
    keyword_clean,
    conceptos_clean,
    topics_clean,
    clean
  )
]

# ==========================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/01d_objetivo1_v5_2_embedding_text_sin_concepts.R")

summary_v52
summary_quality_v52
cat("Casos de revisión temática v5.2:", nrow(check_cases_v52), "\n")

check_cases_v52[
  grepl(
    "data sharing|data citation|data reuse|dryad|coronavirus research before 2020",
    title,
    ignore.case = TRUE
  ),
  .(
    title,
    source,
    keyword_clean,
    topics_clean,
    conceptos_clean,
    text_v5_2
  )
]

# =====================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/02b_objetivo1_v5_2_generar_embeddings_specter2.R")

# al terminar mostrar
meta_out <- fread("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/embeddings/specter2_classification_v5_2_docs_interest_no_concepts_sample_1000/docs_embeddings_meta.csv")
nrow(meta_out)
head(meta_out)


python_bin <- path.expand("~/env_cluster/bin/python")

emb_file <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/embeddings/specter2_classification_v5_2_docs_interest_no_concepts_sample_1000/docs_embeddings.npy"

system2(
  python_bin,
  args = c(
    "-c",
    shQuote(paste0(
      "import numpy as np\n",
      "x=np.load(r'", emb_file, "')\n",
      "print(x.shape)\n"
    ))
  ),
  stdout = TRUE,
  stderr = TRUE
)

# ===============================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/02b_objetivo1_v5_2_generar_embeddings_specter2.R")
# al terminar
meta_full <- fread("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/embeddings/specter2_classification_v5_2_docs_interest_no_concepts/docs_embeddings_meta.csv")
nrow(meta_full)
head(meta_full)

# ================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/03_objetivo1_v5_2_zero_shot_multiprototype.R")
nrow(pred)
head(pred)
pred[, .N, by = top1_area_name][order(-N)][1:30]
pred[, .N, by = top1_prototype_language][order(-N)]

# ================

pred_full <- fread(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/objetivo1_v5_2_zero_shot/zero_shot_v5_2_multiprototype_full.csv"
)

nrow(pred_full)

summary_area_v52 <- pred_full[
  ,
  .N,
  by = top1_area_name
][order(-N)]

summary_area_v52[
  ,
  pct := round(N / sum(N) * 100, 2)
]

head(summary_area_v52, 30)

pred_full[
  ,
  .N,
  by = top1_prototype_language
][order(-N)]

pred_full[
  ,
  .N,
  by = confidence_level
][order(-N)]

pred_full[
  ,
  .N,
  by = zero_shot_status
][order(-N)]


pred_full[
  grepl(
    "data sharing|data citation|data reuse|dryad|coronavirus research before 2020|covid",
    text_for_embedding,
    ignore.case = TRUE
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
][1:50]

# ==============================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/04_objetivo1_v5_2_diagnostico_zero_shot_full.R")

summary_global
head(summary_area, 30)
summary_confidence
summary_status
head(boundary_conflicts, 30)
summary_biomed_veterinary
summary_method_domain
cat("Casos críticos:", nrow(critical_cases), "\n")
cat("Muestra de validación:", nrow(validation_sample), "\n")

# ==============================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/05_objetivo1_v5_2_crear_semillas_y_validacion.R")

summary_decision
head(summary_seed_area, 30)
head(summary_review_reason, 30)
summary_validation

cat("seed_strict:", nrow(seed_strict), "\n")
cat("seed_expanded:", nrow(seed_expanded), "\n")
cat("review_set:", nrow(review_set), "\n")
cat("validation_sample:", nrow(validation_sample), "\n")

# ======================================================================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/06_objetivo1_v5_2_muestra_validacion_balanceada.R")
summary_balanced
cat("Total muestra:", nrow(validation_sample), "\n")

# ====================================================================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/07_objetivo1_v5_2_preparar_archivo_validacion_manual.R")

# ===================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/08_objetivo1_v5_2_evaluar_validacion_manual.R")

summary_completeness
summary_overall
summary_group
summary_decision_layer
summary_manual_decision
head(area_confusion, 30)
training_recommendation

# ==================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/09_objetivo1_v5_2_crear_dataset_supervisado.R")

summary_training
head(summary_area, 40)


# ====================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/10_objetivo1_v5_2_train_supervised_models.R")

summary_supervised <- fread(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/objetivo1_v5_2_supervised_models/summary_supervised_v5_2.csv"
)
summary_supervised

# =================================
area_confusion <- fread(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/objetivo1_v5_2_supervised_models/confusion_area_supervised_v5_2.csv"
)
head(area_confusion, 30)

# 3 ====================================================================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/11_objetivo1_v5_2_comparar_zero_shot_supervisado.R")
summary_overall
summary_group
summary_decision
summary_conflict
summary_improvement_area

cat("Casos zero correcto y supervisado incorrecto:", nrow(zero_correct_supervised_wrong), "\n")
cat("Casos zero incorrecto y supervisado correcto:", nrow(zero_wrong_supervised_correct), "\n")

# 3 =========================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/12_objetivo1_v5_2_crear_salida_final_integrada.R")

summary_final
head(summary_area, 40)
head(summary_review, 30)
summary_conflicts

cat("Total documentos:", nrow(final_dt), "\n")
cat("Aceptados automáticamente:", nrow(auto_dt), "\n")
cat("Requieren revisión:", nrow(review_dt), "\n")
cat("Conflictos críticos:", nrow(conflict_dt), "\n")

# 3 =========================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/13_objetivo1_v5_2_cerrar_universo_final.R")

summary_universe
summary_decision
summary_coverage
head(summary_area_auto, 40)

# 3 =========================================================================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/13b_objetivo1_v5_2_corregir_universo_984318.R")

summary_corrected
summary_decision_corrected

# ========================================================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/14_objetivo1_v5_2_validacion_perfiles_catalogo_matching.R")

head(summary_profiles[order(-matched_publications)], 30)

candidate_profiles[
  ,
  .(
    profile_option,
    researcher_name,
    profile_file,
    profile_publications,
    matched_publications,
    match_rate_pct,
    matched_auto_accept,
    auto_accept_pct_of_matched,
    matched_review_required,
    dominant_area,
    dominant_subarea,
    dominant_area_pct_auto,
    validation_priority
  )
][1:60]

# =============================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/14_objetivo1_v5_2_validacion_perfiles_catalogo_matching.R")
head(summary_profiles[order(-matched_publications)], 30)

candidate_profiles[
  ,
  .(
    profile_option,
    researcher_name,
    profile_file,
    profile_publications,
    matched_publications,
    match_rate_pct,
    matched_auto_accept,
    auto_accept_pct_of_matched,
    matched_review_required,
    dominant_area,
    dominant_subarea,
    dominant_area_pct_auto,
    validation_priority
  )
][1:15]

# ===============================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/15_objetivo1_v5_2_preparar_validacion_perfiles_seleccionados.R")

summary_sample
summary_profile
cat("Registros en muestra:", nrow(validation_out), "\n")

# 3 =================================================================

source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/16_objetivo1_v5_2_evaluar_validacion_perfiles.R")
summary_overall
summary_group
summary_profile
summary_decision

cat("Errores:", nrow(errors), "\n")
cat("Fallos top5:", nrow(top5_failures), "\n")

# ===========================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/17_objetivo1_v5_2_analizar_errores_validacion_perfiles.R")

summary_error_profile
summary_error_group
summary_error_area
summary_top5_failure_profile
summary_top5_failure_area
cat("Casos a revisar antes del informe:", nrow(cases_to_review), "\n")

# ===========================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/18_objetivo1_v5_2_diagnostico_bibliometria_metaciencia.R")

summary_biblio
summary_by_profile
summary_predicted_area
recommendations

cat("Fallos top5 bibliometría:", nrow(biblio_top5_failures), "\n")
cat("Top5 contiene bibliometría pero no top1:", nrow(biblio_top5_contains_but_not_top1), "\n")

#==========================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/19_objetivo1_v5_3_reforzar_bibliometria_prueba_local.R")

summary_test
cases_report[
  ,
  .(
    researcher_name,
    profile_title,
    corrected_area_label,
    corrected_subarea_label,
    old_top1_area,
    old_top1_subarea,
    old_top5_biblio,
    new_top1_area,
    new_top1_subarea,
    new_top5_biblio,
    improved_top5_biblio,
    new_false_positive_biblio
  )
]
# ===========================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/20_objetivo1_v5_2_agregar_flag_metaciencia_bibliometria.R")
summary_qa
summary_decision_qa
head(summary_area_qa, 40)
summary_topk_qa
# ==============================================================
source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/20b_objetivo1_v5_2_qa_metaciencia_sin_sobrescribir_decision.R")

summary_qa
summary_original_decision
summary_qa_decision
summary_cross
summary_topk_qa

# ==========================================================
# PRUEBA DE PERFIL INDIVIDUAL
# ==========================================================
selected_profile_query <- "martin"

selected_profile_file <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/perfiles_research/gs-martin.csv"

source(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/21b_objetivo1_v5_2_evaluar_perfil_individual_fast.R"
)
# source("/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/21_objetivo1_v5_2_evaluar_perfil_individual.R")


file.exists(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/tables/objetivo1_v5_2_profile_individual_evaluation_fast/martin/profile_publications_unmatched.csv"
)

summary_matching
summary_match_method
summary_original_decision
summary_qa_decision
summary_qa_effect
summary_area_qa_auto
head(summary_top1_area_all_matches, 30)
summary_review_groups

# ============================================================
researcher_safe_id <- "martin"

source(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/22b_objetivo1_v5_2_clasificar_no_encontrados_perfil_pipeline_oficial.R"
)

source(
  "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN/script/22_objetivo1_v5_2_clasificar_no_encontrados_perfil_v2.R"
)
summary_profile_coverage
summary_profile_output
summary_area_unmatched
summary_status_unmatched
summary_metascience_unmatched
summary_profile_accepted_area
summary_profile_suggested_area


# 3 ===============================================

library(data.table)

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")
researcher_safe_id <- "martin"

unmatched_pred_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_unmatched_classified",
  researcher_safe_id,
  "unmatched_zero_shot_title_only_suggestions.csv"
)

matched_reranked_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_evaluation_fast",
  researcher_safe_id,
  "profile_publications_matched_final_qa_v2_reranked.csv"
)

pred <- fread(unmatched_pred_file)
matched <- fread(matched_reranked_file)

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

pred[, profile_cites := suppressWarnings(as.numeric(profile_cites))]
matched[, profile_cites := suppressWarnings(as.numeric(profile_cites))]

# No emparejadas: sugerencia title-only post-reranking
pred[, profile_classification_origin := "profile_unmatched_title_only_suggestion"]
pred[, profile_output_type := "review_required_title_only_suggestion"]
pred[, profile_suggested_area_label := clean_text(top1_area_name)]
pred[, profile_suggested_subarea_label := clean_text(top1_subarea_name)]
pred[, profile_accepted_area_label := NA_character_]
pred[, profile_accepted_subarea_label := NA_character_]
pred[, profile_requires_review := TRUE]

# Emparejadas: corpus v5.2 + QA
matched[, profile_classification_origin := "matched_in_universe_v5_2_qa"]
matched[, profile_output_type := fifelse(
  qa_auto_accept == TRUE,
  "accepted_from_corpus_qa",
  "review_required_from_corpus_qa"
)]

matched[, profile_suggested_area_label := clean_text(top1_area_name)]
matched[, profile_suggested_subarea_label := clean_text(top1_subarea_name)]

matched[, profile_accepted_area_label := fifelse(
  qa_auto_accept == TRUE,
  qa_final_area_label,
  NA_character_
)]

matched[, profile_accepted_subarea_label := fifelse(
  qa_auto_accept == TRUE,
  qa_final_subarea_label,
  NA_character_
)]

matched[, profile_requires_review := qa_requires_review]

common_cols <- c(
  "profile_row_id",
  "profile_title",
  "profile_year",
  "profile_cites",
  "profile_classification_origin",
  "profile_output_type",
  "profile_suggested_area_label",
  "profile_suggested_subarea_label",
  "profile_accepted_area_label",
  "profile_accepted_subarea_label",
  "profile_requires_review",
  "top1_score",
  "top1_top2_margin",
  "reranked_to_bibliometrics",
  "rerank_reason"
)

for (col in common_cols) {
  if (!col %in% names(pred)) pred[, (col) := NA]
  if (!col %in% names(matched)) matched[, (col) := NA]
}

profile_combined_reranked <- rbindlist(
  list(
    matched[, ..common_cols],
    pred[, ..common_cols]
  ),
  fill = TRUE
)

summary_profile_suggested_area_reranked <- profile_combined_reranked[
  ,
  .(
    publications = .N,
    citations = sum(profile_cites, na.rm = TRUE),
    accepted = sum(!is.na(profile_accepted_area_label) & profile_accepted_area_label != "", na.rm = TRUE),
    review_required = sum(profile_requires_review == TRUE, na.rm = TRUE),
    reranked_cases = sum(reranked_to_bibliometrics == TRUE, na.rm = TRUE)
  ),
  by = .(
    profile_suggested_area_label,
    profile_suggested_subarea_label
  )
][order(-publications)]

summary_profile_suggested_area_reranked[
  ,
  pct := round(publications / sum(publications) * 100, 2)
]

summary_profile_suggested_area_reranked

