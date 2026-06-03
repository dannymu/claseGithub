# ============================================================
# 01_objetivo1_v5_preparar_dataset_enriquecido_GS_OpenAlex.R
# Objetivo 1 v5
# Preparar dataset enriquecido GS + OpenAlex para clasificación
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

out_dir <- file.path(tables_dir, "objetivo1_v5_enriched")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

candidate_files <- c(
  file.path(tables_dir, "dt_documents_All_GS_OpenA.rds"),
  file.path(tables_dir, "dt_documents_All_GS_OpenA.csv"),
  file.path(base_dir, "dt_documents_All_GS_OpenA.rds"),
  file.path(base_dir, "dt_documents_All_GS_OpenA.csv")
)

input_file <- candidate_files[file.exists(candidate_files)][1]

if (is.na(input_file)) {
  stop("No se encontró dt_documents_All_GS_OpenA. Guarda primero el objeto como RDS o CSV.")
}

out_all <- file.path(out_dir, "dataset_objetivo1_v5_enriched_all.csv")
out_embeddings <- file.path(out_dir, "dataset_objetivo1_v5_for_embeddings.csv")
out_title_only <- file.path(out_dir, "dataset_objetivo1_v5_title_only_not_for_training.csv")
out_summary_quality <- file.path(out_dir, "summary_text_quality_v5.csv")
out_summary_sources <- file.path(out_dir, "summary_sources_v5.csv")
out_summary_dedup <- file.path(out_dir, "summary_dedup_v5.csv")

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

has_content <- function(x) {
  x <- clean_text(x)
  x_lower <- tolower(x)
  !(x_lower %in% c("", "na", "nan", "null", "none", "sin resumen", "no abstract"))
}

normalize_title <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- stringi::stri_trans_general(x, "Latin-ASCII")
  x <- gsub("[[:punct:]]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

normalize_doi <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- gsub("^https?://(dx\\.)?doi\\.org/", "", x)
  x <- gsub("^doi:", "", x)
  trimws(x)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))
safe_integer <- function(x) suppressWarnings(as.integer(x))

ensure_col <- function(dt, col, default = "") {
  if (!col %in% names(dt)) {
    dt[, (col) := default]
  }
  invisible(dt)
}

coalesce_text <- function(...) {
  values <- list(...)
  n <- length(values[[1]])
  out <- rep("", n)
  
  for (v in values) {
    v <- clean_text(v)
    idx <- out == "" & v != ""
    out[idx] <- v[idx]
  }
  
  out
}

coalesce_numeric <- function(...) {
  values <- list(...)
  n <- length(values[[1]])
  out <- rep(NA_real_, n)
  
  for (v in values) {
    v <- safe_numeric(v)
    idx <- is.na(out) & !is.na(v)
    out[idx] <- v[idx]
  }
  
  out
}

build_embedding_text <- function(title, abstract, keyword, conceptos, topics, source, type) {
  
  title <- clean_text(title)
  abstract <- clean_text(abstract)
  keyword <- clean_text(keyword)
  conceptos <- clean_text(conceptos)
  topics <- clean_text(topics)
  source <- clean_text(source)
  type <- clean_text(type)
  
  parts <- list(
    title = ifelse(title != "", paste0("[TITLE] ", title), ""),
    keyword = ifelse(keyword != "", paste0("[KEYWORDS] ", keyword), ""),
    conceptos = ifelse(conceptos != "", paste0("[CONCEPTS] ", conceptos), ""),
    topics = ifelse(topics != "", paste0("[TOPICS] ", topics), ""),
    source = ifelse(source != "", paste0("[SOURCE] ", source), ""),
    type = ifelse(type != "", paste0("[TYPE] ", type), ""),
    abstract = ifelse(abstract != "", paste0("[ABSTRACT] ", abstract), "")
  )
  
  mat <- do.call(cbind, parts)
  
  apply(mat, 1, function(row) {
    row <- row[row != ""]
    paste(row, collapse = " [SEP] ")
  })
}

# ============================================================
# 3) Leer datos
# ============================================================

cat("Archivo usado:\n")
cat(input_file, "\n")

if (grepl("\\.rds$", input_file, ignore.case = TRUE)) {
  dt <- readRDS(input_file)
  dt <- as.data.table(dt)
} else {
  dt <- fread(input_file)
}

setDT(dt)

cat("Registros iniciales:", nrow(dt), "\n")

# ============================================================
# 4) Asegurar columnas esperadas
# ============================================================

expected_cols <- c(
  "cited_by_url", "pub_info", "title.x", "pub_year2", "authors",
  "cited_by", "pub_year", "title_url", "cluster_list",
  "total_el_cluster", "cluster_list_ordered", "pub_info_processed",
  "publication", "volume", "issue", "first_page", "last_page",
  "doc_id", "cluster_1e_5", "title_lang", "title_clean",
  "title.y", "keyword", "keyword_score", "conceptos", "topics",
  "doi", "abstract", "year", "revista", "type", "lenguage", "citas"
)

for (col in expected_cols) {
  ensure_col(dt, col, "")
}

# ============================================================
# 5) Construir campos normalizados
# ============================================================

dt[, title_final := coalesce_text(title_clean, `title.y`, `title.x`)]
dt[, title_norm := normalize_title(title_final)]

dt[, doi_norm := normalize_doi(doi)]

dt[, abstract_final := clean_text(abstract)]
dt[, keyword_final := clean_text(keyword)]
dt[, conceptos_final := clean_text(conceptos)]
dt[, topics_final := clean_text(topics)]

dt[, source_final := coalesce_text(revista, publication)]
dt[, type_final := clean_text(type)]
dt[, authors_final := clean_text(authors)]

dt[, year_final := coalesce_numeric(year, pub_year, pub_year2)]
dt[, year_final := safe_integer(year_final)]

dt[, citations_final := coalesce_numeric(citas, cited_by)]

dt[, language_final := coalesce_text(lenguage, title_lang)]

dt[, doc_id_original := clean_text(doc_id)]

# ============================================================
# 6) Indicadores de metadatos
# ============================================================

dt[, has_title := has_content(title_final)]
dt[, has_abstract := has_content(abstract_final)]
dt[, has_keyword := has_content(keyword_final)]
dt[, has_conceptos := has_content(conceptos_final)]
dt[, has_topics := has_content(topics_final)]
dt[, has_source := has_content(source_final)]
dt[, has_doi := has_content(doi_norm)]

dt[, contextual_metadata_count :=
     as.integer(has_abstract) +
     as.integer(has_keyword) +
     as.integer(has_conceptos) +
     as.integer(has_topics)]

dt[, weak_context_count :=
     as.integer(has_source) +
     as.integer(has_doi)]

# ============================================================
# 7) Calidad textual enriquecida
# ============================================================

dt[, text_quality_v5 := fcase(
  
  has_title == TRUE &
    has_abstract == TRUE &
    (has_keyword == TRUE | has_conceptos == TRUE | has_topics == TRUE),
  "title_abstract_keywords_concepts_topics",
  
  has_title == TRUE &
    has_abstract == TRUE,
  "title_abstract",
  
  has_title == TRUE &
    has_abstract == FALSE &
    (has_keyword == TRUE | has_conceptos == TRUE | has_topics == TRUE),
  "title_keywords_concepts_topics",
  
  has_title == TRUE &
    contextual_metadata_count == 0 &
    has_source == TRUE,
  "title_source_only",
  
  has_title == TRUE &
    contextual_metadata_count == 0 &
    has_source == FALSE,
  "title_only",
  
  default = "insufficient_text"
)]

# Regla para embeddings fuertes:
# Se aceptan documentos con título y al menos un metadato contextual fuerte:
# abstract, keyword, conceptos o topics.
dt[, embedding_eligible_v5 := has_title == TRUE & contextual_metadata_count >= 1]

# Los title_only quedan separados, no como base fuerte de entrenamiento.
dt[, title_only_or_weak_v5 := has_title == TRUE & contextual_metadata_count == 0]

# ============================================================
# 8) Construir text_for_embedding enriquecido
# ============================================================

dt[, text_for_embedding_v5 := build_embedding_text(
  title = title_final,
  abstract = abstract_final,
  keyword = keyword_final,
  conceptos = conceptos_final,
  topics = topics_final,
  source = source_final,
  type = type_final
)]

dt[, n_words_embedding_v5 := stringi::stri_count_words(text_for_embedding_v5)]

# ============================================================
# 9) Deduplicación
# ============================================================

dt[, dedup_key := fcase(
  has_doi == TRUE,
  paste0("doi::", doi_norm),
  
  has_doi == FALSE & !is.na(year_final),
  paste0("title_year::", title_norm, "::", year_final),
  
  has_doi == FALSE & is.na(year_final),
  paste0("title::", title_norm),
  
  default = paste0("row::", .I)
)]

# Puntaje de riqueza para conservar mejor registro en duplicados
dt[, metadata_score :=
     10 * as.integer(has_abstract) +
     6  * as.integer(has_keyword) +
     6  * as.integer(has_conceptos) +
     6  * as.integer(has_topics) +
     3  * as.integer(has_doi) +
     2  * as.integer(has_source) +
     1  * as.integer(!is.na(year_final)) +
     pmin(n_words_embedding_v5, 500) / 500]

dt[, row_original := .I]

setorder(dt, dedup_key, -metadata_score, -citations_final, row_original)

dt_dedup <- dt[
  ,
  .SD[1],
  by = dedup_key
]

# ============================================================
# 10) Crear datasets finales
# ============================================================

dt_all <- copy(dt_dedup)

dt_embeddings <- dt_all[
  embedding_eligible_v5 == TRUE &
    title_norm != "" &
    text_for_embedding_v5 != ""
]

dt_title_only <- dt_all[
  title_only_or_weak_v5 == TRUE |
    embedding_eligible_v5 == FALSE
]

# Crear doc_id interno v5
dt_all[, doc_id_v5 := .I]
dt_embeddings[, doc_id_v5 := .I]
dt_title_only[, doc_id_v5 := .I]

# Dataset para embeddings con columnas limpias
dt_embeddings_out <- dt_embeddings[
  ,
  .(
    doc_id_v5,
    doc_id_original,
    dedup_key,
    doi_norm,
    title = title_final,
    abstract = abstract_final,
    keyword = keyword_final,
    conceptos = conceptos_final,
    topics = topics_final,
    source = source_final,
    type = type_final,
    authors = authors_final,
    year = year_final,
    citations = citations_final,
    language = language_final,
    text_quality_v5,
    contextual_metadata_count,
    text_for_embedding = text_for_embedding_v5,
    n_words_embedding_v5,
    metadata_score
  )
]

dt_all_out <- dt_all[
  ,
  .(
    doc_id_v5,
    doc_id_original,
    dedup_key,
    doi_norm,
    title = title_final,
    abstract = abstract_final,
    keyword = keyword_final,
    conceptos = conceptos_final,
    topics = topics_final,
    source = source_final,
    type = type_final,
    authors = authors_final,
    year = year_final,
    citations = citations_final,
    language = language_final,
    text_quality_v5,
    embedding_eligible_v5,
    title_only_or_weak_v5,
    contextual_metadata_count,
    text_for_embedding = text_for_embedding_v5,
    n_words_embedding_v5,
    metadata_score
  )
]

dt_title_only_out <- dt_title_only[
  ,
  .(
    doc_id_v5,
    doc_id_original,
    dedup_key,
    doi_norm,
    title = title_final,
    source = source_final,
    type = type_final,
    authors = authors_final,
    year = year_final,
    citations = citations_final,
    language = language_final,
    text_quality_v5,
    embedding_eligible_v5,
    title_only_or_weak_v5,
    text_for_embedding = text_for_embedding_v5,
    n_words_embedding_v5,
    metadata_score
  )
]

# ============================================================
# 11) Resúmenes
# ============================================================

summary_quality <- dt_all_out[
  ,
  .N,
  by = .(
    text_quality_v5,
    embedding_eligible_v5
  )
][order(-N)]

summary_quality[, pct := round(N / sum(N) * 100, 2)]

summary_sources <- dt_all_out[
  ,
  .(
    documents = .N,
    with_abstract = sum(text_quality_v5 %in% c(
      "title_abstract",
      "title_abstract_keywords_concepts_topics"
    )),
    with_keywords = sum(keyword != ""),
    with_concepts = sum(conceptos != ""),
    with_topics = sum(topics != ""),
    eligible_embeddings = sum(embedding_eligible_v5 == TRUE)
  ),
  by = source
][order(-documents)]

summary_dedup <- data.table(
  metric = c(
    "registros_iniciales",
    "registros_unicos_deduplicados",
    "duplicados_removidos",
    "documentos_elegibles_embeddings_v5",
    "documentos_title_only_o_debiles",
    "porcentaje_elegible_embeddings_v5"
  ),
  value = c(
    nrow(dt),
    nrow(dt_all_out),
    nrow(dt) - nrow(dt_all_out),
    nrow(dt_embeddings_out),
    nrow(dt_title_only_out),
    round(nrow(dt_embeddings_out) / nrow(dt_all_out) * 100, 2)
  )
)

# ============================================================
# 12) Exportar
# ============================================================

fwrite(dt_all_out, out_all)
fwrite(dt_embeddings_out, out_embeddings)
fwrite(dt_title_only_out, out_title_only)

fwrite(summary_quality, out_summary_quality)
fwrite(summary_sources, out_summary_sources)
fwrite(summary_dedup, out_summary_dedup)

cat("\n================ OBJETIVO 1 v5: DATASET ENRIQUECIDO ================\n")
cat("Archivo entrada:", input_file, "\n")

cat("\n================ DEDUP ================\n")
print(summary_dedup)

cat("\n================ CALIDAD TEXTUAL ================\n")
print(summary_quality)

cat("\nArchivos creados:\n")
cat(out_all, "\n")
cat(out_embeddings, "\n")
cat(out_title_only, "\n")
cat(out_summary_quality, "\n")
cat(out_summary_dedup, "\n")