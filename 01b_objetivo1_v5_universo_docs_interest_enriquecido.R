# ============================================================
# 01b_objetivo1_v5_universo_docs_interest_enriquecido.R
# Objetivo 1 v5
# Usar docs_interest como universo oficial de documentos únicos
# y enriquecerlo con metadatos GS + OpenAlex
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

docs_interest_dir <- file.path(
  tables_dir,
  "nodos-gs-open"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------
# Archivos candidatos
# ------------------------------------------------------------

docs_interest_candidates <- c(
  file.path(docs_interest_dir, "docs_interest.rds"),
  file.path(docs_interest_dir, "docs_interest.csv"),
  file.path(docs_interest_dir, "docs_interest.tsv"),
  file.path(tables_dir, "docs_interest.rds"),
  file.path(tables_dir, "docs_interest.csv")
)

metadata_candidates <- c(
  file.path(tables_dir, "dt_documents_All_GS_OpenA.rds"),
  file.path(tables_dir, "dt_documents_All_GS_OpenA.csv"),
  file.path(base_dir, "dt_documents_All_GS_OpenA.rds"),
  file.path(base_dir, "dt_documents_All_GS_OpenA.csv")
)

docs_interest_file <- docs_interest_candidates[file.exists(docs_interest_candidates)][1]
metadata_file <- metadata_candidates[file.exists(metadata_candidates)][1]

if (is.na(docs_interest_file)) {
  stop("No se encontró docs_interest. Revisa la ruta /tables/nodos-gs-open.")
}

if (is.na(metadata_file)) {
  stop("No se encontró dt_documents_All_GS_OpenA. Guarda primero ese objeto como RDS o CSV.")
}

# ------------------------------------------------------------
# Salidas
# ------------------------------------------------------------

out_all <- file.path(
  out_dir,
  "dataset_objetivo1_v5_docs_interest_enriched_all.csv"
)

out_embeddings <- file.path(
  out_dir,
  "dataset_objetivo1_v5_docs_interest_for_embeddings.csv"
)

out_not_eligible <- file.path(
  out_dir,
  "dataset_objetivo1_v5_docs_interest_not_eligible.csv"
)

out_summary_enrichment <- file.path(
  out_dir,
  "summary_enrichment_docs_interest_v5.csv"
)

out_summary_quality <- file.path(
  out_dir,
  "summary_text_quality_docs_interest_v5.csv"
)

out_summary_match_method <- file.path(
  out_dir,
  "summary_match_method_docs_interest_v5.csv"
)

out_problematic <- file.path(
  out_dir,
  "problematic_records_docs_interest_v5.csv"
)

# ============================================================
# 2) Funciones auxiliares
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

normalize_id <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- gsub("^https?://openalex\\.org/", "", x)
  x <- gsub("^https?://doi\\.org/", "", x)
  x <- gsub("^https?://dx\\.doi\\.org/", "", x)
  x <- gsub("^doi:", "", x)
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
# 3) Leer docs_interest
# ============================================================

cat("\nArchivo docs_interest:\n")
cat(docs_interest_file, "\n")

if (grepl("\\.rds$", docs_interest_file, ignore.case = TRUE)) {
  docs_interest <- readRDS(docs_interest_file)
  docs_interest <- as.data.table(docs_interest)
} else if (grepl("\\.tsv$", docs_interest_file, ignore.case = TRUE)) {
  docs_interest <- fread(docs_interest_file, sep = "\t")
} else {
  docs_interest <- fread(docs_interest_file)
}

setDT(docs_interest)

required_docs_cols <- c("doc_id", "work_id", "title", "title_norm")
missing_docs_cols <- setdiff(required_docs_cols, names(docs_interest))

if (length(missing_docs_cols) > 0) {
  stop(
    paste0(
      "docs_interest no tiene columnas requeridas: ",
      paste(missing_docs_cols, collapse = ", ")
    )
  )
}

cat("docs_interest filas:", nrow(docs_interest), "\n")

# ============================================================
# 4) Leer metadata GS + OpenAlex
# ============================================================

cat("\nArchivo metadata GS + OpenAlex:\n")
cat(metadata_file, "\n")

if (grepl("\\.rds$", metadata_file, ignore.case = TRUE)) {
  meta <- readRDS(metadata_file)
  meta <- as.data.table(meta)
} else {
  meta <- fread(metadata_file)
}

setDT(meta)

cat("metadata filas:", nrow(meta), "\n")

# ============================================================
# 5) Asegurar columnas metadata
# ============================================================

expected_meta_cols <- c(
  "cited_by_url", "pub_info", "title.x", "pub_year2", "authors",
  "cited_by", "pub_year", "title_url", "cluster_list",
  "total_el_cluster", "cluster_list_ordered", "pub_info_processed",
  "publication", "volume", "issue", "first_page", "last_page",
  "doc_id", "cluster_1e_5", "title_lang", "title_clean",
  "title.y", "keyword", "keyword_score", "conceptos", "topics",
  "doi", "abstract", "year", "revista", "type", "lenguage", "citas"
)

for (col in expected_meta_cols) {
  ensure_col(meta, col, "")
}

# ============================================================
# 6) Normalizar docs_interest
# ============================================================

docs_interest[, doc_idx := .I]

docs_interest[, doc_id_key := normalize_id(doc_id)]
docs_interest[, work_id_key := normalize_id(work_id)]

docs_interest[, title_final_interest := clean_text(title)]

docs_interest[, title_norm_key := fifelse(
  clean_text(title_norm) != "",
  clean_text(title_norm),
  normalize_title(title)
)]

docs_interest[, title_norm_key := normalize_title(title_norm_key)]

docs_interest <- docs_interest[
  title_norm_key != "" |
    doc_id_key != "" |
    work_id_key != ""
]

cat("docs_interest válidos:", nrow(docs_interest), "\n")

# ============================================================
# 7) Preparar metadata enriquecida
# ============================================================

meta[, meta_row_id := .I]

meta[, meta_doc_id_key := normalize_id(doc_id)]

meta[, meta_title := coalesce_text(
  get("title_clean"),
  get("title.y"),
  get("title.x")
)]

meta[, meta_title_norm := normalize_title(meta_title)]

meta[, meta_doi_norm := normalize_doi(doi)]

meta[, meta_abstract := clean_text(abstract)]
meta[, meta_keyword := clean_text(keyword)]
meta[, meta_conceptos := clean_text(conceptos)]
meta[, meta_topics := clean_text(topics)]

meta[, meta_source := coalesce_text(revista, publication)]
meta[, meta_type := clean_text(type)]
meta[, meta_authors := clean_text(authors)]

meta[, meta_year := coalesce_numeric(year, pub_year, pub_year2)]
meta[, meta_year := safe_integer(meta_year)]

meta[, meta_citations := coalesce_numeric(citas, cited_by)]

meta[, meta_language := coalesce_text(lenguage, title_lang)]

meta[, has_title := has_content(meta_title)]
meta[, has_abstract := has_content(meta_abstract)]
meta[, has_keyword := has_content(meta_keyword)]
meta[, has_conceptos := has_content(meta_conceptos)]
meta[, has_topics := has_content(meta_topics)]
meta[, has_source := has_content(meta_source)]
meta[, has_doi := has_content(meta_doi_norm)]

meta[, contextual_metadata_count :=
       as.integer(has_abstract) +
       as.integer(has_keyword) +
       as.integer(has_conceptos) +
       as.integer(has_topics)]

meta[, meta_text_for_score := build_embedding_text(
  title = meta_title,
  abstract = meta_abstract,
  keyword = meta_keyword,
  conceptos = meta_conceptos,
  topics = meta_topics,
  source = meta_source,
  type = meta_type
)]

meta[, meta_n_words := stringi::stri_count_words(meta_text_for_score)]

meta[, metadata_score :=
       10 * as.integer(has_abstract) +
       6  * as.integer(has_keyword) +
       6  * as.integer(has_conceptos) +
       6  * as.integer(has_topics) +
       3  * as.integer(has_doi) +
       2  * as.integer(has_source) +
       1  * as.integer(!is.na(meta_year)) +
       pmin(meta_n_words, 500) / 500]

meta_small <- meta[
  has_title == TRUE,
  .(
    meta_row_id,
    meta_doc_id_key,
    meta_title_norm,
    meta_title,
    meta_doi_norm,
    meta_abstract,
    meta_keyword,
    meta_conceptos,
    meta_topics,
    meta_source,
    meta_type,
    meta_authors,
    meta_year,
    meta_citations,
    meta_language,
    contextual_metadata_count,
    metadata_score
  )
]

# ============================================================
# 8) Crear índices únicos de metadata
# ============================================================

setorder(meta_small, meta_doc_id_key, -metadata_score, -meta_citations)

meta_by_docid <- meta_small[
  meta_doc_id_key != "",
  .SD[1],
  by = meta_doc_id_key
]

setorder(meta_small, meta_title_norm, -metadata_score, -meta_citations)

meta_by_title <- meta_small[
  meta_title_norm != "",
  .SD[1],
  by = meta_title_norm
]

cat("metadata única por doc_id:", nrow(meta_by_docid), "\n")
cat("metadata única por título:", nrow(meta_by_title), "\n")

# ============================================================
# 9) Matching docs_interest → metadata
# ============================================================

matches <- list()

# ------------------------------------------------------------
# 9.1 Match por doc_id
# ------------------------------------------------------------

m_doc <- merge(
  docs_interest[doc_id_key != ""],
  meta_by_docid,
  by.x = "doc_id_key",
  by.y = "meta_doc_id_key",
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_doc) > 0) {
  m_doc[, enrichment_match_method := "doc_id_exact"]
  m_doc[, enrichment_match_priority := 1]
  m_doc[, enrichment_match_score := 1.0]
  matches[["doc_id_exact"]] <- m_doc
}

# ------------------------------------------------------------
# 9.2 Match por work_id contra meta doc_id
# ------------------------------------------------------------

m_work <- merge(
  docs_interest[work_id_key != ""],
  meta_by_docid,
  by.x = "work_id_key",
  by.y = "meta_doc_id_key",
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_work) > 0) {
  m_work[, enrichment_match_method := "work_id_exact"]
  m_work[, enrichment_match_priority := 2]
  m_work[, enrichment_match_score := 1.0]
  matches[["work_id_exact"]] <- m_work
}

# ------------------------------------------------------------
# 9.3 Match por title_norm
# ------------------------------------------------------------

m_title <- merge(
  docs_interest[title_norm_key != ""],
  meta_by_title,
  by.x = "title_norm_key",
  by.y = "meta_title_norm",
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_title) > 0) {
  m_title[, enrichment_match_method := "title_norm_exact"]
  m_title[, enrichment_match_priority := 3]
  m_title[, enrichment_match_score := 1.0]
  matches[["title_norm_exact"]] <- m_title
}

# ------------------------------------------------------------
# 9.4 Consolidar matches
# ------------------------------------------------------------

all_matches <- if (length(matches) > 0) {
  rbindlist(matches, fill = TRUE)
} else {
  data.table()
}

if (nrow(all_matches) == 0) {
  stop("No hubo ningún match entre docs_interest y metadata GS + OpenAlex.")
}

setorder(
  all_matches,
  doc_idx,
  enrichment_match_priority,
  -metadata_score,
  -meta_citations
)

best_matches <- all_matches[
  ,
  .SD[1],
  by = doc_idx
]

# Columnas metadata seleccionadas para unir al universo
meta_match_cols <- c(
  "doc_idx",
  "enrichment_match_method",
  "enrichment_match_priority",
  "enrichment_match_score",
  "meta_row_id",
  "meta_title",
  "meta_doi_norm",
  "meta_abstract",
  "meta_keyword",
  "meta_conceptos",
  "meta_topics",
  "meta_source",
  "meta_type",
  "meta_authors",
  "meta_year",
  "meta_citations",
  "meta_language",
  "contextual_metadata_count",
  "metadata_score"
)

best_matches_small <- best_matches[, ..meta_match_cols]

dt <- merge(
  docs_interest,
  best_matches_small,
  by = "doc_idx",
  all.x = TRUE
)

# ============================================================
# 10) Construir campos finales enriquecidos
# ============================================================

dt[, title_final := fifelse(
  !is.na(meta_title) & meta_title != "",
  meta_title,
  title_final_interest
)]

dt[, abstract_final := clean_text(meta_abstract)]
dt[, keyword_final := clean_text(meta_keyword)]
dt[, conceptos_final := clean_text(meta_conceptos)]
dt[, topics_final := clean_text(meta_topics)]
dt[, source_final := clean_text(meta_source)]
dt[, type_final := clean_text(meta_type)]
dt[, authors_final := clean_text(meta_authors)]
dt[, doi_final := clean_text(meta_doi_norm)]
dt[, year_final := safe_integer(meta_year)]
dt[, citations_final := safe_numeric(meta_citations)]
dt[, language_final := clean_text(meta_language)]

dt[, has_title := has_content(title_final)]
dt[, has_abstract := has_content(abstract_final)]
dt[, has_keyword := has_content(keyword_final)]
dt[, has_conceptos := has_content(conceptos_final)]
dt[, has_topics := has_content(topics_final)]
dt[, has_source := has_content(source_final)]
dt[, has_doi := has_content(doi_final)]

dt[, contextual_metadata_count_final :=
     as.integer(has_abstract) +
     as.integer(has_keyword) +
     as.integer(has_conceptos) +
     as.integer(has_topics)]

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
    contextual_metadata_count_final == 0 &
    has_source == TRUE,
  "title_source_only",
  
  has_title == TRUE &
    contextual_metadata_count_final == 0 &
    has_source == FALSE,
  "title_only",
  
  default = "insufficient_text"
)]

dt[, embedding_eligible_v5 := has_title == TRUE &
     contextual_metadata_count_final >= 1]

dt[, text_for_embedding := build_embedding_text(
  title = title_final,
  abstract = abstract_final,
  keyword = keyword_final,
  conceptos = conceptos_final,
  topics = topics_final,
  source = source_final,
  type = type_final
)]

dt[, n_words_embedding_v5 := stringi::stri_count_words(text_for_embedding)]

# Mantener doc_id_v5 consistente con el universo oficial
setorder(dt, doc_idx)
dt[, doc_id_v5 := .I]

# ============================================================
# 11) Crear datasets finales
# ============================================================

dt_all_out <- dt[
  ,
  .(
    doc_id_v5,
    doc_id_interest = doc_id,
    work_id,
    title_interest = title,
    title_norm_interest = title_norm,
    enrichment_match_method,
    enrichment_match_score,
    meta_row_id,
    doi = doi_final,
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
    contextual_metadata_count = contextual_metadata_count_final,
    text_for_embedding,
    n_words_embedding_v5,
    metadata_score
  )
]

dt_embeddings_out <- dt_all_out[
  embedding_eligible_v5 == TRUE &
    title != "" &
    text_for_embedding != ""
]

dt_not_eligible <- dt_all_out[
  embedding_eligible_v5 == FALSE |
    title == "" |
    text_for_embedding == ""
]

# ============================================================
# 12) Resúmenes
# ============================================================

summary_enrichment <- data.table(
  metric = c(
    "docs_interest_universe",
    "matched_with_metadata",
    "not_matched_with_metadata",
    "match_rate_pct",
    "embedding_eligible_v5",
    "not_eligible_v5",
    "embedding_eligible_pct"
  ),
  value = c(
    nrow(dt_all_out),
    dt_all_out[!is.na(enrichment_match_method) & enrichment_match_method != "", .N],
    dt_all_out[is.na(enrichment_match_method) | enrichment_match_method == "", .N],
    round(dt_all_out[!is.na(enrichment_match_method) & enrichment_match_method != "", .N] / nrow(dt_all_out) * 100, 2),
    nrow(dt_embeddings_out),
    nrow(dt_not_eligible),
    round(nrow(dt_embeddings_out) / nrow(dt_all_out) * 100, 2)
  )
)

summary_quality <- dt_all_out[
  ,
  .N,
  by = .(
    text_quality_v5,
    embedding_eligible_v5
  )
][order(-N)]

summary_quality[, pct := round(N / sum(N) * 100, 2)]

summary_match_method <- dt_all_out[
  ,
  .N,
  by = enrichment_match_method
][order(-N)]

summary_match_method[, pct := round(N / sum(N) * 100, 2)]

problematic_records <- dt_all_out[
  embedding_eligible_v5 == FALSE |
    n_words_embedding_v5 < 8 |
    title == "",
  .(
    doc_id_v5,
    doc_id_interest,
    work_id,
    title,
    text_quality_v5,
    embedding_eligible_v5,
    enrichment_match_method,
    n_words_embedding_v5,
    source,
    year
  )
][order(embedding_eligible_v5, n_words_embedding_v5)]

# ============================================================
# 13) Exportar
# ============================================================

fwrite(dt_all_out, out_all)
fwrite(dt_embeddings_out, out_embeddings)
fwrite(dt_not_eligible, out_not_eligible)

fwrite(summary_enrichment, out_summary_enrichment)
fwrite(summary_quality, out_summary_quality)
fwrite(summary_match_method, out_summary_match_method)
fwrite(problematic_records, out_problematic)

cat("\n================ OBJETIVO 1 v5 CON docs_interest ================\n")
cat("Universo oficial docs_interest:", nrow(dt_all_out), "\n")

cat("\n================ ENRIQUECIMIENTO ================\n")
print(summary_enrichment)

cat("\n================ CALIDAD TEXTUAL ================\n")
print(summary_quality)

cat("\n================ MÉTODO DE MATCH ================\n")
print(summary_match_method)

cat("\nRegistros problemáticos:", nrow(problematic_records), "\n")

cat("\nArchivos creados:\n")
cat(out_all, "\n")
cat(out_embeddings, "\n")
cat(out_not_eligible, "\n")
cat(out_summary_enrichment, "\n")
cat(out_summary_quality, "\n")
cat(out_summary_match_method, "\n")