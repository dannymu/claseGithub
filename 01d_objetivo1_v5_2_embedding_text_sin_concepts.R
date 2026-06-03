# ============================================================
# 01d_objetivo1_v5_2_embedding_text_sin_concepts.R
# Objetivo 1 v5.2
# Crear texto final para embeddings excluyendo Concepts del texto
# principal, pero conservándolos como metadato auxiliar.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

v5_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

input_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_docs_interest_for_embeddings_clean_text.csv"
)

if (!file.exists(input_file)) {
  stop("No existe dataset_objetivo1_v5_docs_interest_for_embeddings_clean_text.csv. Ejecuta primero 01c.")
}

out_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_2_for_embeddings_no_concepts.csv"
)

out_all_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_2_all_no_concepts.csv"
)

out_summary <- file.path(
  v5_dir,
  "summary_text_for_embedding_v5_2_no_concepts.csv"
)

out_check <- file.path(
  v5_dir,
  "check_cases_v5_2_no_concepts.csv"
)

# ============================================================
# Funciones
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

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

build_embedding_text_v52 <- function(
    title,
    abstract,
    keyword_clean,
    topics_clean,
    source_clean,
    type
) {
  
  title <- clean_text(title)
  abstract <- clean_text(abstract)
  keyword_clean <- clean_text(keyword_clean)
  topics_clean <- clean_text(topics_clean)
  source_clean <- clean_text(source_clean)
  type <- clean_text(type)
  
  parts <- list(
    title = ifelse(title != "", paste0("Title: ", title), ""),
    abstract = ifelse(abstract != "", paste0("Abstract: ", abstract), ""),
    keywords = ifelse(keyword_clean != "", paste0("Keywords: ", keyword_clean), ""),
    topics = ifelse(topics_clean != "", paste0("Research topics: ", topics_clean), ""),
    source = ifelse(source_clean != "", paste0("Source: ", source_clean), ""),
    type = ifelse(type != "", paste0("Document type: ", type), "")
  )
  
  mat <- do.call(cbind, parts)
  
  apply(mat, 1, function(row) {
    row <- row[row != ""]
    paste(row, collapse = " [SEP] ")
  })
}

# ============================================================
# Leer datos
# ============================================================

dt <- fread(input_file)
setDT(dt)

cat("Registros leídos:", nrow(dt), "\n")

needed_cols <- c(
  "doc_id_v5",
  "doc_id_interest",
  "work_id",
  "doi",
  "title",
  "abstract",
  "keyword",
  "keyword_clean",
  "conceptos",
  "conceptos_clean",
  "topics",
  "topics_clean",
  "source",
  "source_clean",
  "type",
  "authors",
  "year",
  "citations",
  "language",
  "text_quality_v5",
  "text_quality_v5_clean",
  "embedding_eligible_v5",
  "embedding_eligible_v5_clean",
  "contextual_metadata_count",
  "contextual_metadata_count_clean",
  "text_for_embedding_original",
  "text_for_embedding",
  "n_words_embedding_v5_original",
  "n_words_embedding_v5_clean",
  "metadata_score"
)

for (col in needed_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := ""]
  }
}

# ============================================================
# Construir texto v5.2 sin Concepts
# ============================================================

dt[, text_for_embedding_v5_2 := build_embedding_text_v52(
  title = title,
  abstract = abstract,
  keyword_clean = keyword_clean,
  topics_clean = topics_clean,
  source_clean = source_clean,
  type = type
)]

dt[, n_words_embedding_v5_2 := stringi::stri_count_words(text_for_embedding_v5_2)]

dt[, has_title := has_content(title)]
dt[, has_abstract := has_content(abstract)]
dt[, has_keyword_clean := has_content(keyword_clean)]
dt[, has_topics_clean := has_content(topics_clean)]
dt[, has_source_clean := has_content(source_clean)]

dt[, contextual_metadata_count_v5_2 :=
     as.integer(has_abstract) +
     as.integer(has_keyword_clean) +
     as.integer(has_topics_clean)]

dt[, embedding_eligible_v5_2 := has_title == TRUE &
     contextual_metadata_count_v5_2 >= 1]

dt[, text_quality_v5_2 := fcase(
  
  has_title == TRUE &
    has_abstract == TRUE &
    (has_keyword_clean == TRUE | has_topics_clean == TRUE),
  "title_abstract_keywords_topics",
  
  has_title == TRUE &
    has_abstract == TRUE,
  "title_abstract",
  
  has_title == TRUE &
    has_abstract == FALSE &
    (has_keyword_clean == TRUE | has_topics_clean == TRUE),
  "title_keywords_topics",
  
  has_title == TRUE &
    contextual_metadata_count_v5_2 == 0 &
    has_source_clean == TRUE,
  "title_source_only",
  
  has_title == TRUE &
    contextual_metadata_count_v5_2 == 0 &
    has_source_clean == FALSE,
  "title_only",
  
  default = "insufficient_text"
)]

# ============================================================
# Salida para embeddings
# ============================================================

dt_embeddings <- dt[
  embedding_eligible_v5_2 == TRUE &
    text_for_embedding_v5_2 != ""
]

dt_out <- dt_embeddings[
  ,
  .(
    doc_id_v5,
    doc_id_interest,
    work_id,
    doi,
    title,
    abstract,
    keyword,
    keyword_clean,
    conceptos,
    conceptos_clean,
    topics,
    topics_clean,
    source,
    source_clean,
    type,
    authors,
    year,
    citations,
    language,
    text_quality_v5,
    text_quality_v5_clean,
    text_quality_v5_2,
    embedding_eligible_v5,
    embedding_eligible_v5_clean,
    embedding_eligible_v5_2,
    contextual_metadata_count,
    contextual_metadata_count_clean,
    contextual_metadata_count_v5_2,
    text_for_embedding_original,
    text_for_embedding_v5_1 = text_for_embedding,
    text_for_embedding = text_for_embedding_v5_2,
    n_words_embedding_v5_original,
    n_words_embedding_v5_clean,
    n_words_embedding_v5_2,
    metadata_score
  )
]

dt_all <- copy(dt)
dt_all[, text_for_embedding := text_for_embedding_v5_2]

# ============================================================
# Resúmenes
# ============================================================

summary_v52 <- data.table(
  metric = c(
    "documents_input",
    "documents_eligible_v5_2",
    "documents_not_eligible_v5_2",
    "mean_words_v5_1",
    "mean_words_v5_2",
    "median_words_v5_1",
    "median_words_v5_2",
    "documents_with_abstract",
    "documents_with_keywords",
    "documents_with_topics",
    "documents_with_concepts_auxiliary"
  ),
  value = c(
    nrow(dt),
    nrow(dt_out),
    nrow(dt) - nrow(dt_out),
    round(mean(safe_numeric(dt$n_words_embedding_v5_clean), na.rm = TRUE), 2),
    round(mean(safe_numeric(dt$n_words_embedding_v5_2), na.rm = TRUE), 2),
    round(median(safe_numeric(dt$n_words_embedding_v5_clean), na.rm = TRUE), 2),
    round(median(safe_numeric(dt$n_words_embedding_v5_2), na.rm = TRUE), 2),
    sum(dt$has_abstract == TRUE, na.rm = TRUE),
    sum(dt$has_keyword_clean == TRUE, na.rm = TRUE),
    sum(dt$has_topics_clean == TRUE, na.rm = TRUE),
    sum(has_content(dt$conceptos_clean), na.rm = TRUE)
  )
)

summary_quality_v52 <- dt[
  ,
  .N,
  by = .(
    text_quality_v5_2,
    embedding_eligible_v5_2
  )
][order(-N)]

summary_quality_v52[, pct := round(N / sum(N) * 100, 2)]

check_patterns <- "data sharing|data citation|data reuse|dryad|biodiversity|coronavirus|covid"

check_cases_v52 <- dt_out[
  grepl(check_patterns, text_for_embedding_original, ignore.case = TRUE) |
    grepl(check_patterns, text_for_embedding, ignore.case = TRUE),
  .(
    doc_id_v5,
    work_id,
    title,
    year,
    source,
    source_clean,
    text_quality_v5_2,
    keyword_clean,
    topics_clean,
    conceptos_clean,
    text_v5_1 = substr(text_for_embedding_v5_1, 1, 700),
    text_v5_2 = substr(text_for_embedding, 1, 700)
  )
]

# ============================================================
# Exportar
# ============================================================

fwrite(dt_out, out_file)
fwrite(dt_all, out_all_file)
fwrite(summary_v52, out_summary)
fwrite(summary_quality_v52, file.path(v5_dir, "summary_quality_v5_2_no_concepts.csv"))
fwrite(check_cases_v52, out_check)

cat("\n================ OBJETIVO 1 v5.2 SIN CONCEPTS EN EMBEDDING ================\n")

cat("\nResumen v5.2:\n")
print(summary_v52)

cat("\nCalidad v5.2:\n")
print(summary_quality_v52)

cat("\nCasos de revisión temática v5.2:", nrow(check_cases_v52), "\n")

cat("\nArchivos creados:\n")
cat(out_file, "\n")
cat(out_all_file, "\n")
cat(out_summary, "\n")
cat(out_check, "\n")