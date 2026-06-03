# ============================================================
# 01c_objetivo1_v5_limpiar_text_for_embedding.R
# Objetivo 1 v5.1
# Limpiar y ponderar text_for_embedding antes de generar embeddings
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

v5_dir <- file.path(
  tables_dir,
  "objetivo1_v5_docs_interest_enriched"
)

input_file <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_docs_interest_enriched_all.csv"
)

if (!file.exists(input_file)) {
  stop("No existe dataset_objetivo1_v5_docs_interest_enriched_all.csv.")
}

out_file_all <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_docs_interest_enriched_all_clean_text.csv"
)

out_file_embeddings <- file.path(
  v5_dir,
  "dataset_objetivo1_v5_docs_interest_for_embeddings_clean_text.csv"
)

out_summary_quality <- file.path(
  v5_dir,
  "summary_text_quality_v5_clean_text.csv"
)

out_summary_compare <- file.path(
  v5_dir,
  "summary_compare_text_for_embedding_v5_vs_clean.csv"
)

out_check_cases <- file.path(
  v5_dir,
  "check_problematic_cases_clean_text.csv"
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

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

split_terms <- function(x) {
  x <- clean_text(x)
  if (x == "") return(character())
  
  terms <- unlist(strsplit(x, ";", fixed = TRUE))
  terms <- clean_text(terms)
  terms <- terms[terms != ""]
  unique(terms)
}

collapse_terms <- function(x, max_terms = 12) {
  x <- unique(clean_text(x))
  x <- x[x != ""]
  if (length(x) == 0) return("")
  x <- x[seq_len(min(length(x), max_terms))]
  paste(x, collapse = "; ")
}

has_covid_context_fun <- function(title, abstract, keyword) {
  txt <- paste(
    clean_text(title),
    clean_text(abstract),
    clean_text(keyword),
    collapse = " "
  )
  grepl(
    "covid|coronavirus|sars[- ]?cov[- ]?2|pandemic|pandemia",
    txt,
    ignore.case = TRUE
  )
}

clean_keywords_one <- function(x, max_terms = 10) {
  
  terms <- split_terms(x)
  
  if (length(terms) == 0) return("")
  
  # Las keywords suelen ser más precisas que concepts/topics.
  # Solo se eliminan términos claramente absurdos o técnicos mal interpretados.
  bad_patterns <- c(
    "\\(programming language\\)",
    "\\(typography\\)",
    "\\(fluid\\)",
    "\\(optics\\)",
    "\\(material\\)"
  )
  
  keep <- rep(TRUE, length(terms))
  
  for (pat in bad_patterns) {
    keep <- keep & !grepl(pat, terms, ignore.case = TRUE)
  }
  
  terms <- terms[keep]
  
  collapse_terms(terms, max_terms = max_terms)
}

clean_concepts_one <- function(x, max_terms = 14) {
  
  terms <- split_terms(x)
  
  if (length(terms) == 0) return("")
  
  # Conceptos muy generales. Se conservan solo si no queda nada más específico.
  generic_exact <- tolower(c(
    "Humanities",
    "Philosophy",
    "Political science",
    "Geography",
    "Biology",
    "Chemistry",
    "Physics",
    "Medicine",
    "Computer science",
    "Engineering",
    "Sociology",
    "Psychology",
    "Art",
    "Business",
    "Management",
    "Economics",
    "Law",
    "Mathematics",
    "Environmental science"
  ))
  
  bad_patterns <- c(
    "\\(programming language\\)",
    "\\(typography\\)",
    "\\(fluid\\)",
    "\\(optics\\)",
    "\\(material\\)",
    "\\(computer security\\)",
    "\\(archaeology\\)",
    "\\(finance\\)",
    "\\(physics\\)"
  )
  
  terms_lower <- tolower(terms)
  
  is_generic <- terms_lower %in% generic_exact
  
  is_bad <- rep(FALSE, length(terms))
  for (pat in bad_patterns) {
    is_bad <- is_bad | grepl(pat, terms, ignore.case = TRUE)
  }
  
  specific_terms <- terms[!is_generic & !is_bad]
  generic_terms <- terms[is_generic & !is_bad]
  
  # Si hay términos específicos, se priorizan.
  if (length(specific_terms) > 0) {
    return(collapse_terms(specific_terms, max_terms = max_terms))
  }
  
  # Si no hay específicos, conservar pocos conceptos amplios.
  collapse_terms(generic_terms, max_terms = 3)
}

clean_topics_one <- function(x, title, abstract, keyword, max_terms = 8) {
  
  terms <- split_terms(x)
  
  if (length(terms) == 0) return("")
  
  has_covid_context <- has_covid_context_fun(title, abstract, keyword)
  
  bad_exact <- tolower(c(
    "Diverse Applied Research Studies"
  ))
  
  terms_lower <- tolower(terms)
  
  keep <- !(terms_lower %in% bad_exact)
  
  # Eliminar topics tipo COVID si el título/abstract/keywords no hablan realmente de COVID.
  covid_topic <- grepl(
    "covid|pandemic|pandemia",
    terms,
    ignore.case = TRUE
  )
  
  if (!has_covid_context) {
    keep <- keep & !covid_topic
  }
  
  terms <- terms[keep]
  
  collapse_terms(terms, max_terms = max_terms)
}

clean_source_one <- function(x) {
  x <- clean_text(x)
  
  if (x == "") return("")
  
  # Índices o fuentes genéricas que no son realmente revista/congreso.
  generic_sources <- c(
    "PubMed",
    "DOAJ",
    "DOAJ (DOAJ: Directory of Open Access Journals)"
  )
  
  if (tolower(x) %in% tolower(generic_sources)) {
    return("")
  }
  
  x
}

build_clean_embedding_text <- function(
    title,
    abstract,
    keyword_clean,
    conceptos_clean,
    topics_clean,
    source_clean,
    type
) {
  
  title <- clean_text(title)
  abstract <- clean_text(abstract)
  keyword_clean <- clean_text(keyword_clean)
  conceptos_clean <- clean_text(conceptos_clean)
  topics_clean <- clean_text(topics_clean)
  source_clean <- clean_text(source_clean)
  type <- clean_text(type)
  
  parts <- list(
    title = ifelse(title != "", paste0("Title: ", title), ""),
    abstract = ifelse(abstract != "", paste0("Abstract: ", abstract), ""),
    keywords = ifelse(keyword_clean != "", paste0("Keywords: ", keyword_clean), ""),
    topics = ifelse(topics_clean != "", paste0("Research topics: ", topics_clean), ""),
    concepts = ifelse(conceptos_clean != "", paste0("Concepts: ", conceptos_clean), ""),
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
# 3) Leer dataset
# ============================================================

dt <- fread(input_file)
setDT(dt)

cat("Filas leídas:", nrow(dt), "\n")

needed_cols <- c(
  "doc_id_v5",
  "doc_id_interest",
  "work_id",
  "title",
  "abstract",
  "keyword",
  "conceptos",
  "topics",
  "source",
  "type",
  "year",
  "citations",
  "language",
  "text_quality_v5",
  "embedding_eligible_v5",
  "text_for_embedding",
  "n_words_embedding_v5"
)

for (col in needed_cols) {
  if (!col %in% names(dt)) {
    dt[, (col) := ""]
  }
}

# ============================================================
# 4) Limpiar campos de metadatos
# ============================================================

dt[, title := clean_text(title)]
dt[, abstract := clean_text(abstract)]
dt[, keyword := clean_text(keyword)]
dt[, conceptos := clean_text(conceptos)]
dt[, topics := clean_text(topics)]
dt[, source := clean_text(source)]
dt[, type := clean_text(type)]

dt[, keyword_clean := vapply(
  keyword,
  clean_keywords_one,
  character(1),
  max_terms = 10
)]

dt[, conceptos_clean := vapply(
  conceptos,
  clean_concepts_one,
  character(1),
  max_terms = 14
)]

dt[, topics_clean := mapply(
  clean_topics_one,
  x = topics,
  title = title,
  abstract = abstract,
  keyword = keyword,
  MoreArgs = list(max_terms = 8),
  SIMPLIFY = TRUE,
  USE.NAMES = FALSE
)]

dt[, source_clean := vapply(
  source,
  clean_source_one,
  character(1)
)]

# ============================================================
# 5) Construir text_for_embedding limpio
# ============================================================

dt[, text_for_embedding_v5_clean := build_clean_embedding_text(
  title = title,
  abstract = abstract,
  keyword_clean = keyword_clean,
  conceptos_clean = conceptos_clean,
  topics_clean = topics_clean,
  source_clean = source_clean,
  type = type
)]

dt[, n_words_embedding_v5_clean := stringi::stri_count_words(text_for_embedding_v5_clean)]

dt[, has_title := has_content(title)]
dt[, has_abstract := has_content(abstract)]
dt[, has_keyword_clean := has_content(keyword_clean)]
dt[, has_conceptos_clean := has_content(conceptos_clean)]
dt[, has_topics_clean := has_content(topics_clean)]
dt[, has_source_clean := has_content(source_clean)]

dt[, contextual_metadata_count_clean :=
     as.integer(has_abstract) +
     as.integer(has_keyword_clean) +
     as.integer(has_conceptos_clean) +
     as.integer(has_topics_clean)]

dt[, text_quality_v5_clean := fcase(
  
  has_title == TRUE &
    has_abstract == TRUE &
    (has_keyword_clean == TRUE | has_conceptos_clean == TRUE | has_topics_clean == TRUE),
  "title_abstract_clean_metadata",
  
  has_title == TRUE &
    has_abstract == TRUE,
  "title_abstract",
  
  has_title == TRUE &
    has_abstract == FALSE &
    (has_keyword_clean == TRUE | has_conceptos_clean == TRUE | has_topics_clean == TRUE),
  "title_clean_metadata",
  
  has_title == TRUE &
    contextual_metadata_count_clean == 0 &
    has_source_clean == TRUE,
  "title_source_only",
  
  has_title == TRUE &
    contextual_metadata_count_clean == 0 &
    has_source_clean == FALSE,
  "title_only",
  
  default = "insufficient_text"
)]

dt[, embedding_eligible_v5_clean := has_title == TRUE &
     contextual_metadata_count_clean >= 1]

# ============================================================
# 6) Crear dataset limpio para embeddings
# ============================================================

dt_all_clean <- copy(dt)

dt_embeddings_clean <- dt_all_clean[
  embedding_eligible_v5_clean == TRUE &
    text_for_embedding_v5_clean != ""
]

dt_embeddings_out <- dt_embeddings_clean[
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
    embedding_eligible_v5,
    embedding_eligible_v5_clean,
    contextual_metadata_count,
    contextual_metadata_count_clean,
    text_for_embedding_original = text_for_embedding,
    text_for_embedding = text_for_embedding_v5_clean,
    n_words_embedding_v5_original = n_words_embedding_v5,
    n_words_embedding_v5_clean,
    metadata_score
  )
]

# ============================================================
# 7) Resúmenes
# ============================================================

summary_quality_clean <- dt_all_clean[
  ,
  .N,
  by = .(
    text_quality_v5_clean,
    embedding_eligible_v5_clean
  )
][order(-N)]

summary_quality_clean[, pct := round(N / sum(N) * 100, 2)]

summary_compare <- data.table(
  metric = c(
    "documents_total",
    "eligible_original",
    "eligible_clean",
    "not_eligible_clean",
    "mean_words_original",
    "mean_words_clean",
    "median_words_original",
    "median_words_clean",
    "documents_with_abstract",
    "documents_with_clean_keywords",
    "documents_with_clean_concepts",
    "documents_with_clean_topics"
  ),
  value = c(
    nrow(dt_all_clean),
    sum(dt_all_clean$embedding_eligible_v5 == TRUE, na.rm = TRUE),
    sum(dt_all_clean$embedding_eligible_v5_clean == TRUE, na.rm = TRUE),
    sum(dt_all_clean$embedding_eligible_v5_clean == FALSE, na.rm = TRUE),
    round(mean(safe_numeric(dt_all_clean$n_words_embedding_v5), na.rm = TRUE), 2),
    round(mean(safe_numeric(dt_all_clean$n_words_embedding_v5_clean), na.rm = TRUE), 2),
    round(median(safe_numeric(dt_all_clean$n_words_embedding_v5), na.rm = TRUE), 2),
    round(median(safe_numeric(dt_all_clean$n_words_embedding_v5_clean), na.rm = TRUE), 2),
    sum(dt_all_clean$has_abstract == TRUE, na.rm = TRUE),
    sum(dt_all_clean$has_keyword_clean == TRUE, na.rm = TRUE),
    sum(dt_all_clean$has_conceptos_clean == TRUE, na.rm = TRUE),
    sum(dt_all_clean$has_topics_clean == TRUE, na.rm = TRUE)
  )
)

check_patterns <- "data sharing|data citation|data reuse|dryad|biodiversity|coronavirus|covid"

check_cases <- dt_embeddings_out[
  grepl(check_patterns, text_for_embedding_original, ignore.case = TRUE) |
    grepl(check_patterns, text_for_embedding, ignore.case = TRUE),
  .(
    doc_id_v5,
    work_id,
    title,
    year,
    source,
    source_clean,
    text_quality_v5,
    text_quality_v5_clean,
    keyword_clean,
    conceptos_clean,
    topics_clean,
    original = substr(text_for_embedding_original, 1, 800),
    clean = substr(text_for_embedding, 1, 800)
  )
]

# ============================================================
# 8) Exportar
# ============================================================

fwrite(dt_all_clean, out_file_all)
fwrite(dt_embeddings_out, out_file_embeddings)
fwrite(summary_quality_clean, out_summary_quality)
fwrite(summary_compare, out_summary_compare)
fwrite(check_cases, out_check_cases)

cat("\n================ OBJETIVO 1 v5.1 TEXTO LIMPIO ================\n")

cat("\n================ RESUMEN COMPARATIVO ================\n")
print(summary_compare)

cat("\n================ CALIDAD TEXTUAL LIMPIA ================\n")
print(summary_quality_clean)

cat("\nCasos de revisión temática encontrados:", nrow(check_cases), "\n")

cat("\nArchivos creados:\n")
cat(out_file_all, "\n")
cat(out_file_embeddings, "\n")
cat(out_summary_quality, "\n")
cat(out_summary_compare, "\n")
cat(out_check_cases, "\n")