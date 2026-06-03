# ============================================================
# 21_objetivo1_v5_2_evaluar_perfil_individual.R
# Objetivo 1 v5.2 + QA
# Evaluar un perfil individual de Google Scholar contra
# el universo final clasificado.
#
# Uso:
# - Cambiar selected_profile_query para evaluar otro perfil.
# - Ejemplo: "alberto", "lopez", "torres", "corchado".
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

# ============================================================
# 1) Configuración
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

profiles_dir <- file.path(
  base_dir,
  "perfiles_research"
)

final_qa_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe_qa",
  "classification_final_universe_v5_2_corrected_984318_with_metascience_qa_preserve_original.csv"
)

# ------------------------------------------------------------
# Perfil a evaluar
# ------------------------------------------------------------
# Para Alberto Martín-Martín puedes usar:
selected_profile_query <- selected_profile_

# Si quieres forzar un archivo específico, escribe la ruta aquí.
# Si queda NA, el script busca por selected_profile_query.
selected_profile_file <- selected_profile_file_

# Cantidad de registros sugeridos para revisión manual
n_review_auto <- 20
n_review_top5 <- 30
n_review_critical <- 20
n_review_metascience <- 30
n_review_unmatched <- 30

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

find_col <- function(dt, candidates) {
  nms <- names(dt)
  nms_lower <- tolower(nms)
  candidates_lower <- tolower(candidates)
  
  for (cand in candidates_lower) {
    idx <- which(nms_lower == cand)
    if (length(idx) > 0) return(nms[idx[1]])
  }
  
  for (cand in candidates_lower) {
    idx <- grep(cand, nms_lower, fixed = TRUE)
    if (length(idx) > 0) return(nms[idx[1]])
  }
  
  NA_character_
}

guess_researcher_name <- function(file_path) {
  x <- basename(file_path)
  x <- sub("\\.csv$", "", x, ignore.case = TRUE)
  x <- sub("^gs[-_]", "", x, ignore.case = TRUE)
  x <- gsub("[-_]+", " ", x)
  x <- trimws(x)
  stringi::stri_trans_totitle(x)
}

make_safe_id <- function(x) {
  x <- clean_text(x)
  x <- tolower(x)
  x <- stringi::stri_trans_general(x, "Latin-ASCII")
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x
}

sample_n_safe <- function(dt, n) {
  if (nrow(dt) == 0) return(dt)
  dt[sample(.N, min(.N, n))]
}

short_text <- function(x, n = 1500) {
  x <- clean_text(x)
  ifelse(nchar(x) > n, paste0(substr(x, 1, n), " ..."), x)
}

make_top5_text <- function(dt) {
  paste(
    paste0("1) ", clean_text(dt$candidate_top1)),
    paste0("2) ", clean_text(dt$candidate_top2)),
    paste0("3) ", clean_text(dt$candidate_top3)),
    paste0("4) ", clean_text(dt$candidate_top4)),
    paste0("5) ", clean_text(dt$candidate_top5)),
    sep = "\n"
  )
}

# ============================================================
# 3) Validaciones iniciales
# ============================================================

if (!dir.exists(profiles_dir)) {
  stop("No existe profiles_dir: ", profiles_dir)
}

if (!file.exists(final_qa_file)) {
  stop("No existe final_qa_file: ", final_qa_file)
}

profile_files <- list.files(
  profiles_dir,
  pattern = "\\.csv$",
  full.names = TRUE
)

if (length(profile_files) == 0) {
  stop("No se encontraron perfiles CSV en: ", profiles_dir)
}

# ============================================================
# 4) Seleccionar perfil
# ============================================================

if (!is.na(selected_profile_file) && selected_profile_file != "") {
  
  profile_file <- selected_profile_file
  
  if (!file.exists(profile_file)) {
    stop("No existe selected_profile_file: ", profile_file)
  }
  
} else {
  
  candidates <- profile_files[
    grepl(
      selected_profile_query,
      basename(profile_files),
      ignore.case = TRUE
    )
  ]
  
  if (length(candidates) == 0) {
    stop(
      "No se encontró ningún perfil que contenga: ",
      selected_profile_query,
      "\nArchivos disponibles:\n",
      paste(basename(profile_files), collapse = "\n")
    )
  }
  
  if (length(candidates) > 1) {
    cat("\nSe encontraron varios perfiles candidatos:\n")
    print(basename(candidates))
    cat("\nSe usará el primero. Si quieres otro, usa selected_profile_file.\n")
  }
  
  profile_file <- candidates[1]
}

researcher_name <- guess_researcher_name(profile_file)
researcher_safe_id <- make_safe_id(researcher_name)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_evaluation",
  researcher_safe_id
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n================ PERFIL SELECCIONADO ================\n")
cat("Investigador:", researcher_name, "\n")
cat("Safe ID:", researcher_safe_id, "\n")
cat("Archivo:", profile_file, "\n")
cat("Salida:", out_dir, "\n")

# ============================================================
# 5) Leer perfil GS
# ============================================================

profile_raw <- fread(profile_file)
setDT(profile_raw)

title_col <- find_col(profile_raw, c("Title", "title", "titulo", "título", "publication_title"))
year_col <- find_col(profile_raw, c("Year", "year", "publication_year", "pub_year", "anio", "año"))
cites_col <- find_col(profile_raw, c("Cites", "cites", "citations", "cited_by", "citas"))
doi_col <- find_col(profile_raw, c("DOI", "doi"))
authors_col <- find_col(profile_raw, c("Authors", "authors", "author"))
source_col <- find_col(profile_raw, c("Source", "source", "journal", "revista", "publication"))
abstract_col <- find_col(profile_raw, c("Abstract", "abstract", "resumen"))
type_col <- find_col(profile_raw, c("Type", "type", "document_type"))
article_url_col <- find_col(profile_raw, c("ArticleURL", "article_url", "title_url", "url"))
fulltext_url_col <- find_col(profile_raw, c("FullTextURL", "fulltext_url", "pdf_url"))

if (is.na(title_col)) {
  stop("No se encontró columna de título en el perfil.")
}

profile <- data.table(
  researcher_name = researcher_name,
  researcher_safe_id = researcher_safe_id,
  profile_file = basename(profile_file),
  profile_path = profile_file,
  profile_row_id = seq_len(nrow(profile_raw)),
  profile_title = clean_text(profile_raw[[title_col]]),
  profile_year = if (!is.na(year_col)) safe_integer(profile_raw[[year_col]]) else NA_integer_,
  profile_cites = if (!is.na(cites_col)) safe_numeric(profile_raw[[cites_col]]) else NA_real_,
  profile_doi = if (!is.na(doi_col)) clean_text(profile_raw[[doi_col]]) else "",
  profile_authors = if (!is.na(authors_col)) clean_text(profile_raw[[authors_col]]) else "",
  profile_source = if (!is.na(source_col)) clean_text(profile_raw[[source_col]]) else "",
  profile_abstract = if (!is.na(abstract_col)) clean_text(profile_raw[[abstract_col]]) else "",
  profile_type = if (!is.na(type_col)) clean_text(profile_raw[[type_col]]) else "",
  profile_article_url = if (!is.na(article_url_col)) clean_text(profile_raw[[article_url_col]]) else "",
  profile_fulltext_url = if (!is.na(fulltext_url_col)) clean_text(profile_raw[[fulltext_url_col]]) else ""
)

profile <- profile[profile_title != ""]

profile[, profile_publication_key := paste(researcher_safe_id, profile_row_id, sep = "___")]
profile[, profile_title_norm := normalize_title(profile_title)]
profile[, profile_doi_norm := normalize_doi(profile_doi)]

profile[, profile_text_quality := fcase(
  profile_title != "" & profile_abstract != "" & profile_doi_norm != "",
  "title_abstract_doi",
  
  profile_title != "" & profile_abstract != "",
  "title_abstract",
  
  profile_title != "" & profile_doi_norm != "",
  "title_doi",
  
  profile_title != "",
  "title_only",
  
  default = "insufficient"
)]

# ============================================================
# 6) Leer universo final QA
# ============================================================

final_qa <- fread(final_qa_file)
setDT(final_qa)

needed_final_cols <- c(
  "doc_id_v5",
  "doc_id_interest",
  "work_id",
  "doi",
  "title",
  "year",
  "citations",
  "source",
  "source_clean",
  "type",
  "language",
  "original_final_decision",
  "original_final_confidence_tier",
  "original_auto_accept",
  "original_requires_review",
  "original_critical_conflict",
  "original_final_area_label",
  "original_final_subarea_label",
  "original_reportable_label",
  "qa_final_decision",
  "qa_final_confidence_tier",
  "qa_auto_accept",
  "qa_requires_review",
  "qa_final_area_label",
  "qa_final_subarea_label",
  "qa_reportable_label",
  "qa_metascience_flag",
  "qa_metascience_priority_review",
  "qa_metascience_review_type",
  "qa_changed_auto_accept",
  "qa_changed_decision",
  "top1_area_name",
  "top1_subarea_name",
  "top1_score",
  "top2_area_name",
  "top2_subarea_name",
  "top2_score",
  "top3_area_name",
  "top3_subarea_name",
  "top3_score",
  "top4_area_name",
  "top4_subarea_name",
  "top4_score",
  "top5_area_name",
  "top5_subarea_name",
  "top5_score",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "top1_top2_margin",
  "zero_shot_status",
  "confidence_level",
  "review_type",
  "review_reason",
  "has_metascience_bibliometrics_terms",
  "top1_is_bibliometrics",
  "top5_contains_bibliometrics",
  "possible_bibliometrics_metascience_conflict",
  "possible_bibliometrics_metascience_top5_failure",
  "possible_bibliometrics_metascience_rerank_case",
  "text_quality_v5_2",
  "embedding_eligible_v5_2",
  "keyword_clean",
  "topics_clean",
  "conceptos_clean",
  "text_for_embedding"
)

for (col in needed_final_cols) {
  if (!col %in% names(final_qa)) {
    final_qa[, (col) := NA]
  }
}

final_idx <- final_qa[
  ,
  .(
    final_doc_id_v5 = as.integer(doc_id_v5),
    doc_id_interest,
    final_work_id = clean_text(work_id),
    final_doi = clean_text(doi),
    final_title = clean_text(title),
    final_year = safe_integer(year),
    final_citations = safe_numeric(citations),
    final_source = clean_text(source),
    final_source_clean = clean_text(source_clean),
    final_type = clean_text(type),
    final_language = clean_text(language),
    
    original_final_decision = clean_text(original_final_decision),
    original_final_confidence_tier = clean_text(original_final_confidence_tier),
    original_auto_accept = as.logical(original_auto_accept),
    original_requires_review = as.logical(original_requires_review),
    original_critical_conflict = as.logical(original_critical_conflict),
    original_final_area_label = clean_text(original_final_area_label),
    original_final_subarea_label = clean_text(original_final_subarea_label),
    original_reportable_label = clean_text(original_reportable_label),
    
    qa_final_decision = clean_text(qa_final_decision),
    qa_final_confidence_tier = clean_text(qa_final_confidence_tier),
    qa_auto_accept = as.logical(qa_auto_accept),
    qa_requires_review = as.logical(qa_requires_review),
    qa_final_area_label = clean_text(qa_final_area_label),
    qa_final_subarea_label = clean_text(qa_final_subarea_label),
    qa_reportable_label = clean_text(qa_reportable_label),
    
    qa_metascience_flag = as.logical(qa_metascience_flag),
    qa_metascience_priority_review = as.logical(qa_metascience_priority_review),
    qa_metascience_review_type = clean_text(qa_metascience_review_type),
    qa_changed_auto_accept = as.logical(qa_changed_auto_accept),
    qa_changed_decision = as.logical(qa_changed_decision),
    
    top1_area_name = clean_text(top1_area_name),
    top1_subarea_name = clean_text(top1_subarea_name),
    top1_score = safe_numeric(top1_score),
    top2_area_name = clean_text(top2_area_name),
    top2_subarea_name = clean_text(top2_subarea_name),
    top2_score = safe_numeric(top2_score),
    top3_area_name = clean_text(top3_area_name),
    top3_subarea_name = clean_text(top3_subarea_name),
    top3_score = safe_numeric(top3_score),
    top4_area_name = clean_text(top4_area_name),
    top4_subarea_name = clean_text(top4_subarea_name),
    top4_score = safe_numeric(top4_score),
    top5_area_name = clean_text(top5_area_name),
    top5_subarea_name = clean_text(top5_subarea_name),
    top5_score = safe_numeric(top5_score),
    candidate_top1 = clean_text(candidate_top1),
    candidate_top2 = clean_text(candidate_top2),
    candidate_top3 = clean_text(candidate_top3),
    candidate_top4 = clean_text(candidate_top4),
    candidate_top5 = clean_text(candidate_top5),
    top1_top2_margin = safe_numeric(top1_top2_margin),
    zero_shot_status = clean_text(zero_shot_status),
    confidence_level = clean_text(confidence_level),
    review_type = clean_text(review_type),
    review_reason = clean_text(review_reason),
    
    has_metascience_bibliometrics_terms = as.logical(has_metascience_bibliometrics_terms),
    top1_is_bibliometrics = as.logical(top1_is_bibliometrics),
    top5_contains_bibliometrics = as.logical(top5_contains_bibliometrics),
    possible_bibliometrics_metascience_conflict = as.logical(possible_bibliometrics_metascience_conflict),
    possible_bibliometrics_metascience_top5_failure = as.logical(possible_bibliometrics_metascience_top5_failure),
    possible_bibliometrics_metascience_rerank_case = as.logical(possible_bibliometrics_metascience_rerank_case),
    
    text_quality_v5_2 = clean_text(text_quality_v5_2),
    embedding_eligible_v5_2 = as.logical(embedding_eligible_v5_2),
    keyword_clean = clean_text(keyword_clean),
    topics_clean = clean_text(topics_clean),
    conceptos_clean = clean_text(conceptos_clean),
    text_for_embedding = clean_text(text_for_embedding)
  )
]

final_idx[, final_title_norm := normalize_title(final_title)]
final_idx[, final_doi_norm := normalize_doi(final_doi)]

final_idx[, final_rank := fcase(
  qa_auto_accept == TRUE, 1L,
  qa_requires_review == TRUE, 2L,
  default = 3L
)]

final_by_doi <- final_idx[
  final_doi_norm != "",
  .SD[order(final_rank, -top1_score)][1],
  by = final_doi_norm
]

final_by_title_year <- final_idx[
  final_title_norm != "" & !is.na(final_year),
  .SD[order(final_rank, -top1_score)][1],
  by = .(final_title_norm, final_year)
]

final_by_title <- final_idx[
  final_title_norm != "",
  .SD[order(final_rank, -top1_score)][1],
  by = final_title_norm
]

# ============================================================
# 7) Matching perfil → universo final QA
# ============================================================

matches <- list()

# 7.1 DOI exacto
m_doi <- merge(
  profile[profile_doi_norm != ""],
  final_by_doi,
  by.x = "profile_doi_norm",
  by.y = "final_doi_norm",
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_doi) > 0) {
  m_doi[, profile_match_method := "doi_exact"]
  m_doi[, profile_match_priority := 1L]
  matches[["doi_exact"]] <- m_doi
}

matched_keys <- if (length(matches) > 0) {
  unique(rbindlist(matches, fill = TRUE)$profile_publication_key)
} else character()

# 7.2 Título + año exacto
remaining <- profile[
  !profile_publication_key %in% matched_keys
]

m_title_year <- merge(
  remaining[profile_title_norm != "" & !is.na(profile_year)],
  final_by_title_year,
  by.x = c("profile_title_norm", "profile_year"),
  by.y = c("final_title_norm", "final_year"),
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_title_year) > 0) {
  m_title_year[, profile_match_method := "title_year_exact"]
  m_title_year[, profile_match_priority := 2L]
  matches[["title_year_exact"]] <- m_title_year
}

matched_keys <- if (length(matches) > 0) {
  unique(rbindlist(matches, fill = TRUE)$profile_publication_key)
} else character()

# 7.3 Título exacto normalizado
remaining <- profile[
  !profile_publication_key %in% matched_keys
]

m_title <- merge(
  remaining[profile_title_norm != ""],
  final_by_title,
  by.x = "profile_title_norm",
  by.y = "final_title_norm",
  all = FALSE,
  allow.cartesian = FALSE
)

if (nrow(m_title) > 0) {
  m_title[, profile_match_method := "title_norm_exact"]
  m_title[, profile_match_priority := 3L]
  matches[["title_norm_exact"]] <- m_title
}

profile_matches <- if (length(matches) > 0) {
  rbindlist(matches, fill = TRUE)
} else {
  data.table()
}

if (nrow(profile_matches) > 0) {
  setorder(
    profile_matches,
    profile_publication_key,
    profile_match_priority,
    final_rank,
    -top1_score
  )
  
  profile_matches <- profile_matches[
    ,
    .SD[1],
    by = profile_publication_key
  ]
}

matched_keys <- unique(profile_matches$profile_publication_key)

profile_unmatched <- profile[
  !profile_publication_key %in% matched_keys
]

# ============================================================
# 8) Fuzzy candidates opcional para no encontrados
# ============================================================

fuzzy_candidates <- data.table()

if (requireNamespace("stringdist", quietly = TRUE) && nrow(profile_unmatched) > 0) {
  
  cat("\nGenerando candidatos fuzzy para no encontrados...\n")
  
  # Para no hacerlo muy pesado, comparar solo títulos con longitud razonable
  unmatched_tmp <- profile_unmatched[
    nchar(profile_title_norm) >= 15
  ]
  
  final_title_pool <- final_by_title[
    nchar(final_title_norm) >= 15,
    .(
      final_doc_id_v5,
      final_title,
      final_title_norm,
      final_year,
      final_work_id,
      qa_final_decision,
      qa_final_area_label,
      qa_final_subarea_label,
      top1_area_name,
      top1_subarea_name,
      top1_score
    )
  ]
  
  if (nrow(unmatched_tmp) > 0 && nrow(final_title_pool) > 0) {
    
    fuzzy_list <- lapply(seq_len(nrow(unmatched_tmp)), function(i) {
      
      q <- unmatched_tmp[i]
      
      # Reducir búsqueda por año si existe
      pool <- if (!is.na(q$profile_year)) {
        final_title_pool[
          is.na(final_year) |
            final_year %in% c(q$profile_year - 1L, q$profile_year, q$profile_year + 1L)
        ]
      } else {
        final_title_pool
      }
      
      if (nrow(pool) == 0) return(data.table())
      
      d <- stringdist::stringsim(
        q$profile_title_norm,
        pool$final_title_norm,
        method = "jw"
      )
      
      idx <- order(d, decreasing = TRUE)[1:min(5, length(d))]
      
      data.table(
        profile_publication_key = q$profile_publication_key,
        profile_row_id = q$profile_row_id,
        profile_title = q$profile_title,
        profile_year = q$profile_year,
        profile_cites = q$profile_cites,
        fuzzy_rank = seq_along(idx),
        fuzzy_similarity = round(d[idx], 4),
        final_doc_id_v5 = pool$final_doc_id_v5[idx],
        final_title = pool$final_title[idx],
        final_year = pool$final_year[idx],
        final_work_id = pool$final_work_id[idx],
        qa_final_decision = pool$qa_final_decision[idx],
        qa_final_area_label = pool$qa_final_area_label[idx],
        qa_final_subarea_label = pool$qa_final_subarea_label[idx],
        top1_area_name = pool$top1_area_name[idx],
        top1_subarea_name = pool$top1_subarea_name[idx],
        top1_score = pool$top1_score[idx]
      )
    })
    
    fuzzy_candidates <- rbindlist(fuzzy_list, fill = TRUE)
    fuzzy_candidates <- fuzzy_candidates[
      fuzzy_similarity >= 0.92
    ][order(profile_row_id, -fuzzy_similarity)]
  }
}

# ============================================================
# 9) Preparar tabla para revisión manual del perfil
# ============================================================

if (nrow(profile_matches) > 0) {
  
  profile_matches[, candidate_top5_text := make_top5_text(.SD)]
  
  profile_matches[, text_for_review := paste(
    paste0("Título perfil: ", profile_title),
    paste0("Título universo: ", final_title),
    paste0("Fuente perfil: ", profile_source),
    paste0("Fuente universo: ", final_source),
    paste0("Año perfil: ", profile_year),
    paste0("Año universo: ", final_year),
    paste0("Candidatos top-k:\n", candidate_top5_text),
    sep = "\n\n"
  )]
  
  profile_matches[, text_for_review_short := short_text(text_for_review, 1800)]
  
  profile_matches[, validation_group := fcase(
    qa_auto_accept == TRUE,
    "01_profile_qa_auto_accepted",
    
    qa_metascience_priority_review == TRUE,
    "02_profile_metascience_qa_priority",
    
    original_critical_conflict == TRUE,
    "03_profile_critical_conflict",
    
    qa_requires_review == TRUE,
    "04_profile_review_top5",
    
    default = "05_profile_other"
  )]
  
  review_auto <- sample_n_safe(
    profile_matches[validation_group == "01_profile_qa_auto_accepted"],
    n_review_auto
  )
  
  review_top5 <- sample_n_safe(
    profile_matches[validation_group == "04_profile_review_top5"],
    n_review_top5
  )
  
  review_critical <- sample_n_safe(
    profile_matches[validation_group == "03_profile_critical_conflict"],
    n_review_critical
  )
  
  review_metascience <- sample_n_safe(
    profile_matches[validation_group == "02_profile_metascience_qa_priority"],
    n_review_metascience
  )
  
  review_sample <- rbindlist(
    list(
      review_auto,
      review_metascience,
      review_critical,
      review_top5
    ),
    fill = TRUE
  )
  
  review_sample <- unique(
    review_sample,
    by = c("profile_publication_key", "final_doc_id_v5")
  )
  
} else {
  
  review_sample <- data.table()
}

# Agregar no encontrados a revisión
if (nrow(profile_unmatched) > 0) {
  
  unmatched_review <- sample_n_safe(profile_unmatched, n_review_unmatched)
  
  unmatched_review[, validation_group := "06_profile_unmatched"]
  unmatched_review[, final_doc_id_v5 := NA_integer_]
  unmatched_review[, final_work_id := ""]
  unmatched_review[, final_title := ""]
  unmatched_review[, final_year := NA_integer_]
  unmatched_review[, qa_final_decision := "not_matched_in_final_universe"]
  unmatched_review[, qa_final_area_label := ""]
  unmatched_review[, qa_final_subarea_label := ""]
  unmatched_review[, top1_area_name := ""]
  unmatched_review[, top1_subarea_name := ""]
  unmatched_review[, candidate_top1 := ""]
  unmatched_review[, candidate_top2 := ""]
  unmatched_review[, candidate_top3 := ""]
  unmatched_review[, candidate_top4 := ""]
  unmatched_review[, candidate_top5 := ""]
  unmatched_review[, text_for_review_short := paste0(
    "Título perfil: ", profile_title,
    "\n\nNo emparejado con el universo final v5.2 + QA."
  )]
  
  review_sample <- rbindlist(
    list(review_sample, unmatched_review),
    fill = TRUE
  )
}

if (nrow(review_sample) > 0) {
  
  review_sample[, manual_area_correct := ""]
  review_sample[, manual_subarea_correct := ""]
  review_sample[, manual_area_label := ""]
  review_sample[, manual_subarea_label := ""]
  review_sample[, manual_decision := ""]
  review_sample[, manual_comment := ""]
  
  review_cols <- c(
    "validation_group",
    "researcher_name",
    "researcher_safe_id",
    "profile_file",
    "profile_row_id",
    "profile_title",
    "profile_year",
    "profile_cites",
    "profile_source",
    "profile_match_method",
    "final_doc_id_v5",
    "final_work_id",
    "final_title",
    "final_year",
    "qa_final_decision",
    "qa_final_confidence_tier",
    "qa_auto_accept",
    "qa_requires_review",
    "qa_final_area_label",
    "qa_final_subarea_label",
    "qa_metascience_flag",
    "qa_metascience_priority_review",
    "qa_metascience_review_type",
    "original_final_decision",
    "original_auto_accept",
    "original_final_area_label",
    "original_final_subarea_label",
    "candidate_top1",
    "candidate_top2",
    "candidate_top3",
    "candidate_top4",
    "candidate_top5",
    "top1_area_name",
    "top1_subarea_name",
    "top1_score",
    "top2_area_name",
    "top2_subarea_name",
    "top2_score",
    "top1_top2_margin",
    "text_for_review_short",
    "manual_area_correct",
    "manual_subarea_correct",
    "manual_area_label",
    "manual_subarea_label",
    "manual_decision",
    "manual_comment"
  )
  
  for (col in review_cols) {
    if (!col %in% names(review_sample)) {
      review_sample[, (col) := ""]
    }
  }
  
  review_sample <- review_sample[, ..review_cols]
  
  setorder(
    review_sample,
    validation_group,
    -profile_cites,
    profile_year,
    profile_title
  )
}

# ============================================================
# 10) Resúmenes
# ============================================================

summary_matching <- data.table(
  metric = c(
    "profile_publications_clean",
    "matched_total",
    "unmatched_total",
    "match_rate_pct",
    "fuzzy_candidates_review",
    "profile_total_citations"
  ),
  value = c(
    nrow(profile),
    nrow(profile_matches),
    nrow(profile_unmatched),
    round(nrow(profile_matches) / nrow(profile) * 100, 2),
    nrow(fuzzy_candidates),
    sum(profile$profile_cites, na.rm = TRUE)
  )
)

summary_match_method <- if (nrow(profile_matches) > 0) {
  profile_matches[, .N, by = profile_match_method][order(-N)]
} else data.table(profile_match_method = character(), N = integer())

summary_original_decision <- if (nrow(profile_matches) > 0) {
  profile_matches[
    ,
    .N,
    by = .(
      original_final_decision,
      original_final_confidence_tier,
      original_auto_accept,
      original_requires_review
    )
  ][order(-N)]
} else data.table()

if (nrow(summary_original_decision) > 0) {
  summary_original_decision[, pct := round(N / sum(N) * 100, 2)]
}

summary_qa_decision <- if (nrow(profile_matches) > 0) {
  profile_matches[
    ,
    .N,
    by = .(
      qa_final_decision,
      qa_final_confidence_tier,
      qa_auto_accept,
      qa_requires_review
    )
  ][order(-N)]
} else data.table()

if (nrow(summary_qa_decision) > 0) {
  summary_qa_decision[, pct := round(N / sum(N) * 100, 2)]
}

summary_qa_effect <- if (nrow(profile_matches) > 0) {
  data.table(
    metric = c(
      "original_auto_accept",
      "qa_auto_accept",
      "auto_removed_by_qa",
      "qa_metascience_priority_review",
      "qa_changed_decision",
      "critical_conflict_original"
    ),
    value = c(
      profile_matches[original_auto_accept == TRUE, .N],
      profile_matches[qa_auto_accept == TRUE, .N],
      profile_matches[original_auto_accept == TRUE & qa_auto_accept == FALSE, .N],
      profile_matches[qa_metascience_priority_review == TRUE, .N],
      profile_matches[qa_changed_decision == TRUE, .N],
      profile_matches[original_critical_conflict == TRUE, .N]
    )
  )
} else data.table(metric = character(), value = numeric())

summary_area_qa_auto <- if (nrow(profile_matches) > 0) {
  profile_matches[
    qa_auto_accept == TRUE & qa_final_area_label != "",
    .(
      publications = .N,
      profile_citations = sum(profile_cites, na.rm = TRUE),
      mean_top1_score = round(mean(top1_score, na.rm = TRUE), 4),
      mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4)
    ),
    by = .(
      qa_final_area_label,
      qa_final_subarea_label
    )
  ][order(-publications)]
} else data.table()

if (nrow(summary_area_qa_auto) > 0) {
  summary_area_qa_auto[, pct := round(publications / sum(publications) * 100, 2)]
}

summary_top1_area_all_matches <- if (nrow(profile_matches) > 0) {
  profile_matches[
    ,
    .(
      publications = .N,
      profile_citations = sum(profile_cites, na.rm = TRUE),
      auto_accept_qa = sum(qa_auto_accept == TRUE, na.rm = TRUE),
      review_qa = sum(qa_requires_review == TRUE, na.rm = TRUE),
      metascience_priority = sum(qa_metascience_priority_review == TRUE, na.rm = TRUE)
    ),
    by = .(
      top1_area_name,
      top1_subarea_name
    )
  ][order(-publications)]
} else data.table()

summary_year <- profile[
  !is.na(profile_year),
  .(
    profile_publications = .N,
    profile_citations = sum(profile_cites, na.rm = TRUE)
  ),
  by = profile_year
][order(profile_year)]

summary_year_matched <- if (nrow(profile_matches) > 0) {
  profile_matches[
    !is.na(profile_year),
    .(
      matched = .N,
      qa_auto_accept = sum(qa_auto_accept == TRUE, na.rm = TRUE),
      qa_review = sum(qa_requires_review == TRUE, na.rm = TRUE),
      metascience_priority = sum(qa_metascience_priority_review == TRUE, na.rm = TRUE)
    ),
    by = profile_year
  ][order(profile_year)]
} else data.table()

summary_year <- merge(
  summary_year,
  summary_year_matched,
  by = "profile_year",
  all.x = TRUE
)

for (col in c("matched", "qa_auto_accept", "qa_review", "metascience_priority")) {
  if (col %in% names(summary_year)) {
    summary_year[is.na(get(col)), (col) := 0L]
  }
}

summary_review_groups <- if (nrow(review_sample) > 0) {
  review_sample[, .N, by = validation_group][order(validation_group)]
} else data.table(validation_group = character(), N = integer())

# ============================================================
# 11) Exportar tablas
# ============================================================

fwrite(profile, file.path(out_dir, "profile_publications_clean.csv"))
fwrite(profile_matches, file.path(out_dir, "profile_publications_matched_final_qa.csv"))
fwrite(profile_unmatched, file.path(out_dir, "profile_publications_unmatched.csv"))
fwrite(fuzzy_candidates, file.path(out_dir, "profile_fuzzy_candidates_review.csv"))
fwrite(review_sample, file.path(out_dir, "profile_manual_review_sample.csv"))

fwrite(summary_matching, file.path(out_dir, "summary_matching.csv"))
fwrite(summary_match_method, file.path(out_dir, "summary_match_method.csv"))
fwrite(summary_original_decision, file.path(out_dir, "summary_original_decision.csv"))
fwrite(summary_qa_decision, file.path(out_dir, "summary_qa_decision.csv"))
fwrite(summary_qa_effect, file.path(out_dir, "summary_qa_effect.csv"))
fwrite(summary_area_qa_auto, file.path(out_dir, "summary_area_qa_auto.csv"))
fwrite(summary_top1_area_all_matches, file.path(out_dir, "summary_top1_area_all_matches.csv"))
fwrite(summary_year, file.path(out_dir, "summary_year.csv"))
fwrite(summary_review_groups, file.path(out_dir, "summary_review_groups.csv"))

# Guardar configuración
profile_config <- data.table(
  researcher_name = researcher_name,
  researcher_safe_id = researcher_safe_id,
  profile_file = basename(profile_file),
  profile_path = profile_file,
  output_dir = out_dir,
  title_col = title_col,
  year_col = year_col,
  cites_col = cites_col,
  doi_col = doi_col,
  authors_col = authors_col,
  source_col = source_col,
  abstract_col = abstract_col,
  type_col = type_col
)

fwrite(profile_config, file.path(out_dir, "profile_config.csv"))

# ============================================================
# 12) XLSX para revisión manual
# ============================================================

xlsx_file <- file.path(out_dir, "profile_manual_review_sample.xlsx")

if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  wb <- openxlsx::createWorkbook()
  
  openxlsx::addWorksheet(wb, "revision_perfil")
  openxlsx::addWorksheet(wb, "resumen_matching")
  openxlsx::addWorksheet(wb, "resumen_qa")
  openxlsx::addWorksheet(wb, "areas_qa_auto")
  openxlsx::addWorksheet(wb, "instrucciones")
  
  openxlsx::writeData(wb, "revision_perfil", review_sample)
  openxlsx::writeData(wb, "resumen_matching", summary_matching)
  openxlsx::writeData(wb, "resumen_qa", summary_qa_decision)
  openxlsx::writeData(wb, "areas_qa_auto", summary_area_qa_auto)
  
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
      "TRUE si el área aceptada o candidata es correcta; FALSE si es incorrecta.",
      "TRUE si la subárea aceptada o candidata es correcta; FALSE si es incorrecta.",
      "Si el área es incorrecta, escribir el área correcta.",
      "Si la subárea es incorrecta, escribir la subárea correcta.",
      "Usar: correct, area_correct_subarea_wrong, area_wrong, insufficient_information, ambiguous_or_both_valid, top5_contains_correct_label, top5_does_not_contain_correct_label, not_matched.",
      "Comentario breve sobre la corrección o duda."
    )
  )
  
  openxlsx::writeData(wb, "instrucciones", instrucciones)
  
  openxlsx::freezePane(wb, "revision_perfil", firstActiveRow = 2, firstActiveCol = 6)
  openxlsx::setColWidths(wb, "revision_perfil", cols = 1:ncol(review_sample), widths = "auto")
  
  text_col <- which(names(review_sample) == "text_for_review_short")
  if (length(text_col) > 0) {
    openxlsx::setColWidths(wb, "revision_perfil", cols = text_col, widths = 90)
  }
  
  openxlsx::saveWorkbook(wb, xlsx_file, overwrite = TRUE)
}

# ============================================================
# 13) Gráficos básicos
# ============================================================

if (requireNamespace("ggplot2", quietly = TRUE)) {
  
  plots_dir <- file.path(out_dir, "plots")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  
  # 13.1 Publicaciones por año
  if (nrow(summary_year) > 0) {
    
    p_year <- ggplot2::ggplot(
      summary_year,
      ggplot2::aes(x = profile_year, y = profile_publications)
    ) +
      ggplot2::geom_col() +
      ggplot2::labs(
        title = paste0("Publicaciones por año - ", researcher_name),
        x = "Año",
        y = "Publicaciones en perfil"
      ) +
      ggplot2::theme_minimal()
    
    ggplot2::ggsave(
      file.path(plots_dir, "profile_publications_by_year.png"),
      p_year,
      width = 10,
      height = 5,
      dpi = 150
    )
  }
  
  # 13.2 Decisión QA
  if (nrow(summary_qa_decision) > 0) {
    
    p_decision <- ggplot2::ggplot(
      summary_qa_decision,
      ggplot2::aes(x = reorder(qa_final_decision, N), y = N)
    ) +
      ggplot2::geom_col() +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = paste0("Decisión QA v5.2 - ", researcher_name),
        x = "Decisión QA",
        y = "Publicaciones emparejadas"
      ) +
      ggplot2::theme_minimal()
    
    ggplot2::ggsave(
      file.path(plots_dir, "profile_qa_decision_distribution.png"),
      p_decision,
      width = 10,
      height = 6,
      dpi = 150
    )
  }
  
  # 13.3 Áreas aceptadas automáticamente
  if (nrow(summary_area_qa_auto) > 0) {
    
    p_area <- ggplot2::ggplot(
      head(summary_area_qa_auto, 20),
      ggplot2::aes(
        x = reorder(
          paste(qa_final_area_label, qa_final_subarea_label, sep = " / "),
          publications
        ),
        y = publications
      )
    ) +
      ggplot2::geom_col() +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = paste0("Áreas/subáreas aceptadas automáticamente QA - ", researcher_name),
        x = "Área / subárea",
        y = "Publicaciones"
      ) +
      ggplot2::theme_minimal()
    
    ggplot2::ggsave(
      file.path(plots_dir, "profile_qa_auto_areas.png"),
      p_area,
      width = 11,
      height = 7,
      dpi = 150
    )
  }
}

# ============================================================
# 14) Imprimir resultados
# ============================================================

cat("\n================ EVALUACIÓN PERFIL INDIVIDUAL v5.2 + QA ================\n")

cat("\nPerfil:\n")
print(profile_config)

cat("\nResumen matching:\n")
print(summary_matching)

cat("\nMatching por método:\n")
print(summary_match_method)

cat("\nDecisión original v5.2:\n")
print(summary_original_decision)

cat("\nDecisión QA v5.2:\n")
print(summary_qa_decision)

cat("\nEfecto QA:\n")
print(summary_qa_effect)

cat("\nÁreas aceptadas automáticamente QA:\n")
print(summary_area_qa_auto)

cat("\nTop1 áreas en todos los emparejados:\n")
print(head(summary_top1_area_all_matches, 30))

cat("\nMuestra de revisión:\n")
print(summary_review_groups)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")