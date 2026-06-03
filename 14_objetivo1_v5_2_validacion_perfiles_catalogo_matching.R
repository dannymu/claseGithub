# ============================================================
# 14_objetivo1_v5_2_validacion_perfiles_catalogo_matching.R
# Objetivo 1 v5.2
# Validación externa por perfiles de investigadores
#
# Propósito:
# - Leer todos los perfiles de Google Scholar disponibles.
# - Emparejarlos con el universo final v5.2.
# - Calcular cobertura por perfil.
# - Identificar perfiles candidatos para validación externa.
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

profiles_dir <- file.path(
  base_dir,
  "perfiles_research"
)

final_universe_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_final_universe",
  "classification_final_universe_v5_2_corrected_984318.csv"
)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_validation"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_profiles_catalog <- file.path(out_dir, "profiles_catalog_v5_2.csv")
out_profile_publications <- file.path(out_dir, "profile_publications_clean_v5_2.csv")
out_profile_matches <- file.path(out_dir, "profile_publications_matched_final_v5_2.csv")
out_profile_unmatched <- file.path(out_dir, "profile_publications_unmatched_final_v5_2.csv")
out_summary_profiles <- file.path(out_dir, "summary_profiles_matching_v5_2.csv")
out_summary_profile_area <- file.path(out_dir, "summary_profile_area_auto_v5_2.csv")
out_candidate_profiles <- file.path(out_dir, "candidate_profiles_for_external_validation_v5_2.csv")

if (!dir.exists(profiles_dir)) {
  stop("No existe la carpeta de perfiles: ", profiles_dir)
}

if (!file.exists(final_universe_file)) {
  stop("No existe el universo final corregido v5.2: ", final_universe_file)
}

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

read_profile_file <- function(file_path, option_id) {
  
  dt <- tryCatch(
    fread(file_path),
    error = function(e) data.table()
  )
  
  if (nrow(dt) == 0) {
    return(data.table())
  }
  
  title_col <- find_col(dt, c("Title", "title", "titulo", "título", "publication_title"))
  year_col <- find_col(dt, c("Year", "year", "publication_year", "pub_year", "anio", "año"))
  cites_col <- find_col(dt, c("Cites", "cites", "citations", "cited_by", "citas"))
  doi_col <- find_col(dt, c("DOI", "doi"))
  authors_col <- find_col(dt, c("Authors", "authors", "author"))
  source_col <- find_col(dt, c("Source", "source", "journal", "revista", "publication"))
  abstract_col <- find_col(dt, c("Abstract", "abstract", "resumen"))
  type_col <- find_col(dt, c("Type", "type", "document_type"))
  url_col <- find_col(dt, c("ArticleURL", "article_url", "title_url", "url"))
  
  if (is.na(title_col)) {
    return(data.table())
  }
  
  researcher_name <- guess_researcher_name(file_path)
  researcher_safe_id <- make_safe_id(researcher_name)
  
  out <- data.table(
    profile_option = option_id,
    researcher_name = researcher_name,
    researcher_safe_id = researcher_safe_id,
    profile_file = basename(file_path),
    profile_path = file_path,
    profile_row_id = seq_len(nrow(dt)),
    profile_title = clean_text(dt[[title_col]]),
    profile_year = if (!is.na(year_col)) safe_integer(dt[[year_col]]) else NA_integer_,
    profile_cites = if (!is.na(cites_col)) safe_numeric(dt[[cites_col]]) else NA_real_,
    profile_doi = if (!is.na(doi_col)) clean_text(dt[[doi_col]]) else "",
    profile_authors = if (!is.na(authors_col)) clean_text(dt[[authors_col]]) else "",
    profile_source = if (!is.na(source_col)) clean_text(dt[[source_col]]) else "",
    profile_abstract = if (!is.na(abstract_col)) clean_text(dt[[abstract_col]]) else "",
    profile_type = if (!is.na(type_col)) clean_text(dt[[type_col]]) else "",
    profile_url = if (!is.na(url_col)) clean_text(dt[[url_col]]) else ""
  )
  
  out <- out[profile_title != ""]
  
  out[, profile_publication_key := paste(researcher_safe_id, profile_row_id, sep = "___")]
  out[, profile_title_norm := normalize_title(profile_title)]
  out[, profile_doi_norm := normalize_doi(profile_doi)]
  
  out
}

# ============================================================
# 3) Leer universo final v5.2
# ============================================================

final_universe <- fread(final_universe_file)
setDT(final_universe)

cat("Universo final v5.2:", nrow(final_universe), "\n")

needed_final_cols <- c(
  "doc_id_v5",
  "work_id",
  "doi",
  "title",
  "year",
  "final_area_label",
  "final_subarea_label",
  "reportable_label",
  "final_decision",
  "final_confidence_tier",
  "auto_accept",
  "requires_review",
  "critical_conflict",
  "review_type",
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
  "top1_top2_margin"
)

for (col in needed_final_cols) {
  if (!col %in% names(final_universe)) {
    final_universe[, (col) := NA]
  }
}

final_idx <- final_universe[
  ,
  .(
    final_doc_id_v5 = doc_id_v5,
    final_work_id = clean_text(work_id),
    final_doi = clean_text(doi),
    final_title = clean_text(title),
    final_year = safe_integer(year),
    final_area_label = clean_text(final_area_label),
    final_subarea_label = clean_text(final_subarea_label),
    reportable_label = clean_text(reportable_label),
    final_decision = clean_text(final_decision),
    final_confidence_tier = clean_text(final_confidence_tier),
    auto_accept = as.logical(auto_accept),
    requires_review = as.logical(requires_review),
    critical_conflict = as.logical(critical_conflict),
    review_type = clean_text(review_type),
    candidate_top1 = clean_text(candidate_top1),
    candidate_top2 = clean_text(candidate_top2),
    candidate_top3 = clean_text(candidate_top3),
    candidate_top4 = clean_text(candidate_top4),
    candidate_top5 = clean_text(candidate_top5),
    top1_area_name = clean_text(top1_area_name),
    top1_subarea_name = clean_text(top1_subarea_name),
    top1_score = safe_numeric(top1_score),
    top2_area_name = clean_text(top2_area_name),
    top2_subarea_name = clean_text(top2_subarea_name),
    top2_score = safe_numeric(top2_score),
    top1_top2_margin = safe_numeric(top1_top2_margin)
  )
]

final_idx[, final_title_norm := normalize_title(final_title)]
final_idx[, final_doi_norm := normalize_doi(final_doi)]

final_idx[, final_rank := fifelse(auto_accept == TRUE, 1L, 2L)]
final_idx[is.na(final_rank), final_rank := 3L]

# Índices únicos para matching
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
# 4) Leer perfiles
# ============================================================

profile_files <- list.files(
  profiles_dir,
  pattern = "\\.csv$",
  full.names = TRUE
)

if (length(profile_files) == 0) {
  stop("No se encontraron archivos CSV en perfiles_research.")
}

profiles_catalog <- rbindlist(
  lapply(seq_along(profile_files), function(i) {
    fp <- profile_files[i]
    
    preview <- tryCatch(fread(fp, nrows = 5), error = function(e) data.table())
    
    data.table(
      profile_option = i,
      researcher_name = guess_researcher_name(fp),
      researcher_safe_id = make_safe_id(guess_researcher_name(fp)),
      profile_file = basename(fp),
      profile_path = fp,
      rows_detected = tryCatch(nrow(fread(fp, select = 1)), error = function(e) NA_integer_),
      detected_columns = if (nrow(preview) > 0) paste(names(preview), collapse = " | ") else ""
    )
  }),
  fill = TRUE
)

profile_publications <- rbindlist(
  lapply(seq_along(profile_files), function(i) {
    read_profile_file(profile_files[i], i)
  }),
  fill = TRUE
)

cat("Perfiles detectados:", nrow(profiles_catalog), "\n")
cat("Publicaciones de perfiles:", nrow(profile_publications), "\n")

# ============================================================
# 5) Matching perfil → universo final
# ============================================================

matches <- list()

# 5.1 DOI exacto
m_doi <- merge(
  profile_publications[profile_doi_norm != ""],
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
} else {
  character()
}

# 5.2 Título + año exacto
remaining <- profile_publications[
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
} else {
  character()
}

# 5.3 Título exacto normalizado
remaining <- profile_publications[
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

profile_unmatched <- profile_publications[
  !profile_publication_key %in% matched_keys
]

# ============================================================
# 6) Resúmenes por perfil
# ============================================================

summary_base <- profile_publications[
  ,
  .(
    profile_publications = .N,
    with_doi = sum(profile_doi_norm != ""),
    with_year = sum(!is.na(profile_year)),
    total_profile_cites = sum(profile_cites, na.rm = TRUE)
  ),
  by = .(
    profile_option,
    researcher_name,
    researcher_safe_id,
    profile_file
  )
]

summary_match <- profile_matches[
  ,
  .(
    matched_publications = .N,
    matched_auto_accept = sum(auto_accept == TRUE, na.rm = TRUE),
    matched_review_required = sum(requires_review == TRUE, na.rm = TRUE),
    matched_critical_conflict = sum(critical_conflict == TRUE, na.rm = TRUE),
    matched_not_classified = sum(final_decision == "not_classified_insufficient_metadata", na.rm = TRUE),
    mean_top1_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4)
  ),
  by = .(
    profile_option,
    researcher_name,
    researcher_safe_id,
    profile_file
  )
]

summary_profiles <- merge(
  summary_base,
  summary_match,
  by = c("profile_option", "researcher_name", "researcher_safe_id", "profile_file"),
  all.x = TRUE
)

num_cols <- c(
  "matched_publications",
  "matched_auto_accept",
  "matched_review_required",
  "matched_critical_conflict",
  "matched_not_classified"
)

for (col in num_cols) {
  summary_profiles[is.na(get(col)), (col) := 0L]
}

summary_profiles[
  ,
  unmatched_publications := profile_publications - matched_publications
]

summary_profiles[
  ,
  match_rate_pct := round(matched_publications / profile_publications * 100, 2)
]

summary_profiles[
  ,
  auto_accept_pct_of_matched := fifelse(
    matched_publications > 0,
    round(matched_auto_accept / matched_publications * 100, 2),
    NA_real_
  )
]

summary_profiles[
  ,
  review_pct_of_matched := fifelse(
    matched_publications > 0,
    round(matched_review_required / matched_publications * 100, 2),
    NA_real_
  )
]

# ============================================================
# 7) Áreas automáticas por perfil
# ============================================================

summary_profile_area <- profile_matches[
  auto_accept == TRUE &
    final_area_label != "",
  .(
    publications = .N,
    citations = sum(profile_cites, na.rm = TRUE),
    mean_score = round(mean(top1_score, na.rm = TRUE), 4),
    mean_margin = round(mean(top1_top2_margin, na.rm = TRUE), 4)
  ),
  by = .(
    profile_option,
    researcher_name,
    researcher_safe_id,
    profile_file,
    final_area_label,
    final_subarea_label
  )
][order(researcher_name, -publications)]

summary_profile_area[
  ,
  pct_within_profile_auto := round(publications / sum(publications) * 100, 2),
  by = researcher_safe_id
]

# Área dominante por perfil
dominant_area <- summary_profile_area[
  ,
  .SD[order(-publications)][1],
  by = researcher_safe_id
][
  ,
  .(
    researcher_safe_id,
    dominant_area = final_area_label,
    dominant_subarea = final_subarea_label,
    dominant_area_publications = publications,
    dominant_area_pct_auto = pct_within_profile_auto
  )
]

summary_profiles <- merge(
  summary_profiles,
  dominant_area,
  by = "researcher_safe_id",
  all.x = TRUE
)

# ============================================================
# 8) Candidatos para validación externa
# ============================================================

candidate_profiles <- summary_profiles[
  matched_publications >= 20
][order(dominant_area, -matched_auto_accept, -match_rate_pct)]

candidate_profiles[
  ,
  validation_priority := fcase(
    matched_auto_accept >= 20 & auto_accept_pct_of_matched >= 40,
    "alta_prioridad_validacion_externa",
    
    matched_publications >= 20,
    "prioridad_media_validacion_externa",
    
    default = "baja_prioridad"
  )
]

# ============================================================
# 9) Exportar
# ============================================================

fwrite(profiles_catalog, out_profiles_catalog)
fwrite(profile_publications, out_profile_publications)
fwrite(profile_matches, out_profile_matches)
fwrite(profile_unmatched, out_profile_unmatched)
fwrite(summary_profiles, out_summary_profiles)
fwrite(summary_profile_area, out_summary_profile_area)
fwrite(candidate_profiles, out_candidate_profiles)

# ============================================================
# 10) Imprimir
# ============================================================

cat("\n================ VALIDACIÓN POR PERFILES v5.2 ================\n")

cat("\nPerfiles detectados:", nrow(profiles_catalog), "\n")
cat("Publicaciones en perfiles:", nrow(profile_publications), "\n")
cat("Publicaciones emparejadas:", nrow(profile_matches), "\n")
cat("Publicaciones no emparejadas:", nrow(profile_unmatched), "\n")

cat("\nResumen perfiles top 30:\n")
print(head(summary_profiles[order(-matched_publications)], 30))

cat("\nCandidatos para validación externa top 40:\n")
print(head(candidate_profiles, 40))

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")