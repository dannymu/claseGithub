# ============================================================
# 21b_objetivo1_v5_2_evaluar_perfil_individual_fast.R
# Evaluar un perfil individual usando índices precomputados.
# Sin fuzzy global.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")

profiles_dir <- file.path(base_dir, "perfiles_research")

index_file <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_indexes",
  "final_qa_search_indexes_v5_2.rds"
)

# Cambia esto para otro perfil
selected_profile_query <- selected_profile_

# Si quieres forzar archivo exacto:
selected_profile_file <- selected_profile_file_

# No usar fuzzy global en producción
use_fuzzy <- FALSE

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

if (!file.exists(index_file)) {
  stop("No existe índice. Ejecuta primero el script 21a.")
}

profile_files <- list.files(
  profiles_dir,
  pattern = "\\.csv$",
  full.names = TRUE
)

if (!is.na(selected_profile_file) && selected_profile_file != "") {
  profile_file <- selected_profile_file
} else {
  candidates <- profile_files[
    grepl(selected_profile_query, basename(profile_files), ignore.case = TRUE)
  ]
  
  if (length(candidates) == 0) {
    stop("No se encontró perfil con query: ", selected_profile_query)
  }
  
  if (length(candidates) > 1) {
    cat("\nVarios perfiles encontrados. Se usará el primero:\n")
    print(basename(candidates))
  }
  
  profile_file <- candidates[1]
}

researcher_name <- guess_researcher_name(profile_file)
researcher_safe_id <- make_safe_id(researcher_name)

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_evaluation_fast",
  researcher_safe_id
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("\n================ PERFIL ================\n")
cat("Investigador:", researcher_name, "\n")
cat("Archivo:", profile_file, "\n")
cat("Salida:", out_dir, "\n")

# ============================================================
# Leer perfil
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

if (is.na(title_col)) {
  stop("No se encontró columna de título.")
}

profile <- data.table(
  researcher_name = researcher_name,
  researcher_safe_id = researcher_safe_id,
  profile_file = basename(profile_file),
  profile_row_id = seq_len(nrow(profile_raw)),
  profile_title = clean_text(profile_raw[[title_col]]),
  profile_year = if (!is.na(year_col)) safe_integer(profile_raw[[year_col]]) else NA_integer_,
  profile_cites = if (!is.na(cites_col)) safe_numeric(profile_raw[[cites_col]]) else NA_real_,
  profile_doi = if (!is.na(doi_col)) clean_text(profile_raw[[doi_col]]) else "",
  profile_authors = if (!is.na(authors_col)) clean_text(profile_raw[[authors_col]]) else "",
  profile_source = if (!is.na(source_col)) clean_text(profile_raw[[source_col]]) else "",
  profile_abstract = if (!is.na(abstract_col)) clean_text(profile_raw[[abstract_col]]) else "",
  profile_type = if (!is.na(type_col)) clean_text(profile_raw[[type_col]]) else ""
)

profile <- profile[profile_title != ""]

profile[, profile_publication_key := paste(researcher_safe_id, profile_row_id, sep = "___")]
profile[, profile_title_norm := normalize_title(profile_title)]
profile[, profile_doi_norm := normalize_doi(profile_doi)]

# ============================================================
# Cargar índices
# ============================================================

cat("\nCargando índices precomputados...\n")
idx <- readRDS(index_file)

final_by_doi <- idx$final_by_doi
final_by_title_year <- idx$final_by_title_year
final_by_title <- idx$final_by_title

# ============================================================
# Matching rápido
# ============================================================

matches <- list()

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

remaining <- profile[!profile_publication_key %in% matched_keys]

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

remaining <- profile[!profile_publication_key %in% matched_keys]

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
} else data.table()

if (nrow(profile_matches) > 0) {
  setorder(profile_matches, profile_publication_key, profile_match_priority, final_rank, -top1_score)
  profile_matches <- profile_matches[, .SD[1], by = profile_publication_key]
}

matched_keys <- unique(profile_matches$profile_publication_key)

profile_unmatched <- profile[
  !profile_publication_key %in% matched_keys
]

# ============================================================
# Resúmenes
# ============================================================

summary_matching <- data.table(
  metric = c(
    "profile_publications_clean",
    "matched_total",
    "unmatched_total",
    "match_rate_pct",
    "profile_total_citations"
  ),
  value = c(
    nrow(profile),
    nrow(profile_matches),
    nrow(profile_unmatched),
    round(nrow(profile_matches) / nrow(profile) * 100, 2),
    sum(profile$profile_cites, na.rm = TRUE)
  )
)

summary_match_method <- if (nrow(profile_matches) > 0) {
  profile_matches[, .N, by = profile_match_method][order(-N)]
} else data.table()

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
} else data.table()

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
      qa_auto_accept = sum(qa_auto_accept == TRUE, na.rm = TRUE),
      qa_review = sum(qa_requires_review == TRUE, na.rm = TRUE),
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

summary_year <- merge(summary_year, summary_year_matched, by = "profile_year", all.x = TRUE)

for (col in c("matched", "qa_auto_accept", "qa_review", "metascience_priority")) {
  if (col %in% names(summary_year)) {
    summary_year[is.na(get(col)), (col) := 0L]
  }
}

# ============================================================
# Muestra de revisión manual
# ============================================================

if (nrow(profile_matches) > 0) {
  
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
  
  review_sample <- rbindlist(
    list(
      sample_n_safe(profile_matches[validation_group == "01_profile_qa_auto_accepted"], 20),
      sample_n_safe(profile_matches[validation_group == "02_profile_metascience_qa_priority"], 30),
      sample_n_safe(profile_matches[validation_group == "03_profile_critical_conflict"], 20),
      sample_n_safe(profile_matches[validation_group == "04_profile_review_top5"], 30)
    ),
    fill = TRUE
  )
  
  review_sample <- unique(review_sample, by = c("profile_publication_key", "final_doc_id_v5"))
  
} else {
  review_sample <- data.table()
}

if (nrow(profile_unmatched) > 0) {
  unmatched_review <- sample_n_safe(profile_unmatched, 30)
  unmatched_review[, validation_group := "06_profile_unmatched"]
  review_sample <- rbindlist(list(review_sample, unmatched_review), fill = TRUE)
}

if (nrow(review_sample) > 0) {
  review_sample[, manual_area_correct := ""]
  review_sample[, manual_subarea_correct := ""]
  review_sample[, manual_area_label := ""]
  review_sample[, manual_subarea_label := ""]
  review_sample[, manual_decision := ""]
  review_sample[, manual_comment := ""]
}

summary_review_groups <- if (nrow(review_sample) > 0) {
  review_sample[, .N, by = validation_group][order(validation_group)]
} else data.table()

# ============================================================
# Exportar
# ============================================================

fwrite(profile, file.path(out_dir, "profile_publications_clean.csv"))
fwrite(profile_matches, file.path(out_dir, "profile_publications_matched_final_qa.csv"))
fwrite(profile_unmatched, file.path(out_dir, "profile_publications_unmatched.csv"))
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

profile_config <- data.table(
  researcher_name = researcher_name,
  researcher_safe_id = researcher_safe_id,
  profile_file = basename(profile_file),
  profile_path = profile_file,
  output_dir = out_dir,
  title_col = title_col,
  year_col = year_col,
  cites_col = cites_col,
  doi_col = doi_col
)

fwrite(profile_config, file.path(out_dir, "profile_config.csv"))

# XLSX ligero
if (requireNamespace("openxlsx", quietly = TRUE)) {
  
  xlsx_file <- file.path(out_dir, "profile_manual_review_sample.xlsx")
  
  wb <- openxlsx::createWorkbook()
  
  openxlsx::addWorksheet(wb, "revision_perfil")
  openxlsx::addWorksheet(wb, "resumen_matching")
  openxlsx::addWorksheet(wb, "resumen_qa")
  openxlsx::addWorksheet(wb, "areas_qa_auto")
  
  openxlsx::writeData(wb, "revision_perfil", review_sample)
  openxlsx::writeData(wb, "resumen_matching", summary_matching)
  openxlsx::writeData(wb, "resumen_qa", summary_qa_decision)
  openxlsx::writeData(wb, "areas_qa_auto", summary_area_qa_auto)
  
  openxlsx::saveWorkbook(wb, xlsx_file, overwrite = TRUE)
}

cat("\n================ EVALUACIÓN PERFIL INDIVIDUAL FAST ================\n")

cat("\nResumen matching:\n")
print(summary_matching)

cat("\nMatching por método:\n")
print(summary_match_method)

cat("\nDecisión original:\n")
print(summary_original_decision)

cat("\nDecisión QA:\n")
print(summary_qa_decision)

cat("\nEfecto QA:\n")
print(summary_qa_effect)

cat("\nÁreas QA auto:\n")
print(summary_area_qa_auto)

cat("\nTop1 áreas todos los emparejados:\n")
print(head(summary_top1_area_all_matches, 30))

cat("\nMuestra revisión:\n")
print(summary_review_groups)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")