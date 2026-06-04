# ============================================================
# 22_objetivo1_v5_2_clasificar_no_encontrados_perfil_v2.R
#
# Paso intermedio entre 21b y 0054:
# Clasifica las publicaciones del perfil que NO fueron encontradas
# en el corpus principal (profile_publications_unmatched.csv),
# usando SPECTER2 + zero-shot v4 sobre el título solamente.
#
# Novedades v2:
# - Usa 0060_classify_profile_unmatched_titles_v5.py (reranking bibliométrico)
# - Post-proceso de reranking bibliométrico también sobre publicaciones
#   MATCHEADAS del corpus (corrección de casos NLP/Data Science → Bibliometrics)
# - Reporte detallado de casos rerrankeados
# - Generación de combined_profile_all_publications.csv con columna
#   classification_origin para 0054
#
# Prerequisitos:
#   - Haber ejecutado 21b (genera profile_publications_unmatched.csv)
#   - Tener disponibles los embeddings de taxonomía v4
#   - Python con: transformers, adapters, numpy, pandas, scikit-learn, tqdm
#
# Uso:
#   researcher_safe_id <- "martin_alberto"
#   source("22_objetivo1_v5_2_clasificar_no_encontrados_perfil_v2.R")
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(stringi)
})

base_dir   <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")
script_dir <- file.path(base_dir, "script")

# ── parámetros ────────────────────────────────────────────────

if (!exists("researcher_safe_id") || is.null(researcher_safe_id)) {
  stop("Define researcher_safe_id antes de ejecutar este script.")
}

# Directorio de salida del script 21b
profile_eval_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_individual_evaluation_fast",
  researcher_safe_id
)

# Embeddings y metadata de taxonomía v4 (misma ruta que usa 22b y 03)
tax_emb_file  <- file.path(
  base_dir,
  "embeddings",
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings.npy"
)
tax_meta_file <- file.path(
  base_dir,
  "embeddings",
  "specter2_taxonomy_v4_multiprototype",
  "docs_embeddings_meta_fixed.csv"
)

# Directorio de salida
unmatched_out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_unmatched_classified",
  researcher_safe_id
)

# Python y script
python_bin    <- path.expand("~/env_cluster/bin/python")
python_script <- file.path(script_dir, "0060_classify_profile_unmatched_titles_v5.py")

# Umbral de reranking bibliométrico (igual que v5 Python)
rerank_margin_threshold <- 0.05

# ── términos bibliométricos para reranking en R ───────────────

biblio_title_terms <- c(
  "bibliometr", "scientometr", "metascience", "informetr",
  "webometr", "altmetr", "technomet",
  "h-index", "h index", "impact factor", "citation",
  "journal ranking", "journal metric", "research metric",
  "scholar metric", "academic metric", "publication metric",
  "google scholar", "web of science", "scopus", "pubmed",
  "academic search", "academic database", "research database",
  "semantic scholar", "dimensions data",
  "research evaluation", "research assessment",
  "academic ranking", "university ranking",
  "research output", "scientific output",
  "research impact", "academic impact",
  "open access", "open science",
  "institutional repository", "research profile",
  "impactstory", "orcid", "citation analysis", "citation network",
  "co-citation", "bibliographic coupling",
  "publication trend"
)

biblio_area_terms <- c(
  "bibliometr", "scientometr", "metascience",
  "library", "information science", "research evaluation", "informetr"
)

has_biblio_signal <- function(title) {
  tl <- tolower(title)
  any(vapply(biblio_title_terms, function(t) grepl(t, tl, fixed = TRUE), logical(1)))
}

is_biblio_area <- function(area_name) {
  al <- tolower(as.character(area_name))
  any(vapply(biblio_area_terms, function(t) grepl(t, al, fixed = TRUE), logical(1)))
}

# Vectorized version for data.table column operations
is_biblio_area_vec <- function(area_vec) {
  vapply(area_vec, function(x) {
    if (is.na(x) || x == "") return(FALSE)
    is_biblio_area(x)
  }, logical(1))
}

# ── validar inputs ────────────────────────────────────────────

unmatched_csv <- file.path(profile_eval_dir, "profile_publications_unmatched.csv")

if (!file.exists(unmatched_csv)) {
  stop(
    "No se encontró profile_publications_unmatched.csv en:\n  ", profile_eval_dir,
    "\n\nEjecuta primero 21b con researcher_safe_id = '", researcher_safe_id, "'"
  )
}

unmatched_check <- fread(unmatched_csv)

if (nrow(unmatched_check) == 0) {
  cat("\nNo hay publicaciones sin match para '", researcher_safe_id, "'. Nada que clasificar.\n", sep = "")
  unmatched_classified <- data.table()
} else {

  cat("\nPublicaciones sin match a clasificar:", nrow(unmatched_check), "\n")
  cat("Porcentaje título vacío:",
      round(mean(unmatched_check$profile_title == "" | is.na(unmatched_check$profile_title)) * 100, 1), "%\n")

  for (f in c(tax_emb_file, tax_meta_file, python_script)) {
    if (!file.exists(f)) stop("No se encontró:\n  ", f)
  }

  dir.create(unmatched_out_dir, recursive = TRUE, showWarnings = FALSE)

  # ── llamar al script Python v5 ──────────────────────────────

  cat("\nEjecutando clasificación SPECTER2 zero-shot v4 + reranking bibliométrico...\n")
  cat("Salida en:", unmatched_out_dir, "\n\n")

  py_args <- c(
    python_script,
    "--unmatched_csv",          shQuote(unmatched_csv),
    "--tax_emb",                shQuote(tax_emb_file),
    "--tax_meta",               shQuote(tax_meta_file),
    "--output_dir",             shQuote(unmatched_out_dir),
    "--rerank_bibliometrics",
    "--rerank_margin_threshold", rerank_margin_threshold,
    "--overwrite"
  )

  t0 <- proc.time()
  ret <- system2(command = python_bin, args = py_args, stdout = TRUE, stderr = TRUE)
  elapsed <- proc.time() - t0

  cat(paste(ret, collapse = "\n"), "\n")
  cat(sprintf("\nTiempo transcurrido: %.1f segundos\n", elapsed["elapsed"]))

  # ── leer resultados ─────────────────────────────────────────

  pred_file <- file.path(unmatched_out_dir, "unmatched_zero_shot_title_only_suggestions.csv")

  if (!file.exists(pred_file)) {
    stop("El script Python no generó el archivo esperado:\n  ", pred_file)
  }

  unmatched_classified <- fread(pred_file)

  cat("\n================ CLASIFICACIÓN TÍTULOS SIN MATCH ================\n")
  cat("Publicaciones clasificadas:", nrow(unmatched_classified), "\n")

  if ("top1_area_name" %in% names(unmatched_classified)) {
    summary_area_unmatched <- unmatched_classified[, .N, by = top1_area_name][order(-N)]
    summary_area_unmatched[, pct := round(N / sum(N) * 100, 2)]
    cat("\nDistribución de áreas (título-solo, post-reranking):\n")
    print(summary_area_unmatched)
  }

  if ("reranked_to_bibliometrics" %in% names(unmatched_classified)) {
    n_reranked <- sum(unmatched_classified$reranked_to_bibliometrics == TRUE, na.rm = TRUE)
    cat("\nCasos rerrankeados a Bibliometrics:", n_reranked, "\n")
    if (n_reranked > 0) {
      reranked_dt <- unmatched_classified[reranked_to_bibliometrics == TRUE,
                                          .(profile_title, orig_top1_area_before_rerank,
                                            top1_area_name, top1_score, top1_top2_margin)]
      print(reranked_dt)
    }
  }

  if ("confidence_level" %in% names(unmatched_classified)) {
    cat("\nConfianza:\n")
    print(unmatched_classified[, .N, by = confidence_level][order(-N)])
  }

  # Guardar resúmenes
  if (exists("summary_area_unmatched")) {
    fwrite(summary_area_unmatched,
           file.path(unmatched_out_dir, "summary_area_unmatched.csv"))
  }
}

# ── reranking bibliométrico para publicaciones MATCHEADAS ─────

cat("\n================ POST-PROCESO RERANKING MATCHEADAS ================\n")

matched_file <- file.path(profile_eval_dir, "profile_publications_matched_final_qa.csv")

if (!file.exists(matched_file)) {
  cat("AVISO: No se encontró profile_publications_matched_final_qa.csv\n")
  cat("Solo se tienen las publicaciones clasificadas por título.\n")
  matched_reranked <- data.table()
} else {
  matched_dt <- fread(matched_file)
  cat("Publicaciones matcheadas leídas:", nrow(matched_dt), "\n")

  # Detectar qué columna tiene el área asignada
  area_col <- if ("qa_final_area_label" %in% names(matched_dt)) "qa_final_area_label" else
              if ("top1_area_name" %in% names(matched_dt)) "top1_area_name" else NA_character_

  if (is.na(area_col)) {
    cat("AVISO: No se encontró columna de área en matched_dt. Saltando reranking.\n")
    matched_reranked <- matched_dt
  } else {

    # Condiciones para reranking:
    # 1. Área actual NO es bibliométrica
    # 2. El título tiene señal bibliométrica
    # 3. top2_area_name (si existe) es bibliométrica o está en top2-5
    matched_dt[, biblio_signal_in_title := vapply(
      profile_title, has_biblio_signal, logical(1)
    )]

    matched_dt[, current_area_is_biblio := is_biblio_area_vec(get(area_col))]

    # Buscar si hay área bibliométrica en top2..top5
    top_area_cols <- intersect(
      paste0("top", 2:5, "_area_name"),
      names(matched_dt)
    )

    if (length(top_area_cols) > 0) {
      matched_dt[, biblio_in_top2_5 := Reduce("|", lapply(top_area_cols, function(col) {
        is_biblio_area_vec(get(col))
      }))]
    } else {
      matched_dt[, biblio_in_top2_5 := FALSE]
    }

    # Margen original
    margin_col <- if ("top1_top2_margin" %in% names(matched_dt)) "top1_top2_margin" else NA_character_

    if (!is.na(margin_col)) {
      matched_dt[, margin_ok_for_rerank := get(margin_col) < rerank_margin_threshold | is.na(get(margin_col))]
    } else {
      matched_dt[, margin_ok_for_rerank := TRUE]
    }

    rerank_flag <- (
      matched_dt$biblio_signal_in_title &
      !matched_dt$current_area_is_biblio &
      matched_dt$biblio_in_top2_5 &
      matched_dt$margin_ok_for_rerank
    )

    n_matched_reranked <- sum(rerank_flag, na.rm = TRUE)
    cat("Publicaciones matcheadas con señal bibliométrica en título:", sum(matched_dt$biblio_signal_in_title), "\n")
    cat("Publicaciones matcheadas rerrankeadas a Bibliometrics:", n_matched_reranked, "\n")

    if (n_matched_reranked > 0) {
      # Identificar qué columna bibliométrica del top2-5 usar
      matched_dt[, biblio_rerank_area := NA_character_]
      for (col in top_area_cols) {
        matched_dt[rerank_flag & is.na(biblio_rerank_area) & is_biblio_area_vec(get(col)),
                   biblio_rerank_area := get(col)]
      }

      matched_dt[rerank_flag, orig_area_before_rerank := get(area_col)]
      matched_dt[rerank_flag, (area_col) := biblio_rerank_area]

      # Si existe qa_final_subarea_label, intentar actualizar
      if ("top2_subarea_name" %in% names(matched_dt) && "qa_final_subarea_label" %in% names(matched_dt)) {
        for (rank_i in 2:5) {
          subarea_col <- paste0("top", rank_i, "_subarea_name")
          area_col_i  <- paste0("top", rank_i, "_area_name")
          if (subarea_col %in% names(matched_dt) && area_col_i %in% names(matched_dt)) {
            matched_dt[
              rerank_flag & is_biblio_area_vec(get(area_col_i)),
              qa_final_subarea_label := get(subarea_col)
            ]
          }
        }
      }

      matched_dt[rerank_flag, reranked_to_bibliometrics := TRUE]
      matched_dt[!rerank_flag | is.na(rerank_flag), reranked_to_bibliometrics := FALSE]
      matched_dt[rerank_flag, final_label_source := "corpus_match_reranked_biblio_signal"]

      cat("\nPublicaciones matcheadas rerrankeadas:\n")
      reranked_matched <- matched_dt[rerank_flag == TRUE,
                                     .(profile_title, orig_area_before_rerank,
                                       new_area = get(area_col))]
      print(reranked_matched)
    } else {
      matched_dt[, reranked_to_bibliometrics := FALSE]
    }

    matched_reranked <- matched_dt
    fwrite(matched_reranked,
           file.path(profile_eval_dir, "profile_publications_matched_final_qa_v2_reranked.csv"))
    cat("Archivo actualizado guardado:", file.path(profile_eval_dir, "profile_publications_matched_final_qa_v2_reranked.csv"), "\n")
  }
}

# ── combinar y resumen general ────────────────────────────────

cat("\n================ PERFIL COMPLETO COMBINADO ================\n")

n_matched   <- if (nrow(matched_reranked) > 0) nrow(matched_reranked) else 0
n_unmatched <- if (nrow(unmatched_classified) > 0) nrow(unmatched_classified) else 0
n_total     <- n_matched + n_unmatched

if (n_matched > 0 && "qa_final_area_label" %in% names(matched_reranked)) {
  cat("\nPublicaciones MATCHEADAS (por área, post-reranking):\n")
  matched_summary <- matched_reranked[
    !is.na(qa_final_area_label) & qa_final_area_label != "",
    .(publicaciones = .N, citas = sum(profile_cites, na.rm = TRUE)),
    by = .(area = qa_final_area_label)
  ][order(-publicaciones)]
  matched_summary[, pct := round(publicaciones / sum(publicaciones) * 100, 1)]
  print(matched_summary)
}

if (n_unmatched > 0 && "top1_area_name" %in% names(unmatched_classified)) {
  cat("\nPublicaciones NO MATCHEADAS (por área, title-only + reranking):\n")
  unmatched_summary <- unmatched_classified[
    !is.na(top1_area_name),
    .(publicaciones = .N, citas = sum(profile_cites, na.rm = TRUE)),
    by = .(area = top1_area_name)
  ][order(-publicaciones)]
  unmatched_summary[, pct := round(publicaciones / sum(publicaciones) * 100, 1)]
  print(unmatched_summary)
}

profile_coverage <- data.table(
  metrica = c(
    "publicaciones_matcheadas",
    "publicaciones_titulo_solo_clasificadas",
    "total_perfil_cubierto",
    "pct_matcheadas",
    "pct_titulo_solo"
  ),
  valor = c(
    n_matched,
    n_unmatched,
    n_total,
    if (n_total > 0) round(n_matched / n_total * 100, 2) else NA_real_,
    if (n_total > 0) round(n_unmatched / n_total * 100, 2) else NA_real_
  )
)

cat("\nCobertura total del perfil:\n")
print(profile_coverage)

fwrite(profile_coverage, file.path(unmatched_out_dir, "profile_coverage_summary.csv"))

cat("\n\nNEXT STEP: Ejecuta 0054_combine_profile_embeddings.py con:\n")
cat("  --unmatched_emb:", file.path(unmatched_out_dir, "unmatched_embeddings.npy"), "\n")
cat("  --unmatched_meta:", file.path(unmatched_out_dir, "unmatched_embeddings_meta.csv"), "\n")
