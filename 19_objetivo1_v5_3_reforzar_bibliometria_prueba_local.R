# ============================================================
# 19_objetivo1_v5_3_reforzar_bibliometria_prueba_local.R
# Objetivo 1 v5.3 experimental
# Reforzar prototipos de Bibliometrics and Scientometrics
# y probar localmente sobre validación por perfiles.
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# 1) Rutas
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"
tables_dir <- file.path(base_dir, "tables")
embeddings_dir <- file.path(base_dir, "embeddings")
script_dir <- file.path(base_dir, "script")

python_bin <- path.expand("~/env_cluster/bin/python")

script_embeddings <- file.path(
  script_dir,
  "0002-generar_embeddings_specter2.py"
)

script_zero <- file.path(
  script_dir,
  "0030_zero_shot_v4_multiprototype.py"
)

profile_val_results_dir <- file.path(
  tables_dir,
  "objetivo1_v5_2_profile_validation",
  "selected_profiles_external_validation",
  "validation_results_selected_profiles_v5_2"
)

clean_file <- file.path(
  profile_val_results_dir,
  "manual_validation_selected_profiles_clean_v5_2.csv"
)

tax_old_dir <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v4_multiprototype"
)

tax_old_meta <- file.path(
  tax_old_dir,
  "docs_embeddings_meta_fixed.csv"
)

tax_old_emb <- file.path(
  tax_old_dir,
  "docs_embeddings.npy"
)

docs_emb_dir <- file.path(
  embeddings_dir,
  "specter2_classification_v5_2_docs_interest_no_concepts"
)

doc_emb_full <- file.path(
  docs_emb_dir,
  "docs_embeddings.npy"
)

doc_meta_full_candidates <- c(
  file.path(docs_emb_dir, "docs_embeddings_meta_fixed.csv"),
  file.path(docs_emb_dir, "docs_embeddings_meta.csv")
)

doc_meta_full <- doc_meta_full_candidates[file.exists(doc_meta_full_candidates)][1]

out_dir <- file.path(
  tables_dir,
  "objetivo1_v5_3_biblio_prototype_test"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

tax_aug_input <- file.path(
  out_dir,
  "taxonomy_v5_3_biblio_augmented.csv"
)

tax_aug_dir <- file.path(
  embeddings_dir,
  "specter2_taxonomy_v5_3_biblio_augmented"
)

local_ids_file <- file.path(
  out_dir,
  "local_profile_validation_ids_for_biblio_test.csv"
)

local_doc_dir <- file.path(
  out_dir,
  "local_doc_embeddings_profile_validation"
)

dir.create(local_doc_dir, recursive = TRUE, showWarnings = FALSE)

local_doc_emb <- file.path(local_doc_dir, "docs_embeddings.npy")
local_doc_meta <- file.path(local_doc_dir, "docs_embeddings_meta.csv")

out_old <- file.path(
  out_dir,
  "zero_shot_local_old_taxonomy_v4.csv"
)

out_new <- file.path(
  out_dir,
  "zero_shot_local_new_taxonomy_v5_3_biblio_augmented.csv"
)

out_comparison <- file.path(
  out_dir,
  "comparison_old_vs_new_biblio_augmented.csv"
)

out_summary <- file.path(
  out_dir,
  "summary_biblio_augmented_test.csv"
)

out_cases <- file.path(
  out_dir,
  "cases_biblio_augmented_test.csv"
)

# ============================================================
# 2) Validaciones
# ============================================================

required_files <- c(
  clean_file,
  tax_old_meta,
  tax_old_emb,
  doc_emb_full,
  doc_meta_full,
  script_embeddings,
  script_zero,
  python_bin
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    "Faltan archivos requeridos:\n",
    paste(missing_files, collapse = "\n")
  )
}

# ============================================================
# 3) Funciones
# ============================================================

clean_text <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  x <- enc2utf8(x)
  x <- gsub("[\r\n\t]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

has_pattern <- function(x, pattern) {
  grepl(pattern, x, ignore.case = TRUE, perl = TRUE)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(x))

pick_col <- function(dt, candidates) {
  nms <- names(dt)
  nms_low <- tolower(nms)
  cand_low <- tolower(candidates)
  
  for (cand in cand_low) {
    idx <- which(nms_low == cand)
    if (length(idx) > 0) return(nms[idx[1]])
  }
  
  NA_character_
}

run_command <- function(command, args, step_name) {
  
  cat("\n============================================================\n")
  cat("Ejecutando:", step_name, "\n")
  cat("Comando:", command, "\n")
  cat("Args:", paste(args, collapse = " "), "\n")
  cat("============================================================\n")
  
  res <- system2(
    command = command,
    args = args,
    stdout = TRUE,
    stderr = TRUE
  )
  
  cat(paste(res, collapse = "\n"), "\n")
  
  status <- attr(res, "status")
  if (!is.null(status) && status != 0) {
    stop("Falló el paso: ", step_name)
  }
  
  invisible(res)
}

contains_biblio_topk <- function(dt) {
  
  out <- rep(FALSE, nrow(dt))
  
  for (k in 1:5) {
    area_col <- paste0("top", k, "_area_name")
    sub_col  <- paste0("top", k, "_subarea_name")
    
    if (area_col %in% names(dt) && sub_col %in% names(dt)) {
      out <- out |
        clean_text(dt[[area_col]]) == "Social Sciences" &
        clean_text(dt[[sub_col]]) == "Bibliometrics and Scientometrics"
    }
  }
  
  out
}

is_biblio_top1 <- function(dt) {
  
  if (!all(c("top1_area_name", "top1_subarea_name") %in% names(dt))) {
    return(rep(FALSE, nrow(dt)))
  }
  
  clean_text(dt$top1_area_name) == "Social Sciences" &
    clean_text(dt$top1_subarea_name) == "Bibliometrics and Scientometrics"
}

# ============================================================
# 4) Crear taxonomía aumentada con nuevos prototipos
# ============================================================

tax <- fread(tax_old_meta)
setDT(tax)

# Asegurar doc_id
if (!"doc_id" %in% names(tax)) {
  tax[, doc_id := .I]
}

tax[, doc_id := suppressWarnings(as.integer(doc_id))]
if (any(is.na(tax$doc_id))) {
  tax[, doc_id := .I]
}

area_col <- pick_col(tax, c("area_name", "area_label", "area"))
subarea_col <- pick_col(tax, c("subarea_name", "subarea_label", "subarea"))
text_col <- pick_col(tax, c("text_for_embedding", "text", "prototype_text"))

if (is.na(area_col)) stop("No se encontró columna de área en taxonomía.")
if (is.na(subarea_col)) stop("No se encontró columna de subárea en taxonomía.")
if (is.na(text_col)) stop("No se encontró columna de texto para embeddings en taxonomía.")

# Renombrar/crear text_for_embedding para compatibilidad
if (text_col != "text_for_embedding") {
  tax[, text_for_embedding := clean_text(get(text_col))]
} else {
  tax[, text_for_embedding := clean_text(text_for_embedding)]
}

if (!"prototype_language" %in% names(tax)) tax[, prototype_language := ""]
if (!"prototype_type" %in% names(tax)) tax[, prototype_type := ""]
if (!"prototype_id" %in% names(tax)) tax[, prototype_id := paste0("prototype_", .I)]

new_prototypes <- data.table(
  prototype_language = c(
    rep("en", 12),
    rep("es", 12)
  ),
  text_for_embedding = c(
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Studies citation analysis, bibliometric indicators, scientific productivity, h-index, journal impact, research evaluation, science mapping, co-citation and scholarly communication.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Research about Google Scholar, Scopus, Web of Science, citation databases, journal rankings, citation impact, academic search engines and scientific visibility.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Analysis of scholarly communication, scientific publishing, research assessment, open science, research metrics, altmetrics and scientometric indicators.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Bibliometric studies of medical literature, biomedical publications, clinical research output, citation patterns and scientific impact in health sciences.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Bibliometric studies of computer science, artificial intelligence, information retrieval, NLP publications, citation networks and research trends.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Research data management, data citation, data sharing, data reuse, scholarly records, open citations and scientific communication infrastructure.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Evaluation of researchers, journals, institutions, scientific collaboration, co-authorship networks, productivity indicators and research performance.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Studies on scientific journals, editorial quality, impact factor, journal rankings, publication patterns, indexing systems and research visibility.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Mapping science through bibliographic coupling, co-citation analysis, co-word analysis, citation networks and thematic evolution.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Analysis of open access, repositories, scholarly communication, research dissemination, citation databases and academic visibility.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Scientometric analysis of scientific production, citation behavior, research fronts, publication trends and knowledge domains.",
    "Area: Social Sciences. Subarea: Bibliometrics and Scientometrics. Use of bibliometric methods to evaluate disciplines, journals, countries, universities and research groups.",
    
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Estudios sobre análisis de citas, indicadores bibliométricos, producción científica, índice h, impacto de revistas, evaluación científica, mapas de ciencia y comunicación científica.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Investigaciones sobre Google Scholar, Scopus, Web of Science, bases de datos de citas, rankings de revistas, impacto científico, buscadores académicos y visibilidad científica.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Análisis de comunicación científica, publicación académica, evaluación de la investigación, ciencia abierta, métricas científicas, altmétricas e indicadores cienciométricos.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Estudios bibliométricos de literatura médica, publicaciones biomédicas, producción científica en salud, patrones de citación e impacto científico.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Estudios bibliométricos de informática, inteligencia artificial, recuperación de información, PLN, redes de citación y tendencias de investigación.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Gestión de datos de investigación, citación de datos, intercambio de datos, reutilización de datos, registros académicos, citas abiertas e infraestructura de comunicación científica.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Evaluación de investigadores, revistas, instituciones, colaboración científica, redes de coautoría, productividad científica y desempeño investigador.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Estudios sobre revistas científicas, calidad editorial, factor de impacto, rankings, patrones de publicación, sistemas de indexación y visibilidad científica.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Mapeo de la ciencia mediante acoplamiento bibliográfico, análisis de cocitación, co-palabras, redes de citación y evolución temática.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Análisis de acceso abierto, repositorios, comunicación académica, difusión científica, bases de datos de citas y visibilidad investigadora.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Análisis cienciométrico de producción científica, comportamiento de citación, frentes de investigación, tendencias de publicación y dominios de conocimiento.",
    "Área: Ciencias Sociales. Subárea: Bibliometría y Cienciometría. Uso de métodos bibliométricos para evaluar disciplinas, revistas, países, universidades y grupos de investigación."
  )
)

new_rows <- tax[0][rep(1, nrow(new_prototypes))]

new_rows[, doc_id := max(tax$doc_id, na.rm = TRUE) + seq_len(.N)]
new_rows[, (area_col) := "Social Sciences"]
new_rows[, (subarea_col) := "Bibliometrics and Scientometrics"]
new_rows[, text_for_embedding := new_prototypes$text_for_embedding]
new_rows[, prototype_language := new_prototypes$prototype_language]
new_rows[, prototype_type := "augmented_bibliometrics_metascience_v5_3"]
new_rows[, prototype_id := paste0("biblio_metascience_v5_3_", seq_len(.N))]

tax_aug <- rbindlist(
  list(tax, new_rows),
  fill = TRUE
)

fwrite(tax_aug, tax_aug_input)

cat("\nTaxonomía original:", nrow(tax), "\n")
cat("Prototipos añadidos:", nrow(new_rows), "\n")
cat("Taxonomía aumentada:", nrow(tax_aug), "\n")

# ============================================================
# 5) Generar embeddings de taxonomía aumentada
# ============================================================

dir.create(tax_aug_dir, recursive = TRUE, showWarnings = FALSE)

args_tax_emb <- c(
  script_embeddings,
  "--input", tax_aug_input,
  "--output_dir", tax_aug_dir,
  "--id_col", "doc_id",
  "--text_col", "text_for_embedding",
  "--model_name", "allenai/specter2_base",
  "--adapter_name", "allenai/specter2_classification",
  "--adapter_load_as", "specter2_classification",
  "--batch_size", "16",
  "--max_length", "512",
  "--normalize",
  "--overwrite"
)

run_command(
  python_bin,
  args_tax_emb,
  "Generar embeddings taxonomía aumentada v5.3"
)

# ============================================================
# 5b) Crear metadata fixed de taxonomía aumentada
# El script zero-shot requiere columnas estándar:
# area_name, subarea_name, subarea_id, prototype_id,
# language, prototype_type
# ============================================================

tax_aug_meta <- file.path(tax_aug_dir, "docs_embeddings_meta.csv")
tax_aug_meta_fixed <- file.path(tax_aug_dir, "docs_embeddings_meta_fixed.csv")
tax_aug_emb <- file.path(tax_aug_dir, "docs_embeddings.npy")

if (!file.exists(tax_aug_meta)) {
  stop("No se creó metadata de taxonomía aumentada.")
}

if (!file.exists(tax_aug_emb)) {
  stop("No se creó docs_embeddings.npy de taxonomía aumentada.")
}

# Usamos tax_aug como fuente de verdad porque conserva todas las columnas
# necesarias de la taxonomía.
tax_aug_fixed <- copy(tax_aug)

# ------------------------------------------------------------
# Asegurar doc_id
# ------------------------------------------------------------

if (!"doc_id" %in% names(tax_aug_fixed)) {
  tax_aug_fixed[, doc_id := .I]
}

tax_aug_fixed[, doc_id := suppressWarnings(as.integer(doc_id))]

if (any(is.na(tax_aug_fixed$doc_id))) {
  tax_aug_fixed[, doc_id := .I]
}

# ------------------------------------------------------------
# Crear columnas estándar requeridas por 0030_zero_shot_v4_multiprototype.py
# ------------------------------------------------------------

if (!"area_name" %in% names(tax_aug_fixed)) {
  tax_aug_fixed[, area_name := clean_text(get(area_col))]
} else {
  tax_aug_fixed[, area_name := clean_text(area_name)]
}

if (!"subarea_name" %in% names(tax_aug_fixed)) {
  tax_aug_fixed[, subarea_name := clean_text(get(subarea_col))]
} else {
  tax_aug_fixed[, subarea_name := clean_text(subarea_name)]
}

# subarea_id puede existir con otro nombre o no existir.
if (!"subarea_id" %in% names(tax_aug_fixed)) {
  
  possible_subarea_id_cols <- c(
    "subarea_code",
    "subarea_label_id",
    "label_id",
    "taxon_id"
  )
  
  found_subarea_id <- possible_subarea_id_cols[
    possible_subarea_id_cols %in% names(tax_aug_fixed)
  ][1]
  
  if (!is.na(found_subarea_id)) {
    tax_aug_fixed[, subarea_id := clean_text(get(found_subarea_id))]
  } else {
    tax_aug_fixed[
      ,
      subarea_id := paste0(
        clean_text(area_name),
        " || ",
        clean_text(subarea_name)
      )
    ]
  }
  
} else {
  tax_aug_fixed[, subarea_id := clean_text(subarea_id)]
}

# prototype_id
if (!"prototype_id" %in% names(tax_aug_fixed)) {
  tax_aug_fixed[, prototype_id := paste0("prototype_", seq_len(.N))]
} else {
  tax_aug_fixed[, prototype_id := clean_text(prototype_id)]
  tax_aug_fixed[
    prototype_id == "" | is.na(prototype_id),
    prototype_id := paste0("prototype_", .I)
  ]
}

# language: el script espera "language", no prototype_language
if (!"language" %in% names(tax_aug_fixed)) {
  
  if ("prototype_language" %in% names(tax_aug_fixed)) {
    tax_aug_fixed[, language := clean_text(prototype_language)]
  } else {
    tax_aug_fixed[, language := ""]
  }
  
} else {
  tax_aug_fixed[, language := clean_text(language)]
}

tax_aug_fixed[
  language == "" | is.na(language),
  language := fifelse(
    grepl("Área:|Subárea:", text_for_embedding, ignore.case = TRUE),
    "es",
    "en"
  )
]

# prototype_type
if (!"prototype_type" %in% names(tax_aug_fixed)) {
  tax_aug_fixed[, prototype_type := ""]
} else {
  tax_aug_fixed[, prototype_type := clean_text(prototype_type)]
}

tax_aug_fixed[
  prototype_type == "" | is.na(prototype_type),
  prototype_type := "taxonomy_prototype"
]

# ------------------------------------------------------------
# emb_row debe apuntar a la fila dentro de docs_embeddings.npy
# La taxonomía aumentada se embebió en el orden de tax_aug_input,
# por lo tanto emb_row = 0:(n-1)
# ------------------------------------------------------------

tax_aug_fixed[, emb_row := seq_len(.N) - 1L]

# ------------------------------------------------------------
# Verificación de columnas obligatorias
# ------------------------------------------------------------

required_tax_cols <- c(
  "doc_id",
  "emb_row",
  "area_name",
  "subarea_name",
  "subarea_id",
  "prototype_id",
  "language",
  "prototype_type",
  "text_for_embedding"
)

missing_tax_cols <- setdiff(required_tax_cols, names(tax_aug_fixed))

if (length(missing_tax_cols) > 0) {
  stop(
    "Faltan columnas en tax_aug_fixed: ",
    paste(missing_tax_cols, collapse = ", ")
  )
}

# Mantener metadata completa, pero con las columnas requeridas garantizadas
fwrite(tax_aug_fixed, tax_aug_meta_fixed)

cat("\nMetadata fixed de taxonomía aumentada creada:\n")
cat(tax_aug_meta_fixed, "\n")
cat("Filas:", nrow(tax_aug_fixed), "\n")
cat("Rango emb_row:", min(tax_aug_fixed$emb_row), "-", max(tax_aug_fixed$emb_row), "\n")
cat("Columnas requeridas OK:\n")
print(required_tax_cols)

# ============================================================
# 6) Preparar casos locales de validación
# ============================================================

val <- fread(clean_file)
setDT(val)

needed_val_cols <- c(
  "final_doc_id_v5",
  "validation_group",
  "researcher_name",
  "researcher_safe_id",
  "profile_title",
  "candidate_top1",
  "candidate_top2",
  "candidate_top3",
  "candidate_top4",
  "candidate_top5",
  "corrected_area_label",
  "corrected_subarea_label",
  "manual_decision_clean",
  "manual_comment",
  "manual_profile_observation",
  "text_for_review_short"
)

for (col in needed_val_cols) {
  if (!col %in% names(val)) val[, (col) := NA]
}

for (col in names(val)) {
  if (is.character(val[[col]])) val[, (col) := clean_text(get(col))]
}

biblio_pattern <- paste(
  c(
    "bibliometric",
    "bibliometr",
    "scientometric",
    "cienciometr",
    "informetric",
    "altmetric",
    "citation analysis",
    "citation index",
    "co-citation",
    "cocitation",
    "co-word",
    "h-index",
    "google scholar",
    "web of science",
    "scopus",
    "journal impact",
    "impact factor",
    "research evaluation",
    "producci[oó]n cient[ií]fica",
    "comunicaci[oó]n cient[ií]fica",
    "scholarly communication",
    "science mapping",
    "data citation",
    "data sharing",
    "open science",
    "ciencia abierta"
  ),
  collapse = "|"
)

val[, text_diagnosis := paste(
  profile_title,
  candidate_top1,
  candidate_top2,
  candidate_top3,
  candidate_top4,
  candidate_top5,
  manual_comment,
  manual_profile_observation,
  text_for_review_short,
  sep = " "
)]

val[, has_biblio_terms := has_pattern(text_diagnosis, biblio_pattern)]

val[, corrected_is_biblio := corrected_area_label == "Social Sciences" &
      corrected_subarea_label == "Bibliometrics and Scientometrics"]

val[, doc_id_v5 := suppressWarnings(as.integer(final_doc_id_v5))]

# Casos de prueba:
# - todos los relacionados con bibliometría
# - más controles no bibliométricos de la misma validación
set.seed(123)

biblio_test <- val[
  corrected_is_biblio == TRUE |
    has_biblio_terms == TRUE
]

controls <- val[
  corrected_is_biblio == FALSE &
    has_biblio_terms == FALSE
]

controls <- controls[
  sample(.N, min(.N, 60))
]

local_test <- rbindlist(
  list(biblio_test, controls),
  fill = TRUE
)

local_test <- unique(
  local_test[!is.na(doc_id_v5)],
  by = "doc_id_v5"
)

local_test[, manual_pair := paste(corrected_area_label, corrected_subarea_label, sep = " / ")]

fwrite(local_test, local_ids_file)

cat("\nCasos locales de prueba:", nrow(local_test), "\n")
cat("Casos bibliométricos/metaciencia:", local_test[corrected_is_biblio == TRUE, .N], "\n")
cat("Controles:", local_test[corrected_is_biblio == FALSE, .N], "\n")

# ============================================================
# 7) Extraer embeddings documentales locales
# ============================================================

extract_py <- file.path(out_dir, "extract_local_doc_embeddings.py")

extract_code <- paste0(
  "import numpy as np\n",
  "import pandas as pd\n",
  "\n",
  "doc_emb_file = r'", doc_emb_full, "'\n",
  "doc_meta_file = r'", doc_meta_full, "'\n",
  "ids_file = r'", local_ids_file, "'\n",
  "out_emb_file = r'", local_doc_emb, "'\n",
  "out_meta_file = r'", local_doc_meta, "'\n",
  "\n",
  "emb = np.load(doc_emb_file, mmap_mode='r')\n",
  "meta = pd.read_csv(doc_meta_file)\n",
  "ids = pd.read_csv(ids_file)\n",
  "\n",
  "if 'doc_id_v5' not in meta.columns:\n",
  "    if 'doc_id' in meta.columns:\n",
  "        meta['doc_id_v5'] = meta['doc_id']\n",
  "    else:\n",
  "        raise ValueError('metadata no tiene doc_id_v5 ni doc_id')\n",
  "\n",
  "meta['doc_id_v5'] = pd.to_numeric(meta['doc_id_v5'], errors='coerce').astype('Int64')\n",
  "ids['doc_id_v5'] = pd.to_numeric(ids['doc_id_v5'], errors='coerce').astype('Int64')\n",
  "meta['emb_row'] = range(len(meta))\n",
  "\n",
  "m = ids.merge(meta, on='doc_id_v5', how='inner', suffixes=('', '_embmeta'))\n",
  "m = m.dropna(subset=['emb_row']).copy()\n",
  "m['emb_row'] = m['emb_row'].astype(int)\n",
  "\n",
  "X = np.asarray(emb[m['emb_row'].values], dtype=np.float32)\n",
  "np.save(out_emb_file, X)\n",
  "\n",
  "# Para el zero-shot, doc_id debe existir\n",
  "m['doc_id'] = m['doc_id_v5'].astype(int)\n",
  "m.to_csv(out_meta_file, index=False)\n",
  "\n",
  "print('local embeddings:', X.shape)\n",
  "print('local meta rows:', len(m))\n"
)

writeLines(extract_code, extract_py)

run_command(
  python_bin,
  extract_py,
  "Extraer embeddings locales de documentos"
)


# ============================================================
# 7b) Corregir emb_row local
# El script zero-shot espera que emb_row apunte a filas del NPY usado.
# Como ahora usamos un NPY local, emb_row debe ir de 0 a n-1.
# ============================================================

local_meta_fix <- fread(local_doc_meta)
setDT(local_meta_fix)

if (!"doc_id_v5" %in% names(local_meta_fix)) {
  if ("doc_id" %in% names(local_meta_fix)) {
    local_meta_fix[, doc_id_v5 := as.integer(doc_id)]
  } else {
    stop("La metadata local no tiene doc_id_v5 ni doc_id.")
  }
}

local_meta_fix[, doc_id_v5 := as.integer(doc_id_v5)]

# Guardar el emb_row original del corpus completo, por trazabilidad
if ("emb_row" %in% names(local_meta_fix)) {
  local_meta_fix[, emb_row_full_corpus := emb_row]
}

# Crear emb_row correcto para el embedding local
local_meta_fix[, emb_row := seq_len(.N) - 1L]

# Asegurar doc_id para el script zero-shot
local_meta_fix[, doc_id := as.integer(doc_id_v5)]

fwrite(local_meta_fix, local_doc_meta)

cat("\nMetadata local corregida:\n")
cat("Filas:", nrow(local_meta_fix), "\n")
cat("Rango emb_row:", min(local_meta_fix$emb_row), "-", max(local_meta_fix$emb_row), "\n")

# ============================================================
# 8) Ejecutar zero-shot local con taxonomía vieja y nueva
# ============================================================

args_zero_old <- c(
  script_zero,
  "--doc_emb", local_doc_emb,
  "--doc_meta", local_doc_meta,
  "--tax_emb", tax_old_emb,
  "--tax_meta", tax_old_meta,
  "--output", out_old,
  "--top_k", "5",
  "--batch_size", "500",
  "--sample_n", "0",
  "--seed", "123",
  "--min_score", "0.35",
  "--min_margin", "0.03"
)

run_command(
  python_bin,
  args_zero_old,
  "Zero-shot local con taxonomía original v4"
)

args_zero_new <- c(
  script_zero,
  "--doc_emb", local_doc_emb,
  "--doc_meta", local_doc_meta,
  "--tax_emb", tax_aug_emb,
  "--tax_meta", tax_aug_meta_fixed,
  "--output", out_new,
  "--top_k", "5",
  "--batch_size", "500",
  "--sample_n", "0",
  "--seed", "123",
  "--min_score", "0.35",
  "--min_margin", "0.03"
)

run_command(
  python_bin,
  args_zero_new,
  "Zero-shot local con taxonomía aumentada v5.3"
)

# ============================================================
# 9) Comparar viejo vs nuevo
# ============================================================

old <- fread(out_old)
new <- fread(out_new)

setDT(old)
setDT(new)

# Asegurar doc_id_v5
if (!"doc_id_v5" %in% names(old)) old[, doc_id_v5 := as.integer(doc_id)]
if (!"doc_id_v5" %in% names(new)) new[, doc_id_v5 := as.integer(doc_id)]

old[, old_top1_biblio := is_biblio_top1(.SD)]
old[, old_top5_biblio := contains_biblio_topk(.SD)]

new[, new_top1_biblio := is_biblio_top1(.SD)]
new[, new_top5_biblio := contains_biblio_topk(.SD)]

old_small <- old[
  ,
  .(
    doc_id_v5,
    old_top1_area = top1_area_name,
    old_top1_subarea = top1_subarea_name,
    old_top1_score = top1_score,
    old_top1_biblio,
    old_top5_biblio
  )
]

new_small <- new[
  ,
  .(
    doc_id_v5,
    new_top1_area = top1_area_name,
    new_top1_subarea = top1_subarea_name,
    new_top1_score = top1_score,
    new_top1_biblio,
    new_top5_biblio
  )
]

comparison <- merge(
  local_test,
  old_small,
  by = "doc_id_v5",
  all.x = TRUE
)

comparison <- merge(
  comparison,
  new_small,
  by = "doc_id_v5",
  all.x = TRUE
)

comparison[, improved_top5_biblio := corrected_is_biblio == TRUE &
             old_top5_biblio == FALSE &
             new_top5_biblio == TRUE]

comparison[, improved_top1_biblio := corrected_is_biblio == TRUE &
             old_top1_biblio == FALSE &
             new_top1_biblio == TRUE]

comparison[, new_false_positive_biblio := corrected_is_biblio == FALSE &
             old_top1_biblio == FALSE &
             new_top1_biblio == TRUE]

summary_test <- data.table(
  metric = c(
    "local_test_rows",
    "corrected_biblio_cases",
    "control_non_biblio_cases",
    "old_top1_biblio_rate_on_corrected_biblio",
    "new_top1_biblio_rate_on_corrected_biblio",
    "old_top5_biblio_rate_on_corrected_biblio",
    "new_top5_biblio_rate_on_corrected_biblio",
    "improved_top5_biblio_cases",
    "improved_top1_biblio_cases",
    "new_false_positive_biblio_cases",
    "new_false_positive_biblio_rate_controls"
  ),
  value = c(
    nrow(comparison),
    comparison[corrected_is_biblio == TRUE, .N],
    comparison[corrected_is_biblio == FALSE, .N],
    round(mean(comparison[corrected_is_biblio == TRUE]$old_top1_biblio, na.rm = TRUE), 4),
    round(mean(comparison[corrected_is_biblio == TRUE]$new_top1_biblio, na.rm = TRUE), 4),
    round(mean(comparison[corrected_is_biblio == TRUE]$old_top5_biblio, na.rm = TRUE), 4),
    round(mean(comparison[corrected_is_biblio == TRUE]$new_top5_biblio, na.rm = TRUE), 4),
    comparison[improved_top5_biblio == TRUE, .N],
    comparison[improved_top1_biblio == TRUE, .N],
    comparison[new_false_positive_biblio == TRUE, .N],
    round(mean(comparison[corrected_is_biblio == FALSE]$new_false_positive_biblio, na.rm = TRUE), 4)
  )
)

cases_report <- comparison[
  corrected_is_biblio == TRUE |
    improved_top5_biblio == TRUE |
    new_false_positive_biblio == TRUE,
  .(
    doc_id_v5,
    researcher_name,
    researcher_safe_id,
    profile_title,
    corrected_area_label,
    corrected_subarea_label,
    old_top1_area,
    old_top1_subarea,
    old_top1_score,
    old_top1_biblio,
    old_top5_biblio,
    new_top1_area,
    new_top1_subarea,
    new_top1_score,
    new_top1_biblio,
    new_top5_biblio,
    improved_top5_biblio,
    improved_top1_biblio,
    new_false_positive_biblio,
    manual_decision_clean,
    manual_comment
  )
][order(researcher_name, -improved_top5_biblio)]

# ============================================================
# 10) Exportar
# ============================================================

fwrite(comparison, out_comparison)
fwrite(summary_test, out_summary)
fwrite(cases_report, out_cases)

cat("\n================ PRUEBA LOCAL BIBLIOMETRÍA v5.3 ================\n")

cat("\nResumen prueba:\n")
print(summary_test)

cat("\nCasos relevantes:\n")
print(cases_report)

cat("\nArchivos creados en:\n")
cat(out_dir, "\n")