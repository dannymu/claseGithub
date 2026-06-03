
# ============================================================
# 0003-zero_shot_classification.py
# Clasificación zero-shot por similitud coseno
# ============================================================
#
# Entrada:
#   - Embeddings SPECTER2 de documentos (.npy)
#   - Metadata de documentos (.csv)
#   - Embeddings SPECTER2 de taxonomía (.npy)
#   - Metadata de taxonomía (.csv)
#
# Salida:
#   - CSV con top-k áreas/subáreas sugeridas por documento
#
# Idea central:
#   Documento científico -> embedding SPECTER2
#   Subárea taxonómica   -> embedding SPECTER2
#
#   Luego se calcula similitud coseno:
#
#   score = coseno(documento, subárea)
#
#   La subárea con mayor score se propone como etiqueta preliminar.
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Función auxiliar: normalizar matriz
# ============================================================
# La similitud coseno entre dos vectores normalizados equivale
# al producto punto entre ellos.
#
# Esto permite calcular:
#
#   similitud = documento @ taxonomia.T
#
# de forma eficiente.
# ============================================================

def normalize_matrix(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm


# ============================================================
# Argumentos del script
# ============================================================

parser = argparse.ArgumentParser(
    description="Clasificación zero-shot de documentos científicos usando embeddings SPECTER2"
)

parser.add_argument(
    "--doc_emb",
    required=True,
    help="Ruta del archivo .npy con embeddings de documentos"
)

parser.add_argument(
    "--doc_meta",
    required=True,
    help="Ruta del CSV con metadata de documentos"
)

parser.add_argument(
    "--tax_emb",
    required=True,
    help="Ruta del archivo .npy con embeddings de la taxonomía"
)

parser.add_argument(
    "--tax_meta",
    required=True,
    help="Ruta del CSV con metadata de la taxonomía"
)

parser.add_argument(
    "--output",
    required=True,
    help="Ruta del CSV de salida con predicciones zero-shot"
)

parser.add_argument(
    "--top_k",
    type=int,
    default=5,
    help="Número de etiquetas candidatas a guardar por documento"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=5000,
    help="Tamaño de lote para procesar documentos"
)

parser.add_argument(
    "--sample_n",
    type=int,
    default=0,
    help="Si es mayor que 0, clasifica solo una muestra aleatoria"
)

parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Semilla para muestra reproducible"
)

parser.add_argument(
    "--min_score",
    type=float,
    default=0.35,
    help="Score mínimo para aceptar una etiqueta preliminar"
)

parser.add_argument(
    "--min_margin",
    type=float,
    default=0.03,
    help="Diferencia mínima entre top1 y top2 para evitar ambigüedad"
)

args = parser.parse_args()


# ============================================================
# 1) Leer metadata
# ============================================================

print("Leyendo metadata de documentos...")
doc_meta = pd.read_csv(args.doc_meta)

print("Leyendo metadata de taxonomía...")
tax_meta = pd.read_csv(args.tax_meta)

print("Filas metadata documentos:", len(doc_meta))
print("Filas metadata taxonomía:", len(tax_meta))


# ============================================================
# 2) Verificar columnas mínimas
# ============================================================

required_doc_cols = [
    "emb_row",
    "doc_id"
]

required_tax_cols = [
    "emb_row",
    "subarea_id",
    "area_name",
    "subarea_name"
]

for col in required_doc_cols:
    if col not in doc_meta.columns:
        raise ValueError(f"Falta columna en doc_meta: {col}")

for col in required_tax_cols:
    if col not in tax_meta.columns:
        raise ValueError(f"Falta columna en tax_meta: {col}")


# ============================================================
# 3) Ordenar metadata por emb_row
# ============================================================
# Esto es crítico.
#
# emb_row indica la fila exacta en el archivo .npy.
# Por eso siempre se ordena por emb_row antes de clasificar.
# ============================================================

doc_meta = doc_meta.sort_values("emb_row").reset_index(drop=True)
tax_meta = tax_meta.sort_values("emb_row").reset_index(drop=True)


# ============================================================
# 4) Tomar muestra si se solicita
# ============================================================
# Para pruebas iniciales se recomienda sample_n = 10000.
# Para correr todo el corpus se usa sample_n = 0.
# ============================================================

if args.sample_n > 0 and args.sample_n < len(doc_meta):
    doc_meta = (
        doc_meta
        .sample(n=args.sample_n, random_state=args.seed)
        .sort_values("emb_row")
        .reset_index(drop=True)
    )

print("Documentos a clasificar:", len(doc_meta))


# ============================================================
# 5) Cargar embeddings
# ============================================================

print("Cargando embeddings de documentos...")
doc_emb = np.load(args.doc_emb, mmap_mode="r")

print("Cargando embeddings de taxonomía...")
tax_emb = np.load(args.tax_emb, mmap_mode="r")

print("Shape documentos:", doc_emb.shape)
print("Shape taxonomía:", tax_emb.shape)


# ============================================================
# 6) Validaciones de dimensiones
# ============================================================

if doc_emb.shape[1] != tax_emb.shape[1]:
    raise ValueError(
        "La dimensión de embeddings de documentos y taxonomía no coincide."
    )

if tax_emb.shape[0] != len(tax_meta):
    raise ValueError(
        "El número de filas de tax_emb no coincide con tax_meta."
    )

if doc_meta["emb_row"].max() >= doc_emb.shape[0]:
    raise ValueError(
        "Hay emb_row en doc_meta fuera del rango del archivo doc_emb."
    )

if tax_meta["emb_row"].max() >= tax_emb.shape[0]:
    raise ValueError(
        "Hay emb_row en tax_meta fuera del rango del archivo tax_emb."
    )


# ============================================================
# 7) Preparar matriz de taxonomía
# ============================================================
# La taxonomía es pequeña, por eso se carga completa en memoria.
# ============================================================

tax_rows = tax_meta["emb_row"].to_numpy(dtype=np.int64)

tax_matrix = np.array(tax_emb[tax_rows, :], dtype=np.float32)
tax_matrix = normalize_matrix(tax_matrix)

top_k = min(args.top_k, len(tax_meta))


# ============================================================
# 8) Definir columnas de documento que se conservarán
# ============================================================
# Evitamos guardar text_for_embedding porque puede hacer el CSV
# demasiado pesado.
# ============================================================

preferred_doc_cols = [
    "emb_row",
    "doc_id",
    "work_id",
    "title",
    "title_norm",
    "language_group",
    "quality_flag",
    "n_words_text",
    "year",
    "revista",
    "type",
    "citas"
]

doc_cols_out = [
    col for col in preferred_doc_cols
    if col in doc_meta.columns
]


# ============================================================
# 9) Clasificación zero-shot por lotes
# ============================================================
# Para cada documento:
#
# 1. Tomamos su embedding.
# 2. Lo normalizamos.
# 3. Calculamos similitud contra todas las subáreas.
# 4. Guardamos las top-k subáreas más cercanas.
# ============================================================

results = []

doc_rows = doc_meta["emb_row"].to_numpy(dtype=np.int64)

print("Iniciando clasificación zero-shot...")

for start in tqdm(range(0, len(doc_meta), args.batch_size)):
    
    end = min(start + args.batch_size, len(doc_meta))
    
    batch_meta = doc_meta.iloc[start:end].reset_index(drop=True)
    batch_rows = doc_rows[start:end]
    
    # Embeddings del lote
    batch_emb = np.array(doc_emb[batch_rows, :], dtype=np.float32)
    batch_emb = normalize_matrix(batch_emb)
    
    # Matriz de similitud:
    # filas = documentos del lote
    # columnas = subáreas de la taxonomía
    scores = np.matmul(batch_emb, tax_matrix.T)
    
    # Obtener top-k sin ordenar completamente todas las columnas
    top_idx_unsorted = np.argpartition(
        -scores,
        kth=top_k - 1,
        axis=1
    )[:, :top_k]
    
    # Ordenar top-k de mayor a menor score
    top_scores_unsorted = np.take_along_axis(
        scores,
        top_idx_unsorted,
        axis=1
    )
    
    order = np.argsort(-top_scores_unsorted, axis=1)
    
    top_idx = np.take_along_axis(
        top_idx_unsorted,
        order,
        axis=1
    )
    
    top_scores = np.take_along_axis(
        top_scores_unsorted,
        order,
        axis=1
    )
    
    # Construir resultados por documento
    for i in range(len(batch_meta)):
        
        row = {}
        
        # Copiar metadata básica del documento
        for col in doc_cols_out:
            row[col] = batch_meta.loc[i, col]
        
        # Top 1
        idx1 = int(top_idx[i, 0])
        score1 = float(top_scores[i, 0])
        
        # Top 2
        if top_k > 1:
            idx2 = int(top_idx[i, 1])
            score2 = float(top_scores[i, 1])
            margin = score1 - score2
        else:
            idx2 = None
            score2 = np.nan
            margin = np.nan
        
        # Regla de estado preliminar
        #
        # low_similarity_review:
        #   el documento no se parece suficientemente a ninguna subárea.
        #
        # ambiguous_review:
        #   top1 y top2 están demasiado cerca.
        #
        # preliminary_label:
        #   etiqueta preliminar aceptable para revisión posterior.
        #
        if score1 < args.min_score:
            status = "low_similarity_review"
        elif top_k > 1 and margin < args.min_margin:
            status = "ambiguous_review"
        else:
            status = "preliminary_label"
        
        row["top1_area_name"] = tax_meta.loc[idx1, "area_name"]
        row["top1_subarea_id"] = tax_meta.loc[idx1, "subarea_id"]
        row["top1_subarea_name"] = tax_meta.loc[idx1, "subarea_name"]
        row["top1_score"] = score1
        
        if top_k > 1:
            row["top2_area_name"] = tax_meta.loc[idx2, "area_name"]
            row["top2_subarea_id"] = tax_meta.loc[idx2, "subarea_id"]
            row["top2_subarea_name"] = tax_meta.loc[idx2, "subarea_name"]
            row["top2_score"] = score2
            row["top1_top2_margin"] = float(margin)
        
        # Top 3 en adelante
        for k in range(2, top_k):
            idxk = int(top_idx[i, k])
            scorek = float(top_scores[i, k])
            kk = k + 1
            
            row[f"top{kk}_area_name"] = tax_meta.loc[idxk, "area_name"]
            row[f"top{kk}_subarea_id"] = tax_meta.loc[idxk, "subarea_id"]
            row[f"top{kk}_subarea_name"] = tax_meta.loc[idxk, "subarea_name"]
            row[f"top{kk}_score"] = scorek
        
        row["zero_shot_status"] = status
        
        results.append(row)


# ============================================================
# 10) Exportar resultados
# ============================================================

out = pd.DataFrame(results)

os.makedirs(os.path.dirname(args.output), exist_ok=True)

out.to_csv(args.output, index=False)

print("\nClasificación zero-shot finalizada.")
print("Archivo de salida:", args.output)
print("Filas exportadas:", len(out))

print("\nDistribución de estados:")
print(out["zero_shot_status"].value_counts(dropna=False))

print("\nDistribución top1_area_name:")
print(out["top1_area_name"].value_counts(dropna=False).head(30))

print("\nDistribución top1_subarea_name:")
print(out["top1_subarea_name"].value_counts(dropna=False).head(30))

