
# ============================================================
# 0054_combine_profile_embeddings.py
# Combinar embeddings del corpus completo + embeddings del perfil
# ============================================================

import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--profile", required=True)
parser.add_argument("--full_emb", required=True)
parser.add_argument("--full_meta", required=True)
parser.add_argument("--unmatched_emb", required=True)
parser.add_argument("--unmatched_meta", required=True)
parser.add_argument("--out_emb", required=True)
parser.add_argument("--out_meta", required=True)
parser.add_argument("--out_pred", required=True)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.out_emb), exist_ok=True)
os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)
os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)

print("Leyendo perfil depurado...")
profile = pd.read_csv(args.profile)

if "classification_origin" not in profile.columns:
    raise ValueError("El perfil no contiene classification_origin")

if "profile_row_id" not in profile.columns:
    raise ValueError("El perfil no contiene profile_row_id")

if "doc_id" not in profile.columns:
    raise ValueError("El perfil no contiene doc_id")

profile = profile.copy()
profile["profile_analysis_id"] = np.arange(1, len(profile) + 1)

# Conservar doc_id original antes de crear id interno
profile["original_doc_id"] = profile["doc_id"]

print("Leyendo metadata de embeddings del corpus...")
full_meta = pd.read_csv(args.full_meta)

print("Leyendo metadata de embeddings de no encontrados...")
unmatched_meta = pd.read_csv(args.unmatched_meta)

if "emb_row" not in full_meta.columns:
    raise ValueError("full_meta no tiene emb_row")

if "doc_id" not in full_meta.columns:
    raise ValueError("full_meta no tiene doc_id")

if "emb_row" not in unmatched_meta.columns:
    raise ValueError("unmatched_meta no tiene emb_row")

if "doc_id" not in unmatched_meta.columns:
    raise ValueError("unmatched_meta no tiene doc_id")

full_meta = full_meta.dropna(subset=["doc_id", "emb_row"]).copy()
unmatched_meta = unmatched_meta.dropna(subset=["doc_id", "emb_row"]).copy()

full_meta["doc_id_key"] = full_meta["doc_id"].astype(str)
unmatched_meta["doc_id_key"] = unmatched_meta["doc_id"].astype(str)

full_meta["emb_row"] = full_meta["emb_row"].astype(int)
unmatched_meta["emb_row"] = unmatched_meta["emb_row"].astype(int)

full_lookup = dict(zip(full_meta["doc_id_key"], full_meta["emb_row"]))
unmatched_lookup = dict(zip(unmatched_meta["doc_id_key"], unmatched_meta["emb_row"]))

print("Cargando embeddings...")
full_emb = np.load(args.full_emb, mmap_mode="r")
unmatched_emb = np.load(args.unmatched_emb, mmap_mode="r")

vectors = []
missing = []

for idx, row in profile.iterrows():
    
    origin = row.get("classification_origin", "")
    
    if origin == "matched_in_corpus":
        key_raw = row.get("original_doc_id")
        try:
            key = str(int(float(key_raw)))
        except Exception:
            key = str(key_raw)
        
        if key not in full_lookup:
            missing.append((idx, origin, key, row.get("profile_title", "")))
            continue
        
        emb_row = full_lookup[key]
        vec = np.array(full_emb[emb_row, :], dtype=np.float32)
        vectors.append(vec)
    
    elif origin == "profile_only_classified":
        key_raw = row.get("profile_row_id")
        try:
            key = str(int(float(key_raw)))
        except Exception:
            key = str(key_raw)
        
        if key not in unmatched_lookup:
            missing.append((idx, origin, key, row.get("profile_title", "")))
            continue
        
        emb_row = unmatched_lookup[key]
        vec = np.array(unmatched_emb[emb_row, :], dtype=np.float32)
        vectors.append(vec)
    
    else:
        missing.append((idx, origin, "unknown_origin", row.get("profile_title", "")))

if missing:
    print("Ejemplos de embeddings faltantes:")
    for m in missing[:20]:
        print(m)
    raise ValueError(f"No se encontraron embeddings para {len(missing)} publicaciones del perfil.")

X = np.vstack(vectors).astype(np.float32)

np.save(args.out_emb, X)

# Metadata alineada con la matriz combinada
meta_cols = [
    "profile_analysis_id",
    "profile_row_id",
    "original_doc_id",
    "classification_origin",
    "profile_title",
    "profile_year",
    "profile_cites",
    "final_area_label",
    "subarea_clean_label",
    "subarea_status",
    "final_label_source",
    "final_label_source_detail",
    "analysis_level",
    "final_confidence_score",
    "profile_evidence_level",
    "needs_profile_review"
]

meta_cols = [c for c in meta_cols if c in profile.columns]

meta = profile[meta_cols].copy()
meta.insert(0, "emb_row", np.arange(len(meta), dtype=int))
meta["doc_id"] = profile["profile_analysis_id"]

meta.to_csv(args.out_meta, index=False)

# Archivo de predicciones para mapas
pred = profile.copy()
pred["doc_id_for_maps"] = profile["profile_analysis_id"]

# Asegurar columnas usadas por mapas
if "profile_title" in pred.columns:
    pred["title_for_maps"] = pred["profile_title"]
else:
    pred["title_for_maps"] = ""

if "profile_year" in pred.columns:
    pred["year_for_maps"] = pred["profile_year"]
else:
    pred["year_for_maps"] = np.nan

if "final_area_label" not in pred.columns:
    pred["final_area_label"] = "Unknown"

if "final_confidence_score" not in pred.columns:
    pred["final_confidence_score"] = np.nan

if "final_label_source" not in pred.columns:
    pred["final_label_source"] = "unknown"

pred.to_csv(args.out_pred, index=False)

print("Embeddings combinados guardados:", args.out_emb)
print("Metadata combinada guardada:", args.out_meta)
print("Predicciones para mapas guardadas:", args.out_pred)
print("Shape embeddings:", X.shape)
print("Filas perfil:", len(profile))

