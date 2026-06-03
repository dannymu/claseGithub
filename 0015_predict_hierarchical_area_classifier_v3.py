
# ============================================================
# 0015_predict_hierarchical_area_classifier_v3.py
# Aplicar modelo jerárquico v3 a documentos nuevos
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Aplicar clasificador jerárquico v3 de áreas científicas"
)

parser.add_argument("--emb", required=True, help="Archivo .npy con embeddings")
parser.add_argument("--meta", required=True, help="CSV con metadata y emb_row")
parser.add_argument("--model_dir", required=True, help="Directorio del modelo v3")
parser.add_argument("--output", required=True, help="CSV de salida")
parser.add_argument("--batch_size", type=int, default=20000)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

print("Leyendo metadata...")
meta = pd.read_csv(args.meta)

if "emb_row" not in meta.columns:
    raise ValueError("La metadata debe tener emb_row")

if "doc_id" not in meta.columns:
    raise ValueError("La metadata debe tener doc_id")

meta = meta.dropna(subset=["emb_row", "doc_id"]).copy()
meta["emb_row"] = meta["emb_row"].astype(int)
meta["doc_id"] = meta["doc_id"].astype(int)

print("Filas metadata:", len(meta))

print("Cargando embeddings...")
emb = np.load(args.emb, mmap_mode="r")
print("Shape embeddings:", emb.shape)

if meta["emb_row"].max() >= emb.shape[0]:
    raise ValueError("Hay emb_row fuera del rango del archivo .npy")

print("Cargando modelos...")
macro_model = joblib.load(os.path.join(args.model_dir, "macroarea_model.joblib"))
macro_encoder = joblib.load(os.path.join(args.model_dir, "macroarea_encoder.joblib"))

area_models = joblib.load(os.path.join(args.model_dir, "area_models_by_macro.joblib"))
area_encoders = joblib.load(os.path.join(args.model_dir, "area_encoders_by_macro.joblib"))

results = []

rows = meta["emb_row"].to_numpy(dtype=np.int64)

print("Prediciendo por lotes...")

for start in tqdm(range(0, len(meta), args.batch_size)):
    
    end = min(start + args.batch_size, len(meta))
    
    batch_meta = meta.iloc[start:end].copy()
    batch_rows = rows[start:end]
    
    X = np.array(emb[batch_rows, :], dtype=np.float32)
    
    # ----------------------------
    # 1) Predicción de macroárea
    # ----------------------------
    macro_proba = macro_model.predict_proba(X)
    macro_idx = np.argmax(macro_proba, axis=1)
    macro_pred = macro_encoder.inverse_transform(macro_idx)
    macro_score = macro_proba[np.arange(len(batch_meta)), macro_idx]
    
    batch_meta["pred_macroarea_label"] = macro_pred
    batch_meta["pred_macroarea_score"] = macro_score
    
    # ----------------------------
    # 2) Predicción de área dentro de macroárea predicha
    # ----------------------------
    pred_area = []
    pred_area_score = []
    pred_area_top2 = []
    pred_area_top2_score = []
    
    for i in range(len(batch_meta)):
        
        macro_i = macro_pred[i]
        x_i = X[i:i+1]
        
        if macro_i not in area_models:
            pred_area.append(None)
            pred_area_score.append(np.nan)
            pred_area_top2.append(None)
            pred_area_top2_score.append(np.nan)
            continue
        
        clf = area_models[macro_i]
        le = area_encoders[macro_i]
        
        proba = clf.predict_proba(x_i)[0]
        order = np.argsort(-proba)
        
        idx1 = int(order[0])
        area1 = le.inverse_transform([idx1])[0]
        score1 = float(proba[idx1])
        
        pred_area.append(area1)
        pred_area_score.append(score1)
        
        if len(order) > 1:
            idx2 = int(order[1])
            area2 = le.inverse_transform([idx2])[0]
            score2 = float(proba[idx2])
        else:
            area2 = None
            score2 = np.nan
        
        pred_area_top2.append(area2)
        pred_area_top2_score.append(score2)
    
    batch_meta["pred_area_hierarchical"] = pred_area
    batch_meta["pred_area_hierarchical_score"] = pred_area_score
    batch_meta["pred_area_hierarchical_top2"] = pred_area_top2
    batch_meta["pred_area_hierarchical_top2_score"] = pred_area_top2_score
    
    results.append(batch_meta)

out = pd.concat(results, ignore_index=True)

# Mantener columnas principales si existen
preferred_cols = [
    "doc_id",
    "emb_row",
    "title",
    "title_norm",
    "language_group",
    "quality_flag",
    "n_words_text",
    "year",
    "revista",
    "type",
    "citas",
    "pred_macroarea_label",
    "pred_macroarea_score",
    "pred_area_hierarchical",
    "pred_area_hierarchical_score",
    "pred_area_hierarchical_top2",
    "pred_area_hierarchical_top2_score"
]

cols_out = [c for c in preferred_cols if c in out.columns]
other_cols = [c for c in out.columns if c not in cols_out]

out = out[cols_out + other_cols]

out.to_csv(args.output, index=False)

print("Predicciones guardadas en:", args.output)
print("Filas:", len(out))

