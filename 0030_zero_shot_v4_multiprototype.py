
# ============================================================
# 0030_zero_shot_v4_multiprototype.py
# Zero-shot v4 con taxonomía bilingüe multiprototipo
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize


parser = argparse.ArgumentParser(
    description="Zero-shot v4 multiprototipo: documentos vs taxonomía"
)

parser.add_argument("--doc_emb", required=True)
parser.add_argument("--doc_meta", required=True)
parser.add_argument("--tax_emb", required=True)
parser.add_argument("--tax_meta", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=5000)
parser.add_argument("--sample_n", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--min_score", type=float, default=0.35)
parser.add_argument("--min_margin", type=float, default=0.03)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

print("Cargando embeddings de documentos...")
doc_emb = np.load(args.doc_emb, mmap_mode="r")

print("Cargando metadata de documentos...")
doc_meta = pd.read_csv(args.doc_meta)

print("Cargando embeddings de taxonomía v4...")
tax_emb = np.load(args.tax_emb, mmap_mode="r")

print("Cargando metadata de taxonomía v4...")
tax_meta = pd.read_csv(args.tax_meta)

if "emb_row" not in doc_meta.columns:
    raise ValueError("doc_meta debe contener emb_row")

if "emb_row" not in tax_meta.columns:
    raise ValueError("tax_meta debe contener emb_row")

required_tax_cols = [
    "area_name",
    "subarea_name",
    "subarea_id",
    "prototype_id",
    "language",
    "prototype_type"
]

missing_tax = [c for c in required_tax_cols if c not in tax_meta.columns]

if missing_tax:
    raise ValueError("Faltan columnas en tax_meta: " + ", ".join(missing_tax))

doc_meta = doc_meta.dropna(subset=["emb_row"]).copy()
tax_meta = tax_meta.dropna(subset=["emb_row"]).copy()

doc_meta["emb_row"] = doc_meta["emb_row"].astype(int)
tax_meta["emb_row"] = tax_meta["emb_row"].astype(int)

if doc_meta["emb_row"].max() >= doc_emb.shape[0]:
    raise ValueError("Hay emb_row de documentos fuera del rango de doc_emb")

if tax_meta["emb_row"].max() >= tax_emb.shape[0]:
    raise ValueError("Hay emb_row de taxonomía fuera del rango de tax_emb")

# Ordenar taxonomía por emb_row para alinear embeddings
tax_meta = tax_meta.sort_values("emb_row").reset_index(drop=True)
tax_rows = tax_meta["emb_row"].to_numpy(dtype=np.int64)

T = np.array(tax_emb[tax_rows, :], dtype=np.float32)
T = normalize(T)

print("Shape documentos:", doc_emb.shape)
print("Shape taxonomía:", T.shape)
print("Prototipos taxonomía:", len(tax_meta))
print("Subáreas únicas:", tax_meta["subarea_id"].nunique())

# Crear grupos por subárea
groups = []

for subarea_id, idx in tax_meta.groupby("subarea_id").groups.items():
    idx = np.array(list(idx), dtype=np.int64)
    
    first = tax_meta.iloc[idx[0]]
    
    groups.append({
        "subarea_id": subarea_id,
        "area_name": first["area_name"],
        "subarea_name": first["subarea_name"],
        "indices": idx
    })

print("Grupos de subáreas:", len(groups))

# Selección de muestra si aplica
if args.sample_n and args.sample_n > 0 and args.sample_n < len(doc_meta):
    doc_meta_run = doc_meta.sample(
        n=args.sample_n,
        random_state=args.seed
    ).copy()
else:
    doc_meta_run = doc_meta.copy()

doc_meta_run = doc_meta_run.sort_values("emb_row").reset_index(drop=True)

print("Documentos a clasificar:", len(doc_meta_run))

results = []

doc_rows = doc_meta_run["emb_row"].to_numpy(dtype=np.int64)

for start in tqdm(range(0, len(doc_meta_run), args.batch_size)):
    
    end = min(start + args.batch_size, len(doc_meta_run))
    
    batch_meta = doc_meta_run.iloc[start:end].copy()
    batch_rows = doc_rows[start:end]
    
    X = np.array(doc_emb[batch_rows, :], dtype=np.float32)
    X = normalize(X)
    
    sim = X @ T.T
    
    # Matrices agregadas documento x subárea
    n_batch = sim.shape[0]
    n_groups = len(groups)
    
    sub_scores = np.zeros((n_batch, n_groups), dtype=np.float32)
    sub_best_proto_idx = np.zeros((n_batch, n_groups), dtype=np.int64)
    
    for j, g in enumerate(groups):
        idx = g["indices"]
        sim_g = sim[:, idx]
        best_local = np.argmax(sim_g, axis=1)
        best_scores = sim_g[np.arange(n_batch), best_local]
        best_global_proto = idx[best_local]
        
        sub_scores[:, j] = best_scores
        sub_best_proto_idx[:, j] = best_global_proto
    
    order = np.argsort(-sub_scores, axis=1)
    
    for i in range(n_batch):
        
        row = batch_meta.iloc[i].to_dict()
        
        top_indices = order[i, :args.top_k]
        
        for rank, gidx in enumerate(top_indices, start=1):
            
            g = groups[int(gidx)]
            score = float(sub_scores[i, int(gidx)])
            proto_idx = int(sub_best_proto_idx[i, int(gidx)])
            proto = tax_meta.iloc[proto_idx]
            
            row[f"top{rank}_area_name"] = g["area_name"]
            row[f"top{rank}_subarea_id"] = g["subarea_id"]
            row[f"top{rank}_subarea_name"] = g["subarea_name"]
            row[f"top{rank}_score"] = score
            
            row[f"top{rank}_prototype_id"] = proto.get("prototype_id", None)
            row[f"top{rank}_prototype_language"] = proto.get("language", None)
            row[f"top{rank}_prototype_type"] = proto.get("prototype_type", None)
        
        # Margen top1-top2
        if args.top_k >= 2:
            row["top1_top2_margin"] = row["top1_score"] - row["top2_score"]
        else:
            row["top1_top2_margin"] = np.nan
        
        # Estado zero-shot
        top1_score = row["top1_score"]
        margin = row["top1_top2_margin"]
        
        if top1_score < args.min_score:
            status = "low_similarity_review"
        elif margin < args.min_margin:
            status = "ambiguous_review"
        else:
            status = "preliminary_label"
        
        row["zero_shot_status"] = status
        
        # Confianza más detallada
        if top1_score < 0.35:
            conf = "baja_similitud"
        elif margin < 0.03:
            conf = "ambiguo"
        elif top1_score >= 0.70 and margin >= 0.07:
            conf = "alta_confianza"
        elif top1_score >= 0.55 and margin >= 0.03:
            conf = "confianza_media"
        else:
            conf = "revisar"
        
        row["confidence_level"] = conf
        
        results.append(row)

out = pd.DataFrame(results)

out.to_csv(args.output, index=False)

print("Archivo guardado:", args.output)
print("Filas:", len(out))

print("\nEstados zero-shot:")
print(out["zero_shot_status"].value_counts(dropna=False))

print("\nConfianza:")
print(out["confidence_level"].value_counts(dropna=False))

print("\nTop áreas:")
print(out["top1_area_name"].value_counts(dropna=False).head(30))

print("\nTop prototipos usados:")
if "top1_prototype_type" in out.columns:
    print(out["top1_prototype_type"].value_counts(dropna=False).head(30))

