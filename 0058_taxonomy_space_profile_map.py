
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()

parser.add_argument("--profile", required=True)
parser.add_argument("--doc_emb", required=True)
parser.add_argument("--tax_emb", required=True)
parser.add_argument("--tax_meta", required=True)
parser.add_argument("--out_points", required=True)
parser.add_argument("--out_area_matrix", required=True)
parser.add_argument("--out_subarea_matrix", required=True)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--max_k", type=int, default=8)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.out_points), exist_ok=True)

def find_col(df, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"No se encontró ninguna columna entre: {candidates}")

def reduce_tsne_pca_kmeans(X, prefix, seed=123, max_k=8):
    
    n = X.shape[0]
    
    # PCA
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(X)
    
    # t-SNE
    if n >= 4:
        perplexity = min(10, max(2, n // 4))
        perplexity = min(perplexity, n - 1)
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="pca",
            learning_rate="auto"
        )
        
        tsne_coords = tsne.fit_transform(X)
    else:
        tsne_coords = pca_coords.copy()
    
    # K-means
    best_k = 1
    best_score = -1
    best_labels = np.zeros(n, dtype=int)
    
    if n >= 6:
        max_k = min(max_k, n - 1)
        
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(X)
            
            try:
                score = silhouette_score(X, labels)
            except Exception:
                score = -1
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    
    out = pd.DataFrame({
        f"{prefix}_tsne_x": tsne_coords[:, 0],
        f"{prefix}_tsne_y": tsne_coords[:, 1],
        f"{prefix}_pca_x": pca_coords[:, 0],
        f"{prefix}_pca_y": pca_coords[:, 1],
        f"{prefix}_kmeans_cluster": best_labels,
        f"{prefix}_kmeans_best_k": best_k,
        f"{prefix}_kmeans_silhouette": best_score
    })
    
    return out

print("Leyendo perfil...")
profile = pd.read_csv(args.profile)

print("Cargando embeddings documentos...")
X = np.load(args.doc_emb)

print("Cargando embeddings taxonomía...")
T = np.load(args.tax_emb)

print("Leyendo metadata taxonomía...")
tax = pd.read_csv(args.tax_meta)

if X.shape[0] != len(profile):
    raise ValueError(f"Embeddings perfil {X.shape[0]} != filas perfil {len(profile)}")

area_col = find_col(tax, ["area_name", "area_label", "top1_area_name"])
subarea_col = find_col(tax, ["subarea_name", "subarea_label", "top1_subarea_name"])

tax = tax.copy()
tax["area_key"] = tax[area_col].astype(str)
tax["subarea_key"] = tax[area_col].astype(str) + " || " + tax[subarea_col].astype(str)

# Normalizar
Xn = normalize(X)
Tn = normalize(T)

print("Calculando matriz de similitud documento-taxonomía...")
sim = Xn @ Tn.T

# ============================================================
# Vector temático por área
# ============================================================

area_groups = tax.groupby("area_key").indices
area_names = sorted(area_groups.keys())

area_matrix = np.zeros((X.shape[0], len(area_names)), dtype=np.float32)

for j, area in enumerate(area_names):
    idx = list(area_groups[area])
    area_matrix[:, j] = sim[:, idx].max(axis=1)

area_df = pd.DataFrame(area_matrix, columns=area_names)
area_df.insert(0, "profile_row_number", np.arange(1, X.shape[0] + 1))

# ============================================================
# Vector temático por subárea
# ============================================================

subarea_groups = tax.groupby("subarea_key").indices
subarea_names = sorted(subarea_groups.keys())

subarea_matrix = np.zeros((X.shape[0], len(subarea_names)), dtype=np.float32)

for j, subarea in enumerate(subarea_names):
    idx = list(subarea_groups[subarea])
    subarea_matrix[:, j] = sim[:, idx].max(axis=1)

subarea_df = pd.DataFrame(subarea_matrix, columns=subarea_names)
subarea_df.insert(0, "profile_row_number", np.arange(1, X.shape[0] + 1))

# ============================================================
# Reducción dimensional en espacio temático
# ============================================================

area_coords = reduce_tsne_pca_kmeans(
    normalize(area_matrix),
    prefix="topic_area",
    seed=args.seed,
    max_k=args.max_k
)

subarea_coords = reduce_tsne_pca_kmeans(
    normalize(subarea_matrix),
    prefix="topic_subarea",
    seed=args.seed,
    max_k=args.max_k
)

# Top área/subárea según espacio temático
area_top_idx = area_matrix.argmax(axis=1)
subarea_top_idx = subarea_matrix.argmax(axis=1)

profile_out = profile.copy()
profile_out["profile_row_number"] = np.arange(1, len(profile_out) + 1)

profile_out["taxonomy_space_top_area"] = [area_names[i] for i in area_top_idx]
profile_out["taxonomy_space_top_area_score"] = area_matrix[np.arange(X.shape[0]), area_top_idx]

profile_out["taxonomy_space_top_subarea_pair"] = [subarea_names[i] for i in subarea_top_idx]
profile_out["taxonomy_space_top_subarea_score"] = subarea_matrix[np.arange(X.shape[0]), subarea_top_idx]

points = pd.concat(
    [
        profile_out.reset_index(drop=True),
        area_coords.reset_index(drop=True),
        subarea_coords.reset_index(drop=True)
    ],
    axis=1
)

points.to_csv(args.out_points, index=False)
area_df.to_csv(args.out_area_matrix, index=False)
subarea_df.to_csv(args.out_subarea_matrix, index=False)

print("Archivo puntos creado:", args.out_points)
print("Matriz área:", args.out_area_matrix, area_matrix.shape)
print("Matriz subárea:", args.out_subarea_matrix, subarea_matrix.shape)
print("Best k área:", int(area_coords["topic_area_kmeans_best_k"].iloc[0]))
print("Best k subárea:", int(subarea_coords["topic_subarea_kmeans_best_k"].iloc[0]))

