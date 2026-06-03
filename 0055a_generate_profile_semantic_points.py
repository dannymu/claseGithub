
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()

parser.add_argument("--emb", required=True)
parser.add_argument("--meta", required=True)
parser.add_argument("--profile", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max_k", type=int, default=8)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

print("Cargando embeddings...")
X = np.load(args.emb)

print("Cargando metadata...")
meta = pd.read_csv(args.meta)

print("Cargando perfil...")
profile = pd.read_csv(args.profile)

if "profile_analysis_id" not in meta.columns:
    if "doc_id" in meta.columns:
        meta["profile_analysis_id"] = meta["doc_id"]
    else:
        meta["profile_analysis_id"] = np.arange(1, len(meta) + 1)

if "profile_analysis_id" not in profile.columns:
    profile["profile_analysis_id"] = np.arange(1, len(profile) + 1)

if X.shape[0] != len(meta):
    raise ValueError(f"Filas embeddings {X.shape[0]} != filas metadata {len(meta)}")

X = normalize(X)

# PCA
pca = PCA(n_components=2, random_state=args.seed)
pca_coords = pca.fit_transform(X)

# t-SNE
n = X.shape[0]

if n >= 4:
    perplexity = min(10, max(2, n // 4))
    perplexity = min(perplexity, n - 1)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto"
    )
    
    tsne_coords = tsne.fit_transform(X)
else:
    tsne_coords = pca_coords.copy()

# K-means: seleccionar k por silhouette
best_k = 1
best_score = -1
best_labels = np.zeros(n, dtype=int)

if n >= 6:
    max_k = min(args.max_k, n - 1)
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(X)
        
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

points = meta.copy()

points["tsne_x"] = tsne_coords[:, 0]
points["tsne_y"] = tsne_coords[:, 1]
points["pca_x"] = pca_coords[:, 0]
points["pca_y"] = pca_coords[:, 1]
points["kmeans_cluster"] = best_labels
points["kmeans_best_k"] = best_k
points["kmeans_silhouette"] = best_score

# Unir con perfil para tener área, fuente, evidencia, idioma luego
profile_small = profile.copy()

# Evitar columnas duplicadas excepto profile_analysis_id
cols_to_add = [c for c in profile_small.columns if c not in points.columns or c == "profile_analysis_id"]
profile_small = profile_small[cols_to_add]

points = points.merge(
    profile_small,
    on="profile_analysis_id",
    how="left"
)

points.to_csv(args.output, index=False)

print("Archivo creado:", args.output)
print("Filas:", len(points))
print("Best k:", best_k)
print("Silhouette:", best_score)

