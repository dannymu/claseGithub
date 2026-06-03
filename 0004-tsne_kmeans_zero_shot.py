
# ============================================================
# 0004-tsne_kmeans_zero_shot.py
# Visualización t-SNE + clustering K-means sobre embeddings
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


# ============================================================
# Argumentos
# ============================================================

parser = argparse.ArgumentParser(
    description="t-SNE + K-means para evaluar clusters temáticos sobre embeddings SPECTER2"
)
parser.add_argument("--title_col", default="title", help="Columna de título para anotaciones")
parser.add_argument("--year_col", default="year", help="Columna de año para colorear")

parser.add_argument("--emb", required=True, help="Archivo .npy con embeddings")
parser.add_argument("--meta", required=True, help="Metadata CSV de embeddings")
parser.add_argument("--pred", required=True, help="CSV con predicciones zero-shot")
parser.add_argument("--output_dir", required=True, help="Directorio de salida")

parser.add_argument("--id_col", default="doc_id")
parser.add_argument("--k", type=int, default=0, help="Número de clusters. Si k=0, se estima automáticamente")
parser.add_argument("--max_k", type=int, default=8)
parser.add_argument("--sample_n", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ============================================================
# 1) Cargar datos
# ============================================================

print("Cargando metadata...")
meta = pd.read_csv(args.meta)

print("Cargando predicciones zero-shot...")
pred = pd.read_csv(args.pred)

print("Cargando embeddings...")
emb = np.load(args.emb, mmap_mode="r")

print("Shape embeddings:", emb.shape)
print("Filas metadata:", len(meta))
print("Filas predicciones:", len(pred))


# ============================================================
# 2) Validaciones básicas
# ============================================================

if "emb_row" not in meta.columns:
    raise ValueError("La metadata debe tener emb_row")

if args.id_col not in meta.columns:
    raise ValueError(f"La metadata debe tener {args.id_col}")

if args.id_col not in pred.columns:
    raise ValueError(f"Las predicciones deben tener {args.id_col}")

if len(meta) != emb.shape[0]:
    raise ValueError("El número de filas de metadata no coincide con embeddings")


# ============================================================
# 3) Unir metadata con predicciones zero-shot
# ============================================================
# Usamos doc_id para traer:
# - top1_area_name
# - top1_subarea_name
# - scores
# - zero_shot_status
# ============================================================

cols_pred_keep = [
    args.id_col,
    "top1_area_name",
    "top1_subarea_name",
    "top1_score",
    "top2_area_name",
    "top2_subarea_name",
    "top2_score",
    "top1_top2_margin",
    "zero_shot_status"
]

cols_pred_keep = [c for c in cols_pred_keep if c in pred.columns]

df = meta.merge(
    pred[cols_pred_keep],
    on=args.id_col,
    how="left"
)

df = df.sort_values("emb_row").reset_index(drop=True)

print("Filas después de unir:", len(df))

# ============================================================
# Preparar columnas de título y año
# ============================================================

if args.title_col in df.columns:
    titulos = df[args.title_col].fillna("").astype(str).tolist()
else:
    titulos = [f"Doc {i+1}" for i in range(len(df))]

if args.year_col in df.columns:
    anios = pd.to_numeric(df[args.year_col], errors="coerce")
else:
    anios = pd.Series([np.nan] * len(df))


# ============================================================
# 4) Muestra opcional
# ============================================================
# Para perfiles de investigador se usa todo.
# Para corpus grandes se recomienda sample_n.
# ============================================================

if args.sample_n > 0 and args.sample_n < len(df):
    df = (
        df.sample(n=args.sample_n, random_state=args.seed)
        .sort_values("emb_row")
        .reset_index(drop=True)
    )

print("Documentos a visualizar:", len(df))

if len(df) < 3:
    raise ValueError("Se necesitan al menos 3 documentos para t-SNE/K-means")


# ============================================================
# 5) Extraer embeddings correspondientes
# ============================================================

rows = df["emb_row"].to_numpy(dtype=np.int64)
X = np.array(emb[rows, :], dtype=np.float32)

# Normalización L2 para estabilizar distancias
X = normalize(X)


# ============================================================
# 6) PCA previo
# ============================================================
# PCA reduce ruido y acelera t-SNE.
# Si hay pocos documentos, se ajusta automáticamente.
# ============================================================

n_docs, n_dim = X.shape

pca_components = min(50, n_dim, n_docs - 1)

print("Componentes PCA:", pca_components)

pca = PCA(
    n_components=pca_components,
    random_state=args.seed
)

X_pca = pca.fit_transform(X)

explained = float(np.sum(pca.explained_variance_ratio_))
print("Varianza explicada PCA:", explained)


# ============================================================
# 7) Selección de k para K-means
# ============================================================
# Si el usuario no define k, se prueba k=2 hasta max_k
# y se elige el mejor silhouette score.
# ============================================================

if args.k > 0:
    best_k = args.k
    best_silhouette = np.nan

else:
    max_k = min(args.max_k, n_docs - 1)
    
    candidate_ks = list(range(2, max_k + 1))
    
    best_k = 2
    best_silhouette = -1
    
    if len(candidate_ks) == 0:
        best_k = 1
        best_silhouette = np.nan
    else:
        for k in candidate_ks:
            km_tmp = KMeans(
                n_clusters=k,
                random_state=args.seed,
                n_init=20
            )
            
            labels_tmp = km_tmp.fit_predict(X_pca)
            
            if len(set(labels_tmp)) > 1:
                sil = silhouette_score(X_pca, labels_tmp)
            else:
                sil = -1
            
            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

print("K seleccionado:", best_k)
print("Silhouette:", best_silhouette)


# ============================================================
# 8) K-means final
# ============================================================

if best_k == 1:
    kmeans_labels = np.zeros(n_docs, dtype=int)
else:
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=args.seed,
        n_init=50
    )
    
    kmeans_labels = kmeans.fit_predict(X_pca)

df["kmeans_cluster"] = kmeans_labels


# ============================================================
# 9) t-SNE 2D
# ============================================================
# t-SNE requiere que perplexity sea menor que el número de documentos.
# Para perfiles pequeños se ajusta automáticamente.
# ============================================================

if n_docs < 5:
    print("Muy pocos documentos para t-SNE estable. Se usa PCA 2D como fallback.")
    
    if X_pca.shape[1] >= 2:
        coords = X_pca[:, :2]
    else:
        coords = np.column_stack([X_pca[:, 0], np.zeros(n_docs)])

else:
    perplexity = min(30, max(2, (n_docs - 1) // 3))
    
    print("Perplexity t-SNE:", perplexity)
    
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=perplexity,
    #     init="pca",
    #     learning_rate="auto",
    #     random_state=args.seed
    # )
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        max_iter=1000
    )
        
    coords = tsne.fit_transform(X_pca)

df["tsne_x"] = coords[:, 0]
df["tsne_y"] = coords[:, 1]


# ============================================================
# 10) Resúmenes
# ============================================================

cluster_summary = (
    df.groupby("kmeans_cluster")
    .agg(
        publications=(args.id_col, "count"),
        mean_top1_score=("top1_score", "mean"),
        mean_margin=("top1_top2_margin", "mean")
    )
    .reset_index()
)

if "top1_area_name" in df.columns:
    cluster_area = (
        df.groupby(["kmeans_cluster", "top1_area_name"])
        .size()
        .reset_index(name="N")
        .sort_values(["kmeans_cluster", "N"], ascending=[True, False])
    )
else:
    cluster_area = pd.DataFrame()

if "top1_subarea_name" in df.columns:
    cluster_subarea = (
        df.groupby(["kmeans_cluster", "top1_area_name", "top1_subarea_name"])
        .size()
        .reset_index(name="N")
        .sort_values(["kmeans_cluster", "N"], ascending=[True, False])
    )
else:
    cluster_subarea = pd.DataFrame()


# ============================================================
# 11) Exportar CSV
# ============================================================

points_file = os.path.join(args.output_dir, "tsne_kmeans_points.csv")
cluster_summary_file = os.path.join(args.output_dir, "tsne_kmeans_cluster_summary.csv")
cluster_area_file = os.path.join(args.output_dir, "tsne_kmeans_cluster_area_matrix.csv")
cluster_subarea_file = os.path.join(args.output_dir, "tsne_kmeans_cluster_subarea_matrix.csv")

df.to_csv(points_file, index=False)
cluster_summary.to_csv(cluster_summary_file, index=False)
cluster_area.to_csv(cluster_area_file, index=False)
cluster_subarea.to_csv(cluster_subarea_file, index=False)

print("CSV puntos:", points_file)
print("CSV resumen clusters:", cluster_summary_file)


# ============================================================
# 12) Gráficos estilo mapa semántico
# ============================================================

def short_title(text, max_chars=35):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

# ------------------------------------------------------------
# Gráfico 1: t-SNE coloreado por año de publicación
# ------------------------------------------------------------
if anios.notna().sum() > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        df["tsne_x"],
        df["tsne_y"],
        c=anios,
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidths=1.2
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Año de publicación")
    
    for i, t in enumerate(titulos):
        ax.annotate(
            short_title(t, 10),
            (df["tsne_x"].iloc[i], df["tsne_y"].iloc[i]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=4,
            alpha=0.7
        )
    
    ax.set_title(
        "Mapa semántico del corpus — t-SNE (color = año de publicación)",
        fontweight="bold"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_year = os.path.join(args.output_dir, "mapa_semantico_corpus_anio.png")
    plt.savefig(plot_year, dpi=150, bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------
# Gráfico 2: t-SNE coloreado por cluster K-means
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))

scatter = ax.scatter(
    df["tsne_x"],
    df["tsne_y"],
    c=df["kmeans_cluster"],
    cmap="tab10",
    s=100,
    alpha=0.8,
    edgecolors="white"
)

legend1 = ax.legend(
    *scatter.legend_elements(),
    title="Clusters Temáticos",
    loc="best"
)
ax.add_artist(legend1)

for i, t in enumerate(titulos):
    ax.annotate(
        short_title(t, 10),
        (df["tsne_x"].iloc[i], df["tsne_y"].iloc[i]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=5,
        alpha=0.6
    )

ax.set_title(
    "Mapa Semántico por Agrupamiento Temático (K-Means)",
    fontweight="bold",
    fontsize=15
)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.grid(True, alpha=0.2)
plt.tight_layout()

plot_cluster = os.path.join(args.output_dir, "mapa_semantico_corpus_kmeans.png")
plt.savefig(plot_cluster, dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Gráfico 3: t-SNE coloreado por área zero-shot
# ------------------------------------------------------------
if "top1_area_name" in df.columns:
    area_cat = pd.Categorical(df["top1_area_name"].fillna("Unknown"))
    area_codes = area_cat.codes
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    scatter = ax.scatter(
        df["tsne_x"],
        df["tsne_y"],
        c=area_codes,
        cmap="tab20",
        s=100,
        alpha=0.8,
        edgecolors="white"
    )
    
    handles, _ = scatter.legend_elements()
    labels = list(area_cat.categories)

    if len(labels) <= 30:
        ax.legend(
            handles,
            labels,
            title="Área zero-shot",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )
    

    
    
    
    for i, t in enumerate(titulos):
        ax.annotate(
            short_title(t, 10),
            (df["tsne_x"].iloc[i], df["tsne_y"].iloc[i]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=5,
            alpha=0.6
        )
    
    ax.set_title(
        "Mapa Semántico por Área Sugerida por Zero-shot",
        fontweight="bold",
        fontsize=15
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    plot_area = os.path.join(args.output_dir, "mapa_semantico_corpus_zero_shot_area.png")
    plt.savefig(plot_area, dpi=150, bbox_inches="tight")
    plt.close()

print("Gráficos creados en:", args.output_dir)
print("Proceso finalizado.")

