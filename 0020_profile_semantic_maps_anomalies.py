
# ============================================================
# 0020_profile_semantic_maps_anomalies.py
# Mapas semánticos y detección exploratoria de anomalías
# para perfiles de investigadores
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Generar mapas semánticos, K-means y anomalías para perfil de investigador"
)

parser.add_argument("--emb", required=True)
parser.add_argument("--meta", required=True)
parser.add_argument("--pred", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--id_col", default="doc_id")
parser.add_argument("--title_col", default="title")
parser.add_argument("--year_col", default="year")
parser.add_argument("--area_col", default="final_area_label")
parser.add_argument("--confidence_col", default="final_confidence_score")
parser.add_argument("--source_col", default="final_label_source")
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--max_k", type=int, default=8)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================
# 1) Cargar datos
# ============================================================

print("Cargando metadata...")
meta = pd.read_csv(args.meta)

print("Cargando predicciones...")
pred = pd.read_csv(args.pred)

print("Cargando embeddings...")
emb = np.load(args.emb, mmap_mode="r")

if "emb_row" not in meta.columns:
    raise ValueError("La metadata debe contener emb_row")

if args.id_col not in meta.columns:
    raise ValueError(f"Metadata sin columna {args.id_col}")

if args.id_col not in pred.columns:
    raise ValueError(f"Predicciones sin columna {args.id_col}")

meta["emb_row"] = meta["emb_row"].astype(int)
meta[args.id_col] = meta[args.id_col].astype(int)
pred[args.id_col] = pred[args.id_col].astype(int)

# Unir metadata + predicciones
df = pred.merge(
    meta[[args.id_col, "emb_row"]],
    on=args.id_col,
    how="left",
    suffixes=("", "_meta")
)

df = df.dropna(subset=["emb_row"]).copy()
df["emb_row"] = df["emb_row"].astype(int)

df = df.sort_values("emb_row").reset_index(drop=True)

print("Documentos del perfil:", len(df))

if len(df) < 3:
    raise ValueError("Se requieren al menos 3 publicaciones para mapas semánticos.")

rows = df["emb_row"].to_numpy(dtype=np.int64)

if rows.max() >= emb.shape[0]:
    raise ValueError("emb_row fuera de rango del archivo .npy")

X = np.array(emb[rows, :], dtype=np.float32)
X = normalize(X)

n_docs, n_dim = X.shape

# ============================================================
# 2) Preparar variables
# ============================================================

def short_title(text, max_chars=40):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

if args.title_col in df.columns:
    titles = df[args.title_col].fillna("").astype(str).tolist()
else:
    titles = [f"Doc {i+1}" for i in range(len(df))]

if args.year_col in df.columns:
    years = pd.to_numeric(df[args.year_col], errors="coerce")
else:
    years = pd.Series([np.nan] * len(df))

if args.area_col in df.columns:
    areas = df[args.area_col].fillna("Unknown").astype(str)
else:
    areas = pd.Series(["Unknown"] * len(df))

if args.confidence_col in df.columns:
    confidence = pd.to_numeric(df[args.confidence_col], errors="coerce")
else:
    confidence = pd.Series([np.nan] * len(df))

if args.source_col in df.columns:
    label_source = df[args.source_col].fillna("unknown").astype(str)
else:
    label_source = pd.Series(["unknown"] * len(df))

# ============================================================
# 3) PCA
# ============================================================

pca_components = min(50, n_dim, n_docs - 1)

pca = PCA(
    n_components=pca_components,
    random_state=args.seed
)

X_pca = pca.fit_transform(X)

pca_2d = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.column_stack([X_pca[:, 0], np.zeros(n_docs)])

explained = float(np.sum(pca.explained_variance_ratio_))

print("PCA components:", pca_components)
print("PCA explained variance:", explained)

df["pca_x"] = pca_2d[:, 0]
df["pca_y"] = pca_2d[:, 1]

# ============================================================
# 4) K-means
# ============================================================

if args.k > 0:
    best_k = args.k
    best_silhouette = np.nan
else:
    max_k = min(args.max_k, n_docs - 1)
    candidate_ks = list(range(2, max_k + 1))
    
    best_k = 1
    best_silhouette = np.nan
    
    if len(candidate_ks) > 0:
        best_score = -999
        for k in candidate_ks:
            km_tmp = KMeans(
                n_clusters=k,
                random_state=args.seed,
                n_init=30
            )
            labels_tmp = km_tmp.fit_predict(X_pca)
            if len(set(labels_tmp)) > 1:
                score = silhouette_score(X_pca, labels_tmp)
            else:
                score = -999
            
            if score > best_score:
                best_score = score
                best_k = k
                best_silhouette = score

print("K seleccionado:", best_k)
print("Silhouette:", best_silhouette)

if best_k <= 1:
    kmeans_labels = np.zeros(n_docs, dtype=int)
    centers = np.array([X_pca.mean(axis=0)])
else:
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=args.seed,
        n_init=50
    )
    kmeans_labels = kmeans.fit_predict(X_pca)
    centers = kmeans.cluster_centers_

df["kmeans_cluster"] = kmeans_labels

# Distancia al centro del cluster
cluster_distances = []

for i in range(n_docs):
    cluster_i = kmeans_labels[i]
    center_i = centers[cluster_i]
    dist_i = np.linalg.norm(X_pca[i] - center_i)
    cluster_distances.append(dist_i)

df["distance_to_cluster_centroid"] = cluster_distances

# Distancia al centro global
global_center = X_pca.mean(axis=0)
df["distance_to_global_centroid"] = np.linalg.norm(X_pca - global_center, axis=1)

# ============================================================
# 5) t-SNE
# ============================================================

if n_docs < 5:
    coords = pca_2d
else:
    perplexity = min(30, max(2, (n_docs - 1) // 3))
    
    print("Perplexity t-SNE:", perplexity)
    
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
            max_iter=1000
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
            n_iter=1000
        )
    
    coords = tsne.fit_transform(X_pca)

df["tsne_x"] = coords[:, 0]
df["tsne_y"] = coords[:, 1]

# ============================================================
# 6) Detección exploratoria de anomalías
# ============================================================

# Outlier por distancia global
if n_docs >= 10:
    global_threshold = np.percentile(df["distance_to_global_centroid"], 90)
else:
    global_threshold = np.percentile(df["distance_to_global_centroid"], 80)

df["global_semantic_outlier"] = df["distance_to_global_centroid"] >= global_threshold

# Outlier por distancia dentro del cluster
df["cluster_semantic_outlier"] = False

for cl in sorted(df["kmeans_cluster"].unique()):
    idx = df["kmeans_cluster"] == cl
    sub = df.loc[idx]
    if len(sub) >= 5:
        th = np.percentile(sub["distance_to_cluster_centroid"], 90)
        df.loc[idx, "cluster_semantic_outlier"] = sub["distance_to_cluster_centroid"] >= th

# Baja confianza
df["low_confidence_flag"] = False

if args.confidence_col in df.columns:
    df["low_confidence_flag"] = confidence < 0.75

# Fuente no aceptada automáticamente
df["not_automatic_label_flag"] = label_source != "supervised_v3"

# Mismatch cluster-área dominante
cluster_area = (
    df.groupby(["kmeans_cluster", args.area_col])
    .size()
    .reset_index(name="N")
    if args.area_col in df.columns
    else pd.DataFrame()
)

dominant_area_by_cluster = {}

if not cluster_area.empty:
    for cl in sorted(df["kmeans_cluster"].unique()):
        sub = cluster_area[cluster_area["kmeans_cluster"] == cl].sort_values("N", ascending=False)
        if len(sub) > 0:
            dominant_area_by_cluster[cl] = sub.iloc[0][args.area_col]

df["dominant_area_in_cluster"] = df["kmeans_cluster"].map(dominant_area_by_cluster)

if args.area_col in df.columns:
    df["cluster_area_mismatch"] = (
        df[args.area_col].fillna("Unknown").astype(str) !=
        df["dominant_area_in_cluster"].fillna("Unknown").astype(str)
    )
else:
    df["cluster_area_mismatch"] = False

# Isolation Forest como detector adicional
if n_docs >= 20:
    iso = IsolationForest(
        contamination=min(0.15, max(0.05, 3 / n_docs)),
        random_state=args.seed
    )
    iso_pred = iso.fit_predict(X_pca)
    df["isolation_forest_outlier"] = iso_pred == -1
else:
    df["isolation_forest_outlier"] = False

# Anomalía temporal simple: documentos muy alejados en año dentro del cluster
df["temporal_outlier_in_cluster"] = False

if years.notna().sum() >= 5:
    df["_year_num"] = years
    for cl in sorted(df["kmeans_cluster"].unique()):
        sub = df[df["kmeans_cluster"] == cl]
        if sub["_year_num"].notna().sum() >= 5:
            y_mean = sub["_year_num"].mean()
            y_sd = sub["_year_num"].std()
            if y_sd > 0:
                idx = (df["kmeans_cluster"] == cl) & (abs(df["_year_num"] - y_mean) > 1.5 * y_sd)
                df.loc[idx, "temporal_outlier_in_cluster"] = True

# Score agregado de anomalía
anomaly_cols = [
    "global_semantic_outlier",
    "cluster_semantic_outlier",
    "low_confidence_flag",
    "not_automatic_label_flag",
    "cluster_area_mismatch",
    "isolation_forest_outlier",
    "temporal_outlier_in_cluster"
]

df["anomaly_score"] = df[anomaly_cols].sum(axis=1)

def anomaly_type(row):
    flags = [col for col in anomaly_cols if bool(row[col])]
    if len(flags) == 0:
        return "sin_senales_anomalas"
    return ";".join(flags)

df["anomaly_type"] = df.apply(anomaly_type, axis=1)

df["anomaly_level"] = pd.cut(
    df["anomaly_score"],
    bins=[-1, 0, 1, 2, 10],
    labels=[
        "sin_senal",
        "baja",
        "media",
        "alta"
    ]
)

# ============================================================
# 7) Resúmenes
# ============================================================

cluster_summary = (
    df.groupby("kmeans_cluster")
    .agg(
        publications=(args.id_col, "count"),
        mean_confidence=(args.confidence_col, "mean") if args.confidence_col in df.columns else (args.id_col, "count"),
        mean_year=(args.year_col, "mean") if args.year_col in df.columns else (args.id_col, "count"),
        mean_global_distance=("distance_to_global_centroid", "mean"),
        mean_cluster_distance=("distance_to_cluster_centroid", "mean"),
        anomaly_mean=("anomaly_score", "mean")
    )
    .reset_index()
)

if args.area_col in df.columns:
    cluster_area_matrix = (
        df.groupby(["kmeans_cluster", args.area_col])
        .size()
        .reset_index(name="N")
        .sort_values(["kmeans_cluster", "N"], ascending=[True, False])
    )
else:
    cluster_area_matrix = pd.DataFrame()

anomalies = df[df["anomaly_score"] >= 2].copy()
anomalies = anomalies.sort_values(
    ["anomaly_score", "distance_to_global_centroid"],
    ascending=[False, False]
)

# ============================================================
# 8) Guardar CSV
# ============================================================

points_file = os.path.join(args.output_dir, "profile_semantic_points.csv")
cluster_summary_file = os.path.join(args.output_dir, "profile_kmeans_cluster_summary.csv")
cluster_area_file = os.path.join(args.output_dir, "profile_kmeans_area_matrix.csv")
anomalies_file = os.path.join(args.output_dir, "profile_possible_anomalies.csv")

df.to_csv(points_file, index=False)
cluster_summary.to_csv(cluster_summary_file, index=False)
cluster_area_matrix.to_csv(cluster_area_file, index=False)
anomalies.to_csv(anomalies_file, index=False)

print("Puntos:", points_file)
print("Clusters:", cluster_summary_file)
print("Anomalías:", anomalies_file)

# ============================================================
# 9) Funciones de gráfico
# ============================================================

def annotate_points(ax, xcol, ycol, max_labels=120):
    if len(df) <= max_labels:
        for i, t in enumerate(titles):
            ax.annotate(
                short_title(t, 35),
                (df[xcol].iloc[i], df[ycol].iloc[i]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=5,
                alpha=0.65
            )

def plot_area_map(xcol, ycol, filename, title):
    area_cat = pd.Categorical(areas.fillna("Unknown"))
    area_codes = area_cat.codes
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    scatter = ax.scatter(
        df[xcol],
        df[ycol],
        c=area_codes,
        cmap="tab20",
        s=120,
        alpha=0.82,
        edgecolors="white",
        linewidths=1.0
    )
    
    labels = list(area_cat.categories)
    handles, _ = scatter.legend_elements()
    
    if len(labels) <= 20:
        ax.legend(
            handles,
            labels,
            title="Área",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )
    
    annotate_points(ax, xcol, ycol)
    
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename), dpi=160, bbox_inches="tight")
    plt.close()

def plot_year_map(xcol, ycol, filename, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if years.notna().sum() > 0:
        scatter = ax.scatter(
            df[xcol],
            df[ycol],
            c=years,
            cmap="viridis",
            s=110,
            alpha=0.82,
            edgecolors="white",
            linewidths=1.0
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Año de publicación")
    else:
        ax.scatter(
            df[xcol],
            df[ycol],
            s=110,
            alpha=0.82,
            edgecolors="white",
            linewidths=1.0
        )
    
    annotate_points(ax, xcol, ycol)
    
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename), dpi=160, bbox_inches="tight")
    plt.close()

def plot_kmeans_map(xcol, ycol, filename, title):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    scatter = ax.scatter(
        df[xcol],
        df[ycol],
        c=df["kmeans_cluster"],
        cmap="tab10",
        s=120,
        alpha=0.82,
        edgecolors="white",
        linewidths=1.0
    )
    
    legend1 = ax.legend(
        *scatter.legend_elements(),
        title="K-means",
        loc="best"
    )
    ax.add_artist(legend1)
    
    annotate_points(ax, xcol, ycol)
    
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename), dpi=160, bbox_inches="tight")
    plt.close()

def plot_anomaly_map(xcol, ycol, filename, title):
    fig, ax = plt.subplots(figsize=(14, 9))
    
    scatter = ax.scatter(
        df[xcol],
        df[ycol],
        c=df["anomaly_score"],
        cmap="plasma",
        s=130,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.0
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Anomaly score")
    
    annotate_points(ax, xcol, ycol)
    
    ax.set_title(title, fontweight="bold", fontsize=15)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, filename), dpi=160, bbox_inches="tight")
    plt.close()

# ============================================================
# 10) Crear mapas
# ============================================================

plot_area_map(
    "tsne_x",
    "tsne_y",
    "mapa_tsne_por_area.png",
    "Mapa semántico del perfil — t-SNE por área temática"
)

plot_year_map(
    "tsne_x",
    "tsne_y",
    "mapa_tsne_por_anio.png",
    "Mapa semántico del perfil — t-SNE por año de publicación"
)

plot_kmeans_map(
    "tsne_x",
    "tsne_y",
    "mapa_tsne_por_kmeans.png",
    "Mapa semántico del perfil — K-means temático"
)

plot_anomaly_map(
    "tsne_x",
    "tsne_y",
    "mapa_tsne_anomalias.png",
    "Mapa semántico del perfil — señales de posible anomalía"
)

plot_area_map(
    "pca_x",
    "pca_y",
    "mapa_pca_por_area.png",
    "Mapa PCA del perfil — área temática"
)

plot_anomaly_map(
    "pca_x",
    "pca_y",
    "mapa_pca_anomalias.png",
    "Mapa PCA del perfil — señales de posible anomalía"
)

# ============================================================
# 11) Gráficos agregados
# ============================================================

# Barras por área
area_counts = areas.value_counts().reset_index()
area_counts.columns = ["area", "N"]

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(area_counts["area"], area_counts["N"])
ax.invert_yaxis()
ax.set_title("Distribución temática del perfil", fontweight="bold", fontsize=15)
ax.set_xlabel("Número de publicaciones")
ax.set_ylabel("Área")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "barras_distribucion_areas.png"), dpi=160, bbox_inches="tight")
plt.close()

# Barras por anomalía
anom_counts = df["anomaly_level"].value_counts().reset_index()
anom_counts.columns = ["anomaly_level", "N"]

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(anom_counts["anomaly_level"].astype(str), anom_counts["N"])
ax.set_title("Distribución de niveles de anomalía exploratoria", fontweight="bold", fontsize=15)
ax.set_xlabel("Nivel")
ax.set_ylabel("Número de publicaciones")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "barras_niveles_anomalia.png"), dpi=160, bbox_inches="tight")
plt.close()

print("Gráficos creados en:", args.output_dir)
print("Proceso finalizado.")

