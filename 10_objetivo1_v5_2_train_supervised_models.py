# ============================================================
# 10_objetivo1_v5_2_train_supervised_models.py
# Objetivo 1 v5.2
# Entrenar modelos supervisados usando embeddings SPECTER2 v5.2
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1) Rutas
# ============================================================

base_dir = "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"

emb_dir = os.path.join(
    base_dir,
    "embeddings",
    "specter2_classification_v5_2_docs_interest_no_concepts"
)

tables_dir = os.path.join(base_dir, "tables")

supervised_dir = os.path.join(
    tables_dir,
    "objetivo1_v5_2_supervised_dataset"
)

out_dir = os.path.join(
    tables_dir,
    "objetivo1_v5_2_supervised_models"
)

os.makedirs(out_dir, exist_ok=True)

emb_file = os.path.join(emb_dir, "docs_embeddings.npy")
emb_meta_file = os.path.join(emb_dir, "docs_embeddings_meta.csv")

train_file = os.path.join(
    supervised_dir,
    "dataset_supervised_train_v5_2.csv"
)

test_file = os.path.join(
    supervised_dir,
    "dataset_manual_test_v5_2.csv"
)

out_predictions = os.path.join(
    out_dir,
    "manual_test_predictions_supervised_v5_2.csv"
)

out_summary = os.path.join(
    out_dir,
    "summary_supervised_v5_2.csv"
)

out_area_report = os.path.join(
    out_dir,
    "classification_report_area_v5_2.csv"
)

out_pair_report = os.path.join(
    out_dir,
    "classification_report_area_subarea_pair_v5_2.csv"
)

out_area_confusion = os.path.join(
    out_dir,
    "confusion_area_supervised_v5_2.csv"
)

out_pair_confusion = os.path.join(
    out_dir,
    "confusion_area_subarea_pair_supervised_v5_2.csv"
)

model_area_file = os.path.join(out_dir, "model_area_sgd_v5_2.joblib")
model_pair_file = os.path.join(out_dir, "model_area_subarea_pair_sgd_v5_2.joblib")
encoder_area_file = os.path.join(out_dir, "label_encoder_area_v5_2.joblib")
encoder_pair_file = os.path.join(out_dir, "label_encoder_area_subarea_pair_v5_2.joblib")

# ============================================================
# 2) Funciones
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def make_pair(area, subarea):
    return clean_text(area) + " || " + clean_text(subarea)

def split_pair(pair):
    parts = str(pair).split(" || ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""

def build_row_lookup(meta):
    if "emb_row" not in meta.columns:
        meta["emb_row"] = np.arange(len(meta))

    if "doc_id_v5" not in meta.columns:
        raise ValueError("La metadata de embeddings no contiene doc_id_v5.")

    meta["doc_id_v5"] = meta["doc_id_v5"].astype(int)
    return dict(zip(meta["doc_id_v5"], meta["emb_row"].astype(int)))

def attach_embedding_rows(df, row_lookup, label):
    df = df.copy()
    df["doc_id_v5"] = df["doc_id_v5"].astype(int)
    df["emb_row"] = df["doc_id_v5"].map(row_lookup)

    missing = df["emb_row"].isna().sum()
    print(f"{label}: filas = {len(df):,}; sin embedding = {missing:,}")

    df = df.dropna(subset=["emb_row"]).copy()
    df["emb_row"] = df["emb_row"].astype(int)

    return df

def report_to_dataframe(y_true, y_pred, labels):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "score": values})
    return pd.DataFrame(rows)

def confusion_to_dataframe(y_true, y_pred, labels, label_name="label"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            n = int(cm[i, j])
            if n > 0:
                rows.append({
                    f"true_{label_name}": true_label,
                    f"predicted_{label_name}": pred_label,
                    "N": n
                })
    return pd.DataFrame(rows).sort_values("N", ascending=False)

# ============================================================
# 3) Cargar datos
# ============================================================

print("Cargando embeddings...")
emb = np.load(emb_file, mmap_mode="r")

print("Embeddings shape:", emb.shape)

print("Cargando metadata embeddings...")
emb_meta = pd.read_csv(emb_meta_file)

row_lookup = build_row_lookup(emb_meta)

print("Cargando entrenamiento...")
train = pd.read_csv(train_file)

print("Cargando test manual...")
test = pd.read_csv(test_file)

# ============================================================
# 4) Preparar etiquetas
# ============================================================

required_cols = [
    "doc_id_v5",
    "label_area",
    "label_subarea",
    "sample_weight"
]

for col in required_cols:
    if col not in train.columns:
        raise ValueError(f"Falta columna en train: {col}")
    if col not in test.columns:
        raise ValueError(f"Falta columna en test: {col}")

train["label_area"] = train["label_area"].map(clean_text)
train["label_subarea"] = train["label_subarea"].map(clean_text)
test["label_area"] = test["label_area"].map(clean_text)
test["label_subarea"] = test["label_subarea"].map(clean_text)

train = train[
    (train["label_area"] != "") &
    (train["label_subarea"] != "")
].copy()

test = test[
    (test["label_area"] != "") &
    (test["label_subarea"] != "")
].copy()

train["label_pair"] = [
    make_pair(a, s) for a, s in zip(train["label_area"], train["label_subarea"])
]

test["label_pair"] = [
    make_pair(a, s) for a, s in zip(test["label_area"], test["label_subarea"])
]

train["sample_weight"] = pd.to_numeric(train["sample_weight"], errors="coerce").fillna(1.0)

# ============================================================
# 5) Vincular filas con embeddings
# ============================================================

train = attach_embedding_rows(train, row_lookup, "Train")
test = attach_embedding_rows(test, row_lookup, "Manual test")

X_train = np.asarray(emb[train["emb_row"].values], dtype=np.float32)
X_test = np.asarray(emb[test["emb_row"].values], dtype=np.float32)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ============================================================
# 6) Codificar etiquetas
# ============================================================

area_encoder = LabelEncoder()
pair_encoder = LabelEncoder()

y_area_train = area_encoder.fit_transform(train["label_area"])
y_pair_train = pair_encoder.fit_transform(train["label_pair"])

# Para test: conservar etiquetas no vistas
known_area = set(area_encoder.classes_)
known_pair = set(pair_encoder.classes_)

test["area_seen_in_train"] = test["label_area"].isin(known_area)
test["pair_seen_in_train"] = test["label_pair"].isin(known_pair)

print("Áreas en entrenamiento:", len(area_encoder.classes_))
print("Pares área/subárea en entrenamiento:", len(pair_encoder.classes_))
print("Test con área no vista:", (~test["area_seen_in_train"]).sum())
print("Test con par no visto:", (~test["pair_seen_in_train"]).sum())

sample_weight = train["sample_weight"].values

# ============================================================
# 7) Entrenar modelo de área
# ============================================================

print("\nEntrenando clasificador de área...")

area_clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=1000,
    tol=1e-3,
    class_weight="balanced",
    random_state=123,
    n_jobs=-1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5
)

area_clf.fit(
    X_train,
    y_area_train,
    sample_weight=sample_weight
)

# ============================================================
# 8) Entrenar modelo de par área/subárea
# ============================================================

print("\nEntrenando clasificador de par área/subárea...")

pair_clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=1000,
    tol=1e-3,
    class_weight="balanced",
    random_state=123,
    n_jobs=-1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5
)

pair_clf.fit(
    X_train,
    y_pair_train,
    sample_weight=sample_weight
)

# ============================================================
# 9) Predicción en test manual
# ============================================================

print("\nPrediciendo test manual...")

area_pred_encoded = area_clf.predict(X_test)
pair_pred_encoded = pair_clf.predict(X_test)

area_pred = area_encoder.inverse_transform(area_pred_encoded)
pair_pred = pair_encoder.inverse_transform(pair_pred_encoded)

pair_pred_area = []
pair_pred_subarea = []

for p in pair_pred:
    a, s = split_pair(p)
    pair_pred_area.append(a)
    pair_pred_subarea.append(s)

test["pred_area_model"] = area_pred
test["pred_pair_model"] = pair_pred
test["pred_pair_area"] = pair_pred_area
test["pred_pair_subarea"] = pair_pred_subarea

# ============================================================
# 10) Evaluación
# ============================================================

test["area_model_correct"] = test["pred_area_model"] == test["label_area"]
test["pair_area_correct"] = test["pred_pair_area"] == test["label_area"]
test["pair_subarea_correct"] = test["pred_pair_subarea"] == test["label_subarea"]
test["pair_exact_correct"] = test["pred_pair_model"] == test["label_pair"]

area_accuracy = accuracy_score(test["label_area"], test["pred_area_model"])
area_balanced_accuracy = balanced_accuracy_score(test["label_area"], test["pred_area_model"])

pair_area_accuracy = accuracy_score(test["label_area"], test["pred_pair_area"])
pair_subarea_accuracy = accuracy_score(test["label_subarea"], test["pred_pair_subarea"])
pair_exact_accuracy = accuracy_score(test["label_pair"], test["pred_pair_model"])

# Evaluación solo en pares vistos
test_seen_pair = test[test["pair_seen_in_train"]].copy()

if len(test_seen_pair) > 0:
    pair_exact_accuracy_seen = accuracy_score(
        test_seen_pair["label_pair"],
        test_seen_pair["pred_pair_model"]
    )
else:
    pair_exact_accuracy_seen = np.nan

summary = pd.DataFrame([
    {"metric": "train_rows", "value": len(train)},
    {"metric": "manual_test_rows", "value": len(test)},
    {"metric": "area_classes_train", "value": len(area_encoder.classes_)},
    {"metric": "area_subarea_pair_classes_train", "value": len(pair_encoder.classes_)},
    {"metric": "manual_test_unseen_area_labels", "value": int((~test["area_seen_in_train"]).sum())},
    {"metric": "manual_test_unseen_pair_labels", "value": int((~test["pair_seen_in_train"]).sum())},
    {"metric": "area_model_accuracy", "value": round(area_accuracy, 4)},
    {"metric": "area_model_balanced_accuracy", "value": round(area_balanced_accuracy, 4)},
    {"metric": "pair_model_area_accuracy", "value": round(pair_area_accuracy, 4)},
    {"metric": "pair_model_subarea_accuracy", "value": round(pair_subarea_accuracy, 4)},
    {"metric": "pair_model_exact_accuracy", "value": round(pair_exact_accuracy, 4)},
    {"metric": "pair_model_exact_accuracy_seen_pairs_only", "value": round(pair_exact_accuracy_seen, 4)}
])

print("\n================ RESUMEN SUPERVISADO v5.2 ================")
print(summary)

# ============================================================
# 11) Reportes
# ============================================================

area_labels = sorted(test["label_area"].unique())
pair_labels = sorted(test["label_pair"].unique())

area_report = report_to_dataframe(
    test["label_area"],
    test["pred_area_model"],
    area_labels
)

pair_report = report_to_dataframe(
    test["label_pair"],
    test["pred_pair_model"],
    pair_labels
)

area_confusion = confusion_to_dataframe(
    test["label_area"],
    test["pred_area_model"],
    area_labels,
    label_name="area"
)

pair_confusion = confusion_to_dataframe(
    test["label_pair"],
    test["pred_pair_model"],
    pair_labels,
    label_name="pair"
)

# ============================================================
# 12) Guardar
# ============================================================

test.to_csv(out_predictions, index=False)
summary.to_csv(out_summary, index=False)
area_report.to_csv(out_area_report, index=False)
pair_report.to_csv(out_pair_report, index=False)
area_confusion.to_csv(out_area_confusion, index=False)
pair_confusion.to_csv(out_pair_confusion, index=False)

joblib.dump(area_clf, model_area_file)
joblib.dump(pair_clf, model_pair_file)
joblib.dump(area_encoder, encoder_area_file)
joblib.dump(pair_encoder, encoder_pair_file)

config = {
    "model_type": "SGDClassifier_log_loss",
    "embeddings": emb_file,
    "train_file": train_file,
    "test_file": test_file,
    "area_model": model_area_file,
    "pair_model": model_pair_file,
    "area_encoder": encoder_area_file,
    "pair_encoder": encoder_pair_file,
    "notes": "Modelo supervisado v5.2 entrenado con seed_strict + seed_expanded. Test independiente manual."
}

with open(os.path.join(out_dir, "model_config_supervised_v5_2.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("\nArchivos creados en:")
print(out_dir)
