
# ============================================================
# 0011_train_supervised_area_classifier.py
# Clasificador supervisado de áreas usando embeddings SPECTER2
# ============================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


parser = argparse.ArgumentParser(
    description="Entrenar clasificador supervisado de áreas con embeddings SPECTER2"
)

parser.add_argument("--emb", required=True, help="Archivo .npy con embeddings")
parser.add_argument("--labels", required=True, help="CSV con emb_row y area_label")
parser.add_argument("--output_dir", required=True, help="Directorio de salida")
parser.add_argument("--label_col", default="area_label")
parser.add_argument("--test_size", type=float, default=0.20)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("Cargando etiquetas...")
labels = pd.read_csv(args.labels)

if "emb_row" not in labels.columns:
    raise ValueError("El archivo de etiquetas debe tener emb_row")

if args.label_col not in labels.columns:
    raise ValueError(f"No existe la columna de etiqueta: {args.label_col}")

labels = labels.dropna(subset=["emb_row", args.label_col]).copy()
labels["emb_row"] = labels["emb_row"].astype(int)
labels[args.label_col] = labels[args.label_col].astype(str)

print("Filas etiquetas:", len(labels))
print("Clases:", labels[args.label_col].nunique())

print("Cargando embeddings...")
emb = np.load(args.emb, mmap_mode="r")

print("Shape embeddings:", emb.shape)

if labels["emb_row"].max() >= emb.shape[0]:
    raise ValueError("Hay emb_row fuera del rango del archivo .npy")

X = np.array(emb[labels["emb_row"].to_numpy(), :], dtype=np.float32)
y_text = labels[args.label_col].to_numpy()

# Codificar etiquetas
le = LabelEncoder()
y = le.fit_transform(y_text)

# División estratificada
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    labels.index.to_numpy(),
    test_size=args.test_size,
    random_state=args.seed,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# Modelo base robusto y eficiente para embeddings
clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
    verbose=1
)

print("Entrenando modelo...")
clf.fit(X_train, y_train)

print("Prediciendo...")
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", acc)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

report_txt = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)

# Guardar modelo
model_file = os.path.join(args.output_dir, "supervised_area_logreg_model.joblib")
encoder_file = os.path.join(args.output_dir, "supervised_area_label_encoder.joblib")

joblib.dump(clf, model_file)
joblib.dump(le, encoder_file)

# Guardar métricas
metrics = {
    "created_at": datetime.now().isoformat(),
    "n_samples": int(len(labels)),
    "n_classes": int(len(le.classes_)),
    "classes": le.classes_.tolist(),
    "test_size": args.test_size,
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "model": "LogisticRegression",
    "embeddings": args.emb,
    "labels": args.labels
}

with open(os.path.join(args.output_dir, "supervised_area_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open(os.path.join(args.output_dir, "supervised_area_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(args.output_dir, "supervised_area_classification_report.csv")
)

pd.DataFrame(
    cm,
    index=le.classes_,
    columns=le.classes_
).to_csv(
    os.path.join(args.output_dir, "supervised_area_confusion_matrix.csv")
)

# Guardar predicciones de test
test_out = labels.loc[idx_test].copy()
test_out["true_area_label"] = le.inverse_transform(y_test)
test_out["pred_area_label"] = le.inverse_transform(y_pred)
test_out["correct"] = test_out["true_area_label"] == test_out["pred_area_label"]

test_out.to_csv(
    os.path.join(args.output_dir, "supervised_area_test_predictions.csv"),
    index=False
)

print("Modelo guardado en:", model_file)
print("Métricas guardadas en:", args.output_dir)

