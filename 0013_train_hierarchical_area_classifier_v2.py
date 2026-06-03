
# ============================================================
# 0013_train_hierarchical_area_classifier_v2.py
# Modelo jerárquico:
# 1) embedding -> macroarea
# 2) embedding -> área dentro de macroarea
# ============================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


parser = argparse.ArgumentParser(
    description="Entrenar clasificador jerárquico de áreas científicas"
)

parser.add_argument("--emb", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("Leyendo etiquetas...")
df = pd.read_csv(args.labels)

required = [
    "emb_row",
    "area_label",
    "macroarea_label",
    "label_source"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Falta columna requerida: {col}")

df = df.dropna(subset=["emb_row", "area_label", "macroarea_label"]).copy()
df["emb_row"] = df["emb_row"].astype(int)
df["area_label"] = df["area_label"].astype(str)
df["macroarea_label"] = df["macroarea_label"].astype(str)
df["label_source"] = df["label_source"].astype(str)

print("Total filas:", len(df))
print("Fuentes:")
print(df["label_source"].value_counts())

print("Cargando embeddings...")
emb = np.load(args.emb, mmap_mode="r")
print("Shape embeddings:", emb.shape)

if df["emb_row"].max() >= emb.shape[0]:
    raise ValueError("Hay emb_row fuera del rango del .npy")

# ============================================================
# Separación metodológica:
# - train: pseudo-etiquetas
# - eval: manual_gold
# ============================================================

train_df = df[df["label_source"] != "manual_gold"].copy()
eval_df = df[df["label_source"] == "manual_gold"].copy()

if len(eval_df) == 0:
    raise ValueError("No hay manual_gold para evaluación externa.")

print("Train pseudo:", len(train_df))
print("Eval manual_gold:", len(eval_df))

X_train = np.array(
    emb[train_df["emb_row"].to_numpy(), :],
    dtype=np.float32
)

X_eval = np.array(
    emb[eval_df["emb_row"].to_numpy(), :],
    dtype=np.float32
)

# ============================================================
# 1) Modelo de macroárea
# ============================================================

macro_encoder = LabelEncoder()
y_macro_train = macro_encoder.fit_transform(train_df["macroarea_label"])

macro_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
    random_state=args.seed,
    verbose=1
)

print("\nEntrenando modelo de macroárea...")
macro_model.fit(X_train, y_macro_train)

macro_pred_eval_id = macro_model.predict(X_eval)
macro_pred_eval = macro_encoder.inverse_transform(macro_pred_eval_id)

eval_df["pred_macroarea_label"] = macro_pred_eval
eval_df["macro_correct"] = (
    eval_df["macroarea_label"] == eval_df["pred_macroarea_label"]
)

macro_accuracy = accuracy_score(
    eval_df["macroarea_label"],
    eval_df["pred_macroarea_label"]
)

macro_macro_f1 = f1_score(
    eval_df["macroarea_label"],
    eval_df["pred_macroarea_label"],
    average="macro"
)

print("Macroarea accuracy:", macro_accuracy)
print("Macroarea macro F1:", macro_macro_f1)

# ============================================================
# 2) Modelos de área por macroárea
# ============================================================

area_models = {}
area_encoders = {}
area_train_counts = {}

for macro in sorted(train_df["macroarea_label"].unique()):
    
    sub_train = train_df[train_df["macroarea_label"] == macro].copy()
    
    n_classes = sub_train["area_label"].nunique()
    
    print(f"\nMacroárea: {macro}")
    print("Filas:", len(sub_train))
    print("Áreas:", n_classes)
    print(sub_train["area_label"].value_counts())
    
    if n_classes < 2:
        print("Saltando: solo una clase.")
        continue
    
    X_sub = np.array(
        emb[sub_train["emb_row"].to_numpy(), :],
        dtype=np.float32
    )
    
    le_area = LabelEncoder()
    y_sub = le_area.fit_transform(sub_train["area_label"])
    
    clf_area = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=args.seed,
        verbose=0
    )
    
    clf_area.fit(X_sub, y_sub)
    
    area_models[macro] = clf_area
    area_encoders[macro] = le_area
    area_train_counts[macro] = sub_train["area_label"].value_counts().to_dict()

# ============================================================
# 3) Predicción jerárquica sobre manual_gold
# ============================================================

hier_area_preds = []
hier_area_scores = []

oracle_area_preds = []
oracle_area_scores = []

for i in range(len(eval_df)):
    
    x_i = X_eval[i:i+1]
    
    # Macro predicha
    pred_macro = eval_df.iloc[i]["pred_macroarea_label"]
    true_macro = eval_df.iloc[i]["macroarea_label"]
    
    # Predicción jerárquica usando macroárea predicha
    if pred_macro in area_models:
        clf = area_models[pred_macro]
        le = area_encoders[pred_macro]
        proba = clf.predict_proba(x_i)[0]
        idx = int(np.argmax(proba))
        hier_area_preds.append(le.inverse_transform([idx])[0])
        hier_area_scores.append(float(proba[idx]))
    else:
        hier_area_preds.append(None)
        hier_area_scores.append(np.nan)
    
    # Predicción usando macroárea verdadera
    # Sirve para saber si el problema está en el nivel macro o en el área interna.
    if true_macro in area_models:
        clf = area_models[true_macro]
        le = area_encoders[true_macro]
        proba = clf.predict_proba(x_i)[0]
        idx = int(np.argmax(proba))
        oracle_area_preds.append(le.inverse_transform([idx])[0])
        oracle_area_scores.append(float(proba[idx]))
    else:
        oracle_area_preds.append(None)
        oracle_area_scores.append(np.nan)

eval_df["pred_area_hierarchical"] = hier_area_preds
eval_df["pred_area_hierarchical_score"] = hier_area_scores

eval_df["pred_area_oracle_macro"] = oracle_area_preds
eval_df["pred_area_oracle_macro_score"] = oracle_area_scores

eval_df["area_correct_hierarchical"] = (
    eval_df["area_label"] == eval_df["pred_area_hierarchical"]
)

eval_df["area_correct_oracle_macro"] = (
    eval_df["area_label"] == eval_df["pred_area_oracle_macro"]
)

# ============================================================
# 4) Modelo plano de comparación
# ============================================================

area_encoder_flat = LabelEncoder()
y_area_train_flat = area_encoder_flat.fit_transform(train_df["area_label"])

flat_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
    random_state=args.seed,
    verbose=1
)

print("\nEntrenando modelo plano de comparación...")
flat_model.fit(X_train, y_area_train_flat)

flat_pred_id = flat_model.predict(X_eval)
flat_pred = area_encoder_flat.inverse_transform(flat_pred_id)

eval_df["pred_area_flat"] = flat_pred
eval_df["area_correct_flat"] = eval_df["area_label"] == eval_df["pred_area_flat"]

# ============================================================
# 5) Métricas
# ============================================================

hier_eval_valid = eval_df.dropna(subset=["pred_area_hierarchical"]).copy()
oracle_eval_valid = eval_df.dropna(subset=["pred_area_oracle_macro"]).copy()

metrics = {
    "created_at": datetime.now().isoformat(),
    "n_train_pseudo": int(len(train_df)),
    "n_eval_manual": int(len(eval_df)),
    "n_macroareas_train": int(train_df["macroarea_label"].nunique()),
    "n_areas_train": int(train_df["area_label"].nunique()),
    "macroarea_accuracy_manual": float(macro_accuracy),
    "macroarea_macro_f1_manual": float(macro_macro_f1),
    "area_accuracy_hierarchical_manual": float(
        accuracy_score(
            hier_eval_valid["area_label"],
            hier_eval_valid["pred_area_hierarchical"]
        )
    ),
    "area_macro_f1_hierarchical_manual": float(
        f1_score(
            hier_eval_valid["area_label"],
            hier_eval_valid["pred_area_hierarchical"],
            average="macro"
        )
    ),
    "area_accuracy_oracle_macro_manual": float(
        accuracy_score(
            oracle_eval_valid["area_label"],
            oracle_eval_valid["pred_area_oracle_macro"]
        )
    ),
    "area_macro_f1_oracle_macro_manual": float(
        f1_score(
            oracle_eval_valid["area_label"],
            oracle_eval_valid["pred_area_oracle_macro"],
            average="macro"
        )
    ),
    "area_accuracy_flat_manual": float(
        accuracy_score(
            eval_df["area_label"],
            eval_df["pred_area_flat"]
        )
    ),
    "area_macro_f1_flat_manual": float(
        f1_score(
            eval_df["area_label"],
            eval_df["pred_area_flat"],
            average="macro"
        )
    )
}

print("\nMétricas:")
for k, v in metrics.items():
    print(k, ":", v)

# ============================================================
# 6) Reportes por clase
# ============================================================

macro_report = classification_report(
    eval_df["macroarea_label"],
    eval_df["pred_macroarea_label"],
    output_dict=True,
    zero_division=0
)

hier_report = classification_report(
    hier_eval_valid["area_label"],
    hier_eval_valid["pred_area_hierarchical"],
    output_dict=True,
    zero_division=0
)

oracle_report = classification_report(
    oracle_eval_valid["area_label"],
    oracle_eval_valid["pred_area_oracle_macro"],
    output_dict=True,
    zero_division=0
)

flat_report = classification_report(
    eval_df["area_label"],
    eval_df["pred_area_flat"],
    output_dict=True,
    zero_division=0
)

# ============================================================
# 7) Guardar
# ============================================================

joblib.dump(macro_model, os.path.join(args.output_dir, "macroarea_model.joblib"))
joblib.dump(macro_encoder, os.path.join(args.output_dir, "macroarea_encoder.joblib"))

joblib.dump(area_models, os.path.join(args.output_dir, "area_models_by_macro.joblib"))
joblib.dump(area_encoders, os.path.join(args.output_dir, "area_encoders_by_macro.joblib"))

joblib.dump(flat_model, os.path.join(args.output_dir, "flat_area_model.joblib"))
joblib.dump(area_encoder_flat, os.path.join(args.output_dir, "flat_area_encoder.joblib"))

with open(os.path.join(args.output_dir, "hierarchical_v2_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open(os.path.join(args.output_dir, "area_train_counts_by_macro.json"), "w", encoding="utf-8") as f:
    json.dump(area_train_counts, f, ensure_ascii=False, indent=2)

pd.DataFrame(macro_report).transpose().to_csv(
    os.path.join(args.output_dir, "macroarea_manual_report.csv")
)

pd.DataFrame(hier_report).transpose().to_csv(
    os.path.join(args.output_dir, "area_hierarchical_manual_report.csv")
)

pd.DataFrame(oracle_report).transpose().to_csv(
    os.path.join(args.output_dir, "area_oracle_macro_manual_report.csv")
)

pd.DataFrame(flat_report).transpose().to_csv(
    os.path.join(args.output_dir, "area_flat_manual_report.csv")
)

eval_df.to_csv(
    os.path.join(args.output_dir, "manual_gold_predictions_hierarchical_v2.csv"),
    index=False
)

# Matrices de confusión
pd.DataFrame(
    confusion_matrix(
        eval_df["macroarea_label"],
        eval_df["pred_macroarea_label"],
        labels=sorted(eval_df["macroarea_label"].unique())
    ),
    index=sorted(eval_df["macroarea_label"].unique()),
    columns=sorted(eval_df["macroarea_label"].unique())
).to_csv(os.path.join(args.output_dir, "macroarea_confusion_matrix.csv"))

pd.DataFrame(
    confusion_matrix(
        eval_df["area_label"],
        eval_df["pred_area_flat"],
        labels=sorted(eval_df["area_label"].unique())
    ),
    index=sorted(eval_df["area_label"].unique()),
    columns=sorted(eval_df["area_label"].unique())
).to_csv(os.path.join(args.output_dir, "flat_area_confusion_matrix.csv"))

print("\nArchivos guardados en:", args.output_dir)

