
import os
import argparse
import numpy as np
import pandas as pd
import joblib

parser = argparse.ArgumentParser(
    description="Aplicar clasificador supervisado de áreas a documentos con embeddings"
)

parser.add_argument("--emb", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--label_col", default="manual_area_label_std")

args = parser.parse_args()

df = pd.read_csv(args.input)

if "emb_row" not in df.columns:
    raise ValueError("El archivo input debe tener emb_row")

emb = np.load(args.emb, mmap_mode="r")

clf = joblib.load(args.model)
le = joblib.load(args.encoder)

df = df.dropna(subset=["emb_row"]).copy()
df["emb_row"] = df["emb_row"].astype(int)

if df["emb_row"].max() >= emb.shape[0]:
    raise ValueError("Hay emb_row fuera del rango del archivo .npy")

X = np.array(emb[df["emb_row"].to_numpy(), :], dtype=np.float32)

pred = clf.predict(X)
proba = clf.predict_proba(X)

df["supervised_pred_area"] = le.inverse_transform(pred)
df["supervised_pred_confidence"] = proba.max(axis=1)

# Top 3 del clasificador
top3_idx = np.argsort(-proba, axis=1)[:, :3]

for k in range(3):
    df[f"supervised_top{k+1}_area"] = le.inverse_transform(top3_idx[:, k])
    df[f"supervised_top{k+1}_score"] = proba[np.arange(len(df)), top3_idx[:, k]]

if args.label_col in df.columns:
    df["manual_area_label_eval"] = df[args.label_col].astype(str)
    df["supervised_area_correct"] = (
        df["supervised_pred_area"] == df["manual_area_label_eval"]
    )

os.makedirs(os.path.dirname(args.output), exist_ok=True)
df.to_csv(args.output, index=False)

print("Predicciones guardadas en:", args.output)
print("Filas:", len(df))

