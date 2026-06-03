
import os
import json
import time
import argparse
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm


parser = argparse.ArgumentParser(
    description="Generar embeddings SPECTER2 de forma reproducible"
)

parser.add_argument("--input", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--id_col", default="doc_id")
parser.add_argument("--text_col", default="text_for_embedding")

parser.add_argument("--model_name", default="allenai/specter2_base")
parser.add_argument("--adapter_name", default="allenai/specter2_classification")
parser.add_argument("--adapter_load_as", default="specter2_classification")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=512)

parser.add_argument("--normalize", action="store_true")
parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

out_emb = os.path.join(args.output_dir, "docs_embeddings.npy")
out_meta = os.path.join(args.output_dir, "docs_embeddings_meta.csv")
out_manifest = os.path.join(args.output_dir, "embedding_manifest.json")

if args.overwrite:
    for f in [out_emb, out_meta, out_manifest]:
        if os.path.exists(f):
            os.remove(f)

print("Leyendo archivo de entrada...")
df = pd.read_csv(args.input)

print("Columnas disponibles:")
print(df.columns.tolist())

if args.id_col not in df.columns:
    raise ValueError(f"No existe la columna ID: {args.id_col}")

if args.text_col not in df.columns:
    raise ValueError(f"No existe la columna de texto: {args.text_col}")

df[args.text_col] = df[args.text_col].fillna("").astype(str)
df = df[df[args.text_col].str.strip() != ""].copy()

df = df.reset_index(drop=True)
df["emb_row"] = np.arange(len(df), dtype=np.int64)

n_docs = len(df)

print(f"Documentos seleccionados: {n_docs:,}")

if n_docs == 0:
    raise ValueError("No hay documentos válidos para generar embeddings.")

meta_cols = ["emb_row", args.id_col, args.text_col]

optional_cols = [
    "work_id",
    "title",
    "title_norm",
    "abstract",
    "keyword",
    "doi",
    "year",
    "revista",
    "type",
    "citas",
    "title_lang",
    "n_chars_text",
    "n_words_text",
    "quality_flag",
    "language_detected",
    "language_group",
    "usable_for_embedding_generation",
    "usable_for_training_initial",
    "usable_for_embedding_eval",
    "usable_for_training",
    "input_row"
]

for col in optional_cols:
    if col in df.columns and col not in meta_cols:
        meta_cols.append(col)

df[meta_cols].to_csv(out_meta, index=False)

print(f"Metadata guardada en: {out_meta}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo usado: {device}")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

print("Cargando modelo base...")
model = AutoAdapterModel.from_pretrained(args.model_name)

print("Cargando adapter...")
model.load_adapter(
    args.adapter_name,
    source="hf",
    load_as=args.adapter_load_as,
    set_active=True
)

model.set_active_adapters(args.adapter_load_as)
model.to(device)
model.eval()

hidden_size = int(model.config.hidden_size)

print(f"Modelo base: {args.model_name}")
print(f"Adapter: {args.adapter_name}")
print(f"Dimensión embedding: {hidden_size}")

embeddings_out = np.lib.format.open_memmap(
    out_emb,
    mode="w+",
    dtype=np.float32,
    shape=(n_docs, hidden_size)
)

manifest = {
    "created_at": datetime.now().isoformat(),
    "status": "running",
    "input_file": args.input,
    "input_sha256": sha256_file(args.input),
    "output_embeddings": out_emb,
    "output_metadata": out_meta,
    "model_name": args.model_name,
    "adapter_name": args.adapter_name,
    "adapter_load_as": args.adapter_load_as,
    "id_col": args.id_col,
    "text_col": args.text_col,
    "n_documents": n_docs,
    "hidden_size": hidden_size,
    "batch_size": args.batch_size,
    "max_length": args.max_length,
    "pooling": "CLS token",
    "normalize": args.normalize,
    "device": device,
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__
}

with open(out_manifest, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

texts = df[args.text_col].tolist()

t0 = time.time()

print("Generando embeddings...")

with torch.no_grad():
    for start in tqdm(range(0, n_docs, args.batch_size)):
        end = min(start + args.batch_size, n_docs)
        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
            return_token_type_ids=False
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model(**inputs)

        batch_emb = output.last_hidden_state[:, 0, :]
        batch_emb = batch_emb.detach().cpu().numpy().astype(np.float32)

        if args.normalize:
            batch_emb = l2_normalize(batch_emb).astype(np.float32)

        embeddings_out[start:end, :] = batch_emb

del embeddings_out

elapsed = time.time() - t0

manifest["status"] = "finished"
manifest["finished_at"] = datetime.now().isoformat()
manifest["elapsed_seconds"] = elapsed

with open(out_manifest, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("\nProceso finalizado correctamente.")
print(f"Embeddings: {out_emb}")
print(f"Metadata: {out_meta}")
print(f"Manifest: {out_manifest}")
print(f"Tiempo total segundos: {elapsed:.2f}")

