# ============================================================
# 0060_classify_profile_unmatched_titles_v2.py
#
# Clasifica publicaciones NO emparejadas de un perfil usando
# SOLO el título, SPECTER2 classification adapter y zero-shot
# multiprototipo.
#
# Regla metodológica:
# - Esta salida NO es clasificación final.
# - Es una sugerencia exploratoria title-only.
# - Todo resultado requiere revisión.
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer
# from adapters import AutoAdapterModel
from adapters import AutoAdapterModel, AdapterSetup


# ============================================================
# Helpers
# ============================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def format_title_for_specter2(title: str, sep_token: str) -> str:
    """
    SPECTER2 espera una estructura tipo:
    title [SEP] abstract

    Para título-solo usamos:
    title [SEP] ""

    No agregamos prefijo 'Title:' para no introducir otra distribución
    distinta a la usada en el corpus principal.
    """
    title = clean_text(title)
    if not title:
        return ""
    return f"{title} {sep_token} "


def bool_str(x):
    return bool(x)


# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser(
    description="Clasificar publicaciones no emparejadas usando título-solo + SPECTER2 zero-shot."
)

parser.add_argument("--unmatched_csv", required=True)
parser.add_argument("--tax_emb", required=True)
parser.add_argument("--tax_meta", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--model_name", default="allenai/specter2_base")
parser.add_argument("--adapter_name", default="allenai/specter2_classification")
parser.add_argument("--adapter_load_as", default="specter2_classification")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--top_k", type=int, default=5)

# Umbrales más conservadores porque título-solo suele tener menor señal
parser.add_argument("--min_score", type=float, default=0.30)
parser.add_argument("--min_margin", type=float, default=0.02)

parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--normalize_emb", action="store_true", default=True)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

out_emb_path = os.path.join(args.output_dir, "unmatched_embeddings.npy")
out_meta_path = os.path.join(args.output_dir, "unmatched_embeddings_meta.csv")
out_pred_path = os.path.join(args.output_dir, "unmatched_zero_shot_title_only_suggestions.csv")

if os.path.exists(out_pred_path) and not args.overwrite:
    print("Ya existe archivo de predicciones y overwrite=False:")
    print(out_pred_path)
    print("Saliendo sin recalcular.")
    exit(0)


# ============================================================
# Leer publicaciones no emparejadas
# ============================================================

print("Leyendo publicaciones sin match...")
unmatched = pd.read_csv(args.unmatched_csv)

required_cols = ["profile_row_id", "profile_title"]
missing = [c for c in required_cols if c not in unmatched.columns]
if missing:
    raise ValueError(f"Faltan columnas en unmatched_csv: {missing}")

unmatched = unmatched.copy()
unmatched["profile_title"] = unmatched["profile_title"].fillna("").astype(str)
unmatched = unmatched[unmatched["profile_title"].str.strip() != ""].reset_index(drop=True)

unmatched["emb_row"] = np.arange(len(unmatched), dtype=np.int64)
unmatched["classification_origin"] = "profile_unmatched_title_only_suggestion"
unmatched["doc_id"] = unmatched["profile_row_id"].astype(str)

n_docs = len(unmatched)
print(f"Publicaciones a clasificar: {n_docs:,}")

if n_docs == 0:
    print("No hay publicaciones sin match con título válido.")
    np.save(out_emb_path, np.zeros((0, 768), dtype=np.float32))
    unmatched.to_csv(out_meta_path, index=False)
    unmatched.to_csv(out_pred_path, index=False)
    exit(0)


# ============================================================
# Cargar modelo SPECTER2 classification adapter
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
sep_token = tokenizer.sep_token or "[SEP]"

print("Cargando modelo base...")
model = AutoAdapterModel.from_pretrained(args.model_name)

print("Cargando adapter de clasificación...")

loaded_adapter = model.load_adapter(
    args.adapter_name,
    source="hf",
    load_as=args.adapter_load_as,
    set_active=True
)

adapter_name = args.adapter_load_as

try:
    model.set_active_adapters(adapter_name)
except Exception as e:
    raise RuntimeError(
        f"No se pudo activar el adapter '{adapter_name}'. Error: {e}"
    )

print("Adapter solicitado:", args.adapter_name)
print("Adapter cargado como:", loaded_adapter)
print("Adapter usado para inferencia:", adapter_name)

try:
    print("Adapter activo:", model.active_adapters)
except Exception:
    print("No se pudo leer model.active_adapters")

try:
    print(model.adapter_summary())
except Exception:
    print("No se pudo imprimir adapter_summary().")

model.to(device)
model.eval()

hidden_size = int(model.config.hidden_size)
print(f"Dimensión embedding: {hidden_size}")

# Verificación mínima antes de generar todos los embeddings
test_text = ["test title " + (tokenizer.sep_token or "[SEP]") + " "]
test_inputs = tokenizer(
    test_text,
    padding=True,
    truncation=True,
    max_length=args.max_length,
    return_tensors="pt",
    return_token_type_ids=False
)
test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

with torch.no_grad():
    with AdapterSetup(adapter_name):
        _ = model(**test_inputs)

print("Verificación adapter: forward ejecutado dentro de AdapterSetup.")


# ============================================================
# Preparar textos: título + SEP + abstract vacío
# ============================================================

texts = [
    format_title_for_specter2(t, sep_token)
    for t in unmatched["profile_title"].tolist()
]


# ============================================================
# Generar embeddings
# ============================================================

print("Generando embeddings SPECTER2 desde títulos...")
all_embeddings = np.zeros((n_docs, hidden_size), dtype=np.float32)

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

        with AdapterSetup(adapter_name):
        output = model(**inputs)

        batch_emb = output.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

        if args.normalize_emb:
            batch_emb = l2_normalize(batch_emb)

        all_embeddings[start:end, :] = batch_emb

np.save(out_emb_path, all_embeddings)
unmatched.to_csv(out_meta_path, index=False)

print(f"Embeddings guardados: {out_emb_path}")
print(f"Metadata guardada: {out_meta_path}")


# ============================================================
# Cargar taxonomía
# ============================================================

print("Cargando taxonomía...")
tax_emb_raw = np.load(args.tax_emb, mmap_mode="r")
tax_meta = pd.read_csv(args.tax_meta)

required_tax = [
    "area_name",
    "subarea_name",
    "subarea_id",
    "prototype_id",
    "language",
    "emb_row"
]

missing_tax = [c for c in required_tax if c not in tax_meta.columns]
if missing_tax:
    raise ValueError(f"Faltan columnas en tax_meta: {missing_tax}")

tax_meta = tax_meta.dropna(subset=["emb_row"]).copy()
tax_meta["emb_row"] = tax_meta["emb_row"].astype(int)

if tax_meta["emb_row"].min() < 0 or tax_meta["emb_row"].max() >= tax_emb_raw.shape[0]:
    raise ValueError("Hay emb_row de tax_meta fuera del rango de tax_emb.")

tax_meta = tax_meta.sort_values("emb_row").reset_index(drop=True)
tax_rows = tax_meta["emb_row"].to_numpy(dtype=np.int64)

T = np.array(tax_emb_raw[tax_rows, :], dtype=np.float32)
T = normalize(T)

print(f"Prototipos taxonomía: {len(tax_meta)}")
print(f"Subáreas únicas: {tax_meta['subarea_id'].nunique()}")


# ============================================================
# Agrupar prototipos por subárea
# ============================================================

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


# ============================================================
# Zero-shot multiprototipo
# ============================================================

print("Aplicando zero-shot title-only multiprototipo...")

X = normalize(all_embeddings.copy())
n_groups = len(groups)
results = []

batch_size_zs = 2000

for start in tqdm(range(0, n_docs, batch_size_zs)):
    end = min(start + batch_size_zs, n_docs)
    X_batch = X[start:end]
    n_batch = X_batch.shape[0]

    sim = X_batch @ T.T

    sub_scores = np.zeros((n_batch, n_groups), dtype=np.float32)
    sub_best_proto_idx = np.zeros((n_batch, n_groups), dtype=np.int64)

    for j, g in enumerate(groups):
        idx = g["indices"]
        sim_g = sim[:, idx]

        best_local = np.argmax(sim_g, axis=1)
        best_scores = sim_g[np.arange(n_batch), best_local]
        best_global = idx[best_local]

        sub_scores[:, j] = best_scores
        sub_best_proto_idx[:, j] = best_global

    order = np.argsort(-sub_scores, axis=1)

    for i in range(n_batch):
        pub_row = unmatched.iloc[start + i].to_dict()
        top_idxs = order[i, :args.top_k]

        for rank, gidx in enumerate(top_idxs, start=1):
            g = groups[int(gidx)]
            score = float(sub_scores[i, int(gidx)])
            pidx = int(sub_best_proto_idx[i, int(gidx)])
            proto = tax_meta.iloc[pidx]

            pub_row[f"top{rank}_area_name"] = g["area_name"]
            pub_row[f"top{rank}_subarea_id"] = g["subarea_id"]
            pub_row[f"top{rank}_subarea_name"] = g["subarea_name"]
            pub_row[f"top{rank}_score"] = score
            pub_row[f"top{rank}_prototype_id"] = proto.get("prototype_id", None)
            pub_row[f"top{rank}_prototype_language"] = proto.get("language", None)

            pub_row[f"candidate_top{rank}"] = (
                f"{g['area_name']} / {g['subarea_name']} ({score:.4f})"
            )

        top1_score = float(pub_row.get("top1_score", 0.0))
        top2_score = float(pub_row.get("top2_score", 0.0))
        margin = top1_score - top2_score if args.top_k >= 2 else np.nan

        pub_row["top1_top2_margin"] = margin

        if top1_score < args.min_score:
            status = "low_similarity_review"
            conf = "baja_similitud"
            strength = "exploratory_title_only_low_similarity"
        elif margin < args.min_margin:
            status = "ambiguous_review"
            conf = "ambiguo"
            strength = "exploratory_title_only_ambiguous"
        elif top1_score >= 0.60 and margin >= 0.05:
            status = "preliminary_title_only_suggestion"
            conf = "alta_confianza_title_only"
            strength = "exploratory_title_only_high_signal"
        elif top1_score >= 0.45 and margin >= 0.02:
            status = "preliminary_title_only_suggestion"
            conf = "confianza_media_title_only"
            strength = "exploratory_title_only_medium_signal"
        else:
            status = "preliminary_title_only_suggestion"
            conf = "revisar_title_only"
            strength = "exploratory_title_only_review"

        # IMPORTANTE:
        # No asignar final_area_label.
        # Solo sugerencias title-only.
        pub_row["zero_shot_status"] = status
        pub_row["confidence_level"] = conf
        pub_row["text_input_type"] = "title_sep_empty_abstract"

        pub_row["suggested_area_title_only"] = pub_row.get("top1_area_name", "")
        pub_row["suggested_subarea_title_only"] = pub_row.get("top1_subarea_name", "")
        pub_row["suggested_score_title_only"] = top1_score
        pub_row["suggested_margin_title_only"] = margin

        pub_row["final_label_source"] = "zero_shot_v4_title_only_suggestion"
        pub_row["profile_result_strength"] = strength

        pub_row["qa_auto_accept"] = False
        pub_row["qa_requires_review"] = True
        pub_row["qa_final_decision"] = "requires_review_title_only_zero_shot"
        pub_row["qa_final_confidence_tier"] = "exploratory_title_only"
        pub_row["needs_profile_review"] = True
        pub_row["review_reason"] = (
            "title_only_no_corpus_match; exploratory_zero_shot_suggestion"
        )

        results.append(pub_row)

out_pred = pd.DataFrame(results)
out_pred.to_csv(out_pred_path, index=False)

print("\n============================================================")
print(f"Publicaciones clasificadas como sugerencia title-only: {len(out_pred):,}")

print("\nDistribución top1 área:")
print(out_pred["top1_area_name"].value_counts(dropna=False).head(20).to_string())

print("\nConfianza:")
print(out_pred["confidence_level"].value_counts(dropna=False).to_string())

print("\nEstado zero-shot:")
print(out_pred["zero_shot_status"].value_counts(dropna=False).to_string())

print("\nArchivos generados:")
print(out_emb_path)
print(out_meta_path)
print(out_pred_path)
