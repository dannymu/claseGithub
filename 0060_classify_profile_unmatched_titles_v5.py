# ============================================================
# 0060_classify_profile_unmatched_titles_v5.py
#
# Clasifica publicaciones NO emparejadas de un perfil usando
# SOLO el título, SPECTER2 classification adapter y zero-shot
# multiprototipo v4.
#
# Novedades v5:
# - Reranking de Bibliometría/Cienciometría:
#   Si el título contiene términos bibliométricos Y el área
#   Bibliometrics/Scientometrics está en top2-5 Y el margen
#   top1-top_biblio < args.rerank_margin_threshold
#   → se mueve Bibliometrics a top1 y se marca el cambio.
# - Campo biblio_signal_in_title siempre presente.
# - Campo reranked_to_bibliometrics y rerank_reason.
# - Umbrales ajustables por argumento.
#
# Corrección heredada de v4:
# - output = model(**inputs) DENTRO de AdapterSetup (indentación correcta)
# - No se usa set_active_adapters global; solo AdapterSetup por forward pass
# ============================================================

import os
import argparse
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer
from adapters import AutoAdapterModel, AdapterSetup


# ============================================================
# Reranking: términos bibliométricos en títulos y en áreas
# ============================================================

BIBLIO_TITLE_TERMS = [
    # campo de estudio
    "bibliometr", "scientometr", "metascience", "informetr",
    "webometr", "altmetr", "technomet",
    # métricas e indicadores
    "h-index", "h index", "impact factor", "citation",
    "journal ranking", "journal metric", "research metric",
    "scholar metric", "academic metric", "publication metric",
    # herramientas y fuentes
    "google scholar", "web of science", "scopus", "pubmed",
    "academic search", "academic database", "research database",
    "semantic scholar", "dimensions data",
    # evaluación de investigación
    "research evaluation", "research assessment",
    "academic ranking", "university ranking",
    "research output", "scientific output",
    "research impact", "academic impact",
    "peer review", "open access", "open science",
    # visibilidad y repositorios
    "institutional repository", "research profile",
    "visibility", "academic visibility",
    "impactstory", "orcid", "researcher id",
    "citation analysis", "citation network",
    "co-citation", "bibliographic coupling",
    "publication trend",
]

# Términos a buscar en el area_name de la taxonomía
BIBLIO_LABEL_TERMS = [
    "bibliometr",
    "scientometr",
    "cienciometr",
    "informetr",
    "metascience",
    "research evaluation",
    "scientific communication",
    "scholarly communication",
    "communication and information",
    "information society",
]

BIBLIO_TITLE_TERMS.extend([
    "scholarly communication",
    "scientific communication",
    "comunicacion cientifica",
    "comunicación científica",
    "produccion cientifica",
    "producción científica",
    "scientific production",
    "scientific output",
    "research visibility",
    "academic visibility",
    "journal impact",
    "journal citation",
    "citation database",
    "research performance",
    "science mapping",
    "open citations",
    "data citation",
    "data sharing",
    "data reuse",
])


def has_biblio_signal(title: str) -> bool:
    title_lower = title.lower()
    return any(term in title_lower for term in BIBLIO_TITLE_TERMS)


def is_biblio_label(area_name: str, subarea_name: str) -> bool:
    text = f"{area_name} {subarea_name}".lower()
    return any(term in text for term in BIBLIO_LABEL_TERMS)

def rerank_biblio(order_i, sub_scores_i, groups, margin_threshold: float):
    """
    Dada la ordenación de grupos para una publicación:
    - Devuelve (new_order, biblio_rank, rerank_reason) si se hizo reranking.
    - Devuelve (order_i, None, None) si no aplica.

    Condiciones:
    1. top1 NO es ya un área bibliométrica
    2. top2..topK contiene un área bibliométrica
    3. margen entre top1_score y biblio_score < margin_threshold
    """
    top1_gidx  = int(order_i[0])
    top1_area = groups[top1_gidx]["area_name"]
    top1_subarea = groups[top1_gidx]["subarea_name"]
    top1_score = float(sub_scores_i[top1_gidx])

    if is_biblio_label(top1_area, top1_subarea):
      return order_i, None, None

    # Buscar la primera aparición de bibliometría en top2..topK
    for rank_pos in range(1, len(order_i)):
        gidx  = int(order_i[rank_pos])
        g     = groups[gidx]
        if is_biblio_label(g["area_name"], g["subarea_name"]):
            biblio_score = float(sub_scores_i[gidx])
            margin_top1_biblio = top1_score - biblio_score
            if margin_top1_biblio < margin_threshold:
                # Mover bibliometría a top1
                new_order = list(order_i)
                new_order.pop(rank_pos)
                new_order.insert(0, gidx)
                new_order = np.array(new_order, dtype=np.int64)
                reason = (
                    f"biblio_signal_in_title; "
                    f"orig_top1={top1_area}({top1_score:.4f}); "
                    f"biblio={g['area_name']}({biblio_score:.4f}); "
                    f"margin={margin_top1_biblio:.4f}<{margin_threshold}"
                )
                return new_order, rank_pos + 1, reason   # rank_pos+1 = 1-based original rank
            else:
                break  # margen demasiado grande, no rerank

    return order_i, None, None


# ============================================================
# Helpers
# ============================================================

def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def format_title_for_specter2(title, sep_token):
    title = clean_text(title)
    if not title:
        return ""
    return f"{title} {sep_token} "


# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser(
    description="Clasificar publicaciones no emparejadas: título-solo + SPECTER2 zero-shot v5."
)

parser.add_argument("--unmatched_csv",  required=True)
parser.add_argument("--tax_emb",        required=True)
parser.add_argument("--tax_meta",       required=True)
parser.add_argument("--output_dir",     required=True)
parser.add_argument("--model_name",     default="allenai/specter2_base")
parser.add_argument("--adapter_name",   default="allenai/specter2_classification")
parser.add_argument("--adapter_load_as",default="specter2_classification")
parser.add_argument("--batch_size",     type=int,   default=32)
parser.add_argument("--max_length",     type=int,   default=512)
parser.add_argument("--top_k",          type=int,   default=5)
parser.add_argument("--min_score",      type=float, default=0.30)
parser.add_argument("--min_margin",     type=float, default=0.02)

# Reranking bibliométrico
parser.add_argument(
    "--rerank_bibliometrics", action="store_true", default=True,
    help="Activar reranking de Bibliometrics cuando hay señal en el título (default True)."
)
parser.add_argument(
    "--no_rerank_bibliometrics", dest="rerank_bibliometrics", action="store_false",
    help="Desactivar reranking bibliométrico."
)
parser.add_argument(
    "--rerank_margin_threshold", type=float, default=0.05,
    help="Umbral de margen top1-bibliometrics para aplicar reranking (default 0.05)."
)

parser.add_argument("--overwrite",      action="store_true")
parser.add_argument("--normalize_emb",  action="store_true", default=True)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

out_emb_path  = os.path.join(args.output_dir, "unmatched_embeddings.npy")
out_meta_path = os.path.join(args.output_dir, "unmatched_embeddings_meta.csv")
out_pred_path = os.path.join(args.output_dir, "unmatched_zero_shot_title_only_suggestions.csv")

if os.path.exists(out_pred_path) and not args.overwrite:
    print("Ya existe archivo de predicciones y overwrite=False. Saliendo.")
    exit(0)


# ============================================================
# Leer publicaciones no emparejadas
# ============================================================

print("Leyendo publicaciones sin match...")
unmatched = pd.read_csv(args.unmatched_csv)

missing = [c for c in ["profile_row_id", "profile_title"] if c not in unmatched.columns]
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
# Cargar modelo SPECTER2
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
sep_token = tokenizer.sep_token or "[SEP]"

print("Cargando modelo base...")
model = AutoAdapterModel.from_pretrained(args.model_name)

print("Cargando adapter...")
model.load_adapter(
    args.adapter_name,
    source="hf",
    load_as=args.adapter_load_as,
    set_active=False   # NO activar globalmente: usamos AdapterSetup en cada forward
)

adapter_name = args.adapter_load_as
model.to(device)
model.eval()

hidden_size = int(model.config.hidden_size)
print(f"Adapter cargado: {adapter_name}")
print(f"Dimensión embedding: {hidden_size}")

# Verificación: forward de prueba
test_inputs = tokenizer(
    ["test title " + sep_token + " "],
    padding=True, truncation=True, max_length=64,
    return_tensors="pt", return_token_type_ids=False
)
test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
with torch.no_grad():
    with AdapterSetup(adapter_name):
        _ = model(**test_inputs)
print("Verificación OK: forward ejecutado correctamente con AdapterSetup.")


# ============================================================
# Preparar textos
# ============================================================

texts = [format_title_for_specter2(t, sep_token) for t in unmatched["profile_title"].tolist()]

# Pre-calcular señal bibliométrica por publicación
biblio_signals = [has_biblio_signal(t) for t in unmatched["profile_title"].tolist()]
n_biblio_signal = sum(biblio_signals)
print(f"Títulos con señal bibliométrica: {n_biblio_signal} / {n_docs}")


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
print(f"Embeddings guardados: {out_emb_path}  shape: {all_embeddings.shape}")
print(f"Metadata guardada:    {out_meta_path}")


# ============================================================
# Cargar taxonomía v4
# ============================================================

print("Cargando taxonomía...")
tax_emb_raw = np.load(args.tax_emb, mmap_mode="r")
tax_meta    = pd.read_csv(args.tax_meta)

required_tax = ["area_name", "subarea_name", "subarea_id", "prototype_id", "language", "emb_row"]
missing_tax  = [c for c in required_tax if c not in tax_meta.columns]
if missing_tax:
    raise ValueError(f"Faltan columnas en tax_meta: {missing_tax}")

tax_meta = tax_meta.dropna(subset=["emb_row"]).copy()
tax_meta["emb_row"] = tax_meta["emb_row"].astype(int)
tax_meta = tax_meta.sort_values("emb_row").reset_index(drop=True)
tax_rows = tax_meta["emb_row"].to_numpy(dtype=np.int64)

T = np.array(tax_emb_raw[tax_rows, :], dtype=np.float32)
T = normalize(T)

print(f"Prototipos taxonomía: {len(tax_meta)}  |  Subáreas: {tax_meta['subarea_id'].nunique()}")

groups = []
for subarea_id, idx in tax_meta.groupby("subarea_id").groups.items():
    idx   = np.array(list(idx), dtype=np.int64)
    first = tax_meta.iloc[idx[0]]
    groups.append({
        "subarea_id":   subarea_id,
        "area_name":    first["area_name"],
        "subarea_name": first["subarea_name"],
        "indices":      idx
    })

print(f"Reranking bibliométrico: {'ACTIVO' if args.rerank_bibliometrics else 'INACTIVO'}")
if args.rerank_bibliometrics:
    biblio_labels_in_tax = [
    f"{g['area_name']} / {g['subarea_name']}"
    for g in groups
    if is_biblio_label(g["area_name"], g["subarea_name"])
    ]

    print(f"  Etiquetas bibliométricas detectadas en taxonomía: {biblio_labels_in_tax}")
    print(f"  Umbral de margen para reranking: {args.rerank_margin_threshold}")


# ============================================================
# Zero-shot v4 multiprototipo + reranking v5
# ============================================================

print("Aplicando zero-shot title-only multiprototipo...")

X        = normalize(all_embeddings.copy())
n_groups = len(groups)
results  = []
n_reranked = 0

for start in tqdm(range(0, n_docs, 2000)):
    end     = min(start + 2000, n_docs)
    X_batch = X[start:end]
    n_batch = X_batch.shape[0]

    sim = X_batch @ T.T

    sub_scores         = np.zeros((n_batch, n_groups), dtype=np.float32)
    sub_best_proto_idx = np.zeros((n_batch, n_groups), dtype=np.int64)

    for j, g in enumerate(groups):
        idx    = g["indices"]
        sim_g  = sim[:, idx]
        best_l = np.argmax(sim_g, axis=1)
        sub_scores[:, j]         = sim_g[np.arange(n_batch), best_l]
        sub_best_proto_idx[:, j] = idx[best_l]

    order_batch = np.argsort(-sub_scores, axis=1)

    for i in range(n_batch):
        global_idx = start + i
        pub_row    = unmatched.iloc[global_idx].to_dict()
        order_i    = order_batch[i, :args.top_k]

        # ── reranking bibliométrico v5 ──────────────────────
        reranked         = False
        rerank_reason    = ""
        orig_top1_area   = ""
        biblio_signal    = bool(biblio_signals[global_idx])

        if args.rerank_bibliometrics and biblio_signal and args.top_k >= 2:
            new_order, biblio_orig_rank, reason = rerank_biblio(
                order_i, sub_scores[i], groups, args.rerank_margin_threshold
            )
            if biblio_orig_rank is not None:
                orig_top1_area = groups[int(order_i[0])]["area_name"]
                order_i        = new_order
                reranked       = True
                rerank_reason  = reason
                n_reranked    += 1

        # ── llenar campos top1..topK ────────────────────────
        for rank, gidx in enumerate(order_i, start=1):
            g     = groups[int(gidx)]
            score = float(sub_scores[i, int(gidx)])
            pidx  = int(sub_best_proto_idx[i, int(gidx)])
            proto = tax_meta.iloc[pidx]

            pub_row[f"top{rank}_area_name"]          = g["area_name"]
            pub_row[f"top{rank}_subarea_id"]         = g["subarea_id"]
            pub_row[f"top{rank}_subarea_name"]       = g["subarea_name"]
            pub_row[f"top{rank}_score"]              = score
            pub_row[f"top{rank}_prototype_id"]       = proto.get("prototype_id", None)
            pub_row[f"top{rank}_prototype_language"] = proto.get("language", None)
            pub_row[f"candidate_top{rank}"]          = (
                f"{g['area_name']} / {g['subarea_name']} ({score:.4f})"
            )

        top1_score = float(pub_row.get("top1_score", 0.0))
        top2_score = float(pub_row.get("top2_score", 0.0))
        margin     = top1_score - top2_score if args.top_k >= 2 else float("nan")
        pub_row["top1_top2_margin"] = margin

        # ── estado y confianza ──────────────────────────────
        if top1_score < args.min_score:
            status, conf, strength = (
                "low_similarity_review",
                "baja_similitud",
                "exploratory_title_only_low_similarity",
            )
        elif margin < args.min_margin:
            status, conf, strength = (
                "ambiguous_review",
                "ambiguo",
                "exploratory_title_only_ambiguous",
            )
        elif top1_score >= 0.60 and margin >= 0.05:
            status, conf, strength = (
                "preliminary_title_only",
                "alta_confianza_title_only",
                "exploratory_title_only_high_signal",
            )
        elif top1_score >= 0.45 and margin >= 0.02:
            status, conf, strength = (
                "preliminary_title_only",
                "confianza_media_title_only",
                "exploratory_title_only_medium_signal",
            )
        else:
            status, conf, strength = (
                "preliminary_title_only",
                "revisar_title_only",
                "exploratory_title_only_review",
            )

        # Tras reranking, aumentar nivel de confianza si era "ambiguo"
        # porque el margen original sí era < threshold pero la señal del título lo justifica
        if reranked and status == "ambiguous_review":
            status = "preliminary_title_only"
            conf   = "reranked_biblio_signal"
            strength = "exploratory_title_only_reranked"

        # ── campos de reranking ─────────────────────────────
        pub_row["biblio_signal_in_title"]     = biblio_signal
        pub_row["reranked_to_bibliometrics"]  = reranked
        pub_row["rerank_reason"]              = rerank_reason
        pub_row["orig_top1_area_before_rerank"] = orig_top1_area if reranked else ""

        # ── campos de salida ────────────────────────────────
        pub_row["zero_shot_status"]             = status
        pub_row["confidence_level"]             = conf
        pub_row["text_input_type"]              = "title_sep_empty_abstract"
        pub_row["suggested_area_title_only"]    = pub_row.get("top1_area_name", "")
        pub_row["suggested_subarea_title_only"] = pub_row.get("top1_subarea_name", "")
        pub_row["suggested_score_title_only"]   = top1_score
        pub_row["suggested_margin_title_only"]  = margin
        pub_row["final_label_source"]           = "zero_shot_v4_title_only_suggestion_v5"
        pub_row["profile_result_strength"]      = strength
        pub_row["qa_auto_accept"]               = False
        pub_row["qa_requires_review"]           = True
        pub_row["qa_final_decision"]            = "requires_review_title_only_zero_shot"
        pub_row["qa_final_confidence_tier"]     = "exploratory_title_only"
        pub_row["needs_profile_review"]         = True
        pub_row["review_reason"] = (
            "title_only_no_corpus_match; exploratory_zero_shot_suggestion"
            + ("; reranked_biblio_signal" if reranked else "")
        )

        results.append(pub_row)

out_pred = pd.DataFrame(results)
out_pred.to_csv(out_pred_path, index=False)

# ============================================================
# Resumen
# ============================================================

print(f"\n{'='*60}")
print(f"Publicaciones clasificadas: {len(out_pred):,}")
print(f"Re-clasificadas por señal bibliométrica: {n_reranked}")

print("\nDistribución áreas (top1 final):")
print(out_pred["top1_area_name"].value_counts(dropna=False).head(20).to_string())

print("\nConfianza:")
print(out_pred["confidence_level"].value_counts(dropna=False).to_string())

print("\nEstado zero-shot:")
print(out_pred["zero_shot_status"].value_counts(dropna=False).to_string())

if args.rerank_bibliometrics:
    print("\nReranking bibliométrico aplicado:")
    print(out_pred["reranked_to_bibliometrics"].value_counts(dropna=False).to_string())
    reranked_df = out_pred[out_pred["reranked_to_bibliometrics"] == True]
    if len(reranked_df) > 0:
        print("\nCasos rerrankeados (título → área original → área final):")
        cols = ["profile_title", "orig_top1_area_before_rerank", "top1_area_name", "top1_score", "top1_top2_margin"]
        cols = [c for c in cols if c in reranked_df.columns]
        print(reranked_df[cols].to_string(index=False))

print(f"\nArchivos generados:")
print(f"  {out_emb_path}")
print(f"  {out_meta_path}")
print(f"  {out_pred_path}")
