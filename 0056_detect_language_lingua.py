
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--title_col", default="profile_title")
args = parser.parse_args()

try:
    from lingua import Language, LanguageDetectorBuilder
except Exception as e:
    raise ImportError(
        "No se pudo importar lingua. Instala con: pip install lingua-language-detector"
    ) from e

os.makedirs(os.path.dirname(args.output), exist_ok=True)

df = pd.read_csv(args.input)

if args.title_col not in df.columns:
    raise ValueError(f"No existe la columna de título: {args.title_col}")

# Detector restringido a español e inglés.
# Esto evita ruido de otros idiomas cuando el universo real del perfil es es/en.
detector = (
    LanguageDetectorBuilder
    .from_languages(Language.SPANISH, Language.ENGLISH)
    .with_preloaded_language_models()
    .build()
)

lang_map = {
    Language.SPANISH: "es",
    Language.ENGLISH: "en"
}

langs = []
confidences = []
status = []

for title in df[args.title_col].fillna("").astype(str):
    
    title_clean = title.strip()
    
    if title_clean == "":
        langs.append("unknown")
        confidences.append(0.0)
        status.append("empty_title")
        continue
    
    detected = detector.detect_language_of(title_clean)
    
    if detected is None:
        langs.append("unknown")
        confidences.append(0.0)
        status.append("needs_review")
        continue
    
    confidence_values = detector.compute_language_confidence_values(title_clean)
    
    conf_dict = {
        lang_map.get(item.language, str(item.language)): item.value
        for item in confidence_values
    }
    
    detected_code = lang_map.get(detected, "unknown")
    conf = conf_dict.get(detected_code, 0.0)
    
    langs.append(detected_code)
    confidences.append(conf)
    
    if conf >= 0.80:
        status.append("accepted_high_confidence")
    elif conf >= 0.60:
        status.append("accepted_medium_confidence")
    else:
        status.append("low_confidence_review")

df["title_language_lingua"] = langs
df["title_language_lingua_confidence"] = confidences
df["title_language_lingua_status"] = status
df["title_language_method"] = "lingua_es_en"

df.to_csv(args.output, index=False)

print("Archivo creado:", args.output)
print(df["title_language_lingua"].value_counts(dropna=False))
print(df["title_language_lingua_status"].value_counts(dropna=False))

