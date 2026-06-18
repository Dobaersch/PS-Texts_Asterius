import os
import re
import pandas as pd
from collections import Counter
from nltk.util import ngrams
import spacy
from bs4 import BeautifulSoup

# Altgriechisches NLP-Modell laden (OdyCy) und GPU bevorzugen
spacy.prefer_gpu()
nlp = spacy.load("grc_odycy_joint_trf")


def extract_text_from_xml(filepath):
    """Liest XML-Dateien aus und entfernt alle Tags."""
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml-xml')
        return soup.get_text(separator=' ')


def clean_and_sample_text(text, sample_size=1000):
    """Bereinigt den Text und teilt ihn in Samples auf."""
    text_clean = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF\s]', '', text)
    words = text_clean.split()

    samples = []
    # Warnung behoben: Vereinfachte Bedingungskette (Simplify chained comparison)
    if sample_size <= len(words) < 2000:
        start = (len(words) - sample_size) // 2
        samples.append(" ".join(words[start:start + sample_size]))

    elif len(words) >= 2000:
        core_words = words[500:-500]
        for i in range(0, len(core_words), sample_size):
            chunk = core_words[i:i + sample_size]
            if len(chunk) == sample_size:
                samples.append(" ".join(chunk))

    return samples


def extract_features(text_sample):
    """Extrahiert MFW, POS-Trigramme und Affixe aus einem Sample."""
    doc = nlp(text_sample)

    words = [token.text.lower() for token in doc]
    word_counts = Counter(words)

    pos_tags = [token.pos_ for token in doc]
    pos_trigrams = list(ngrams(pos_tags, 3))
    pos_trigram_counts = Counter(pos_trigrams)

    affixes = [word[-3:] for word in words if len(word) >= 3]
    affix_counts = Counter(affixes)

    return word_counts, pos_trigram_counts, affix_counts


def process_corpus(input_folder, output_csv):
    """Verarbeitet einen Ordner mit Texten und erstellt die Feature-Matrix."""
    # Phase 1: Roh-Features extrahieren und globale Häufigkeiten zählen
    sample_records = []
    global_word_counts = Counter()
    global_pos_counts = Counter()
    global_affix_counts = Counter()

    valid_files = [f for f in os.listdir(input_folder) if f.endswith(".xml") or f.endswith(".txt")]
    if not valid_files:
        print(f"Keine passenden Dateien in {input_folder} gefunden.")
        return

    print(f"Phase 1: Extrahiere Roh-Features und ermittle die häufigsten Merkmale für {input_folder}...")
    for filename in valid_files:
        filepath = os.path.join(input_folder, filename)
        
        if filename.endswith(".xml"):
            raw_text = extract_text_from_xml(filepath)
            clean_filename = filename.replace(".xml", "")
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            clean_filename = filename.replace(".txt", "")

        parts = clean_filename.split("_", 1)
        author = parts[0]
        title = parts[1] if len(parts) > 1 else "Unknown"

        samples = clean_and_sample_text(raw_text)

        for idx, sample in enumerate(samples):
            w_counts, pos_counts, aff_counts = extract_features(sample)
            
            # Globale Zähler aktualisieren
            global_word_counts.update(w_counts)
            global_pos_counts.update(pos_counts)
            global_affix_counts.update(aff_counts)

            # Record für Phase 2 zwischenspeichern
            sample_records.append({
                "author": author,
                "title": f"{title}_sample_{idx + 1}",
                "w_counts": w_counts,
                "pos_counts": pos_counts,
                "aff_counts": aff_counts,
                "filename": filename,
                "sample_idx": idx + 1
            })

    if not sample_records:
        print("Keine Text-Samples generiert.")
        return

    # Top-Features ermitteln (mehr als ausreichend für die Filterung auf 1000/100/1000)
    top_words = [w for w, _ in global_word_counts.most_common(2000)]
    top_pos = [p for p, _ in global_pos_counts.most_common(200)]
    top_affixes = [a for a, _ in global_affix_counts.most_common(2000)]

    print(f"Phase 2: Baue schlanke Feature-Matrix auf (Top-Features) für {len(sample_records)} Samples...")
    all_features = []
    for record in sample_records:
        print(f"Verarbeite {record['filename']} - Sample {record['sample_idx']}...")
        sample_data = {
            "Auteur": record["author"],
            "Titre": record["title"]
        }

        # Nur Werte für die Top-Features eintragen
        for word in top_words:
            sample_data[f"WORD_{word}"] = record["w_counts"].get(word, 0)
        for trigram in top_pos:
            sample_data[f"POS_{trigram[0]}_{trigram[1]}_{trigram[2]}"] = record["pos_counts"].get(trigram, 0)
        for affix in top_affixes:
            sample_data[f"AFFIX_{affix}"] = record["aff_counts"].get(affix, 0)

        all_features.append(sample_data)

    print("Phase 3: Speichere Feature-Matrix als CSV...")
    df = pd.DataFrame(all_features).fillna(0)
    df.to_csv(output_csv, index=False)
    print(f"Korpus erfolgreich verarbeitet und unter {output_csv} gespeichert.\n")

if __name__ == "__main__":
    process_corpus("data/training_texts", "features_training_corpus.csv")
    process_corpus("data/pseudo_texts", "features_pseudo_chrysostomos.csv")