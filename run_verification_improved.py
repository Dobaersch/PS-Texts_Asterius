import pandas as pd
import numpy as np
import re
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.pipeline import SiamesePipeline
import torch

print("0. Hardware-Diagnose...")
if torch.cuda.is_available():
    print(f"Erfolg: CUDA erkannt. Das Modell nutzt die NVIDIA-GPU: {torch.cuda.get_device_name(0)}")
else:
    print("KRITISCHE WARNUNG: Keine NVIDIA-GPU gefunden oder PyTorch wurde ohne CUDA-Support installiert.")
    print("Das Skript wird auf der wesentlich langsameren CPU ausgeführt!")

def filter_top_features(df, prefix, top_n):
    """ Behält nur die 'top_n' häufigsten Spalten eines bestimmten Feature-Typs (z.B. WORD_). """
    feature_cols = [col for col in df.columns if col.startswith(prefix)]
    if not feature_cols:
        return []
    sums = df[feature_cols].sum().sort_values(ascending=False)
    return sums.head(top_n).index.tolist()

print("1. Lade generierte Feature-Matrizen...")
df_train = pd.read_csv("features_training_corpus.csv")
df_pseudo = pd.read_csv("features_pseudo_chrysostomos.csv")

# Damit beide DataFrames exakt die gleichen Spalten haben, füllen wir fehlende auf
df_train, df_pseudo = df_train.align(df_pseudo, join='outer', axis=1, fill_value=0)

print("2. Initialisiere DataframeWrapper und normalisiere Frequenzen...")
# Übergabe der zwingend benötigten Ziel- und Label-Spalten
data_train = DataframeWrapper(df_train, target="Auteur", label="Titre")
data_pseudo = DataframeWrapper(df_pseudo, target="Auteur", label="Titre")

# Relative Frequenzen bilden (methodisch zwingend für Dokumente unterschiedlicher Länge)
data_train.normalized.make_relative(inplace=True)
# Feature-Räume beider Korpora exakt angleichen
data_pseudo.align_to(data_train)

print("3. Baue das Siamese Network auf...")
pipeline = SiamesePipeline(accelerator="auto")
pipeline.build(
    dimension=64,
    learning_rate=1e-4,
    loss="stn_contrastive",
    margin=1.0,
    batch_size=64,
    patience=20
)

print("4. Trainiere das Modell (dies kann eine Weile dauern)...")
pipeline.fit(
    data=data_train,
    dev_ratio=0.1,
    sample=False,
    model_name="Asterius_Verificator"
)

print("\n--- 5. KALIBRIERUNG: BERECHNUNG DER INTRA- UND INTER-AUTOR-BASELINE ---")

# 5.1. Distanzen aller Trainingsdaten zueinander berechnen
baseline_predictions = pipeline.predict(
        data=data_train,
        comparator=data_train,
        threshold=0.5,
        model="Asterius_Verificator"
)

# 5.2. Mapping von Titre zu Auteur für das Trainingsset erstellen
author_map_train = df_train.set_index('Titre')['Auteur'].to_dict()

# 5.3. Label-Mapping auf die Vorhersagen anwenden
baseline_predictions['ComparedClass'] = baseline_predictions['ComparedLabel'].map(author_map_train)
baseline_predictions['ComparatorClass'] = baseline_predictions['ComparatorLabel'].map(author_map_train)

# 5.4. Positiv-Klasse: Asterius vs. Asterius (ohne Identitätsvergleiche)
asterius_intra = baseline_predictions[
    (baseline_predictions['ComparedClass'].str.lower() == 'asterius') &
    (baseline_predictions['ComparatorClass'].str.lower() == 'asterius') &
    (baseline_predictions['ComparedLabel'] != baseline_predictions['ComparatorLabel'])
].copy()

# 5.5. Negativ-Klasse: Asterius vs. Andere Autoren
asterius_inter = baseline_predictions[
    (baseline_predictions['ComparedClass'].str.lower() == 'asterius') &
    (baseline_predictions['ComparatorClass'].str.lower() != 'asterius')
].copy()

# 5.6. Robuste statistische Auswertung (Perzentile statt Standardabweichung)
intra_90th_percentile = asterius_intra['Distance'].quantile(0.90)
mean_inter_distance = asterius_inter['Distance'].mean()

# 5.7. Diskriminativer Schwellenwert
# Der Threshold ist das 90. Perzentil, darf aber nicht zu nah an die Fremd-Autoren heranreichen (Cap bei 85%)
asterius_threshold = min(intra_90th_percentile, mean_inter_distance * 0.85)

print(f"Anzahl verglichener Asterius-Paare (Positiv): {len(asterius_intra)}")
print(f"Anzahl Asterius vs. Fremd-Autoren (Negativ): {len(asterius_inter)}")
print(f"---")
print(f"90. Perzentil der echten Asterius-Texte: {intra_90th_percentile:.4f}")
print(f"Durchschnittliche Distanz zu FREMDEN Autoren: {mean_inter_distance:.4f}")
print(f"---")
print(f"-> DEFINIERTER SCHWELLENWERT FÜR MATCHES: {asterius_threshold:.4f}\n")

print("6. Verifiziere das Pseudo-Chrysostomos Korpus...")
predictions = pipeline.predict(
    data=data_pseudo,
    comparator=data_train,
    threshold=0.5,
    model="Asterius_Verificator"
)

print("7. Auswertung (Pairing Percentage auf Dokumentenebene)...")
author_map = df_pseudo.set_index('Titre')['Auteur'].to_dict()
predictions['ComparedClass'] = predictions['ComparedLabel'].map(author_map)

# 7.1. Filtere Pseudo-Texte, die gegen Asterius geprüft wurden
asterius_matches = predictions[predictions['ComparatorClass'].str.lower() == 'asterius'].copy()

# 7.2. Document-Level Averaging: _sample_X Suffix bereinigen
asterius_matches['Base_Document'] = asterius_matches['ComparedLabel'].str.replace(r'_sample_\d+', '', regex=True)

# 7.3. Klassifiziere jedes Sample-Paar: 1 wenn Distanz <= Schwellenwert, sonst 0
asterius_matches['Is_Match'] = (asterius_matches['Distance'] <= asterius_threshold).astype(int)

# 7.4. Gruppieren nach dem Basis-Dokument und Berechnen der Match-Quote (Pairing Percentage)
pairing_percentages = asterius_matches.groupby('Base_Document')['Is_Match'].mean().reset_index()
pairing_percentages = pairing_percentages.rename(columns={
    'Is_Match': 'Pairing_Percentage'
})

# 7.5. In % umwandeln für präzise Lesbarkeit und absteigend sortieren
pairing_percentages['Pairing_Percentage'] = pairing_percentages['Pairing_Percentage'] * 100
pairing_percentages = pairing_percentages.sort_values(by='Pairing_Percentage', ascending=False)

print("\n--- PAIRING PERCENTAGE ZU ASTERIUS (Top 15 Kandidaten) ---")
print(pairing_percentages.head(15))

# 8. Ergebnisse speichern
pairing_percentages.to_csv("asterius_aggregated_pairing_percentage.csv", index=False)
asterius_matches.to_csv("asterius_results_raw_distances.csv", index=False)

print("\nAggregierte Ergebnisse wurden in 'asterius_aggregated_pairing_percentage.csv' gespeichert.")
print("Rohdaten für die textkritische Chronologie-Visualisierung liegen in 'asterius_results_raw_distances.csv' bereit.")