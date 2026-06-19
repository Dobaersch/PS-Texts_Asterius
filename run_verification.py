import pandas as pd
from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.pipeline import SiamesePipeline


def filter_top_features(df, prefix, top_n):
    """
    Behält nur die 'top_n' häufigsten Spalten eines bestimmten Feature-Typs (z.B. WORD_).
    """
    # Alle Spalten finden, die mit dem Präfix beginnen
    feature_cols = [col for col in df.columns if col.startswith(prefix)]

    if not feature_cols:
        return []

    # Die Summe über das gesamte Korpus für diese Features berechnen
    sums = df[feature_cols].sum().sort_values(ascending=False)

    # Die Top N Spaltennamen zurückgeben
    return sums.head(top_n).index.tolist()


print("1. Lade generierte Feature-Matrizen...")
df_train = pd.read_csv("features_training_corpus.csv")
df_pseudo = pd.read_csv("features_pseudo_chrysostomos.csv")

# Damit beide DataFrames exakt die gleichen Spalten haben, füllen wir fehlende auf
df_train, df_pseudo = df_train.align(df_pseudo, join='outer', axis=1, fill_value=0)

print("2. Filtere auf Top-Features (1000 MFW, 100 POS, 1000 Affixe)...")
top_words = filter_top_features(df_train, "WORD_", 1000)
top_pos = filter_top_features(df_train, "POS_", 100)
top_affixes = filter_top_features(df_train, "AFFIX_", 1000)

# Metadaten-Spalten behalten
meta_cols = ["Auteur", "Titre"]
selected_columns = meta_cols + top_words + top_pos + top_affixes

df_train = df_train[selected_columns]
df_pseudo = df_pseudo[selected_columns]

print("3. Initialisiere DataframeWrapper und normalisiere Frequenzen...")
data_train = DataframeWrapper(df_train, target="Auteur", label="Titre")
data_pseudo = DataframeWrapper(df_pseudo, target="Auteur", label="Titre")

# Relative Frequenzen bilden (sehr wichtig für Dokumente unterschiedlicher Länge!)
data_train.normalized.make_relative(inplace=True)
data_pseudo.align_to(data_train)  # Feature-Raum exakt angleichen

print("4. Baue das Siamese Network auf...")
# "auto" wählt automatisch die GPU, wenn CUDA verfügbar ist, andernfalls CPU
pipeline = SiamesePipeline(accelerator="auto")

pipeline.build(
    dimension=64,  # Lineare Projektion (nach Paper)
    learning_rate=1e-4,  # Lernrate
    loss="stn_contrastive",  # Signal-to-Noise Ratio (STN) Distanz
    margin=1.0,  # Margin
    batch_size=64,
    patience=20  # Early Stopping
)

print("5. Trainiere das Modell (dies kann eine Weile dauern)...")
pipeline.fit(
    data=data_train,
    dev_ratio=0.1,  # 10% zur Validierung während des Trainings abzweigen
    sample=False,
    model_name="Asterius_Verificator"
)

print("6. Verifiziere das Pseudo-Chrysostomos Korpus...")
# Das Modell berechnet nun alle Distanzen zwischen den unbekannten Texten
# und dem Trainingskorpus (inklusive Asterius)
predictions = pipeline.predict(
    data=data_pseudo,
    comparator=data_train,
    threshold=0.5,
    model="Asterius_Verificator"
)

print("7. Auswertung...")
# 1. Filtere nur die Zeilen, in denen Pseudo-Texte mit Asterius verglichen wurden
asterius_matches = predictions[predictions['ComparatorClass'].str.lower() == 'asterius'].copy()

# 2. Wir ignorieren die 'Probability' und nutzen nur die 'Distance'.
# Je kleiner die Distanz, desto höher die stilistische Ähnlichkeit.

# 3. Aggregieren: Berechne die durchschnittliche Distanz pro Pseudo-Text zu allen Asterius-Samples
mean_distances = asterius_matches.groupby('ComparedLabel')['Distance'].mean().reset_index()

# 4. Zusätzliche Metrik: Minimale Distanz (Welches war das absolut ähnlichste Asterius-Sample?)
min_distances = asterius_matches.groupby('ComparedLabel')['Distance'].min().reset_index()
mean_distances['Min_Distance'] = min_distances['Distance']

# 5. Aufsteigend nach der Durchschnitts-Distanz sortieren (die besten Kandidaten stehen oben)
mean_distances = mean_distances.sort_values(by='Distance', ascending=True)

# Spalten zur besseren Lesbarkeit umbenennen
mean_distances = mean_distances.rename(columns={'ComparedLabel': 'Pseudo_Text_Sample', 'Distance': 'Mean_Distance_to_Asterius'})

print("\n--- DURCHSCHNITTLICHE DISTANZEN ZU ASTERIUS (Top 10) ---")
# Zeige die 10 Samples mit der geringsten Durchschnittsdistanz
print(mean_distances.head(10))

# 6. Ergebnisse speichern
mean_distances.to_csv("asterius_aggregated_distances.csv", index=False)
print("\nAggregierte Ergebnisse wurden in 'asterius_aggregated_distances.csv' gespeichert.")

# (Optional) Speichere zur Sicherheit auch nochmal alle Rohdaten, falls du Einzelvergleiche brauchst
asterius_matches.to_csv("asterius_results_raw_distances.csv", index=False)