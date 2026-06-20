import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1. Daten laden
agg_df = pd.read_csv("asterius_aggregated_distances.csv")

# 2. Dateinamen bereinigen: Wir entfernen die Endung "_sample_X" 
# Aus "273_sample_3" wird wieder einfach nur "273"
agg_df['Base_Text'] = agg_df['Pseudo_Text_Sample'].str.replace(r'_sample_\d+', '', regex=True)

# 3. AUF TEXT-EBENE AGGREGIEREN: 
# Wir fassen alle Samples eines Textes zusammen und berechnen den Gesamt-Durchschnitt
text_level_df = agg_df.groupby(['Auteur', 'Base_Text'])['Mean_Distance_to_Asterius'].mean().reset_index()

# 4. Sortieren: Die besten (geringsten) Distanzen nach oben
text_level_df = text_level_df.sort_values(by='Mean_Distance_to_Asterius', ascending=True).reset_index(drop=True)

# 5. Baseline (Oratio 8) ermitteln
baseline_value = text_level_df[text_level_df['Base_Text'].str.contains('oratio8', case=False)]['Mean_Distance_to_Asterius'].values[0]

# =====================================================================
# PLOT: Das bereinigte Gesamtwerk-Balkendiagramm
# =====================================================================
print("Generiere bereinigtes Baseline-Balkendiagramm auf Textebene...")
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Top 15 der GESAMTTEXTE nehmen
top_n = text_level_df.head(15).copy()

# Farben definieren: Oratio 8 (Baseline) bekommt Rot, der Rest Blau
colors = ['#e74c3c' if 'oratio8' in str(label).lower() else '#3498db' for label in top_n['Base_Text']]

# Balken zeichnen
bars = plt.barh(top_n['Base_Text'], top_n['Mean_Distance_to_Asterius'], color=colors)
plt.gca().invert_yaxis() # Den besten Treffer nach oben

# Baseline einzeichnen
plt.axvline(x=baseline_value, color='#c0392b', linestyle='--', linewidth=3, label=f'Baseline (Oratio 8): {baseline_value:.3f}')

# Beschriftungen
plt.title('Durchschnittliche stilistische Distanz zu Asterius (Gesamtwerke)', fontsize=16, pad=20)
plt.xlabel('Mean Distance', fontsize=14)
plt.ylabel('Predigt / Text', fontsize=14)
plt.legend()
plt.tight_layout()

# Grafik speichern
plt.savefig('01b_Baseline_Balkendiagramm_Gesamttexte.png', dpi=300)
plt.close()

print("Bereinigtes Diagramm erfolgreich erstellt!")