import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Daten laden (Namen ggf. anpassen)
agg_df = pd.read_csv("asterius_aggregated_distances.csv")
raw_df = pd.read_csv("asterius_results_raw_distances.csv")

# Stil für Präsentationen festlegen
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# =====================================================================
# PLOT 1: Das Baseline-Balkendiagramm (Goldstandard-Plot)
# =====================================================================
print("Generiere Baseline-Balkendiagramm...")
plt.figure(figsize=(12, 8))

# Wir nehmen die Top 15 aus der aggregierten Liste für eine saubere Grafik
top_n = agg_df.head(15).copy()

# Farben definieren: Oratio 8 bekommt eine besondere Farbe
colors = ['#e74c3c' if 'oratio8' in str(label).lower() else '#3498db' for label in top_n['Pseudo_Text_Sample']]

# Balken zeichnen (horizontal)
bars = plt.barh(top_n['Pseudo_Text_Sample'], top_n['Mean_Distance_to_Asterius'], color=colors)
plt.gca().invert_yaxis() # Den besten Treffer nach oben

# Die magische Baseline-Linie (Oratio 8) einzeichnen
baseline_value = top_n[top_n['Pseudo_Text_Sample'].str.contains('oratio8', case=False)]['Mean_Distance_to_Asterius'].values[0]
plt.axvline(x=baseline_value, color='#c0392b', linestyle='--', linewidth=3, label=f'Baseline (Oratio 8): {baseline_value:.3f}')

# Beschriftungen
plt.title('Durchschnittliche stilistische Distanz zu Asterius (Siamese Network)', fontsize=16, pad=20)
plt.xlabel('Mean Distance', fontsize=14)
plt.ylabel('Text Sample', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('01_Baseline_Balkendiagramm.png', dpi=300)
plt.close()

# =====================================================================
# PLOT 2: Die Heatmap (Das "Oratio 2"-Phänomen)
# =====================================================================
print("Generiere Heatmap für Oratio 2-Zwilling...")
plt.figure(figsize=(14, 6))

# Den Text mit der unglaublichen Oratio 2 Nähe filtern
# (Den genauen Namen musst du ggf. anpassen, falls er in der CSV leicht abweicht)
target_pseudo = 'DeParabolaUilliciIniquitatis_sample_1' 
subset = raw_df[raw_df['ComparedLabel'] == target_pseudo].copy()

if not subset.empty:
    # Nur Asterius-Vergleichstexte behalten
    subset = subset[subset['ComparatorLabel'].str.contains('oratio', case=False, na=False)]
    
    # Pivot-Tabelle für die Heatmap erstellen
    heatmap_data = subset.pivot_table(index='ComparedLabel', columns='ComparatorLabel', values='Distance')
    
    # Heatmap zeichnen
    sns.heatmap(heatmap_data, cmap='YlOrRd_r', annot=True, fmt=".2f", cbar_kws={'label': 'Distanz (Je roter, desto ähnlicher)'})
    
    plt.title(f'Paarweiser Vergleich: {target_pseudo}', fontsize=16, pad=20)
    plt.xlabel('Asterius Original-Predigten', fontsize=14)
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('02_Oratio2_Zwilling_Heatmap.png', dpi=300)
    plt.close()
else:
    print(f"WARNUNG: Text '{target_pseudo}' in Rohdaten nicht gefunden. Heatmap übersprungen.")

# =====================================================================
# PLOT 3: Density Plot (Verteilungskurven der Distanzen)
# =====================================================================
print("Generiere Density Plot...")
plt.figure(figsize=(10, 6))

# Wir trennen gute Kandidaten (Top 10) vom Rest
top_10_labels = agg_df.head(10)['Pseudo_Text_Sample'].tolist()

# Daten in zwei Gruppen aufteilen (Top 10 vs. Rest)
gute_treffer = raw_df[raw_df['ComparedLabel'].isin(top_10_labels)]['Distance']
schlechte_treffer = raw_df[~raw_df['ComparedLabel'].isin(top_10_labels)]['Distance']

# KDE (Kernel Density Estimate) Plots zeichnen
sns.kdeplot(gute_treffer, fill=True, color='#2ecc71', label='Top 10 Asterius-Kandidaten', alpha=0.5)
sns.kdeplot(schlechte_treffer, fill=True, color='#95a5a6', label='Restliche Pseudo-Texte', alpha=0.5)

# Baseline-Linie
plt.axvline(x=baseline_value, color='#c0392b', linestyle=':', linewidth=2, label='Oratio 8 Baseline')

plt.title('Trennschärfe des Modells: Distanz-Verteilungen', fontsize=16, pad=20)
plt.xlabel('Distanz (Siamese Network)', fontsize=14)
plt.ylabel('Dichte (Häufigkeit)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('03_Density_Plot.png', dpi=300)
plt.close()

print("Visualisierungen erfolgreich erstellt!")