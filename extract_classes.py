import pandas as pd

# 1. Daten einlesen
input_file = "asterius_aggregated_pairing_percentage.csv"
print(f"Lese Daten aus '{input_file}' ein...\n")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Fehler: Die Datei '{input_file}' wurde nicht gefunden.")
    exit()

# 2. Definition der Evidenzklassen (Tiers)
# Klasse 1: Hohe Konfidenz (>= 90%)
class_1 = df[df['Pairing_Percentage'] >= 90].copy()

# Klasse 2: Interpolationsverdacht / Imitation (>= 70% und < 90%)
class_2 = df[(df['Pairing_Percentage'] >= 70) & (df['Pairing_Percentage'] < 90)].copy()

# Klasse 3: Ausschluss (< 70%)
class_3 = df[df['Pairing_Percentage'] < 70].copy()

# 3. Statistiken in der Konsole ausgeben
print("--- ERGEBNISSE DER KLASSIFIZIERUNG ---")
print(f"Gesamtzahl der analysierten Texte: {len(df)}")
print(f"Klasse 1 (Hohe Konfidenz, >= 90%): {len(class_1)} Texte")
print(f"Klasse 2 (Interpolationsverdacht, 70-89%): {len(class_2)} Texte")
print(f"Klasse 3 (Ausschluss, < 70%): {len(class_3)} Texte\n")

print("Top 5 Texte der Klasse 1:")
print(class_1.head(5).to_string(index=False))
print("-" * 40)

# 4. Export der klassifizierten Daten in neue CSV-Dateien
file_class_1 = "asterius_class1_hohe_konfidenz.csv"
file_class_2 = "asterius_class2_ähnlichkeiten.csv"
file_class_3 = "asterius_class3_ausschluss.csv"

class_1.to_csv(file_class_1, index=False)
class_2.to_csv(file_class_2, index=False)
class_3.to_csv(file_class_3, index=False)

print(f"\nExport erfolgreich abgeschlossen.")
print(f"- {file_class_1} (Nutzen Sie diese Datei für die weitere Textkritik)")
print(f"- {file_class_2}")
print(f"- {file_class_3}")