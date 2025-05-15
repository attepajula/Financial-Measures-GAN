import argparse
import pandas as pd
import pickle
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Komentoriviparametrit
# -------------------------------
parser = argparse.ArgumentParser(description="Random Forest analyysi aidon ja generoidun datan erottamiseen.")
parser.add_argument('--drop', type=str, default='', help='Pilkuilla eroteltu lista muuttujista, jotka halutaan jättää pois, esim: X3,X7,X12')
args = parser.parse_args()

drop_cols = [col.strip() for col in args.drop.split(',')] if args.drop else []

print(f"Poistetaan seuraavat muuttujat analyysistä: {drop_cols}")

# -------------------------------
# 2. Ladataan data
# -------------------------------
real_norm = pd.read_csv('data_normalized.csv')
synthetic = pd.read_csv('synthetic.csv')  # Oletetaan, että jo denormalisoitu

# Denormalisointiskaalat
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Poimitaan X-muuttujat
feat_cols = [col for col in real_norm.columns if col.startswith('X') and col in synthetic.columns]

def denorm_minus1_1(df, cols, scalers):
    out = df.copy()
    for col in cols:
        min_val = scalers[col]['min']
        max_val = scalers[col]['max']
        out[col] = ((out[col] + 1) / 2) * (max_val - min_val) + min_val
    return out

real_denorm = denorm_minus1_1(real_norm, feat_cols, scalers)

# Otetaan sama määrä rivejä molemmista
n = min(len(real_denorm), len(synthetic))
real_denorm = real_denorm.sample(n, random_state=42)
synthetic = synthetic.sample(n, random_state=42)

# SEKOITA AITO!!
# Shufflaa aidon datan rivit ennen käyttöä
real_denorm = real_denorm.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------
# 2.5 Koodataan status_label numeeriseksi
# -------------------------------
le = LabelEncoder()
real_denorm['status_label'] = le.fit_transform(real_denorm['status_label'])
synthetic['status_label'] = le.transform(synthetic['status_label'])

# Lisää status_label mukaan featureihin
feat_cols = feat_cols + ['status_label']

# -------------------------------
# 3. Valitaan yhteiset featuret ja sovelletaan pudotuksia
# -------------------------------
common_cols = [col for col in feat_cols if col in synthetic.columns and col not in drop_cols]

# Tarkistus
print(f"Käytetään seuraavia muuttujia: {common_cols}")

# -------------------------------
# 4. Rakennetaan analyysidata
# -------------------------------
real_features = real_denorm[common_cols].copy()
synthetic_features = synthetic[common_cols].copy()

# Pyöristetään kaikki analyysiin otettavat numeeriset muuttujat
real_features[common_cols] = real_features[common_cols].round(0)
synthetic_features[common_cols] = synthetic_features[common_cols].round(0)

real_features.to_csv("real_denorm.csv", index=False)
synthetic_features.to_csv("synthetic_test.csv", index=False)

real_features['label'] = 0
synthetic_features['label'] = 1

combined = pd.concat([real_features, synthetic_features], ignore_index=True)
X = combined.drop(columns=['label'])
y = combined['label']

# -------------------------------
# 5. Treenataan Random Forest
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# 6. Tulokset ja visualisointi
# -------------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTarkkuus (real vs synthetic): {acc:.2f}")

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTärkeimmät muuttujat:")
print(importances.head(10))

#plt.figure(figsize=(10, 6))
#importances.head(20).plot(kind='barh')
#plt.gca().invert_yaxis()
#plt.title('Tärkeimmät muuttujat (feature importance)')
#plt.xlabel('Merkittävyys')
#plt.tight_layout()
#plt.savefig('feature_importance.png')
#print("Tallennettu kuva: feature_importance.png")

#python feature_importance.py --drop X11,X5

