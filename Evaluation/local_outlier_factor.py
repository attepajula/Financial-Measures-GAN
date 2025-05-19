import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

real_norm = pd.read_csv('normalized_test.csv')
synthetic = pd.read_csv('synthetic_test.csv')


with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# muuttujat talteen
feat_cols = [col for col in real_norm.columns if col.startswith('X') and col in synthetic.columns]

def denorm_minus1_1(df, cols, scalers):
    out = df.copy()
    for col in cols:
        min_val = scalers[col]['min']
        max_val = scalers[col]['max']
        out[col] = ((out[col] + 1) / 2) * (max_val - min_val) + min_val
    return out

# Denormalisoidaan aidon datan featuret
real_denorm = denorm_minus1_1(real_norm, feat_cols, scalers)

# Yhtä monta riviä kummastakin
n = min(len(real_denorm), len(synthetic))
real_denorm = real_denorm.sample(n, random_state=42)
synthetic = synthetic.sample(n, random_state=42)

# shuffleshuffle
real_denorm = real_denorm.sample(frac=1, random_state=42).reset_index(drop=True)

#--- Koodataan 'status_label' numeeriseksi molemmissa dataseteissä ---
le = LabelEncoder()
real_denorm['status_label'] = le.fit_transform(real_denorm['status_label'])
synthetic['status_label'] = le.transform(synthetic['status_label'])

# POISTA DESIMAALIVUOTO ❗️
real_denorm[feat_cols] = real_denorm[feat_cols].round(2)
synthetic[feat_cols] = synthetic[feat_cols].round(2)

# Valitaan mukaan myös status_label
lof_features = feat_cols + ['status_label']
lof_features = [col for col in lof_features if col in real_denorm.columns and col in synthetic.columns]

X_real = real_denorm[lof_features]
X_synth = synthetic[lof_features]

# Sovitetaan LOF aidolle datalle
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_real)

# LOF generoidulle datalle
real_lof_scores = -lof.decision_function(X_real)
synth_lof_scores = -lof.decision_function(X_synth)

print(f"Aito LOF-score (keskiarvo): {real_lof_scores.mean():.4f}")
print(f"Feikki LOF-score (keskiarvo): {synth_lof_scores.mean():.4f}")

threshold = np.percentile(real_lof_scores, 95)

n_poikkeavat = np.sum(synth_lof_scores > threshold)
print(f"Feikissä {n_poikkeavat} riviä ({n_poikkeavat/len(synthetic):.2%}), joiden LOF-score > {threshold:.2f} (eli poikkeavampia kuin useimmat aidot)")

plt.figure(figsize=(10, 6))
plt.hist(real_lof_scores, bins=500, alpha=0.8, label='Aito data', density=True)
plt.hist(synth_lof_scores, bins=500, alpha=0.6, label='Generoitu data', density=True)
plt.axvline(x=np.mean(real_lof_scores), color='blue', linestyle='--', label=r'$\mu_{\mathrm{aito}}$')
plt.axvline(x=np.mean(synth_lof_scores), color='orange', linestyle='--', label=r'$\mu_{\mathrm{synt}}$')
plt.xlabel('LOF-score (suurempi = poikkeavampi)')
plt.ylabel('Tiheys')
plt.title('LOF-scorejen jakaumat: aito vs. generoitu data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lof.png")

import seaborn as sns

real_X = real_denorm[feat_cols]
synth_X = synthetic[feat_cols]

corr_real = real_X.corr()
corr_synth = synth_X.corr()

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.05, 1, 1], wspace=0.5)

cbar_ax = fig.add_subplot(gs[0])

ax0 = fig.add_subplot(gs[1])
ax1 = fig.add_subplot(gs[2])

sns.heatmap(corr_real, ax=ax0, cmap='Blues', square=True, cbar=True, cbar_ax=cbar_ax)
ax0.set_title("Aito data – korrelaatiot")

sns.heatmap(corr_synth, ax=ax1, cmap='Blues', square=True, cbar=False)
ax1.set_title("Generoitu data – korrelaatiot")

plt.tight_layout()
plt.savefig("korrelaatio_heatmap_leftbar.png")
plt.show()
