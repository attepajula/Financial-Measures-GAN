import pandas as pd
import pickle
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load data
real_norm = pd.read_csv('data_normalized.csv')
synthetic = pd.read_csv('synthetic.csv')  # already in original scale

# Denormalize data_normalized.csv as needed
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
feat_cols = [col for col in real_norm.columns if col.startswith('X') and col in synthetic.columns]

def denorm_minus1_1(df, cols, scalers):
    out = df.copy()
    for col in cols:
        min_val = scalers[col]['min']
        max_val = scalers[col]['max']
        out[col] = ((out[col] + 1) / 2) * (max_val - min_val) + min_val
    return out

real_denorm = denorm_minus1_1(real_norm, feat_cols, scalers)

# Status contingency
real_counts  = real_denorm['status_label'].value_counts().sort_index()
synth_counts = synthetic['status_label'].value_counts().sort_index()
contingency = pd.DataFrame({
    'real': real_counts,
    'synthetic': synth_counts
}).fillna(0).astype(int)

# 1) Chi-neliö -testi
chi2, p, dof, expected = chi2_contingency(contingency)
print("Status contingency table:\n", contingency)
print(f"Chi2={chi2:.2f}, p-value={p:.2e}, dof={dof}")

# Quick min/max sanity check
print("First feature min/max (real, synth):",
      real_denorm[feat_cols[0]].min(), real_denorm[feat_cols[0]].max(),
      synthetic[feat_cols[0]].min(), synthetic[feat_cols[0]].max())

# 2) KS/stat comparison
ks_results = []
for col in feat_cols:
    stat, pval = ks_2samp(real_denorm[col], synthetic[col])
    real_mean, real_std = real_denorm[col].mean(), real_denorm[col].std()
    synth_mean, synth_std = synthetic[col].mean(), synthetic[col].std()
    ks_results.append(
        f"{col:6} | KS_stat={stat:.4f}, p_value={pval:.2e} | "
        f"Real: mean={real_mean:.4f}, std={real_std:.4f} | "
        f"Synth: mean={synth_mean:.4f}, std={synth_std:.4f}"
    )
print("\nKS Test & Stats Results:")
print('\n'.join(ks_results))

# 3) Random Forest -erottuvuus
X = pd.concat([real_denorm[feat_cols], synthetic[feat_cols]], ignore_index=True)
y = np.array([0] * len(real_denorm) + [1] * len(synthetic))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"Random Forest accuracy (Real vs Synthetic): {rf_accuracy:.4f}")

# 4) PCA varianssiselitys
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_
print(f"PCA explained variance ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")

# Plot histograms + status in one combined figure
total_plots = len(feat_cols) + 1  # +1 for status
ncols = 4
nrows = int(np.ceil(total_plots / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()

# 5) Feature-histogrammit
for i, col in enumerate(feat_cols):
    ax = axes[i]
    bins = np.linspace(
        min(real_denorm[col].min(), synthetic[col].min()),
        max(real_denorm[col].max(), synthetic[col].max()), 1000
    )
    ax.hist(real_denorm[col], bins=bins, alpha=0.6, label='Real',      color='orange', density=True)
    ax.hist(synthetic[col],   bins=bins, alpha=0.6, label='Synthetic', color='blue',   density=True)
    ax.set_title(col)
    ax.legend()

# 6) Status-prosentit
ax = axes[len(feat_cols)]
labels = contingency.index.tolist()
x = np.arange(len(labels))
width = 0.35
real_prop  = contingency['real']      / contingency['real'].sum()
synth_prop = contingency['synthetic'] / contingency['synthetic'].sum()
ax.bar(x - width/2, real_prop,  width, label='Real',      color='orange', alpha=0.6)
ax.bar(x + width/2, synth_prop, width, label='Synthetic', color='blue',   alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Proportion')
ax.set_title('Status (dead vs. alive) Proportions')
ax.legend()

# 7) Poistetaan ylimääräiset akselit
for j in range(total_plots, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('feature_hist_comparison.png')
