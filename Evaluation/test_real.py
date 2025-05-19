import pandas as pd
import numpy as np
import pickle
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

real_norm = pd.read_csv("normalized_test.csv")

with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

feat_cols = [col for col in real_norm.columns if col.startswith('X')]

def denorm_minus1_1(df, cols, scalers):
    out = df.copy()
    for col in cols:
        min_val = scalers[col]['min']
        max_val = scalers[col]['max']
        out[col] = ((out[col] + 1) / 2) * (max_val - min_val) + min_val
    return out

real_denorm = denorm_minus1_1(real_norm, feat_cols, scalers)

le = LabelEncoder()
real_denorm['status_label'] = le.fit_transform(real_denorm['status_label'])

all_feats = feat_cols + ['status_label']

real_denorm[all_feats] = real_denorm[all_feats].round(2)

one_real_df, another_real_df = train_test_split(real_denorm, test_size=0.5, random_state=42)

# 1. KS-testi
ks_lines = []
for col in all_feats:
    stat, p = ks_2samp(one_real_df[col], another_real_df[col])
    ks_lines.append(f"{col:15} | KS_stat={stat:.4f}, p_value={p:.4e}")

# 2. Random Forest -erottuvuus
X = pd.concat([one_real_df[all_feats], another_real_df[all_feats]], ignore_index=True)
y = np.array([0] * len(one_real_df) + [1] * len(another_real_df))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, clf.predict(X_test))

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_
expl_line = f"PCA explained variance ratio: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}"

# 4. Tallennus tiedostoon
with open("real_eval_summary.txt", "w") as f:
    f.write("🧪 CONTROL TEST: Real-vs-Real (split)\n")
    f.write(f"Random Forest accuracy (Real vs Real): {rf_accuracy:.4f}\n")
    f.write(expl_line + "\n")
    f.write("\nKS Test Results:\n")
    for line in ks_lines:
        f.write(line + "\n")

print("✅ Tallennettu: real_eval_summary.txt")
