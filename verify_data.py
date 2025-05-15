import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lataa data
df = pd.read_csv("data_normalized.csv")

# Valitse ominaisuudet
features = ['status'] + [f'X{i}' for i in range(1, 19)]

# Aseta kuva
fig, axes = plt.subplots(4, 5, figsize=(20, 12))
axes = axes.flatten()

# Piirrä histogrammit
for i, feat in enumerate(features):
    ax = axes[i]
    vals = df[feat].dropna()
    bins = np.linspace(vals.min(), vals.max(), 100)
    ax.hist(vals, bins=bins, alpha=0.7, color='orange', density=True)
    ax.set_title(feat)
    ax.tick_params(axis='x', rotation=45)

# Poista tyhjä viimeinen ruutu (20. subplot)
fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig("real_data_overview.png")
plt.close()
print("✅ Kuva tallennettu → real_data_overview.png")
