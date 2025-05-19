import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy import stats
import argparse

def normalize(input_csv, output_csv='data_normalized.csv', scalers_path='scalers.pkl',
              method='iqr', threshold=1.5):
    df = pd.read_csv(input_csv)
    df['status'] = ((df['status_label'] == 'alive').astype(float) * 2) - 1
    feat_cols = [f'X{i}' for i in range(1, 19)]
    if method == 'iqr':
        Q1 = df[feat_cols].quantile(0.25)
        Q3 = df[feat_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        mask = df[feat_cols].ge(lower).all(axis=1) & df[feat_cols].le(upper).all(axis=1)
        df = df[mask]
    elif method == 'zscore':
        z = np.abs(stats.zscore(df[feat_cols], nan_policy='omit'))
        mask = (z < threshold).all(axis=1)
        df = df[mask]
    feats = df[feat_cols].values
    # Per-column min-max normalization to [-1, 1]
    feats_min = feats.min(axis=0)
    feats_max = feats.max(axis=0)
    feats_scaled = 2 * (feats - feats_min) / (feats_max - feats_min) - 1
    df[feat_cols] = feats_scaled
    df.to_csv(output_csv, index=False)
    print(f"✅ Normalisoitu data tallennettu: {output_csv}")
    # Save min/max per column for inverse transform
    scalers = {col: {'min': float(feats_min[i]), 'max': float(feats_max[i])} for i, col in enumerate(feat_cols)}
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"📦 Skaalaimet tallennettu: {scalers_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    default='data.csv')
    parser.add_argument('--output',   default='data_normalized.csv')
    parser.add_argument('--scalers',  default='scalers.pkl')
    parser.add_argument('--method',   default='iqr')
    parser.add_argument('--threshold',type=float, default=1.5)
    args = parser.parse_args()
    normalize(args.input, args.output, args.scalers, args.method, args.threshold)

# python normalize_data.py --input data.csv --output data_normalized.csv --scalers scalers.pkl --method iqr --threshold 1
