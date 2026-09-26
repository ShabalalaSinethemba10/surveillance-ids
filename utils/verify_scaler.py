import numpy as np
import pandas as pd
import joblib

# Load scaled data
X_train = np.load("data/processed/X_train.npy")
X_test  = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")

print(f"X_train shape: {X_train.shape}")
print(f"X_train dtype: {X_train.dtype}")

# Check for NaN or inf
print(f"\nNaNs in X_train:  {np.isnan(X_train).sum()}")
print(f"Infs in X_train:  {np.isinf(X_train).sum()}")

# Check overall stats
print(f"\nOverall mean: {np.nanmean(X_train):.6f}")
print(f"Overall std:  {np.nanstd(X_train):.6f}")
print(f"Min:  {np.nanmin(X_train):.4f}")
print(f"Max:  {np.nanmax(X_train):.4f}")

# Per-feature stats (first 5 features)
print(f"\nFirst 5 features (mean, std):")
for i in range(5):
    print(f"  Feature {i}: mean={X_train[:, i].mean():.4f}, std={X_train[:, i].std():.4f}")

# Check labels
print(f"\ny_train unique: {np.unique(y_train)}")
print(f"y_train counts: {np.bincount(y_train)}")

print("\n✅ Data is ready for training.")