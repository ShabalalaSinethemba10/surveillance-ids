import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Config
DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("Loading train/val/test splits...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", low_memory=False)
    val   = pd.read_csv(f"{DATA_DIR}/val.csv",   low_memory=False)
    test  = pd.read_csv(f"{DATA_DIR}/test.csv",  low_memory=False)
    
    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")
    
    # Identify feature columns
    meta_cols = ["label", "device", "attack_type"]
    feature_cols = [c for c in train.columns if c not in meta_cols]
    print(f"\nFeature columns: {len(feature_cols)}")
    
    # Extract features and labels
    X_train = train[feature_cols].values.astype("float32")
    y_train = train["label"].values.astype("int64")
    
    X_val = val[feature_cols].values.astype("float32")
    y_val = val["label"].values.astype("int64")
    
    X_test = test[feature_cols].values.astype("float32")
    y_test = test["label"].values.astype("int64")
    
    # Fit StandardScaler on TRAINING data only (avoid data leakage)
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Check results
    print(f"\nBefore scaling — mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"After scaling  — mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    # Save scaled data as .npy for fast loading during training
    np.save(f"{DATA_DIR}/X_train.npy", X_train_scaled)
    np.save(f"{DATA_DIR}/y_train.npy", y_train)
    np.save(f"{DATA_DIR}/X_val.npy",   X_val_scaled)
    np.save(f"{DATA_DIR}/y_val.npy",   y_val)
    np.save(f"{DATA_DIR}/X_test.npy",  X_test_scaled)
    np.save(f"{DATA_DIR}/y_test.npy",  y_test)
    
    # Save the scaler for later use in the browser
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    print(f"\nSaved:")
    print(f"  {DATA_DIR}/X_train.npy  -> {X_train_scaled.shape}")
    print(f"  {DATA_DIR}/y_train.npy  -> {y_train.shape}")
    print(f"  {DATA_DIR}/X_val.npy    -> {X_val_scaled.shape}")
    print(f"  {DATA_DIR}/y_val.npy    -> {y_val.shape}")
    print(f"  {DATA_DIR}/X_test.npy   -> {X_test_scaled.shape}")
    print(f"  {DATA_DIR}/y_test.npy   -> {y_test.shape}")
    print(f"  {MODEL_DIR}/scaler.pkl  -> StandardScaler")


if __name__ == "__main__":
    main()