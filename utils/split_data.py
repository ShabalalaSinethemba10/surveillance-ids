import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Config
DATA_PATH = "data/processed/nbaiot_combined.csv"
OUTPUT_DIR = "data/processed"
CHUNK_SIZE = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Pass 1: Counting benign vs attack rows...")
    
    benign_count = 0
    attack_count = 0
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        benign_count += (chunk["label"] == 0).sum()
        attack_count += (chunk["label"] == 1).sum()
    
    print(f"Total benign: {benign_count:,}")
    print(f"Total attack: {attack_count:,}")
    
    # Use the smaller of the two — cap at the minority class size
    target = min(benign_count, attack_count)
    # Use 80% of the minority class for safety margin
    SAMPLE_PER_CLASS = int(target * 0.8)
    print(f"\nTarget samples per class: {SAMPLE_PER_CLASS:,}")
    
    print("\nPass 2: Collecting samples...")
    
    benign_chunks = []
    attack_chunks = []
    benign_collected = 0
    attack_collected = 0
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        # Downcast
        float_cols = chunk.select_dtypes(include=["float64"]).columns
        chunk[float_cols] = chunk[float_cols].astype("float32")
        
        if benign_collected < SAMPLE_PER_CLASS:
            b = chunk[chunk["label"] == 0]
            benign_chunks.append(b)
            benign_collected += len(b)
        
        if attack_collected < SAMPLE_PER_CLASS:
            a = chunk[chunk["label"] == 1]
            attack_chunks.append(a)
            attack_collected += len(a)
        
        print(f"  Benign: {benign_collected:,}/{SAMPLE_PER_CLASS:,} | "
              f"Attack: {attack_collected:,}/{SAMPLE_PER_CLASS:,}")
        
        if benign_collected >= SAMPLE_PER_CLASS and attack_collected >= SAMPLE_PER_CLASS:
            break
    
    # Combine and sample exactly
    benign_df = pd.concat(benign_chunks, ignore_index=True)
    attack_df = pd.concat(attack_chunks, ignore_index=True)
    
    benign_df = benign_df.sample(n=SAMPLE_PER_CLASS, random_state=42)
    attack_df = attack_df.sample(n=SAMPLE_PER_CLASS, random_state=42)
    
    df = pd.concat([benign_df, attack_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset: {df.shape}")
    print(df["label"].value_counts())
    
    # Feature cols
    meta_cols = ["label", "device", "attack_type"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"Feature columns: {len(feature_cols)}")
    
    # Split 70/15/15
    X = df[feature_cols]
    y = df["label"]
    meta = df[meta_cols]
    
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, meta, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.176, stratify=y_temp, random_state=42
    )
    
    train_df = pd.concat([X_train, meta_train], axis=1)
    val_df   = pd.concat([X_val,   meta_val],   axis=1)
    test_df  = pd.concat([X_test,  meta_test],  axis=1)
    
    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    print(f"\nTrain class balance:")
    print(train_df["label"].value_counts())
    print(f"\nFiles saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()