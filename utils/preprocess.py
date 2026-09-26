import os
import pandas as pd
import numpy as np

# Config
DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
SAMPLE_PER_FILE = 20_000  # Cap per file to keep memory manageable

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_label(filepath, label, device_id, attack_type):
    """Load one CSV, label it, return DataFrame."""
    print(f"  Loading {os.path.basename(filepath)}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Add metadata columns
    df["label"] = label           # 0 = benign, 1 = attack
    df["device"] = device_id
    df["attack_type"] = attack_type
    
    # Subsample if too large
    if len(df) > SAMPLE_PER_FILE:
        df = df.sample(n=SAMPLE_PER_FILE, random_state=42)
    
    return df


def main():
    all_dfs = []
    
    # Loop over devices 1-9
    for device_id in range(1, 10):
        print(f"\nDevice {device_id}:")
        
        # Benign
        benign_file = f"{device_id}.benign.csv"
        benign_path = os.path.join(DATA_DIR, benign_file)
        if os.path.exists(benign_path):
            df = load_and_label(benign_path, label=0, device_id=device_id, attack_type="benign")
            all_dfs.append(df)
        
        # Gafgyt attacks
        for attack in ["combo", "junk", "scan", "tcp", "udp"]:
            f = f"{device_id}.gafgyt.{attack}.csv"
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                df = load_and_label(p, label=1, device_id=device_id, attack_type=f"gafgyt_{attack}")
                all_dfs.append(df)
        
        # Mirai attacks
        for attack in ["ack", "scan", "syn", "udp", "udpplain"]:
            f = f"{device_id}.mirai.{attack}.csv"
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                df = load_and_label(p, label=1, device_id=device_id, attack_type=f"mirai_{attack}")
                all_dfs.append(df)
    
    print("\nCombining all data...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    print(f"Total columns: {len(combined.columns)}")
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, "nbaiot_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()