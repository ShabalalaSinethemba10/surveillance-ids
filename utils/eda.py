import pandas as pd
import matplotlib.pyplot as plt
import os

# Config
DATA_PATH = "data/processed/nbaiot_combined.csv"
OUTPUT_DIR = "results"
SAMPLE_SIZE = 100_000   # Load only 100K rows for EDA

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"Loading a sample of {SAMPLE_SIZE:,} rows...")
    
    # Read only the first N rows — fast and memory-safe
    df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE, low_memory=False)
    
    # Downcast float64 -> float32 to save memory
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    
    print(f"Sample shape: {df.shape}")
    
    # Check missing values
    print("\n--- Missing Values ---")
    print(f"Total missing: {df.isnull().sum().sum()}")
    
    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    # Class balance
    print("\n--- Class Balance ---")
    print(df["label"].value_counts())
    print(f"Benign %: {(df['label'] == 0).mean() * 100:.2f}%")
    print(f"Attack %: {(df['label'] == 1).mean() * 100:.2f}%")
    
    # Attack type distribution
    print("\n--- Attack Type Distribution ---")
    print(df["attack_type"].value_counts())
    
    # Device distribution
    print("\n--- Device Distribution ---")
    print(df["device"].value_counts().sort_index())
    
    # Plot class balance
    plt.figure(figsize=(8, 5))
    df["label"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title("Class Balance (0 = Benign, 1 = Attack)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=150)
    plt.close()
    
    # Plot attack type distribution
    plt.figure(figsize=(12, 5))
    df["attack_type"].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Attack Type Distribution (Sample)")
    plt.xlabel("Attack Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attack_distribution.png", dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()