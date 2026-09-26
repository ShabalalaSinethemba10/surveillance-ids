import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
import time
from cnn_bilstm import CNN_BiLSTM

# Config
DATA_DIR = "data/processed"
MODEL_DIR = "models"
DEVICE = torch.device("cpu")
BATCH_SIZE = 512


def main():
    print("Loading test data...")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    print(f"Test shape: {X_test.shape}")

    # DataLoader
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load best model
    print("Loading best CNN-BiLSTM model...")
    model = CNN_BiLSTM().to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/cnn_bilstm_best.pth", map_location=DEVICE))
    model.eval()

    # Predict
    print("Running inference...")
    all_preds = []
    all_probs = []
    all_labels = []

    start = time.time()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    elapsed = time.time() - start

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Latency
    latency_per_sample_ms = (elapsed / len(y_true)) * 1000
    throughput_per_sec = len(y_true) / elapsed

    print("\n" + "=" * 60)
    print("CNN-BiLSTM Test Set Results")
    print("=" * 60)
    print(f"Accuracy:      {acc * 100:.4f}%")
    print(f"Precision:     {prec * 100:.4f}%")
    print(f"Recall:        {rec * 100:.4f}%")
    print(f"F1-Score:      {f1 * 100:.4f}%")
    print(f"AUC-ROC:       {auc:.6f}")
    print(f"MCC:           {mcc:.6f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print(f"\nEfficiency:")
    print(f"  Total time:         {elapsed:.2f}s for {len(y_true):,} samples")
    print(f"  Latency per sample: {latency_per_sample_ms:.4f} ms")
    print(f"  Throughput:         {throughput_per_sec:,.0f} samples/sec")
    print("=" * 60)

    # Save results
    results = {
        "model": "CNN-BiLSTM",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "latency_ms_per_sample": latency_per_sample_ms,
        "throughput_samples_per_sec": throughput_per_sec,
    }
    import json
    with open("results/cnn_bilstm_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nMetrics saved to results/cnn_bilstm_metrics.json")


if __name__ == "__main__":
    main()