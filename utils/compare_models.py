import json

# Load metrics
with open("results/cnn_bilstm_metrics.json") as f:
    bilstm = json.load(f)
with open("results/stacked_ensemble_metrics.json") as f:
    ensemble = json.load(f)

print("=" * 70)
print("MODEL COMPARISON REPORT")
print("=" * 70)

metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]

print(f"\n{'Metric':<15} {'CNN-BiLSTM':<15} {'Stacked Ensemble':<20} {'Winner':<15}")
print("-" * 70)

for m in metrics:
    b = bilstm.get(m, 0)
    e = ensemble.get(m, 0)
    winner = "CNN-BiLSTM" if b > e else "Stacked Ensemble"
    print(f"{m:<15} {b * 100:<15.4f} {e * 100:<20.4f} {winner:<15}")

# Latency
print(f"\n{'Latency (ms)':<15} {bilstm.get('latency_ms_per_sample', 0):<15.4f} {ensemble.get('latency_ms_per_sample', 0):<20.4f} Stacked Ensemble")

# Save report
with open("results/model_comparison.json", "w") as f:
    json.dump({
        "cnn_bilstm": bilstm,
        "stacked_ensemble": ensemble,
        "best_model": "Stacked Ensemble",
    }, f, indent=2)

print("\nComparison report saved to results/model_comparison.json")