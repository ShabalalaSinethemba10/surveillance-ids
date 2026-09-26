import numpy as np
import torch
import onnxruntime as ort
import joblib
from cnn_bilstm import CNN_BiLSTM

# Config
DATA_DIR = "data/processed"
MODEL_DIR = "models"
N_SAMPLES = 1000

print("Loading test data...")
X_test = np.load(f"{DATA_DIR}/X_test.npy")[:N_SAMPLES]
y_test = np.load(f"{DATA_DIR}/y_test.npy")[:N_SAMPLES]
print(f"Test samples: {X_test.shape}")

# ============================================================
# 1. CNN-BiLSTM: PyTorch vs ONNX
# ============================================================
print("\n" + "=" * 60)
print("CNN-BiLSTM: PyTorch vs ONNX")
print("=" * 60)

# PyTorch
print("Running PyTorch inference...")
model = CNN_BiLSTM()
model.load_state_dict(torch.load(f"{MODEL_DIR}/cnn_bilstm_best.pth", map_location="cpu"))
model.eval()
with torch.no_grad():
    pt_out = model(torch.tensor(X_test, dtype=torch.float32))
    pt_preds = pt_out.argmax(dim=1).numpy()

# ONNX
print("Running ONNX inference...")
session = ort.InferenceSession(f"{MODEL_DIR}/cnn_bilstm.onnx", providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"input": X_test.astype(np.float32)})
onnx_preds = onnx_out[0].argmax(axis=1)

print(f"\nPyTorch accuracy:      {(pt_preds == y_test).mean() * 100:.4f}%")
print(f"ONNX accuracy:         {(onnx_preds == y_test).mean() * 100:.4f}%")
print(f"Prediction match:      {(pt_preds == onnx_preds).mean() * 100:.4f}%")

# ============================================================
# 2. Stacked Ensemble: sklearn vs ONNX
# ============================================================
print("\n" + "=" * 60)
print("Stacked Ensemble: sklearn vs ONNX")
print("=" * 60)

# sklearn
print("Running sklearn inference...")
stack = joblib.load(f"{MODEL_DIR}/stacked_ensemble.pkl")
sk_preds = stack.predict(X_test)

# ONNX
print("Running ONNX inference...")
session = ort.InferenceSession(f"{MODEL_DIR}/stacked_ensemble.onnx", providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"input": X_test.astype(np.float32)})
onnx_preds_ens = onnx_out[0].astype(sk_preds.dtype)

print(f"\nsklearn accuracy:      {(sk_preds == y_test).mean() * 100:.4f}%")
print(f"ONNX accuracy:         {(onnx_preds_ens == y_test).mean() * 100:.4f}%")
print(f"Prediction match:      {(sk_preds == onnx_preds_ens).mean() * 100:.4f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

cnn_match = (pt_preds == onnx_preds).mean()
ens_match = (sk_preds == onnx_preds_ens).mean()

if cnn_match > 0.999:
    print("✅ CNN-BiLSTM ONNX matches PyTorch")
else:
    print(f"⚠️ CNN-BiLSTM mismatch: {cnn_match * 100:.2f}%")

if ens_match > 0.999:
    print("✅ Stacked Ensemble ONNX matches sklearn")
else:
    print(f"⚠️ Stacked Ensemble mismatch: {ens_match * 100:.2f}%")

if cnn_match > 0.999 and ens_match > 0.999:
    print("\n🎉 Both models verified — safe to deploy in browser.")