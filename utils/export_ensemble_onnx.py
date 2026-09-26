import numpy as np
import joblib
import os
import onnx
import onnxruntime as ort

# --- Register XGBoost converter BEFORE calling convert_sklearn ---
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost,
)

update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

# Config
MODEL_DIR = "models"
N_FEATURES = 115

print("Loading Stacked Ensemble...")
stack = joblib.load(f"{MODEL_DIR}/stacked_ensemble.pkl")

print("Converting to ONNX...")
initial_type = [("input", FloatTensorType([None, N_FEATURES]))]

onnx_model = convert_sklearn(
    stack,
    initial_types=initial_type,
    target_opset={"": 13, "ai.onnx.ml": 3},   # ← FIX: specify ML domain version
    options={id(stack): {"zipmap": False}},
)

# Save
onnx_path = f"{MODEL_DIR}/stacked_ensemble.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Exported to {onnx_path}")

# Verify
onnx_model_loaded = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model_loaded)
print("✅ ONNX model is valid")

# Test inference
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_input = np.random.randn(1, N_FEATURES).astype(np.float32)
outputs = session.run(None, {"input": test_input})
print(f"✅ ONNX inference works.")
print(f"   Probabilities shape: {outputs[1].shape}")
print(f"   Predictions shape:   {outputs[0].shape}")

# File size
size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"✅ Model size: {size_mb:.2f} MB")