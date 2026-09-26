import onnxruntime as ort

MODEL_DIR = "models"

# Check Stacked Ensemble ONNX
print("=" * 60)
print("Stacked Ensemble ONNX Output Names")
print("=" * 60)

session = ort.InferenceSession(f"{MODEL_DIR}/stacked_ensemble.onnx", providers=["CPUExecutionProvider"])

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  Name: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  Name: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.type}")