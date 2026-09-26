import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"NumPy: {np.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

import sklearn
print(f"Scikit-learn: {sklearn.__version__}")

import xgboost
print(f"XGBoost: {xgboost.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")

import onnx
print(f"ONNX: {onnx.__version__}")

print("\nAll imports successful.")