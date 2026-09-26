import numpy as np
import json

DATA_DIR = "data/processed"
WEB_DIR = "web/samples"

X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

# Find one benign and one attack sample
benign_idx = np.where(y_test == 0)[0][0]
attack_idx = np.where(y_test == 1)[0][0]

# Save benign sample
with open(f"{WEB_DIR}/benign.json", "w") as f:
    json.dump({"features": X_test[benign_idx].tolist()}, f)
print(f"Saved benign sample from index {benign_idx}")

# Save attack sample
with open(f"{WEB_DIR}/attack.json", "w") as f:
    json.dump({"features": X_test[attack_idx].tolist()}, f)
print(f"Saved attack sample from index {attack_idx}")

# Also save 10 random samples as CSV for testing
rows = []
for i in range(10):
    rows.append(",".join(str(v) for v in X_test[i]))

with open(f"{WEB_DIR}/sample_batch.csv", "w") as f:
    f.write("\n".join(rows))
print(f"Saved 10 samples to sample_batch.csv")