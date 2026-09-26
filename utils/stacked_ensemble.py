import numpy as np
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

# Config
DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    print("Loading data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")
    
    # Use a subset for SVM (it scales poorly with n_samples)
    # SVM will use 30K samples; RF and XGBoost use all
    n_svm = min(30000, len(X_train))
    X_train_svm = X_train[:n_svm]
    y_train_svm = y_train[:n_svm]
    
    return X_train, y_train, X_train_svm, y_train_svm, X_test, y_test


def main():
    X_train, y_train, X_train_svm, y_train_svm, X_test, y_test = load_data()
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    # ---- Base learners ----
    print("\nDefining base learners...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
    svm = SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        random_state=42
    )
    
    # ---- Meta learner ----
    meta = LogisticRegression(max_iter=1000, random_state=42)
    
    # ---- Stacked Ensemble ----
    print("Building StackingClassifier...")
    stack = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("svm", svm),
        ],
        final_estimator=meta,
        cv=3,
        n_jobs=-1,
        passthrough=False
    )
    
    # ---- Train ----
    # Note: StackingClassifier trains all base learners internally
    # SVM would be slow on 193K samples, so we can't easily subset here
    # Instead, we use a subsample of training data for the entire stack
    
    n_train_stack = min(50000, len(X_train))
    print(f"Using {n_train_stack:,} samples for ensemble training (SVM constraint)")
    X_stack = X_train[:n_train_stack]
    y_stack = y_train[:n_train_stack]
    
    print("\nTraining Stacked Ensemble (RF + XGBoost + SVM + Logistic Regression meta)...")
    start = time.time()
    stack.fit(X_stack, y_stack)
    train_time = time.time() - start
    print(f"Training complete in {train_time:.2f}s")
    
    # ---- Evaluate on test set ----
    print("\nEvaluating on test set...")
    start = time.time()
    y_pred = stack.predict(X_test)
    y_prob = stack.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    latency_ms = (inference_time / len(y_test)) * 1000
    
    print("\n" + "=" * 60)
    print("Stacked Ensemble Test Set Results")
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
    print(f"  Training time:      {train_time:.2f}s")
    print(f"  Inference time:     {inference_time:.2f}s")
    print(f"  Latency per sample: {latency_ms:.4f} ms")
    print("=" * 60)
    
    # ---- Save model and metrics ----
    import joblib
    joblib.dump(stack, f"{MODEL_DIR}/stacked_ensemble.pkl")
    print(f"\nModel saved to {MODEL_DIR}/stacked_ensemble.pkl")
    
    results = {
        "model": "Stacked Ensemble (RF + XGBoost + SVM)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "train_time_sec": train_time,
        "latency_ms_per_sample": latency_ms,
    }
    with open(f"{RESULTS_DIR}/stacked_ensemble_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {RESULTS_DIR}/stacked_ensemble_metrics.json")


if __name__ == "__main__":
    main()