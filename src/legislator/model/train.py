"""Training pipeline for bill passage prediction model.

Fetches historical bill data from Open States, extracts features,
trains a logistic regression model, and saves weights to JSON.

Usage:
    OPENSTATES_API_KEY=your_key PYTHONPATH=src python -m legislator.model.train

No external ML dependencies — logistic regression is implemented in pure Python
using gradient descent so the model can ship without numpy/sklearn.
"""

import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

from legislator.openstates import OpenStatesAPI
from legislator.model.features import (
    FEATURE_NAMES, extract_from_openstates, label_from_openstates,
)

# States with good data quality and solar/energy policy activity
TARGET_STATES = [
    ("Texas", "88"),        # 2023 session
    ("Texas", "87"),        # 2021 session
    ("California", "20232024"),  # 2023-2024 session
    ("California", "20212022"),  # 2021-2022 session
    ("New York", "2023-2024"),   # 2023-2024 session
    ("New York", "2021-2022"),   # 2021-2022 session
    ("North Carolina", "2023"),  # 2023 session
    ("Virginia", "2024"),        # 2024 session
    ("Virginia", "2023"),        # 2023 session
    ("Colorado", "2024"),        # 2024 session
]

# Where to save training data and model weights
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_PATH = Path(__file__).parent / "weights.json"


def fetch_training_data(api: OpenStatesAPI) -> list[dict]:
    """Fetch bills from target states/sessions via the Open States API.

    Saves raw data to disk as a cache so re-runs don't re-fetch.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / "training_bills.json"

    if cache_path.exists():
        print(f"Loading cached training data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    all_bills = []
    include = ["sponsorships", "actions"]

    for jurisdiction, session in TARGET_STATES:
        print(f"Fetching {jurisdiction} session {session}...")
        try:
            bills = api.fetch_all_bills(
                jurisdiction, session, include=include,
                max_pages=150,  # ~3000 bills max per session
            )
            print(f"  Got {len(bills)} bills")
            all_bills.extend(bills)
        except Exception as e:
            print(f"  Error fetching {jurisdiction}/{session}: {e}")
            continue

        # Rate limit courtesy
        time.sleep(1)

    print(f"\nTotal bills fetched: {len(all_bills)}")

    # Cache to disk
    with open(cache_path, "w") as f:
        json.dump(all_bills, f)
    print(f"Cached to {cache_path}")

    return all_bills


def prepare_dataset(bills: list[dict]) -> tuple[list[list[float]], list[int]]:
    """Extract features and labels from raw Open States bill data.

    Returns (X, y) where X is a list of feature vectors and y is labels.
    Skips bills with indeterminate outcomes.
    """
    X = []
    y = []

    for bill in bills:
        label = label_from_openstates(bill)
        if label is None:
            continue

        # Estimate session dates from the bill's action dates
        actions = bill.get("actions", [])
        if not actions:
            continue

        session_start = None
        session_end = None
        first_str = bill.get("first_action_date")
        if first_str:
            try:
                year = int(first_str[:4])
                session_start = date(year, 1, 1)
                session_end = date(year, 12, 31)
            except (ValueError, TypeError):
                pass

        features = extract_from_openstates(bill, session_start, session_end)
        vec = [features[name] for name in FEATURE_NAMES]
        X.append(vec)
        y.append(label)

    return X, y


# ---------------------------------------------------------------------------
# Pure-Python logistic regression
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _normalize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Z-score normalize features. Returns (X_norm, means, stds)."""
    n = len(X)
    d = len(X[0])

    means = [0.0] * d
    for row in X:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]

    stds = [0.0] * d
    for row in X:
        for j in range(d):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

    X_norm = []
    for row in X:
        X_norm.append([(row[j] - means[j]) / stds[j] for j in range(d)])

    return X_norm, means, stds


def train_logistic_regression(
    X: list[list[float]], y: list[int],
    lr: float = 0.1, epochs: int = 1000,
    l2_lambda: float = 0.01,
    class_weight: Optional[dict] = None,
) -> tuple[list[float], float]:
    """Train logistic regression via gradient descent.

    Returns (weights, bias).
    Uses L2 regularization and optional class weights for imbalanced data.
    """
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    # Compute sample weights
    if class_weight:
        sample_weights = [class_weight.get(yi, 1.0) for yi in y]
    else:
        sample_weights = [1.0] * n

    total_weight = sum(sample_weights)

    for epoch in range(epochs):
        # Gradients
        dw = [0.0] * d
        db = 0.0
        loss = 0.0

        for i in range(n):
            z = _dot(w, X[i]) + b
            pred = _sigmoid(z)
            err = (pred - y[i]) * sample_weights[i]

            for j in range(d):
                dw[j] += err * X[i][j]
            db += err

            # Log loss for monitoring
            pred_clamped = max(1e-7, min(1 - 1e-7, pred))
            loss -= sample_weights[i] * (
                y[i] * math.log(pred_clamped) +
                (1 - y[i]) * math.log(1 - pred_clamped)
            )

        # Update with L2 regularization
        for j in range(d):
            w[j] -= lr * (dw[j] / total_weight + l2_lambda * w[j])
        b -= lr * (db / total_weight)

        if epoch % 100 == 0:
            avg_loss = loss / total_weight
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.4f}")

    return w, b


def evaluate(X: list[list[float]], y: list[int],
             w: list[float], b: float) -> dict:
    """Compute accuracy, precision, recall, and AUC-like metrics."""
    tp = fp = tn = fn = 0
    for i in range(len(X)):
        z = _dot(w, X[i]) + b
        pred = 1 if _sigmoid(z) >= 0.5 else 0
        if pred == 1 and y[i] == 1:
            tp += 1
        elif pred == 1 and y[i] == 0:
            fp += 1
        elif pred == 0 and y[i] == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(X) if X else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": len(X),
        "positive_rate": round(sum(y) / len(y), 4) if y else 0,
    }


def save_weights(w: list[float], b: float,
                 means: list[float], stds: list[float],
                 metrics: dict, path: Path = WEIGHTS_PATH):
    """Save trained model weights, normalization params, and metrics to JSON."""
    model = {
        "feature_names": FEATURE_NAMES,
        "weights": w,
        "bias": b,
        "means": means,
        "stds": stds,
        "metrics": metrics,
        "trained_on": TARGET_STATES,
        "trained_at": date.today().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {path}")


def main():
    api_key = os.environ.get("OPENSTATES_API_KEY")
    if not api_key:
        print("Error: OPENSTATES_API_KEY environment variable required")
        sys.exit(1)

    api = OpenStatesAPI(api_key)

    # 1. Fetch historical data
    print("=" * 60)
    print("Step 1: Fetching historical bill data from Open States")
    print("=" * 60)
    bills = fetch_training_data(api)

    # 2. Extract features and labels
    print("\n" + "=" * 60)
    print("Step 2: Extracting features")
    print("=" * 60)
    X, y = prepare_dataset(bills)
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    print(f"Dataset: {len(y)} bills, {n_pos} passed ({n_pos/len(y)*100:.1f}%), "
          f"{n_neg} did not pass")

    if len(y) < 100:
        print("Warning: very small dataset. Results may not be reliable.")

    # 3. Normalize features
    X_norm, means, stds = _normalize(X)

    # 4. Split into train/test (80/20)
    split = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y[:split], y[split:]

    # 5. Compute class weights to handle imbalance
    pos_rate = sum(y_train) / len(y_train) if y_train else 0.5
    neg_rate = 1 - pos_rate
    # Weight the minority class higher
    class_weight = {
        0: 0.5 / neg_rate if neg_rate > 0 else 1.0,
        1: 0.5 / pos_rate if pos_rate > 0 else 1.0,
    }
    print(f"Class weights: passed={class_weight[1]:.2f}, "
          f"not_passed={class_weight[0]:.2f}")

    # 6. Train
    print("\n" + "=" * 60)
    print("Step 3: Training logistic regression")
    print("=" * 60)
    w, b = train_logistic_regression(X_train, y_train,
                                      lr=0.1, epochs=1000,
                                      l2_lambda=0.01,
                                      class_weight=class_weight)

    # 7. Evaluate
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    train_metrics = evaluate(X_train, y_train, w, b)
    test_metrics = evaluate(X_test, y_test, w, b)
    print(f"Train: {train_metrics}")
    print(f"Test:  {test_metrics}")

    # 8. Print feature importances
    print("\nFeature weights (descending by absolute value):")
    weighted = sorted(zip(FEATURE_NAMES, w), key=lambda x: abs(x[1]), reverse=True)
    for name, weight in weighted:
        direction = "+" if weight > 0 else "-"
        print(f"  {direction} {name:30s} {weight:+.4f}")

    # 9. Save
    save_weights(w, b, means, stds, test_metrics)


if __name__ == "__main__":
    main()
