"""Training pipeline for bill passage prediction model.

Loads historical bill data from local Open States bulk downloads,
extracts features, trains a logistic regression model, and saves
weights to JSON.

Usage:
    PYTHONPATH=src python -m legislator.model.train

Data setup:
    1. Create an account at https://open.pluralpolicy.com/
    2. Download session JSON files from https://open.pluralpolicy.com/data/session-json/
       (e.g. Texas 88th Legislature, California 2023-2024, etc.)
    3. Place the downloaded JSON files (or extracted folders) in:
           src/legislator/model/data/
    4. Run this script

The script accepts any of these file layouts in the data/ directory:
    - Individual JSON files: *.json
    - Zip archives: *.zip (will be extracted automatically)
    - Directories containing bills as individual JSON files

No external ML dependencies — logistic regression is implemented in
pure Python using gradient descent.
"""

import glob
import json
import math
import os
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

from legislator.model.features import (
    FEATURE_NAMES, extract_from_openstates, label_from_openstates,
)

# Where to find training data and save model weights
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_PATH = Path(__file__).parent / "weights.json"


def _load_json_file(path: Path) -> list[dict]:
    """Load bills from a single JSON file.

    Handles both formats:
      - A list of bill dicts: [{"id": ..., "identifier": ...}, ...]
      - A single bill dict: {"id": ..., "identifier": ...}
      - An Open States bulk JSON with top-level keys
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Could be a single bill or a wrapper object
        if "identifier" in data:
            return [data]
        # Try common wrapper keys
        for key in ("bills", "results", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # If it has nested bill-like objects, try to extract them
        bills = []
        for val in data.values():
            if isinstance(val, dict) and "identifier" in val:
                bills.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "identifier" in item:
                        bills.append(item)
        if bills:
            return bills
    return []


def _extract_zip(zip_path: Path) -> list[dict]:
    """Extract bills from a zip archive."""
    bills = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                with zf.open(name) as f:
                    try:
                        data = json.loads(f.read().decode())
                        if isinstance(data, list):
                            bills.extend(data)
                        elif isinstance(data, dict) and "identifier" in data:
                            bills.append(data)
                        elif isinstance(data, dict):
                            for key in ("bills", "results", "data"):
                                if key in data and isinstance(data[key], list):
                                    bills.extend(data[key])
                                    break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
    return bills


def load_training_data() -> list[dict]:
    """Load bill data from local files in the data directory.

    Looks for JSON files and zip archives in DATA_DIR.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created data directory: {DATA_DIR}")
        print(f"Please add Open States bulk JSON files to this directory.")
        print(f"Download from: https://open.pluralpolicy.com/data/session-json/")
        return []

    all_bills = []

    # Load JSON files
    json_files = sorted(DATA_DIR.glob("*.json"))
    for path in json_files:
        print(f"Loading {path.name}...")
        bills = _load_json_file(path)
        print(f"  {len(bills)} bills")
        all_bills.extend(bills)

    # Load zip archives
    zip_files = sorted(DATA_DIR.glob("*.zip"))
    for path in zip_files:
        print(f"Extracting {path.name}...")
        bills = _extract_zip(path)
        print(f"  {len(bills)} bills")
        all_bills.extend(bills)

    # Load from subdirectories (extracted bulk downloads)
    for subdir in sorted(DATA_DIR.iterdir()):
        if subdir.is_dir():
            sub_jsons = sorted(subdir.glob("*.json"))
            if sub_jsons:
                print(f"Loading from {subdir.name}/...")
                count = 0
                for path in sub_jsons:
                    bills = _load_json_file(path)
                    count += len(bills)
                    all_bills.extend(bills)
                print(f"  {count} bills")

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
        "trained_at": date.today().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {path}")


def main():
    # 1. Load local data
    print("=" * 60)
    print("Step 1: Loading training data from local files")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}\n")

    bills = load_training_data()

    if not bills:
        print("\n" + "=" * 60)
        print("No training data found!")
        print("=" * 60)
        print()
        print("To train the model, download Open States bulk JSON files:")
        print()
        print("  1. Create a free account at https://open.pluralpolicy.com/")
        print("  2. Go to https://open.pluralpolicy.com/data/session-json/")
        print("  3. Download session files for several states, e.g.:")
        print("     - Texas 88th Legislature (2023)")
        print("     - California 2023-2024 Regular Session")
        print("     - New York 2023 Regular Session")
        print("     - Virginia 2024 Regular Session")
        print("     - Colorado 2024 Regular Session")
        print(f"  4. Place the .json or .zip files in:")
        print(f"     {DATA_DIR}")
        print(f"  5. Re-run: PYTHONPATH=src python -m legislator.model.train")
        sys.exit(1)

    print(f"\nTotal bills loaded: {len(bills)}")

    # 2. Extract features and labels
    print("\n" + "=" * 60)
    print("Step 2: Extracting features")
    print("=" * 60)
    X, y = prepare_dataset(bills)

    if not y:
        print("Error: Could not extract any labeled bills from the data.")
        print("Make sure the JSON files contain bills with 'actions' data.")
        sys.exit(1)

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    print(f"Dataset: {len(y)} bills, {n_pos} passed ({n_pos/len(y)*100:.1f}%), "
          f"{n_neg} did not pass")

    if len(y) < 100:
        print("Warning: very small dataset. Results may not be reliable.")
        print("Consider downloading more state-session files.")

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
