"""Training pipeline for bill passage prediction model.

Loads historical bill data from local Open States bulk downloads,
extracts features, trains a logistic regression model, and saves
weights to JSON.

Usage:
    PYTHONPATH=src python -m legislator.model.train
    PYTHONPATH=src python -m legislator.model.train --max-bills 50000

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

Requires numpy for vectorized training (pip install numpy).

v2 improvements:
    - K-fold cross-validation (stratified by state)
    - Hyperparameter grid search (lr, l2, l1, class weight beta)
    - Early stopping on validation loss
    - Threshold tuning to maximize F1
    - Session dates estimated from per-session action date ranges
    - State passage rate computed and stored as a feature
    - Elastic net (L1 + L2) regularization

v2.1 performance improvements:
    - Numpy-vectorized training loop (~100x faster than pure Python)
    - --max-bills flag to sample a subset for faster iteration
    - Tighter default hyperparameter grid (24 combos vs 243)
"""

import argparse
import json
import math
import os
import random
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

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


# ---------------------------------------------------------------------------
# Session date estimation and state passage rates
# ---------------------------------------------------------------------------

def _get_bill_state(bill: dict) -> str:
    """Extract state identifier from an Open States bill."""
    # Try jurisdiction.name or jurisdiction_id
    jurisdiction = bill.get("jurisdiction") or {}
    jid = jurisdiction.get("id", "") or bill.get("jurisdiction_id", "")
    # Format: ocd-jurisdiction/country:us/state:ca/government
    if "/state:" in jid:
        return jid.split("/state:")[1].split("/")[0].upper()
    # Try from_organization
    from_org = (bill.get("from_organization") or {}).get("name", "")
    if from_org:
        return from_org[:2].upper()
    return "XX"


def _get_bill_session(bill: dict) -> str:
    """Extract session identifier from an Open States bill."""
    return bill.get("legislative_session", "") or bill.get("session", "")


def _estimate_session_dates(bills: list[dict]) -> dict:
    """Estimate session start/end dates from action date ranges.

    Groups bills by (state, session) and uses the min/max action dates
    across all bills in that session as the estimated session bounds.
    This is much better than the old Jan 1 - Dec 31 approximation.

    Returns dict mapping (state, session) -> (start_date, end_date).
    """
    from legislator.model.features import _parse_date

    session_dates = {}  # (state, session) -> [list of dates]

    for bill in bills:
        state = _get_bill_state(bill)
        session = _get_bill_session(bill)
        if not session:
            continue
        key = (state, session)

        actions = bill.get("actions", [])
        for a in actions:
            d = _parse_date(a.get("date", ""))
            if d:
                if key not in session_dates:
                    session_dates[key] = []
                session_dates[key].append(d)

        # Also check first/last action date fields
        for field in ("first_action_date", "latest_action_date"):
            d = _parse_date(bill.get(field, "") or "")
            if d:
                if key not in session_dates:
                    session_dates[key] = []
                session_dates[key].append(d)

    # Convert to (min_date, max_date) pairs
    result = {}
    for key, dates in session_dates.items():
        if dates:
            result[key] = (min(dates), max(dates))
    return result


def _compute_state_passage_rates(bills: list[dict]) -> dict:
    """Compute historical passage rate per state.

    Returns dict mapping state abbreviation -> passage rate (float 0-1).
    """
    state_total = {}
    state_passed = {}

    for bill in bills:
        state = _get_bill_state(bill)
        label = label_from_openstates(bill)
        if label is None:
            continue
        state_total[state] = state_total.get(state, 0) + 1
        if label == 1:
            state_passed[state] = state_passed.get(state, 0) + 1

    rates = {}
    for state in state_total:
        total = state_total[state]
        passed = state_passed.get(state, 0)
        rates[state] = passed / total if total > 0 else 0.0

    return rates


def prepare_dataset(bills: list[dict]) -> tuple:
    """Extract features and labels from raw Open States bill data.

    Returns (X, y, states, state_passage_rates) where X is a numpy array
    of feature vectors, y is a numpy array of labels, and states is a list
    of state abbreviations (for stratified CV).
    """
    # Pre-compute session dates and state passage rates
    print("  Estimating session dates...")
    session_dates = _estimate_session_dates(bills)
    print("  Computing state passage rates...")
    state_passage_rates = _compute_state_passage_rates(bills)

    X = []
    y = []
    states = []

    total = len(bills)
    for i, bill in enumerate(bills):
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i}/{total} bills ({len(X)} labeled so far)...")

        state = _get_bill_state(bill)
        session = _get_bill_session(bill)

        # Get estimated session dates
        session_key = (state, session)
        session_start = None
        session_end = None
        if session_key in session_dates:
            session_start, session_end = session_dates[session_key]

        # Label with session-end awareness
        label = label_from_openstates(bill, session_end_date=session_end)
        if label is None:
            continue

        actions = bill.get("actions", [])
        if not actions:
            continue

        # State passage rate for this bill's state
        spr = state_passage_rates.get(state, 0.0)

        features = extract_from_openstates(
            bill, session_start, session_end,
            state_passage_rate=spr,
        )
        vec = [features[name] for name in FEATURE_NAMES]
        X.append(vec)
        y.append(label)
        states.append(state)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), states, state_passage_rates


# ---------------------------------------------------------------------------
# Numpy-vectorized logistic regression with elastic net
# ---------------------------------------------------------------------------

def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features. Returns (X_norm, means, stds)."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero
    X_norm = (X - means) / stds
    return X_norm, means, stds


def _normalize_with_stats(X: np.ndarray, means: np.ndarray,
                          stds: np.ndarray) -> np.ndarray:
    """Normalize X using pre-computed means and stds."""
    return (X - means) / stds


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray,
    lr: float = 0.1, epochs: int = 1000,
    l2_lambda: float = 0.01,
    l1_lambda: float = 0.0,
    class_weight: Optional[dict] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    patience: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, float, int]:
    """Train logistic regression via gradient descent with elastic net.

    Returns (weights, bias, best_epoch).
    Uses L2 + L1 regularization (elastic net) and optional class weights.
    Supports early stopping on validation loss.
    All computation is numpy-vectorized.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    # Compute sample weights
    if class_weight:
        sample_weights = np.array([class_weight.get(int(yi), 1.0) for yi in y])
    else:
        sample_weights = np.ones(n)

    total_weight = sample_weights.sum()

    best_val_loss = float('inf')
    best_w = w.copy()
    best_b = b
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Vectorized forward pass
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # stable sigmoid

        # Weighted errors
        err = (pred - y) * sample_weights  # (n,)

        # Vectorized gradients
        dw = X.T @ err  # (d,)
        db = err.sum()

        # Update with L2 regularization
        w -= lr * (dw / total_weight + l2_lambda * w)
        b -= lr * (db / total_weight)

        # L1 proximal step (soft thresholding)
        if l1_lambda > 0:
            thresh = lr * l1_lambda
            w = np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)

        # Early stopping check on validation set
        if X_val is not None and y_val is not None:
            z_val = X_val @ w + b
            pred_val = 1.0 / (1.0 + np.exp(-np.clip(z_val, -500, 500)))
            pred_val = np.clip(pred_val, 1e-7, 1 - 1e-7)
            val_loss = -(
                y_val * np.log(pred_val) +
                (1 - y_val) * np.log(1 - pred_val)
            ).mean()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w = w.copy()
                best_b = b
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch: {best_epoch}, "
                          f"val_loss: {best_val_loss:.4f})")
                return best_w, best_b, best_epoch
        else:
            best_w = w.copy()
            best_b = b
            best_epoch = epoch

        if verbose and epoch % 200 == 0:
            pred_clamped = np.clip(pred, 1e-7, 1 - 1e-7)
            loss = -(sample_weights * (
                y * np.log(pred_clamped) +
                (1 - y) * np.log(1 - pred_clamped)
            )).sum()
            avg_loss = loss / total_weight
            val_str = f", val_loss={best_val_loss:.4f}" if X_val is not None else ""
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.4f}{val_str}")

    return best_w, best_b, best_epoch


def evaluate(X: np.ndarray, y: np.ndarray,
             w: np.ndarray, b: float,
             threshold: float = 0.5) -> dict:
    """Compute accuracy, precision, recall, and F1 metrics."""
    z = X @ w + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    n = len(X)
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": n,
        "positive_rate": round(float(y.sum()) / len(y), 4) if len(y) else 0,
        "threshold": threshold,
    }


def _find_best_threshold(X: np.ndarray, y: np.ndarray,
                         w: np.ndarray, b: float) -> float:
    """Search for the threshold that maximizes F1 on the given data."""
    # Pre-compute probabilities once
    z = X @ w + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    best_f1 = 0.0
    best_thresh = 0.5

    for thresh_int in range(10, 91, 5):
        thresh = thresh_int / 100.0
        preds = (probs >= thresh).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Fine-tune around the best
    for delta in range(-4, 5):
        thresh = best_thresh + delta / 100.0
        if 0.05 <= thresh <= 0.95:
            preds = (probs >= thresh).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

    return best_thresh


# ---------------------------------------------------------------------------
# K-fold cross-validation (stratified by state)
# ---------------------------------------------------------------------------

def _stratified_kfold(states: list[str], k: int = 5,
                      seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Generate k-fold train/test splits stratified by state.

    Distributes bills from each state proportionally across folds,
    ensuring each fold has a representative mix of states.
    """
    rng = random.Random(seed)

    # Group indices by state
    state_indices = {}
    for i, state in enumerate(states):
        if state not in state_indices:
            state_indices[state] = []
        state_indices[state].append(i)

    # Shuffle within each state group
    for indices in state_indices.values():
        rng.shuffle(indices)

    # Assign each state's bills round-robin to folds
    fold_indices = [[] for _ in range(k)]
    for state_idxs in state_indices.values():
        for i, idx in enumerate(state_idxs):
            fold_indices[i % k].append(idx)

    # Shuffle each fold
    for fold in fold_indices:
        rng.shuffle(fold)

    # Generate (train, test) pairs as numpy arrays for efficient indexing
    folds = []
    all_indices = set(range(len(states)))
    for i in range(k):
        test_set = set(fold_indices[i])
        train = sorted(all_indices - test_set)
        folds.append((np.array(train), np.array(fold_indices[i])))

    return folds


def cross_validate(X_norm: np.ndarray, y: np.ndarray,
                   states: list[str], lr: float, l2_lambda: float,
                   l1_lambda: float, epochs: int, beta: float,
                   k: int = 5, seed: int = 42) -> dict:
    """Run k-fold CV and return mean metrics.

    Args:
        beta: Class weight balance parameter. 0.5 = standard inverse
            frequency. Lower values reduce positive class weight.
    """
    folds = _stratified_kfold(states, k=k, seed=seed)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_test, y_test = X_norm[test_idx], y[test_idx]

        # Split training into sub-train and validation for early stopping
        rng = random.Random(seed + fold_idx)
        n_train = len(X_train)
        val_size = max(1, n_train // 10)
        val_indices = sorted(rng.sample(range(n_train), val_size))
        train_mask = np.ones(n_train, dtype=bool)
        train_mask[val_indices] = False
        X_subtrain, y_subtrain = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_train[~train_mask], y_train[~train_mask]

        # Compute class weights with beta parameter
        pos_rate = float(y_subtrain.sum()) / len(y_subtrain) if len(y_subtrain) else 0.5
        neg_rate = 1 - pos_rate
        class_weight = {
            0: beta / neg_rate if neg_rate > 0 else 1.0,
            1: (1 - beta) / pos_rate if pos_rate > 0 else 1.0,
        }

        w, b, _ = train_logistic_regression(
            X_subtrain, y_subtrain,
            lr=lr, epochs=epochs,
            l2_lambda=l2_lambda, l1_lambda=l1_lambda,
            class_weight=class_weight,
            X_val=X_val, y_val=y_val,
            patience=50, verbose=False,
        )

        # Find best threshold on validation set
        threshold = _find_best_threshold(X_val, y_val, w, b)

        # Evaluate on test fold
        metrics = evaluate(X_test, y_test, w, b, threshold=threshold)
        fold_metrics.append(metrics)

    # Aggregate
    mean_f1 = sum(m["f1"] for m in fold_metrics) / k
    std_f1 = math.sqrt(sum((m["f1"] - mean_f1) ** 2 for m in fold_metrics) / k)
    mean_precision = sum(m["precision"] for m in fold_metrics) / k
    mean_recall = sum(m["recall"] for m in fold_metrics) / k
    mean_accuracy = sum(m["accuracy"] for m in fold_metrics) / k

    return {
        "mean_f1": round(mean_f1, 4),
        "std_f1": round(std_f1, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_accuracy": round(mean_accuracy, 4),
        "fold_metrics": fold_metrics,
    }


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def hyperparameter_search(X_norm: np.ndarray, y: np.ndarray,
                          states: list[str], verbose: bool = True) -> dict:
    """Grid search over hyperparameters using cross-validation.

    Returns dict with best hyperparameters and CV metrics.

    The default grid has 24 combos (was 243 in v2) — early stopping
    makes large epoch counts unnecessary, and the grid focuses on
    the parameters that matter most (lr, regularization, class weight).
    """
    # Tighter grid: 2*3*2*1*2 = 24 combos (was 3*3*3*3*3 = 243)
    # Early stopping handles epoch count, so we fix it at 1000.
    lr_grid = [0.05, 0.1]
    l2_grid = [0.001, 0.01, 0.1]
    l1_grid = [0.0, 0.01]
    epoch_grid = [1000]
    beta_grid = [0.4, 0.5]

    best_f1 = 0.0
    best_params = {}
    best_cv = {}
    total_combos = (len(lr_grid) * len(l2_grid) * len(l1_grid)
                    * len(epoch_grid) * len(beta_grid))
    combo_num = 0

    for lr in lr_grid:
        for l2 in l2_grid:
            for l1 in l1_grid:
                for epochs in epoch_grid:
                    for beta in beta_grid:
                        combo_num += 1
                        if verbose and combo_num % 10 == 1:
                            print(f"  Combo {combo_num}/{total_combos}: "
                                  f"lr={lr}, l2={l2}, l1={l1}, "
                                  f"epochs={epochs}, beta={beta}")

                        cv = cross_validate(
                            X_norm, y, states,
                            lr=lr, l2_lambda=l2, l1_lambda=l1,
                            epochs=epochs, beta=beta, k=5,
                        )

                        if cv["mean_f1"] > best_f1:
                            best_f1 = cv["mean_f1"]
                            best_params = {
                                "lr": lr,
                                "l2_lambda": l2,
                                "l1_lambda": l1,
                                "epochs": epochs,
                                "beta": beta,
                            }
                            best_cv = cv
                            if verbose:
                                print(f"    ** New best F1: {best_f1:.4f} "
                                      f"(P={cv['mean_precision']:.4f}, "
                                      f"R={cv['mean_recall']:.4f})")

    return {
        "params": best_params,
        "cv_metrics": best_cv,
    }


def save_weights(w, b: float,
                 means, stds,
                 metrics: dict, threshold: float = 0.5,
                 state_passage_rates: Optional[dict] = None,
                 hyperparameters: Optional[dict] = None,
                 cv_metrics: Optional[dict] = None,
                 path: Path = WEIGHTS_PATH):
    """Save trained model weights, normalization params, and metrics to JSON."""
    # Convert numpy arrays to plain lists for JSON serialization
    w_list = w.tolist() if hasattr(w, 'tolist') else list(w)
    means_list = means.tolist() if hasattr(means, 'tolist') else list(means)
    stds_list = stds.tolist() if hasattr(stds, 'tolist') else list(stds)
    model = {
        "version": 2,
        "feature_names": FEATURE_NAMES,
        "weights": w_list,
        "bias": float(b),
        "means": means_list,
        "stds": stds_list,
        "threshold": threshold,
        "state_passage_rates": state_passage_rates or {},
        "metrics": metrics,
        "hyperparameters": hyperparameters or {},
        "cv_metrics": {
            "mean_f1": cv_metrics.get("mean_f1", 0) if cv_metrics else 0,
            "std_f1": cv_metrics.get("std_f1", 0) if cv_metrics else 0,
            "mean_precision": cv_metrics.get("mean_precision", 0) if cv_metrics else 0,
            "mean_recall": cv_metrics.get("mean_recall", 0) if cv_metrics else 0,
        },
        "text_hash_buckets": 50,
        "trained_at": date.today().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train bill passage prediction model")
    parser.add_argument(
        "--max-bills", type=int, default=None,
        help="Max number of raw bills to use (randomly sampled). "
             "Useful for faster iteration on large datasets.")
    args = parser.parse_args()

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

    # Sample if --max-bills specified
    if args.max_bills and len(bills) > args.max_bills:
        rng = random.Random(42)
        bills = rng.sample(bills, args.max_bills)
        print(f"Sampled {args.max_bills} bills for training")

    # 2. Extract features and labels
    print("\n" + "=" * 60)
    print("Step 2: Extracting features")
    print("=" * 60)
    X, y, states, state_passage_rates = prepare_dataset(bills)

    if len(y) == 0:
        print("Error: Could not extract any labeled bills from the data.")
        print("Make sure the JSON files contain bills with 'actions' data.")
        sys.exit(1)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"Dataset: {len(y)} bills, {n_pos} passed ({n_pos/len(y)*100:.1f}%), "
          f"{n_neg} did not pass")
    print(f"States represented: {len(set(states))}")
    print(f"Features per bill: {X.shape[1]}")

    if len(y) < 100:
        print("Warning: very small dataset. Results may not be reliable.")
        print("Consider downloading more state-session files.")

    # 3. Normalize features
    X_norm, means, stds = _normalize(X)

    # Report feature statistics to diagnose data quality
    print("\nFeature variance check:")
    zero_var_features = []
    for i, name in enumerate(FEATURE_NAMES):
        if stds[i] <= 1.0 and means[i] == 0.0:
            # Check if this is a text hash feature (expected to be sparse)
            if not name.startswith("text_hash_"):
                zero_var_features.append(name)
    if zero_var_features:
        print(f"  Warning: {len(zero_var_features)} non-text features have "
              f"zero variance: {zero_var_features}")
    else:
        print("  All structured features have non-zero variance.")

    # 4. Hyperparameter search with cross-validation
    print("\n" + "=" * 60)
    print("Step 3: Hyperparameter search (5-fold stratified CV)")
    print("=" * 60)

    search_result = hyperparameter_search(X_norm, y, states, verbose=True)
    best_params = search_result["params"]
    cv_metrics = search_result["cv_metrics"]

    print(f"\nBest hyperparameters: {best_params}")
    print(f"CV F1: {cv_metrics['mean_f1']:.4f} ± {cv_metrics['std_f1']:.4f}")
    print(f"CV Precision: {cv_metrics['mean_precision']:.4f}")
    print(f"CV Recall: {cv_metrics['mean_recall']:.4f}")

    # 5. Final training on full dataset with best hyperparameters
    print("\n" + "=" * 60)
    print("Step 4: Final training with best hyperparameters")
    print("=" * 60)

    # Split into train/test (80/20) with shuffling
    rng = random.Random(42)
    n = len(X_norm)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * 0.8)
    train_idx = np.array(indices[:split])
    test_idx = np.array(indices[split:])

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]

    # Validation set from training for early stopping
    val_size = max(1, len(X_train) // 10)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]

    # Compute class weights
    pos_rate = float(y_train_sub.sum()) / len(y_train_sub) if len(y_train_sub) else 0.5
    neg_rate = 1 - pos_rate
    beta = best_params.get("beta", 0.5)
    class_weight = {
        0: beta / neg_rate if neg_rate > 0 else 1.0,
        1: (1 - beta) / pos_rate if pos_rate > 0 else 1.0,
    }
    print(f"Class weights: passed={class_weight[1]:.2f}, "
          f"not_passed={class_weight[0]:.2f}")

    w, b, best_epoch = train_logistic_regression(
        X_train_sub, y_train_sub,
        lr=best_params.get("lr", 0.1),
        epochs=best_params.get("epochs", 1000),
        l2_lambda=best_params.get("l2_lambda", 0.01),
        l1_lambda=best_params.get("l1_lambda", 0.0),
        class_weight=class_weight,
        X_val=X_val, y_val=y_val,
        patience=50,
    )

    # 6. Find best threshold
    threshold = _find_best_threshold(X_val, y_val, w, b)
    print(f"\nOptimal threshold: {threshold:.2f}")

    # 7. Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)
    train_metrics = evaluate(X_train, y_train, w, b, threshold=threshold)
    test_metrics = evaluate(X_test, y_test, w, b, threshold=threshold)
    print(f"Train: {train_metrics}")
    print(f"Test:  {test_metrics}")

    # 8. Print feature importances
    print("\nFeature weights (descending by absolute value):")
    w_list = w.tolist() if hasattr(w, 'tolist') else list(w)
    weighted = sorted(zip(FEATURE_NAMES, w_list), key=lambda x: abs(x[1]), reverse=True)
    nonzero = 0
    for name, weight in weighted:
        if abs(weight) < 1e-6:
            continue
        nonzero += 1
        direction = "+" if weight > 0 else "-"
        print(f"  {direction} {name:30s} {weight:+.4f}")
    print(f"\n{nonzero}/{len(w)} features with non-zero weights "
          f"({len(w) - nonzero} driven to zero by L1)")

    # 9. Save
    save_weights(
        w, b, means, stds, test_metrics,
        threshold=threshold,
        state_passage_rates=state_passage_rates,
        hyperparameters=best_params,
        cv_metrics=cv_metrics,
    )


if __name__ == "__main__":
    main()
