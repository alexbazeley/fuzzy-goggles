# Model Assessment: Bill Passage Prediction

## Overview

This document provides a technical assessment of the logistic regression model used to predict the likelihood of state-level bill passage. The model is trained on historical legislative data from the Open States bulk data archive and deployed as a scoring component within the Legislation Tracker application.

## Training Data

**Source:** Open States bulk session JSON files, downloaded from [open.pluralpolicy.com](https://open.pluralpolicy.com/data/session-json/).

**Dataset size:** Approximately 430,000 bills from completed state legislative sessions across multiple states and years.

**Label definition:**
- **Positive (1):** Bill received executive signature or became law (`executive-signature` or `became-law` action classification).
- **Negative (0):** Bill was vetoed, or reached session end without a signature.
- **Excluded:** Bills with indeterminate outcomes (no actions data) are dropped during preprocessing.

**Class imbalance:** The vast majority of introduced state legislation does not become law. Empirically, passage rates across US state legislatures range from roughly 15% to 30% depending on the state and session. The training pipeline applies inverse-frequency class weighting (`0.5 / class_rate`) to counteract this imbalance during gradient descent, ensuring the model does not simply learn to predict the majority class.

**Train/test split:** 80/20 sequential split. No stratification is applied, and no shuffling is performed prior to the split. This means the test set likely contains bills from different sessions or states than the training set, depending on the file loading order. This is a limitation — temporal or geographic leakage in either direction is possible depending on how the data directory is structured.

## Model Architecture

**Algorithm:** Binary logistic regression (single-layer linear model with sigmoid activation).

**Implementation:** Pure Python — no external ML libraries (NumPy, scikit-learn, etc.). Gradient descent is implemented from scratch using standard batch gradient descent with L2 regularization.

**Hyperparameters (as configured in `train.py`):**
| Parameter | Value |
|-----------|-------|
| Learning rate | 0.1 |
| Epochs | 1000 |
| L2 regularization (lambda) | 0.01 |
| Optimizer | Batch gradient descent |
| Feature normalization | Z-score (mean/std computed on training set) |

**Decision threshold:** 0.5 (fixed; not tuned).

## Feature Set

The model uses 18 numeric features, extracted identically during training (from Open States data) and inference (from LegiScan-tracked bills):

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `sponsor_count` | Continuous | Total number of sponsors |
| 2 | `primary_sponsor_count` | Continuous | Number of primary/lead sponsors |
| 3 | `cosponsor_count` | Continuous | Number of co-sponsors |
| 4 | `is_bipartisan` | Binary | Whether sponsors represent 2+ parties |
| 5 | `minority_party_sponsors` | Continuous | Count of sponsors from the minority party |
| 6 | `senate_origin` | Binary | Whether the bill originated in the Senate |
| 7 | `is_resolution` | Binary | Whether the bill is a resolution (SJR, HJR, etc.) |
| 8 | `num_subjects` | Continuous | Number of subject/topic tags |
| 9 | `focused_scope` | Binary | Whether the bill has 1-2 subject tags |
| 10 | `amends_existing_law` | Binary | Whether the title/description contains amendment keywords |
| 11 | `committee_referral` | Binary | Whether the bill was referred to committee |
| 12 | `committee_passage` | Binary | Whether the bill cleared committee |
| 13 | `passed_one_chamber` | Binary | Whether the bill passed at least one chamber |
| 14 | `num_actions` | Continuous | Total number of legislative actions recorded |
| 15 | `days_since_introduction` | Continuous | Days between first and last recorded action |
| 16 | `session_pct_at_intro` | Continuous | When in the session the bill was introduced (0-100%) |
| 17 | `has_companion` | Binary | Whether a companion bill exists in the other chamber |
| 18 | `action_density_30d` | Continuous | Number of actions in the last 30 days of the bill's history |

### Feature Engineering Observations

**Strengths:**
- The feature set covers the major predictive dimensions of legislative success: procedural progress, coalition breadth, partisan composition, timing, and bill structure.
- Procedural features (`committee_passage`, `passed_one_chamber`) are likely the strongest predictors, as they directly encode prior legislative success.
- Including `action_density_30d` captures momentum, which is a meaningful signal for active versus stalled legislation.

**Limitations:**
- **Collinearity:** Several features are structurally correlated. `sponsor_count` is approximately `primary_sponsor_count + cosponsor_count`. `is_bipartisan` and `minority_party_sponsors` are causally linked. `committee_referral` and `committee_passage` are sequentially dependent. L2 regularization partially mitigates this, but a more parsimonious feature set may perform comparably.
- **Leakage risk in `passed_one_chamber`:** This feature directly encodes a late-stage procedural outcome. For bills already past this point, it is a near-perfect predictor. For bills early in their lifecycle (the most useful prediction target), it contributes nothing. The model may over-weight this feature at the expense of signals that are informative earlier in the process.
- **`has_companion` is always 0 at inference:** The `extract_from_tracked_bill()` function hard-codes `has_companion = 0.0` because companion data is not available from the LegiScan API as consumed. This means the trained weight for this feature is never activated during live prediction — it is dead weight.
- **No state fixed effects:** Legislative passage rates vary substantially by state (e.g., New Hampshire introduces thousands of bills per session with a low passage rate; others introduce fewer). The model has no explicit state indicator, meaning it produces a national average prediction that may be poorly calibrated for specific states.
- **No legislator-level features:** Sponsor identity (e.g., committee chair, majority leader) is a strong predictor of passage in political science literature but is not captured here — only counts and party labels are used.
- **Keyword-based `amends_existing_law`:** This feature relies on string matching against the bill title/description ("amend", "relating to", "revise", "modify"). The phrase "relating to" is extremely common in bill titles by convention and may introduce noise.

## Training Process

1. Bills are loaded from local JSON/ZIP files in `src/legislator/model/data/`.
2. Features and labels are extracted; bills without actions are skipped.
3. Features are Z-score normalized using training set statistics (mean, std per feature).
4. An 80/20 train/test split is performed (no shuffle).
5. Logistic regression is trained via batch gradient descent for 1000 epochs with L2 regularization and class weighting.
6. Weights, bias, normalization parameters, and test set metrics are saved to `weights.json`.

### Computational Considerations

With 430,000 training bills and 18 features, each epoch performs approximately 430,000 forward passes and gradient accumulations in pure Python. At 1000 epochs, this is ~430 million sample-level computations. This is computationally expensive for a pure Python implementation — a NumPy vectorized version would be orders of magnitude faster. However, logistic regression on this feature set converges well before 1000 epochs; loss monitoring at epoch intervals (every 100) suggests diminishing returns past epoch 200-300 for a dataset of this size.

## Evaluation Methodology

The model is evaluated on the held-out 20% test split using:
- **Accuracy:** Overall correct classification rate.
- **Precision:** Of bills predicted to pass, what fraction actually passed.
- **Recall:** Of bills that actually passed, what fraction were predicted to pass.
- **F1 score:** Harmonic mean of precision and recall.
- **Confusion matrix:** True positives, false positives, true negatives, false negatives.
- **Positive rate:** Baseline passage rate in the test set.

### Evaluation Limitations

- **No cross-validation:** A single 80/20 split provides a point estimate of performance. K-fold cross-validation would yield more robust estimates, particularly for assessing variance across different data subsets.
- **No AUC-ROC or calibration analysis:** The model outputs probabilities, but only threshold-based metrics (at 0.5) are reported. AUC-ROC would measure discriminative ability across all thresholds. Calibration plots would reveal whether predicted probabilities match observed frequencies — this is particularly important because the model's outputs are displayed to users as percentage likelihoods.
- **No temporal validation:** Because bills are not split by session date, the test set may contain bills from the same sessions as training data. A true temporal holdout (train on sessions from years X, test on sessions from year Y) would better simulate real-world deployment conditions.
- **No per-state evaluation:** Aggregate metrics may mask poor performance on individual states. A state-level breakdown would reveal whether the model is systematically biased for or against certain legislatures.

## Deployment Integration

At inference time, the model:
1. Extracts the same 18 features from a `TrackedBill` instance (using LegiScan data rather than Open States data).
2. Normalizes features using the stored training-set mean/std.
3. Computes a linear combination plus bias, applies sigmoid, and outputs a 0-100 score.
4. Reports per-feature contributions (weight * normalized value) to support interpretability.

### Train-Serve Skew Considerations

The model is trained on Open States data but applied to LegiScan-sourced data. While both sources describe the same underlying legislation, there are structural differences:
- **Sponsor data format:** Open States uses `sponsorships[].person.party`; LegiScan uses `sponsors[].party_id` with numeric codes. The extraction functions handle these differently, introducing potential mismatches in `is_bipartisan` and `minority_party_sponsors`.
- **Action classifications:** Open States uses standardized action classification strings (`committee-passage`, `referral-committee`). LegiScan uses numeric progress event codes (9, 10, 11). The mapping between them is approximate.
- **Session timing:** Training uses estimated session dates (`year/1/1` to `year/12/31`); inference does the same. This is a coarse approximation — actual session calendars vary significantly.
- **`has_companion`:** Trained on Open States data where companion bills are sometimes recorded; always 0 in LegiScan-based inference.

These discrepancies represent a form of distribution shift between training and serving that could degrade real-world performance relative to test-set metrics.

## Strengths

1. **Simplicity and transparency:** Logistic regression is fully interpretable. Each feature's contribution to the prediction is a simple product of weight and normalized value, making it straightforward to explain individual predictions.
2. **No external dependencies:** The pure-Python implementation eliminates version conflicts and deployment complexity. The model is a single JSON file of weights.
3. **Large training set:** 430,000 bills provides substantial statistical power for a model with only 18 parameters. Overfitting risk is minimal.
4. **Class weighting:** Properly addresses the inherent class imbalance in legislative data.
5. **Graceful fallback:** If the trained model is unavailable, the system falls back to a hand-tuned heuristic, ensuring the feature degrades gracefully.

## Limitations and Risks

1. **Linear decision boundary:** Logistic regression assumes a linear relationship between features and log-odds of passage. Interaction effects (e.g., bipartisan support mattering more late in session) are not captured.
2. **No regularization tuning:** The L2 parameter (0.01) is hard-coded, not selected via cross-validation.
3. **Fixed threshold:** The 0.5 decision threshold is standard but may not be optimal given the class imbalance and the application's use case (where false negatives may be more costly than false positives).
4. **Calibration uncertainty:** The sigmoid outputs are treated as probabilities and displayed directly to users as "X% likely to pass." Without calibration verification, these may be systematically over- or under-confident.
5. **No model versioning or retraining pipeline:** Weights are overwritten in place with no version history. There is no automated retraining schedule or data freshness monitoring.
6. **Single global model:** One model covers all 50 states despite significant interstate variation in legislative process, volume, and passage rates.

## Recommendations for Future Work

1. **Calibration analysis:** Plot reliability diagrams to verify that predicted probabilities match observed passage rates. Apply Platt scaling or isotonic regression if needed.
2. **Temporal validation:** Split data by legislative session date to evaluate true out-of-sample performance.
3. **State-level evaluation:** Report per-state metrics to identify systematic biases.
4. **Feature selection:** Remove or consolidate redundant features (e.g., drop `sponsor_count` in favor of `primary_sponsor_count` + `cosponsor_count`; drop `has_companion` since it's dead at inference).
5. **Interaction features:** Add products of key features (e.g., `is_bipartisan * session_pct_at_intro`) to capture nonlinear effects without moving to a more complex model.
6. **Threshold optimization:** Tune the decision threshold based on a precision-recall curve, optimizing for the application's utility function.
7. **Per-state or hierarchical modeling:** Train state-specific intercepts or use a two-stage model to account for interstate variation.
8. **Vectorized implementation:** Replace pure-Python gradient descent with NumPy to reduce training time by 100x+, enabling faster iteration and cross-validation.

## Summary

The model is a reasonable first-pass approach to bill passage prediction. Its primary strength is simplicity — it is fully interpretable, easy to deploy, and trained on a large dataset. Its primary weaknesses are the linear assumption, the gap between training and serving data sources, and the absence of calibration verification. For a tool intended to surface directional signals ("this bill is more likely to pass than that one"), it is adequate. For precise probability estimation, further validation and calibration work are warranted.
