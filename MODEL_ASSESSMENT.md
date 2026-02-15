# Model Assessment: Bill Passage Prediction (v2)

## Overview

Technical assessment of the logistic regression model used to predict state-level bill passage likelihood. The model is trained on historical Open States data and scores bills tracked via the LegiScan API.

**Model version:** v2 (73 features, elastic net, cross-validated, threshold-tuned)

## Training Data

**Source:** Open States bulk data — either session JSON files from [open.pluralpolicy.com](https://open.pluralpolicy.com/data/session-json/) or monthly PostgreSQL dumps from [data.openstates.org](https://data.openstates.org/postgres/monthly/).

**PostgreSQL advantage:** The PostgreSQL dump includes `person.primary_party` on sponsorships, enabling sponsor party features (bipartisan, majority alignment, etc.) that are often missing from JSON exports.

**Dataset size:** ~430,000 bills from completed state legislative sessions across multiple states and years.

**Label definition:**
- **1 (passed):** `executive-signature`, `became-law`, or veto override (`executive-veto` + `veto-override`). Resolutions with `passage` action also count.
- **0 (did not pass):** Vetoed without override, or session ended without passage.
- **Excluded:** Indeterminate bills (session still active, no actions data).

**Class imbalance:** State passage rates range from ~15% to ~30%. The training pipeline applies weighted gradient descent (configurable beta parameter, default grid: 0.4, 0.5) to avoid majority-class bias.

## Model Architecture

**Algorithm:** Binary logistic regression with elastic net (L1 + L2) regularization.

**Implementation:** Numpy-vectorized gradient descent (~100x faster than the v1 pure-Python loops). No scikit-learn dependency.

**Key changes from v1:**
- Elastic net replaces L2-only regularization — L1 drives irrelevant features to exactly zero
- 5-fold cross-validation stratified by state replaces single 80/20 split
- Hyperparameter grid search (24 combinations) replaces hard-coded values
- Early stopping (patience=50 on validation loss) prevents overfitting
- Threshold tuning maximizes F1 instead of using a fixed 0.5 cutoff
- Session dates estimated from per-session action date ranges (not Jan 1–Dec 31)
- Per-state passage rates computed and stored in the model

**Hyperparameter grid (24 combos):**

| Parameter | Values searched |
|-----------|----------------|
| Learning rate | 0.05, 0.1 |
| L2 regularization | 0.001, 0.01, 0.1 |
| L1 regularization | 0.0, 0.01 |
| Epochs | 1000 (with early stopping) |
| Class weight beta | 0.4, 0.5 |

## Feature Set (v2)

73 features total: 23 structured + 50 text hash.

### Critical changes from v1 (18 features)

| Change | Rationale |
|--------|-----------|
| **Removed** `passed_one_chamber` | Feature leakage — this directly encodes the outcome being predicted. A bill that passed one chamber is tautologically far along. The model was over-weighting this at the expense of earlier, more useful signals. |
| **Removed** `has_companion` | Always 0 at inference (LegiScan doesn't expose companion data). Dead weight. |
| **Log-transformed** `num_actions` | Raw action count conflates activity with outcome (bills that pass accumulate more actions by definition). Log transform reduces this conflation. |
| **Added** `sponsor_party_majority` | Fraction of sponsors in the majority party. Strongest predictor per political science literature. |
| **Added** `early_action_count` | Actions in first 30 days — captures momentum without outcome leakage. |
| **Added** `title_length`, `has_fiscal_note` | Title complexity and fiscal impact signal. |
| **Added** `solar_category_count`, `has_solar_keywords` | Domain-specific features using existing solar.py keyword dictionaries. |
| **Added** `state_passage_rate` | Historical passage rate per state, addressing the "no state fixed effects" limitation. |
| **Added** 50 text hash features | Bill titles and descriptions tokenized with legislative stopword removal, mapped to 50 hash buckets via signed feature hashing (djb2, deterministic). |

### Full feature list

| # | Category | Feature | Type | Description |
|---|----------|---------|------|-------------|
| 1 | Sponsor | `sponsor_count` | Continuous | Total number of sponsors |
| 2 | Sponsor | `primary_sponsor_count` | Continuous | Number of primary/lead sponsors |
| 3 | Sponsor | `cosponsor_count` | Continuous | Number of co-sponsors |
| 4 | Sponsor | `is_bipartisan` | Binary | Sponsors from 2+ parties |
| 5 | Sponsor | `minority_party_sponsors` | Continuous | Count of minority party sponsors |
| 6 | Sponsor | `sponsor_party_majority` | Continuous | Fraction of sponsors in the majority party |
| 7 | Structure | `senate_origin` | Binary | Bill originated in the Senate |
| 8 | Structure | `is_resolution` | Binary | Resolution type (SJR, HJR, etc.) |
| 9 | Structure | `num_subjects` | Continuous | Number of subject/topic tags |
| 10 | Structure | `focused_scope` | Binary | 1–2 subject tags (focused bill) |
| 11 | Structure | `amends_existing_law` | Binary | Title contains amendment keywords |
| 12 | Procedural | `committee_referral` | Binary | Referred to committee |
| 13 | Procedural | `committee_passage` | Binary | Cleared committee |
| 14 | Action | `num_actions` | Continuous | Log-transformed total action count |
| 15 | Action | `early_action_count` | Continuous | Actions within first 30 days |
| 16 | Action | `days_since_introduction` | Continuous | Days between first and last action |
| 17 | Action | `session_pct_at_intro` | Continuous | Introduction timing (% into session) |
| 18 | Action | `action_density_30d` | Continuous | Actions in last 30 days |
| 19 | Text | `title_length` | Continuous | Log-transformed title word count |
| 20 | Text | `has_fiscal_note` | Binary | Fiscal/appropriation keywords found |
| 21 | Text | `solar_category_count` | Continuous | Solar keyword categories matched |
| 22 | Text | `has_solar_keywords` | Binary | Any solar keywords found |
| 23 | State | `state_passage_rate` | Continuous | Historical passage rate for the state |
| 24–73 | Text hash | `text_hash_0` through `text_hash_49` | Continuous | Feature-hashed bill text signal |

### Feature engineering notes

**Sponsor party extraction** (`_extract_party()` in `features.py`) tries 5 JSON paths to handle Open States format variation: `person.party`, `person.current_role.party`, `person.primary_party`, top-level `party`, `organization.name`. Party names are normalized to canonical codes (D, R, I, G, L, NP).

**Text hashing** uses djb2 (deterministic across runs, unlike Python's `hash()`). Tokens are signed-hashed: each token maps to a bucket index, and a second hash bit determines the sign (+1 or -1), reducing collision bias.

**Solar features** reuse the keyword dictionaries in `solar.py` (7 categories: Solar Technology, Net Metering, Incentives, Permitting, Storage, Renewable Standards, Utility/Rate Design).

## Training Process

1. Load bills from JSON/ZIP files in `src/legislator/model/data/`
2. Extract features and labels; skip bills without actions; exclude indeterminate labels (active sessions)
3. Estimate session start/end dates from per-session action date ranges
4. Compute per-state passage rates
5. Z-score normalize features (mean/std from training fold)
6. Run 5-fold cross-validation stratified by state:
   - For each of 24 hyperparameter combos, train and evaluate on each fold
   - Select best combo by mean validation F1
7. Retrain on full training set with best hyperparameters + early stopping
8. Tune classification threshold (search 0.10–0.90, maximize F1)
9. Save to `weights.json` (v2 format)

**v2 `weights.json` format** includes: `version`, `weights`, `bias`, `means`, `stds`, `feature_names`, `threshold`, `state_passage_rates`, `hyperparameters`, `cv_metrics`, `metrics`, `text_hash_buckets`, `trained_at`.

Prediction is backward-compatible with v1 weights files (checks `model.get("version", 1)`).

## Evaluation

### v2 methodology

- **5-fold stratified CV:** Folds are stratified by state (via `_stratified_kfold()`), ensuring each fold has proportional state representation. This addresses the v1 limitation of a single unstratified split.
- **Metrics reported:** Per-fold and mean +/- std for F1, precision, recall, accuracy.
- **Threshold tuning:** Instead of fixed 0.5, the optimal threshold is found by searching 0.10–0.90 in 0.01 increments, selecting the threshold that maximizes F1 on the validation set.

### What improved from v1

| Issue | v1 | v2 |
|-------|----|----|
| Feature leakage | `passed_one_chamber` encoded outcome | Removed |
| Dead features | `has_companion` always 0 | Removed |
| Evaluation | Single 80/20 split, no CV | 5-fold stratified CV |
| Regularization | L2-only, hard-coded lambda | Elastic net, grid-searched |
| Threshold | Fixed 0.5 | Tuned to maximize F1 |
| State effects | None | `state_passage_rate` feature |
| Implementation | Pure Python (~430M ops in Python) | Numpy-vectorized (~100x faster) |
| Session dates | Jan 1 – Dec 31 estimate | Estimated from action date ranges |
| Text signal | None | 50 hash bucket features |

### Remaining limitations

1. **Train-serve skew:** Model trains on Open States data but predicts on LegiScan data. Sponsor formats, action classifications, and session timing differ between sources. The extraction functions handle these differences, but mismatches in `is_bipartisan`, `minority_party_sponsors`, and action features are possible.

2. **Calibration unverified:** Predicted probabilities are displayed to users as "X% likely to pass" but have not been validated against observed frequencies. A reliability diagram would reveal whether the model is over- or under-confident.

3. **Single global model:** One model for all 50 states. Despite the `state_passage_rate` feature, the model cannot capture state-specific interaction effects (e.g., bipartisan support mattering differently in different legislatures). Per-state intercepts or a hierarchical model would help.

4. **No legislator-level features:** Sponsor identity (committee chair, majority leader, etc.) is a strong predictor in political science literature. Only counts and party labels are used.

5. **Linear decision boundary:** Logistic regression assumes linear relationships between features and log-odds. Interaction effects (e.g., bipartisan support mattering more late in session) are not captured.

6. **No temporal validation:** Cross-validation is stratified by state, not time. Training on 2023 sessions and testing on 2024 sessions would better simulate real deployment.

7. **Text hash collision analysis:** The 50-bucket hashing trick is compact but collision rates have not been formally analyzed. Some signal may be lost to collisions.

8. **`amends_existing_law` keyword noise:** "relating to" is extremely common in bill titles by convention and may introduce noise. However, L1 regularization should drive this feature's weight to zero if it's not predictive.

## Strengths

1. **Interpretable:** Per-feature contributions (weight * normalized value) are reported for every prediction, enabling the UI to show "why" a bill scored high or low.
2. **No ML dependencies:** Model is a single JSON file. Inference is multiplication + sigmoid — no scikit-learn, no TensorFlow.
3. **Large training set:** ~430K bills with 73 features. Overfitting risk is minimal.
4. **Robust evaluation:** 5-fold stratified CV with grid search provides reliable performance estimates.
5. **Graceful degradation:** Falls back to hand-tuned heuristic when no model is available.
6. **Feature sparsity via L1:** Elastic net drives irrelevant features to zero, making it clear which features actually contribute.

## Recommendations

1. **Calibration analysis:** Plot reliability diagrams. Apply Platt scaling or isotonic regression if probabilities don't match observed rates.
2. **Temporal holdout:** Train on older sessions, test on recent ones. This better simulates real-world use.
3. **Per-state metrics:** Break down F1/precision/recall by state to identify systematic biases.
4. **Interaction features:** Add products like `is_bipartisan * session_pct_at_intro` to capture nonlinear effects without changing model class.
5. **Retraining pipeline:** Automate periodic retraining as new session data becomes available.
6. **Per-state intercepts:** Add state-specific bias terms to account for interstate variation beyond the passage rate feature.
