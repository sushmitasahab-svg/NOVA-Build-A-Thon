"""
train_model.py

Stage 2-5 (plus literature-guided re-test): baseline model, broad
Logistic Regression / Random Forest, and a focused literature-guided
version, all evaluated with session-aware cross-validation for the Flip
Cup made/missed prediction task.

WHY SESSION-LEVEL GROUPED CROSS-VALIDATION:
    We have 100 trials from ONE participant across 2 sessions (50 each).
    There is no way to hold out a different person, so the least-leaky
    validation we can honestly do is: train on one session, test on the
    other. This is implemented with GroupKFold(n_splits=2), grouping by
    session_id. This means there are only 2 folds - not because we chose
    a small number, but because we only have 2 valid groups.

WHY TWO FEATURE SETS (BROAD vs. LITERATURE-GUIDED):
    Our first attempt (BROAD) let SelectKBest pick 20 features out of
    all 930, automatically, based only on what looks correlated with
    the outcome in our own small dataset. With ~50 training samples and
    930 candidates, some features can look "informative" purely by
    chance.

    The LITERATURE-GUIDED set instead picks features BEFORE looking at
    which ones score well in our data, based on published EEG motor
    performance research (e.g. Meinel et al. 2016, Frontiers in Human
    Neuroscience - pre-trial EEG predicting single-trial motor
    performance in a hand motor task). That literature points to:
      - Motor cortex channels: C3, C1, Cz, C2, C4
      - Mu/alpha band (8-13 Hz) and beta band (13-30 Hz) power, both of
        which are well-established as related to motor preparation
        (sensorimotor rhythm desynchronization before movement).
    This gives 5 channels x 4 features (alpha_power, alpha_relative,
    beta_power, beta_relative) = 20 features - deliberately a similar
    COUNT to the broad version's k=20, so any difference we see is about
    WHICH features, not how many.

    This is a hypothesis-driven test, not a search for whatever performs
    best. We are not trying multiple channel/band combinations and
    keeping whichever scores highest - that would reintroduce the same
    "found by chance" problem we're trying to avoid.

WHY RANDOM FOREST HYPERPARAMETERS:
    max_depth=5 and min_samples_leaf=3 deliberately keep trees small.
    With only ~50 training samples, deep unrestricted trees would
    memorize the training data instead of learning general patterns.

WHAT THIS SCRIPT DOES NOT DO:
    It does not search for the "best" hyperparameters or the "best"
    feature subset by trying many options and picking whichever scored
    highest on the test fold - that itself would be a form of leakage.
    These are fixed, reasoned choices, not tuned choices.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "flip_cup_features.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

ID_COLS = ["participant_id", "session_id", "trial_id", "shot_time",
           "wait_duration", "performance_label"]

# Literature-guided feature set: motor cortex channels x mu/alpha + beta
# band features (see docstring above for the reasoning and citation).
MOTOR_CHANNELS = ["C3", "C1", "Cz", "C2", "C4"]
MOTOR_BAND_FEATURE_SUFFIXES = [
    "alpha_power", "alpha_relative", "beta_power", "beta_relative",
]

# zero_division=0 makes precision/recall/f1 return 0.0 (instead of warning
# and guessing) when a fold predicts only one class - which happens with the
# majority-class baseline, so this keeps the output clean and well-defined.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
}
SCORING_NAMES = list(SCORING.keys())

RANDOM_STATE = 42


def load_dataset():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    X = df[feature_cols].to_numpy(dtype=float)
    # made=1, missed=0. We fix this mapping explicitly rather than trusting
    # alphabetical LabelEncoder ordering, so the meaning of "class 1" is
    # never ambiguous.
    y = (df["performance_label"] == "made").astype(int).to_numpy()
    groups = df["session_id"].to_numpy()

    return df, X, y, groups, feature_cols


def get_motor_feature_indices(feature_cols):
    """
    Build the list of column names (and their positions in X) for the
    literature-guided motor-cortex feature set, and verify every one of
    them actually exists in our dataset before using it.
    """
    motor_feature_names = []
    for ch in MOTOR_CHANNELS:
        for suffix in MOTOR_BAND_FEATURE_SUFFIXES:
            motor_feature_names.append(f"{ch}_{suffix}")

    missing = [name for name in motor_feature_names if name not in feature_cols]
    if missing:
        raise ValueError(f"Expected motor-cortex feature columns are missing "
                          f"from the dataset: {missing}")

    indices = [feature_cols.index(name) for name in motor_feature_names]
    return indices, motor_feature_names


def summarize_cv(name, cv_results):
    """Print per-fold and mean+-std results for one model."""
    print(f"\n--- {name} ---")
    for metric in SCORING:
        key = f"test_{metric}"
        scores = cv_results[key]
        fold_str = ", ".join(f"fold{i+1}={s:.3f}" for i, s in enumerate(scores))
        print(f"  {metric:18s}: mean={scores.mean():.3f}  std={scores.std():.3f}  ({fold_str})")


def cv_results_to_dict(name, cv_results):
    out = {"model": name}
    for metric in SCORING:
        key = f"test_{metric}"
        scores = cv_results[key]
        out[f"{metric}_mean"] = float(scores.mean())
        out[f"{metric}_std"] = float(scores.std())
        for i, s in enumerate(scores):
            out[f"{metric}_fold{i+1}"] = float(s)
    return out


def print_verdict(name, bal_acc, baseline_bal_acc):
    diff = bal_acc - baseline_bal_acc
    if diff > 0.05:
        verdict = f"performs {diff:.3f} better than baseline (balanced accuracy) - some signal detected."
    elif diff < -0.05:
        verdict = f"performs {abs(diff):.3f} WORSE than baseline (balanced accuracy)."
    else:
        verdict = "is within +/-0.05 of baseline (balanced accuracy) - no clear evidence of predictive signal."
    print(f"{name}: {verdict}")


def main():
    df, X, y, groups, feature_cols = load_dataset()
    motor_indices, motor_feature_names = get_motor_feature_indices(feature_cols)
    X_motor = X[:, motor_indices]

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Samples: {X.shape[0]}   Broad features: {X.shape[1]}   "
          f"Literature-guided features: {X_motor.shape[1]}")
    print(f"Target distribution: made={y.sum()}  missed={(y == 0).sum()}")
    print(f"Groups (sessions): {sorted(set(groups.tolist()))}")
    print(f"Literature-guided feature list: {motor_feature_names}")

    n_splits = len(set(groups.tolist()))
    gkf = GroupKFold(n_splits=n_splits)

    # ---- Baseline: always predict the majority class ----
    baseline = DummyClassifier(strategy="most_frequent")
    baseline_cv = cross_validate(baseline, X, y, groups=groups, cv=gkf, scoring=SCORING)
    summarize_cv("Baseline (majority class)", baseline_cv)

    # ---- BROAD Model 1: Logistic Regression, SelectKBest over all 930 ----
    logreg_broad = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=20)),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
    ])
    logreg_broad_cv = cross_validate(logreg_broad, X, y, groups=groups, cv=gkf, scoring=SCORING)
    summarize_cv("Logistic Regression (broad, 930->20 auto-selected)", logreg_broad_cv)

    # ---- BROAD Model 2: Random Forest over all 930 ----
    rf_broad = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                        min_samples_leaf=3, random_state=RANDOM_STATE)),
    ])
    rf_broad_cv = cross_validate(rf_broad, X, y, groups=groups, cv=gkf, scoring=SCORING)
    summarize_cv("Random Forest (broad, all 930)", rf_broad_cv)

    # ---- LITERATURE-GUIDED Model 1: Logistic Regression on motor features only ----
    logreg_motor = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
    ])
    logreg_motor_cv = cross_validate(logreg_motor, X_motor, y, groups=groups, cv=gkf, scoring=SCORING)
    summarize_cv("Logistic Regression (literature-guided, 20 motor features)", logreg_motor_cv)

    # ---- LITERATURE-GUIDED Model 2: Random Forest on motor features only ----
    rf_motor = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                        min_samples_leaf=3, random_state=RANDOM_STATE)),
    ])
    rf_motor_cv = cross_validate(rf_motor, X_motor, y, groups=groups, cv=gkf, scoring=SCORING)
    summarize_cv("Random Forest (literature-guided, 20 motor features)", rf_motor_cv)

    # ---- Comparison table ----
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (mean across folds)")
    print("=" * 60)
    comparison_rows = [
        cv_results_to_dict("baseline_majority_class", baseline_cv),
        cv_results_to_dict("logreg_broad_930to20", logreg_broad_cv),
        cv_results_to_dict("rf_broad_930", rf_broad_cv),
        cv_results_to_dict("logreg_literature_guided_20", logreg_motor_cv),
        cv_results_to_dict("rf_literature_guided_20", rf_motor_cv),
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    display_cols = ["model"] + [f"{m}_mean" for m in SCORING]
    print(comparison_df[display_cols].to_string(index=False))

    # ---- Save metrics ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved metrics to: {METRICS_DIR / 'model_comparison.csv'}")

    # ---- Save fitted pipelines (fit on ALL 100 trials, for later use/inspection) ----
    # NOTE: these "final fit" models are for saving/inspection and later use in
    # predict.py. The performance numbers we trust are the cross-validation
    # scores above, NOT this final fit's training accuracy.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logreg_broad.fit(X, y)
    rf_broad.fit(X, y)
    logreg_motor.fit(X_motor, y)
    rf_motor.fit(X_motor, y)
    joblib.dump(logreg_broad, MODELS_DIR / "logistic_regression_broad_pipeline.joblib")
    joblib.dump(rf_broad, MODELS_DIR / "random_forest_broad_pipeline.joblib")
    joblib.dump(logreg_motor, MODELS_DIR / "logistic_regression_literature_pipeline.joblib")
    joblib.dump(rf_motor, MODELS_DIR / "random_forest_literature_pipeline.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns_broad.joblib")
    joblib.dump(motor_feature_names, MODELS_DIR / "feature_columns_literature.joblib")
    print(f"Saved fitted pipelines to: {MODELS_DIR}")

    # ---- Honest interpretation flags ----
    print("\n" + "=" * 60)
    print("INTERPRETATION CHECKS")
    print("=" * 60)
    baseline_bal_acc = baseline_cv["test_balanced_accuracy"].mean()
    print_verdict("Logistic Regression (broad)", logreg_broad_cv["test_balanced_accuracy"].mean(), baseline_bal_acc)
    print_verdict("Random Forest (broad)", rf_broad_cv["test_balanced_accuracy"].mean(), baseline_bal_acc)
    print_verdict("Logistic Regression (literature-guided)", logreg_motor_cv["test_balanced_accuracy"].mean(), baseline_bal_acc)
    print_verdict("Random Forest (literature-guided)", rf_motor_cv["test_balanced_accuracy"].mean(), baseline_bal_acc)

    print("\nRemember: with only 2 session-based folds, these numbers are based on")
    print("just 2 train/test splits (50 vs 50 trials each). Treat any single")
    print("fold's score with caution - look at both fold values above, not just the mean.")


if __name__ == "__main__":
    main()